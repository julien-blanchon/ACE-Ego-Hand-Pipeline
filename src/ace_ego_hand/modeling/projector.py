"""The hand readout: block-15 DiT features of a window -> per-frame bimanual MANO parameters.

The projector turns the `(B, 3072, G, H, W)` tap of the trunk into hands. It is the one released
configuration of the upstream projector, written as a single path:

1. **Patch tokens.** Every latent cell is projected to the width (384) and gets a learned 16x16
   spatial encoding resized to the grid plus a ray encoding built from the *predicted* ray field
   (a 1x1 head on the tap, pooled over frames): the readout is camera-aware without ever seeing K.
2. **Alternating encoder.** 48 special tokens per latent frame (2 hand, 42 joint, 4 register)
   cross-attend the frame's patch grid, then self-attend across the whole clip with rotary
   relative frame positions, four rounds. Spatial detail stays reachable at every round instead
   of being pooled away once.
3. **Readout.** The refined joint tokens attend the grid once more; the attention map is a
   per-joint heatmap whose soft-argmax gives the 2D anchor, and the attended feature gives the
   wrist-relative 3D joint. Hand tokens go through small heads for global orientation and
   articulation (6D rotations), presence, visibility and the camera anchor (image position plus
   log depth). One shape vector per hand is read from the hand features pooled over the clip.
   Everything is linearly interpolated from latent frames to video frames.
4. **Placement.** The canonical MANO joints of every frame are decoded and the camera-space
   translation is solved against the 2D anchors (`camera.solve_translation`), through the
   calibration (K variant) or the predicted rays (K-free variant).

Shapes: G latent frames, F = 4 (G - 1) + 1 video frames, S = 2 hands (slot 0 left, slot 1 right),
J = 21 joints in OpenPose order. Runs in float32.
Ref: ACE-Ego-Hand arXiv:2608.20308
# Adapted from https://github.com/ggxxii/ACE-Ego-Hand/blob/main/ace_ego_hand/archs/projector/memory_projector.py
# and https://github.com/ggxxii/ACE-Ego-Hand/blob/main/ace_ego_hand/archs/projector/memory_encoder.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, TypedDict, Unpack, override

import torch
import torch.nn.functional as F
from einops import einsum, pack, rearrange, reduce, repeat, unpack
from torch import nn

from ..types import (
    CameraRaw,
    ClipSpecialTokens,
    GlobalOrient,
    HandBetas,
    HandFeatures,
    HandPose,
    HandProbability,
    HandTranslation,
    IntrinsicsBatch,
    JointHeatmaps,
    Joints2D,
    JointsRootRelative,
    LatentHandFeatures,
    PatchTokens,
    RayField,
    RayMap,
    RotaryTable,
    Rotation6D,
    RotationMatrices,
    SpatialEncoding,
    SpecialTokens,
    TapFeatures,
)
from ..utils.hub import HubModule
from ..wan.dit import apply_rotary
from .camera import RayPositionalEncoding, solve_translation
from .mano import ManoLayer

NUM_HANDS = 2
NUM_JOINTS = 21
NUM_ARTICULATIONS = 15  # MANO finger joints
ROTATION_6D = 6
NUM_BETAS = 10
ROPE_THETA = 10_000.0


class Placement(TypedDict):
    """The `device` / `dtype` pair every submodule is built with."""

    device: torch.device | None
    dtype: torch.dtype | None


@dataclass(frozen=True, slots=True)
class ProjectorConfig:
    """Quantities of the readout; defaults are the released ACE-Ego-Hand projector."""

    feature_dim: int = 3072  # width of the trunk's token stream
    hidden_dim: int = 384
    num_heads: int = 8
    ffn_mult: int = 4
    num_layers: int = 4  # alternating rounds
    num_registers: int = 4
    spatial_encoding_size: int = 16  # side of the learned spatial encoding grid
    ray_frequency_bands: int = 8

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def head_hidden(self) -> int:
        return max(128, self.hidden_dim // 2)

    @property
    def num_special(self) -> int:
        return NUM_HANDS + NUM_HANDS * NUM_JOINTS + self.num_registers


@dataclass(frozen=True, slots=True)
class HandPrediction:
    """Per-frame hand parameters of a batch of windows, in the encode-resolution image."""

    global_orient: GlobalOrient
    hand_pose: HandPose
    betas: HandBetas
    translation: HandTranslation
    joints_2d: Joints2D  # normalized [0, 1] over the encode image
    presence: HandProbability  # the hand exists in the frame (3D head)
    visible: HandProbability  # the hand is in view (2D head)
    ray_map: RayMap  # predicted per-cell rays, kept for the K-free camera fit


def rotation_6d_to_matrix(rotation: Rotation6D) -> RotationMatrices:
    """Gram-Schmidt the two 3-vectors of a 6D rotation into the columns of a rotation matrix."""

    a1, a2 = rotation[..., :3].float(), rotation[..., 3:].float()
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def _head(
    in_dim: int, hidden: int, out_dim: int, *, norm: bool, **placement: Unpack[Placement]
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.LayerNorm(in_dim, **placement)] if norm else []
    layers += [
        nn.Linear(in_dim, hidden, **placement),
        nn.GELU(),
        nn.Linear(hidden, out_dim, **placement),
    ]
    return nn.Sequential(*layers)


class CrossAttention(nn.Module):
    """Multi-head attention of query tokens over a key/value grid, with separate projections.

    `forward` returns the attended features; `forward_with_heatmap` also returns the attention
    weights averaged over heads, which the joint readout uses as a heatmap.
    """

    def __init__(
        self, dim: int, num_heads: int, *, device: torch.device | None, dtype: torch.dtype | None
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.key = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.value = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.output = nn.Linear(dim, dim, device=device, dtype=dtype)

    def _heads(
        self, query: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(
            rearrange(project(x), "b n (h d) -> b h n d", h=self.num_heads)
            for project, x in ((self.query, query), (self.key, context), (self.value, context))
        )

    @override
    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q, k, v = self._heads(query, context)
        attended = F.scaled_dot_product_attention(q, k, v)
        return self.output(rearrange(attended, "b h n d -> b n (h d)"))

    def forward_with_heatmap(
        self, query: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, JointHeatmaps]:
        q, k, v = self._heads(query, context)
        # Explicit softmax: the weights themselves are the readout's heatmap
        scores = einsum(q, k, "b h n d, b h m d -> b h n m") / (q.shape[-1] ** 0.5)
        weights = scores.softmax(dim=-1)
        attended = einsum(weights, v, "b h n m, b h m d -> b h n d")
        return self.output(rearrange(attended, "b h n d -> b n (h d)")), weights.mean(dim=1)


class TemporalLayer(nn.Module):
    """Pre-norm self-attention over all special tokens of a clip with rotary frame positions, plus an MLP.

    Tokens of one frame share a position, so attention is relative in frame distance and
    unconstrained in length.
    """

    def __init__(
        self, config: ProjectorConfig, *, device: torch.device | None, dtype: torch.dtype | None
    ) -> None:
        super().__init__()
        dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.norm1 = nn.LayerNorm(dim, device=device, dtype=dtype)
        self.query = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.key = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.value = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.output = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(dim, device=device, dtype=dtype)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * config.ffn_mult, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(dim * config.ffn_mult, dim, device=device, dtype=dtype),
        )

    @override
    def forward(
        self, tokens: ClipSpecialTokens, rotary: tuple[RotaryTable, RotaryTable]
    ) -> ClipSpecialTokens:
        normed = self.norm1(tokens)
        q, k, v = (
            rearrange(project(normed), "b n (h d) -> b n h d", h=self.num_heads)
            for project in (self.query, self.key, self.value)
        )
        q, k = apply_rotary(q, *rotary), apply_rotary(k, *rotary)
        attended = F.scaled_dot_product_attention(
            rearrange(q, "b n h d -> b h n d"),
            rearrange(k, "b n h d -> b h n d"),
            rearrange(v, "b n h d -> b h n d"),
        )
        tokens = tokens + self.output(rearrange(attended, "b h n d -> b n (h d)"))
        return tokens + self.mlp(self.norm2(tokens))


class AlternatingLayer(nn.Module):
    """One round: special tokens read their frame's patch grid, then exchange across the clip."""

    def __init__(
        self, config: ProjectorConfig, *, device: torch.device | None, dtype: torch.dtype | None
    ) -> None:
        super().__init__()
        dim = config.hidden_dim
        self.query_norm = nn.LayerNorm(dim, device=device, dtype=dtype)
        self.context_norm = nn.LayerNorm(dim, device=device, dtype=dtype)
        self.spatial = CrossAttention(dim, config.num_heads, device=device, dtype=dtype)
        self.temporal = TemporalLayer(config, device=device, dtype=dtype)

    @override
    def forward(
        self,
        special: SpecialTokens,
        patches: PatchTokens,
        *,
        frames: int,
        rotary: tuple[RotaryTable, RotaryTable],
    ) -> SpecialTokens:
        special = special + self.spatial(self.query_norm(special), self.context_norm(patches))
        clip = rearrange(special, "(b g) k d -> b (g k) d", g=frames)
        clip = self.temporal(clip, rotary)
        return rearrange(clip, "b (g k) d -> (b g) k d", g=frames)


class HandProjector(nn.Module, HubModule):
    """`(tap features, frame count) -> HandPrediction`; see the module docstring."""

    config_class: ClassVar[type] = ProjectorConfig

    def __init__(
        self,
        config: ProjectorConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        dim, hidden = config.hidden_dim, config.head_hidden
        common = Placement(device=device, dtype=dtype)

        self.ray_head = nn.Conv3d(config.feature_dim, 3, kernel_size=1, **common)
        self.patch_projection = nn.Sequential(
            nn.Linear(config.feature_dim, dim, **common), nn.LayerNorm(dim, **common)
        )
        side = config.spatial_encoding_size
        self.spatial_encoding = nn.Parameter(torch.randn(1, dim, side, side, **common) * 0.02)
        self.ray_encoding = RayPositionalEncoding(
            dim, num_bands=config.ray_frequency_bands, **common
        )

        # Special tokens: [hand (2) | joint (42) | register (4)] per latent frame
        self.hand_token = nn.Parameter(torch.randn(1, NUM_HANDS, dim, **common) * 0.02)
        self.joint_token = nn.Parameter(
            torch.randn(1, NUM_HANDS * NUM_JOINTS, dim, **common) * 0.02
        )
        self.register_token = nn.Parameter(
            torch.randn(1, config.num_registers, dim, **common) * 0.02
        )
        self.slot_embedding = nn.Parameter(torch.randn(NUM_HANDS, dim, **common) * 0.02)
        self.layers = nn.ModuleList(
            [AlternatingLayer(config, **common) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(dim, **common)

        # Joint readout over the grid and the wrist-relative 3D head
        self.readout_query_norm = nn.LayerNorm(dim, **common)
        self.readout_context_norm = nn.LayerNorm(dim, **common)
        self.readout = CrossAttention(dim, config.num_heads, **common)
        self.head_joint_3d = _head(dim, hidden, 3, norm=True, **common)

        # Per-hand heads on the (fused) hand features
        self.hand_fuse = nn.Sequential(
            nn.LayerNorm(dim, **common), nn.Linear(dim, dim, **common), nn.GELU()
        )
        self.head_global_orient = _head(dim, hidden, ROTATION_6D, norm=False, **common)
        self.head_hand_pose = _head(
            dim, hidden, NUM_ARTICULATIONS * ROTATION_6D, norm=False, **common
        )
        self.head_camera = _head(dim, hidden, 3, norm=False, **common)
        self.head_visible = _head(dim, hidden, 1, norm=False, **common)
        self.head_presence = _head(dim, hidden, 1, norm=False, **common)
        self.head_betas = _head(dim, hidden, NUM_BETAS, norm=True, **common)

    # --- positional encodings ---

    def spatial_encoding_for(self, height: int, width: int) -> SpatialEncoding:
        """Bilinearly resize the learned grid to the token grid, in `(1, HW, D)` layout."""

        grid = F.interpolate(
            self.spatial_encoding.float(), size=(height, width), mode="bilinear", align_corners=True
        )
        return rearrange(grid, "1 d h w -> 1 (h w) d").to(self.spatial_encoding.dtype)

    def rotary_for(self, frames: int, device: torch.device) -> tuple[RotaryTable, RotaryTable]:
        """Cos / sin tables indexed by frame for the `frames * num_special` token sequence."""

        head_dim = self.config.head_dim
        positions = torch.arange(frames, device=device, dtype=torch.float32)
        frequencies = 1.0 / (
            ROPE_THETA ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        )
        angles = torch.outer(positions, frequencies)  # (G, head_dim / 2)
        per_token = repeat(angles, "g d -> (g k) d", k=self.config.num_special)
        return per_token.cos(), per_token.sin()

    # --- forward ---

    def encode_tokens(
        self, features: TapFeatures
    ) -> tuple[PatchTokens, SpecialTokens, RayField, RayMap]:
        """Project the tap onto patch tokens with their encodings and run the alternating rounds."""

        b, _, g, h, w = features.shape
        ray_map = self.ray_head(features)
        rays = reduce(ray_map, "b c g h w -> b c h w", "mean")  # the camera is constant per clip

        patches = self.patch_projection(rearrange(features, "b d g h w -> (b g) (h w) d"))
        patches = patches + self.spatial_encoding_for(h, w)
        patches = patches + repeat(self.ray_encoding(rays), "b n d -> (b g) n d", g=g)

        hands = self.hand_token + self.slot_embedding[None]
        special = torch.cat([hands, self.joint_token, self.register_token], dim=1)
        special = repeat(special, "1 k d -> (b g) k d", b=b, g=g)
        rotary = self.rotary_for(g, features.device)
        for layer in self.layers:
            special = layer(special, patches, frames=g, rotary=rotary)
        return patches, self.final_norm(special), rays, ray_map

    def read_joints(
        self, joint_tokens: SpecialTokens, patches: PatchTokens, height: int, width: int
    ) -> tuple[Joints2D, JointsRootRelative]:
        """Soft-argmax 2D anchors and wrist-relative 3D joints from the joint tokens, per latent frame."""

        attended, heatmap = self.readout.forward_with_heatmap(
            self.readout_query_norm(joint_tokens), self.readout_context_norm(patches)
        )
        heatmap = rearrange(heatmap, "n q (h w) -> n q h w", h=height, w=width)
        us = torch.linspace(0.0, 1.0, width, device=heatmap.device, dtype=heatmap.dtype)
        vs = torch.linspace(0.0, 1.0, height, device=heatmap.device, dtype=heatmap.dtype)
        mass = reduce(heatmap, "n q h w -> n q", "sum").clamp_min(1e-6)
        u = (reduce(heatmap, "n q h w -> n q w", "sum") * us).sum(-1) / mass
        v = (reduce(heatmap, "n q h w -> n q h", "sum") * vs).sum(-1) / mass
        joints_2d = rearrange(torch.stack([u, v], dim=-1), "n (s j) two -> n s j two", s=NUM_HANDS)
        joints_3d = rearrange(self.head_joint_3d(attended), "n (s j) c -> n s j c", s=NUM_HANDS)
        joints_3d = joints_3d - joints_3d[:, :, :1]
        return joints_2d, joints_3d

    @override
    def forward(
        self,
        features: TapFeatures,
        *,
        num_frames: int,
        mano: ManoLayer,
        image_size: tuple[int, int],
        intrinsics: IntrinsicsBatch | None,
    ) -> HandPrediction:
        """Decode one batch of windows into hands over `num_frames` video frames.

        `intrinsics` (per window, in encode pixels) selects the K variant of the translation
        solve; `None` selects the K-free solve through the predicted rays. `image_size` is the
        encode `(width, height)`, needed by both.
        """

        b, _, _, h, w = features.shape
        patches, special, rays, ray_map = self.encode_tokens(features)
        hand_latent: LatentHandFeatures = rearrange(
            special[:, :NUM_HANDS], "(b g) s d -> b g s d", b=b
        )
        joint_tokens = special[:, NUM_HANDS : NUM_HANDS + NUM_HANDS * NUM_JOINTS]
        joints_2d_latent, joints_3d_latent = self.read_joints(joint_tokens, patches, h, w)

        # Latent frames -> video frames: features and coordinates interpolate linearly in time
        hand_features: HandFeatures = _interpolate_frames(hand_latent, num_frames)
        joints_2d = _interpolate_frames(
            rearrange(joints_2d_latent, "(b g) s j c -> b g s j c", b=b), num_frames
        )
        hand_features = hand_features + self.hand_fuse(hand_features)

        global_orient = rotation_6d_to_matrix(self.head_global_orient(hand_features))
        hand_pose = rotation_6d_to_matrix(
            rearrange(self.head_hand_pose(hand_features), "b f s (p r) -> b f s p r", r=ROTATION_6D)
        )
        camera_raw: CameraRaw = self.head_camera(hand_features)
        visible = torch.sigmoid(self.head_visible(hand_features)[..., 0])
        presence = torch.sigmoid(self.head_presence(hand_features)[..., 0])
        betas: HandBetas = self.head_betas(reduce(hand_features, "b f s d -> b s d", "mean"))

        canonical = mano(
            global_orient, hand_pose, repeat(betas, "b s k -> b f s k", f=num_frames)
        ).joints
        translation = solve_translation(
            camera_raw,
            canonical,
            joints_2d,
            intrinsics=intrinsics,
            rays=None if intrinsics is not None else rays,
            image_size=image_size,
        )
        del joints_3d_latent  # the wrist-relative 3D head is not part of the MANO output
        return HandPrediction(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=betas,
            translation=translation,
            joints_2d=joints_2d,
            presence=presence,
            visible=visible,
            ray_map=ray_map,
        )


def _interpolate_frames(values: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Linear interpolation of `(B, G, ...)` from latent frames to video frames, ends aligned."""

    if values.shape[1] == num_frames:
        return values
    # Fold the trailing axes into one channel axis for `interpolate`, then unfold them again
    flat, trailing = pack([rearrange(values, "b g ... -> b ... g")], "b * g")
    resampled = F.interpolate(flat, size=num_frames, mode="linear", align_corners=True)
    [interpolated] = unpack(rearrange(resampled, "b n f -> b f n"), trailing, "b f *")
    return interpolated
