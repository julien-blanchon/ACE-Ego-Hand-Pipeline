"""Wan2.2-5B diffusion transformer trunk: latent video in, block-15 token features out.

ACE-Ego-Hand reads hand geometry from the hidden state of a video diffusion transformer rather
than from its denoising output, so this module is the Wan2.2 TI2V-5B trunk cut down to what that
readout needs: the patch embedding, the timestep and text embedders, and the first 16 of the 30
blocks. There is no output head and no unpatchify; the pruned checkpoint holds exactly what is
built here. The trunk runs in a fixed regime -- clean latents, timestep zero, one constant text
context -- and every forward is deterministic.

Shape walkthrough for one 85-frame window at 832x480 (one token per 2x2 latent cells):

    latents    (B, 48, 22, 30, 52)   normalized VAE latents, 16x spatial and 4x temporal
    patchify   Conv3d(148 -> 3072, kernel = stride = (1, 2, 2)); see `latent_channels` below
    tokens     (B, 8580, 3072)       22 * 15 * 26 tokens in (frame, height, width) raster order
    16 blocks  (B, 8580, 3072)       self-attention over the grid + cross-attention to the caption
    output     the raw residual stream after block 15 (no final norm), folded back onto the grid

The checkpoint's patch embedding takes 148 channels: 48 for the latent and 100 for the
image-control branch of the Fun-Control release. In this pipeline the control channels are always
zero, so the convolution is evaluated over the first 48 input channels only; the weight keeps its
full shape so it loads unchanged.

Position is 3D rotary (`RotaryEmbedding3D`): the 64 rotation pairs of a 128-wide head are split
22 / 21 / 21 across the frame, height and width axes over integer grid indices. Timestep
conditioning is adaLN-single: one shared MLP maps the timestep to six vectors (shift, scale, gate
for attention and for the feed-forward) and each block adds a small learned table to them. Text
enters through ungated cross-attention in every block.

Numerics follow the upstream inference path: the residual stream starts in the parameter dtype
(bf16) and is promoted to float32 by the first gated residual add; layer norms, adaLN arithmetic
and the rotary rotation run in float32; projections and attention run in bf16. The timestep MLP
runs in float32 with upcast weights, as upstream's autocast island does.

Ref: Wan arXiv:2503.20314; ACE-Ego-Hand arXiv:2608.20308
# Adapted from https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py and
# https://github.com/aigc-apps/VideoX-Fun/blob/main/videox_fun/models/wan_transformer3d.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Self, cast, override

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from ..types import (
    AttentionHeads,
    LatentGrid,
    Modulation,
    RotaryTable,
    TapFeatures,
    TextContext,
    TimeEmbedding,
    Timesteps,
    Tokens,
    VideoLatents,
)
from ..utils.hub import HubModule

NUM_BLOCK_MODULATIONS = 6  # shift, scale, gate for self-attention, then for the feed-forward
ROPE_MAX_POSITIONS = 1024  # longest axis (latent frames or patches) the rotary tables cover
ROPE_THETA = 10_000.0
TIMESTEP_MAX_PERIOD = 10_000.0


@dataclass(frozen=True, slots=True)
class WanDiTConfig:
    """Architecture quantities of the Wan2.2-5B trunk; defaults are the pruned ACE checkpoint."""

    in_channels: int = 148  # 48 latent + 100 control channels in the patch embedding weight
    latent_channels: int = 48  # the channels actually fed; the rest are zero in this pipeline
    dim: int = 3072
    ffn_dim: int = 14336
    num_heads: int = 24
    num_layers: int = 16  # blocks 0..15 of the 30-block model; block 15 is the feature tap
    freq_dim: int = 256  # sinusoidal timestep features
    text_dim: int = 4096  # umT5-XXL hidden size
    text_length: int = 512  # the fixed context length the model was trained with
    patch_size: tuple[int, int, int] = (1, 2, 2)
    eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads


def sinusoidal_embedding(timesteps: Timesteps, dim: int) -> TimeEmbedding:
    """Embed timesteps as `[cos | sin]` over geometrically spaced frequencies, in float32."""

    half = dim // 2
    exponent = -math.log(TIMESTEP_MAX_PERIOD) * torch.arange(half, device=timesteps.device) / half
    angles = timesteps.float()[:, None] * torch.exp(exponent)[None, :]
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


def _linear_fp32(x: Tokens, layer: nn.Linear) -> Tokens:
    """Apply a linear layer in float32 whatever dtype its weights are stored in."""

    return F.linear(x.float(), layer.weight.float(), layer.bias.float())


class LayerNorm(nn.Module):
    """LayerNorm evaluated in float32; affine only where the checkpoint carries a weight.

    Non-affine instances feed adaLN, which supplies the scale and shift itself, and return
    float32 so the modulation happens at full precision. The affine cross-attention norm rounds
    its input to the parameter dtype first, as upstream does, and returns the input dtype.
    """

    weight: nn.Parameter | None
    bias: nn.Parameter | None

    def __init__(
        self,
        dim: int,
        *,
        eps: float,
        affine: bool,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.affine = affine
        weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype)) if affine else None
        bias = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype)) if affine else None
        self.register_parameter("weight", weight)
        self.register_parameter("bias", bias)

    @override
    def forward(self, x: Tokens) -> Tokens:
        if not self.affine:
            return F.layer_norm(x.float(), (self.dim,), None, None, self.eps)
        weight, bias = cast(nn.Parameter, self.weight), cast(nn.Parameter, self.bias)
        rounded = x.to(weight.dtype).float()
        return F.layer_norm(rounded, (self.dim,), weight.float(), bias.float(), self.eps).to(
            x.dtype
        )


class RMSNorm(nn.Module):
    """RMS normalization in float32 with a learned gain, returned in the input dtype (qk-norm)."""

    def __init__(
        self,
        dim: int,
        *,
        eps: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    @override
    def forward(self, x: Tokens) -> Tokens:
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


class RotaryEmbedding3D(nn.Module):
    """Rotary position embedding over the `(frame, height, width)` token grid.

    A token's rotation angles are its frame index in the first 22 pairs of the head, its row in
    the next 21 and its column in the last 21, each axis with its own geometric frequency ladder.
    Angles are tabulated once in float64 and kept as float32 cos / sin buffers; `forward` slices
    and broadcasts them onto the current grid and caches the result per grid.
    """

    cos: RotaryTable
    sin: RotaryTable

    def __init__(self, head_dim: int, *, device: torch.device | None = None) -> None:
        super().__init__()
        spatial_pairs = head_dim // 6
        self.pairs_per_axis = (head_dim // 2 - 2 * spatial_pairs, spatial_pairs, spatial_pairs)

        positions = torch.arange(ROPE_MAX_POSITIONS, dtype=torch.float64)
        angles = torch.cat(
            [_rotary_angles(positions, 2 * pairs) for pairs in self.pairs_per_axis], dim=1
        )
        self.register_buffer("cos", angles.cos().float().to(device), persistent=False)
        self.register_buffer("sin", angles.sin().float().to(device), persistent=False)
        self._cached_grid: LatentGrid | None = None
        self._cached_tables: tuple[RotaryTable, RotaryTable] | None = None

    @override
    def forward(self, grid: LatentGrid) -> tuple[RotaryTable, RotaryTable]:
        cached = self._cached_tables
        if self._cached_grid != grid or cached is None or cached[0].device != self.cos.device:
            cached = (
                _grid_table(self.cos, grid, self.pairs_per_axis),
                _grid_table(self.sin, grid, self.pairs_per_axis),
            )
            self._cached_grid = grid
            self._cached_tables = cached
        return cached


def _rotary_angles(positions: torch.Tensor, dim: int) -> RotaryTable:
    frequencies = 1.0 / ROPE_THETA ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    return torch.outer(positions, frequencies)


def _grid_table(
    table: RotaryTable, grid: LatentGrid, pairs_per_axis: tuple[int, int, int]
) -> RotaryTable:
    # Broadcast each axis' angles over the other two, laid out in (f, h, w) raster order
    f, h, w = grid
    per_axis = torch.split(table, list(pairs_per_axis), dim=1)
    return torch.cat(
        [
            repeat(per_axis[0][:f], "f d -> (f h w) d", h=h, w=w),
            repeat(per_axis[1][:h], "h d -> (f h w) d", f=f, w=w),
            repeat(per_axis[2][:w], "w d -> (f h w) d", f=f, h=h),
        ],
        dim=-1,
    )


def apply_rotary(x: AttentionHeads, cos: RotaryTable, sin: RotaryTable) -> AttentionHeads:
    """Rotate adjacent `(even, odd)` channel pairs of a `(B, N, heads, D)` tensor in float32."""

    pairs = rearrange(x.float(), "b n h (d two) -> b n h d two", two=2)
    x1, x2 = pairs.unbind(dim=-1)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rearrange(rotated, "b n h d two -> b n h (d two)").type_as(x)


class Attention(nn.Module):
    """Multi-head attention with RMS-normalized queries and keys.

    Self-attention passes the token stream as `context` with the rotary tables; cross-attention
    passes the text context and no tables. One `scaled_dot_product_attention` call either way.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        eps: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.key = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.value = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.output = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.query_norm = RMSNorm(dim, eps=eps, device=device, dtype=dtype)
        self.key_norm = RMSNorm(dim, eps=eps, device=device, dtype=dtype)

    @override
    def forward(
        self,
        x: Tokens,
        context: Tokens | TextContext,
        rotary: tuple[RotaryTable, RotaryTable] | None = None,
    ) -> Tokens:
        query = self.query_norm(self.query(x))
        key = self.key_norm(self.key(context))
        value = self.value(context)
        query, key, value = (
            rearrange(t, "b n (h d) -> b n h d", h=self.num_heads) for t in (query, key, value)
        )
        if rotary is not None:
            query = apply_rotary(query, *rotary)
            key = apply_rotary(key, *rotary)

        attended = F.scaled_dot_product_attention(
            rearrange(query, "b n h d -> b h n d"),
            rearrange(key, "b n h d -> b h n d"),
            rearrange(value, "b n h d -> b h n d"),
        )
        return self.output(rearrange(attended, "b h n d -> b n (h d)"))


class WanBlock(nn.Module):
    """One DiT block: adaLN self-attention, text cross-attention, adaLN feed-forward.

    Self-attention and the feed-forward are wrapped in adaLN (normalize, scale and shift by the
    timestep's vectors, add back through a gate); cross-attention is normalized and added ungated.
    The projections consume the parameter dtype; the residual adds happen in float32.
    """

    def __init__(
        self,
        config: WanDiTConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dim, eps = config.dim, config.eps
        self.self_attention_norm = LayerNorm(dim, eps=eps, affine=False, device=device, dtype=dtype)
        self.self_attention = Attention(dim, config.num_heads, eps=eps, device=device, dtype=dtype)
        self.cross_attention_norm = LayerNorm(dim, eps=eps, affine=True, device=device, dtype=dtype)
        self.cross_attention = Attention(dim, config.num_heads, eps=eps, device=device, dtype=dtype)
        self.feed_forward_norm = LayerNorm(dim, eps=eps, affine=False, device=device, dtype=dtype)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, config.ffn_dim, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.ffn_dim, dim, device=device, dtype=dtype),
        )
        # Per-block adaLN table added to the shared timestep modulation
        self.modulation = nn.Parameter(
            torch.randn(1, NUM_BLOCK_MODULATIONS, dim, device=device, dtype=dtype) / math.sqrt(dim)
        )

    @property
    def compute_dtype(self) -> torch.dtype:
        return self.modulation.dtype

    @override
    def forward(
        self,
        tokens: Tokens,
        context: TextContext,
        modulation: Modulation,
        rotary: tuple[RotaryTable, RotaryTable],
    ) -> Tokens:
        dtype = self.compute_dtype
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (
            self.modulation.float() + modulation
        ).chunk(NUM_BLOCK_MODULATIONS, dim=1)

        # 1. Self-attention over the whole (f, h, w) grid, modulated in and gated out. The norm
        # output takes the stream's dtype before modulation, which only matters for block 0,
        # where the stream is still bf16; the gated add then promotes the stream to float32
        normed = self.self_attention_norm(tokens).type_as(tokens) * (1 + scale_a) + shift_a
        attended = self.self_attention(normed.to(dtype), normed.to(dtype), rotary)
        tokens = tokens + attended.float() * gate_a

        # 2. Cross-attention to the text context: normalized, neither modulated nor gated
        normed = self.cross_attention_norm(tokens)
        tokens = tokens + self.cross_attention(normed.to(dtype), context)

        # 3. Feed-forward, modulated and gated by the second half of the vectors
        normed = self.feed_forward_norm(tokens).type_as(tokens) * (1 + scale_f) + shift_f
        return tokens + self.feed_forward(normed.to(dtype)).float() * gate_f


class WanDiT(nn.Module, HubModule):
    """Wan2.2-5B trunk: `(latents, timesteps, text context) -> block-15 token stream`."""

    config_class: ClassVar[type] = WanDiTConfig

    def __init__(
        self,
        config: WanDiTConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dim = config.dim
        self.config = config

        self.patch_embedding = nn.Conv3d(
            config.in_channels,
            dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            device=device,
            dtype=dtype,
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(config.text_dim, dim, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim, device=device, dtype=dtype),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(config.freq_dim, dim, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(dim, dim, device=device, dtype=dtype),
        )
        self.time_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, NUM_BLOCK_MODULATIONS * dim, device=device, dtype=dtype),
        )
        self.rotary = RotaryEmbedding3D(config.head_dim, device=device)
        self.blocks = nn.ModuleList(
            [WanBlock(config, device=device, dtype=dtype) for _ in range(config.num_layers)]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embedding.weight.dtype

    def grid(self, latents: VideoLatents) -> LatentGrid:
        """Token grid `(f, h, w)` that `patch_embedding` produces for this latent tensor."""

        _, _, f, h, w = latents.shape
        pt, ph, pw = self.config.patch_size
        return (f // pt, h // ph, w // pw)

    def compile_blocks(self) -> Self:
        """Compile the repeated block in place; the token count stays dynamic across grids."""

        for block in self.blocks:
            block.compile(dynamic=True)
        return self

    def patchify(self, latents: VideoLatents) -> Tokens:
        """Embed the latent video as tokens in (frame, height, width) raster order.

        Only the first `latent_channels` input channels of the convolution are evaluated: the
        remaining control channels are identically zero in this pipeline, so their contribution
        is the bias alone. The stream starts in the parameter dtype, as upstream.
        """

        conv = self.patch_embedding
        weight = conv.weight[:, : self.config.latent_channels]
        patches = F.conv3d(latents.to(weight.dtype), weight, conv.bias, stride=conv.stride)
        return rearrange(patches, "b d f h w -> b (f h w) d")

    def timestep_modulation(self, timesteps: Timesteps) -> Modulation:
        """The six shared adaLN vectors of a timestep, computed in float32 like upstream."""

        first, second = (cast(nn.Linear, self.time_embedding[i]) for i in (0, 2))
        projection = cast(nn.Linear, self.time_modulation[1])
        embedding = sinusoidal_embedding(timesteps, self.config.freq_dim)
        embedding = _linear_fp32(F.silu(_linear_fp32(embedding, first)), second)
        modulated = _linear_fp32(F.silu(embedding), projection)
        return rearrange(modulated, "b (k d) -> b k d", k=NUM_BLOCK_MODULATIONS)

    def embed_context(self, context: TextContext) -> TextContext:
        """Project the padded umT5 context into the model width, in the parameter dtype."""

        return self.text_embedding(context.to(self.dtype))

    @override
    def forward(self, latents: VideoLatents, timesteps: Timesteps, context: TextContext) -> Tokens:
        tokens = self.patchify(latents)
        rotary = self.rotary(self.grid(latents))
        modulation = self.timestep_modulation(timesteps)
        context = self.embed_context(context)
        for block in self.blocks:
            tokens = block(tokens, context, modulation, rotary)
        return tokens

    def fold(self, tokens: Tokens, grid: LatentGrid) -> TapFeatures:
        """Lay the token stream back onto its `(frames, height, width)` grid, channels first."""

        f, h, w = grid
        return rearrange(tokens, "b (f h w) d -> b d f h w", f=f, h=h, w=w)
