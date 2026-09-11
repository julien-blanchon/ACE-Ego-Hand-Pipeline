"""Camera geometry of the readout: ray encodings, the translation solve, and the K-free camera fit.

The projector never sees a camera matrix directly. It reads the DiT features through a ray
positional encoding built from a *predicted* ray field (a 1x1 head on the tap features, pooled
over frames), and it places each hand in the camera frame by solving for the in-plane
translation that aligns the canonical MANO joints with its own 2D joint anchors: a closed-form
weighted least squares per hand ("mixed PnP"), with depth taken from the depth head. Bearings
come from the calibration when one is given (K variant) and from the predicted ray field
otherwise (K-free variant). Rows with too few interior joints or a poor fit fall back to
inverse-projecting the wrist anchor at the predicted depth.

For K-free deployment the predicted ray field is also fitted to an effective pinhole camera, so
the predictions can be projected into the image without any calibration.
Ref: ACE-Ego-Hand arXiv:2608.20308
# Adapted from https://github.com/ggxxii/ACE-Ego-Hand/blob/main/ace_ego_hand/archs/projector/memory_projector.py
# and https://github.com/ggxxii/ACE-Ego-Hand/blob/main/ace_ego_hand/inference.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from ..types import (
    CameraRaw,
    HandTranslation,
    Intrinsics,
    IntrinsicsBatch,
    Joints2D,
    JointsCamera,
    RayEncoding,
    RayField,
    RayMap,
)

# Acceptance gate of the mixed-PnP solve (upstream constants)
NEAR_DEPTH = 0.05  # metres; joints closer than this cannot vote
INTERIOR_MARGIN = 0.02  # normalized image coordinates; border-saturated anchors cannot vote
MIN_INTERIOR_JOINTS = 6
RESIDUAL_FRACTION = 0.25  # of the hand's 2D diagonal
RESIDUAL_MIN_PIXELS = 15.0
EPSILON = 1e-8
DIRECTION_EPSILON = 1e-6


class RayPositionalEncoding(nn.Module):
    """Azimuth / elevation Fourier features of each patch's ray direction, projected to the width.

    The last layer is zero-initialized upstream so the encoding starts as a no-op residual.
    """

    frequency_bands: torch.Tensor

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_bands: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(4 * num_bands, hidden_dim, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype),
        )
        bands = 2.0 ** torch.arange(num_bands, device=device, dtype=torch.float32)
        self.register_buffer("frequency_bands", bands, persistent=True)

    def forward(self, rays: RayField) -> RayEncoding:
        dx, dy, dz = rays.unbind(dim=1)
        depth = dz.abs().clamp_min(DIRECTION_EPSILON)
        azimuth = rearrange(torch.atan2(dx, depth), "b h w -> b (h w) 1")
        elevation = rearrange(torch.atan2(dy, depth), "b h w -> b (h w) 1")
        bands = self.frequency_bands
        features = torch.cat(
            [
                torch.sin(azimuth * bands),
                torch.cos(azimuth * bands),
                torch.sin(elevation * bands),
                torch.cos(elevation * bands),
            ],
            dim=-1,
        )
        return self.projection(features)


def sample_rays(rays: RayField, points: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample the ray field at normalized `(u, v)` points in `[-1, 1]`: `(B, N, 2) -> (B, N, 3)`."""

    grid = rearrange(points, "b n two -> b n 1 two")
    sampled = F.grid_sample(rays, grid, mode="bilinear", align_corners=True)
    return rearrange(sampled, "b c n 1 -> b n c")


def bearings_from_rays(rays: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(x / |z|, y / |z|)` of sampled ray directions, the K-free stand-in for `(u - c) / f`."""

    depth = rays[..., 2].abs().clamp_min(DIRECTION_EPSILON)
    return rays[..., 0] / depth, rays[..., 1] / depth


def inverse_projection(
    camera_raw: CameraRaw,
    *,
    intrinsics: IntrinsicsBatch | None,
    rays: RayField | None,
) -> HandTranslation:
    """Place the wrist anchor `(u_norm, v_norm)` at the predicted depth, through K or the rays.

    The fallback branch of the translation solve. `camera_raw` holds `(u, v)` in `[-1, 1]` and
    the log depth; exactly one of `intrinsics` (per window, in encode pixels) and `rays` is used.
    """

    depth = torch.exp(camera_raw[..., 2:3])
    if rays is not None:
        b = camera_raw.shape[0]
        points = rearrange(camera_raw[..., :2], "b f s two -> b (f s) two")
        directions = rearrange(sample_rays(rays, points), "b (f s) c -> b f s c", b=b, s=2)
        bearing_x, bearing_y = bearings_from_rays(directions)
        return torch.cat([bearing_x[..., None] * depth, bearing_y[..., None] * depth, depth], -1)
    assert intrinsics is not None, "inverse projection needs intrinsics or a ray field"
    fx, fy, cx, cy, width, height = (rearrange(v, "b -> b 1 1 1") for v in intrinsics.unbind(-1))
    u_pixel = (camera_raw[..., 0:1] + 1.0) * 0.5 * width
    v_pixel = (camera_raw[..., 1:2] + 1.0) * 0.5 * height
    return torch.cat([(u_pixel - cx) * depth / fx, (v_pixel - cy) * depth / fy, depth], -1)


type SolvePieces = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


def _weighted_shift(
    votes: torch.Tensor,
    gain_u: torch.Tensor,
    gain_v: torch.Tensor,
    offset_u: torch.Tensor,
    offset_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted least-squares shift per hand: `argmin_t sum_j m_j (gain_j t - offset_j)^2`."""

    tx = (votes * gain_u * offset_u).sum(-1) / (votes * gain_u * gain_u).sum(-1).clamp_min(EPSILON)
    ty = (votes * gain_v * offset_v).sum(-1) / (votes * gain_v * gain_v).sum(-1).clamp_min(EPSILON)
    return tx, ty


def _solve_through_rays(
    votes: torch.Tensor,
    canonical: JointsCamera,
    z: torch.Tensor,
    joints_2d: Joints2D,
    rays: RayField,
    width: float,
) -> SolvePieces:
    """K-free solve: bearings sampled from the ray field; residuals in pseudo-pixels (x width)."""

    cx_j, cy_j = canonical[..., 0], canonical[..., 1]
    points = rearrange(joints_2d * 2.0 - 1.0, "b f s j two -> b (f s j) two")
    directions = rearrange(
        sample_rays(rays, points), "b (f s j) c -> b f s j c", s=2, j=joints_2d.shape[3]
    )
    bearing_x, bearing_y = bearings_from_rays(directions)
    tx, ty = _weighted_shift(votes, 1.0 / z, 1.0 / z, bearing_x - cx_j / z, bearing_y - cy_j / z)
    u_fit = (cx_j + tx[..., None]) / z * width
    v_fit = (cy_j + ty[..., None]) / z * width
    return tx, ty, bearing_x * width, bearing_y * width, u_fit, v_fit


def _solve_through_intrinsics(
    votes: torch.Tensor,
    canonical: JointsCamera,
    z: torch.Tensor,
    joints_2d: Joints2D,
    intrinsics: IntrinsicsBatch,
    image_size: tuple[int, int],
) -> SolvePieces:
    """K solve: anchors in pixels through the calibration."""

    cx_j, cy_j = canonical[..., 0], canonical[..., 1]
    fx, fy, cx, cy = (rearrange(v, "b -> b 1 1 1") for v in intrinsics[:, :4].unbind(-1))
    u_pixel = joints_2d[..., 0] * float(image_size[0])
    v_pixel = joints_2d[..., 1] * float(image_size[1])
    tx, ty = _weighted_shift(
        votes, fx / z, fy / z, (u_pixel - cx) - fx * cx_j / z, (v_pixel - cy) - fy * cy_j / z
    )
    u_fit = fx * (cx_j + tx[..., None]) / z + cx
    v_fit = fy * (cy_j + ty[..., None]) / z + cy
    return tx, ty, u_pixel, v_pixel, u_fit, v_fit


def solve_translation(
    camera_raw: CameraRaw,
    canonical_joints: JointsCamera,
    joints_2d: Joints2D,
    *,
    intrinsics: IntrinsicsBatch | None,
    rays: RayField | None,
    image_size: tuple[int, int],
) -> HandTranslation:
    """Mixed PnP: keep the predicted depth, re-solve `(tx, ty)` from the 2D anchors per hand.

    Each joint votes with weight 1 when it is in front of the camera and strictly inside the
    image; the shift is the weighted least-squares solution of the projected canonical joints
    against the anchors. A hand is accepted when at least `MIN_INTERIOR_JOINTS` voted and the
    post-solve RMS residual is within `max(RESIDUAL_FRACTION * diagonal, RESIDUAL_MIN_PIXELS)`;
    otherwise the inverse projection of the wrist anchor is used. `canonical_joints` are the
    MANO joints before translation, `(B, F, S, J, 3)`; `joints_2d` in `[0, 1]`.
    """

    fallback = inverse_projection(camera_raw, intrinsics=intrinsics, rays=rays)
    depth = torch.exp(camera_raw[..., 2:3])  # (B, F, S, 1)
    z_raw = canonical_joints[..., 2] + depth  # (B, F, S, J)
    z = z_raw.clamp_min(NEAR_DEPTH)

    interior = (
        (joints_2d[..., 0] > INTERIOR_MARGIN)
        & (joints_2d[..., 0] < 1.0 - INTERIOR_MARGIN)
        & (joints_2d[..., 1] > INTERIOR_MARGIN)
        & (joints_2d[..., 1] < 1.0 - INTERIOR_MARGIN)
    )
    votes = ((z_raw > NEAR_DEPTH) & interior).float()

    if rays is not None:
        pieces = _solve_through_rays(
            votes, canonical_joints, z, joints_2d, rays, float(image_size[0])
        )
    else:
        assert intrinsics is not None, "the translation solve needs intrinsics or a ray field"
        pieces = _solve_through_intrinsics(
            votes, canonical_joints, z, joints_2d, intrinsics, image_size
        )
    tx, ty, u_observed, v_observed, u_fit, v_fit = pieces
    solved = torch.cat([tx[..., None], ty[..., None], depth], dim=-1)

    # Acceptance: enough votes and a scale-aware reprojection residual
    residual = (u_fit - u_observed) ** 2 + (v_fit - v_observed) ** 2
    num_votes = votes.sum(-1)
    rms = ((residual * votes).sum(-1) / num_votes.clamp_min(1.0)).sqrt()
    far = torch.full_like(u_observed, 1e9)
    voted = votes > 0
    u_span = torch.where(voted, u_observed, -far).amax(-1) - torch.where(
        voted, u_observed, far
    ).amin(-1)
    v_span = torch.where(voted, v_observed, -far).amax(-1) - torch.where(
        voted, v_observed, far
    ).amin(-1)
    diagonal = (u_span**2 + v_span**2).clamp_min(0.0).sqrt()
    threshold = torch.maximum(
        RESIDUAL_FRACTION * diagonal, torch.full_like(diagonal, RESIDUAL_MIN_PIXELS)
    )
    accept = (num_votes >= MIN_INTERIOR_JOINTS) & (rms <= threshold)
    return torch.where(accept[..., None], solved, fallback)


def fit_pinhole(ray_map: RayMap, image_size: tuple[int, int]) -> Intrinsics:
    """Fit one pinhole camera `(fx, fy, cx, cy)` to a predicted ray field, in `image_size` pixels.

    Per axis, least squares of the pixel centre against the ray's tangent: `u = cx + fx * dx/dz`.
    All frames of all windows contribute samples; the camera is assumed constant over the clip.
    """

    width, height = float(image_size[0]), float(image_size[1])
    directions = F.normalize(ray_map.float(), dim=1)
    _, _, _, h, w = directions.shape
    dz = directions[:, 2].clamp_min(DIRECTION_EPSILON)
    tangent_x = rearrange(directions[:, 0] / dz, "b g h w -> (b g h w)")
    tangent_y = rearrange(directions[:, 1] / dz, "b g h w -> (b g h w)")
    rows = torch.arange(h, device=ray_map.device, dtype=torch.float32)
    cols = torch.arange(w, device=ray_map.device, dtype=torch.float32)
    u = repeat(
        (cols + 0.5) * width / w, "w -> (b g h w)", b=ray_map.shape[0], g=ray_map.shape[2], h=h
    )
    v = repeat(
        (rows + 0.5) * height / h, "h -> (b g h w)", b=ray_map.shape[0], g=ray_map.shape[2], w=w
    )

    def axis_fit(tangent: torch.Tensor, pixel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # A two-column least squares over every ray: solved on the CPU, where lstsq is exact for
        # any size (the MPS kernel falls back there anyway, with a warning)
        design = torch.stack([torch.ones_like(tangent), tangent], dim=1).cpu()
        solution = torch.linalg.lstsq(design, pixel[:, None].cpu()).solution[:, 0]
        return solution[0].to(ray_map.device), solution[1].to(ray_map.device)  # (centre, focal)

    cx, fx = axis_fit(tangent_x, u)
    cy, fy = axis_fit(tangent_y, v)
    intrinsics = torch.eye(3, device=ray_map.device)
    intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2] = fx, fy, cx, cy
    return intrinsics


def intrinsics_batch(
    intrinsics: Intrinsics, image_size: tuple[int, int], batch: int
) -> IntrinsicsBatch:
    """Repeat one camera `(fx, fy, cx, cy, width, height)` for every window of a batch."""

    values = torch.stack(
        [
            intrinsics[0, 0],
            intrinsics[1, 1],
            intrinsics[0, 2],
            intrinsics[1, 2],
            torch.tensor(float(image_size[0]), device=intrinsics.device),
            torch.tensor(float(image_size[1]), device=intrinsics.device),
        ]
    )
    return repeat(values, "k -> b k", b=batch)
