"""Undistort footage to a pinhole view: one sampling grid per camera, `grid_sample` per frame.

The released checkpoints are pinhole-only, so fisheye or distorted footage is resampled first.
`target_pinhole` picks the pinhole the view is rendered with, `sampling_map` maps every target
pixel to the source pixel it samples by forward-distorting the target's unit-plane rays through
the source lens (the OpenCV `initUndistortRectifyMap` construction with `R = I`), and `remap`
resamples uint8 frames through that map bilinearly. Pure torch; the two lens models are the
OpenCV ones: Brown-Conrady with the rational `k4..k6` terms (`projectPoints`) and the
Kannala-Brandt fisheye polynomial (`fisheye.projectPoints`).

Pixel convention: integer pixel coordinates are pixel centres, as in OpenCV, which is
`grid_sample(align_corners=True)` after the `[0, N - 1] -> [-1, 1]` normalisation.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from ..types import ClipArray, PixelMap, Points2D, UnitPlanePoints
from .calibration import CameraModel

INVERSION_ITERATIONS = 20  # fixed-point / Newton steps of the lens inverse (OpenCV uses 5-10)
BORDER_SAMPLES = 64  # points per image edge when fitting the borderless pinhole
THETA_MAX = math.pi / 2 - 1e-6  # a fisheye ray at 90 degrees has no unit-plane point
UINT8_MAX = 255.0


def distort(camera: CameraModel, points: UnitPlanePoints) -> Points2D:
    """Project unit-plane points to pixels through the lens (OpenCV `projectPoints` at z = 1)."""

    match camera.distortion:
        case "none":
            distorted = points
        case "brown_conrady":
            distorted = _brown_conrady(camera.coefficients, points)
        case "kannala_brandt":
            distorted = _kannala_brandt(camera.coefficients, points)
    x, y = distorted.unbind(-1)
    return torch.stack((camera.fx * x + camera.cx, camera.fy * y + camera.cy), dim=-1)


def undistort_points(camera: CameraModel, pixels: Points2D) -> UnitPlanePoints:
    """Invert `distort`: pixels back to the unit plane (OpenCV `undistortPoints`, iterative)."""

    u, v = pixels.unbind(-1)
    distorted = torch.stack(((u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy), dim=-1)
    match camera.distortion:
        case "none":
            return distorted
        case "brown_conrady":
            return _brown_conrady_inverse(camera.coefficients, distorted)
        case "kannala_brandt":
            return _kannala_brandt_inverse(camera.coefficients, distorted)


def target_pinhole(
    source: CameraModel, size: tuple[int, int], *, fov_scale: float = 1.0
) -> CameraModel:
    """Pick the pinhole the undistorted `size = (width, height)` view is rendered with.

    A pinhole source keeps its intrinsics (rescaled to `size`), so the remap is the identity.
    A distorted source gets the largest borderless view with square pixels, as OpenCV's
    `fisheye.estimateNewCameraMatrixForUndistortRectify(balance=0)`: the source image border is
    undistorted onto the unit plane, the inner axis-aligned rectangle (bounded by the innermost
    point of each edge) is the region every view pixel can be sampled from, and the view is
    centred on it with the tighter of the two focal lengths that fit it, so no source-less
    black region shows and pixels stay square (a stretched view would be off-distribution for
    the model). `fov_scale` divides that focal length: > 1 widens the view (black corners
    appear), < 1 crops further in.
    """

    if source.is_pinhole:
        return source.rescaled(size)

    left, top, right, bottom = _inner_rectangle(source)
    centre_x, centre_y = (left + right) / 2, (top + bottom) / 2

    # The rectangle's tighter side spans the first to last pixel centre of the view
    width, height = size
    focal = max((width - 1) / (right - left), (height - 1) / (bottom - top)) / fov_scale
    cx, cy = (width - 1) / 2 - focal * centre_x, (height - 1) / 2 - focal * centre_y
    return CameraModel(focal, focal, cx, cy, size)


def sampling_map(source: CameraModel, target: CameraModel, device: torch.device) -> PixelMap:
    """For each target pixel centre, the source pixel it samples (`initUndistortRectifyMap`, R = I).

    Built once per camera pair in float64 and stored as float32, like OpenCV's `CV_32FC2` map.
    """

    width, height = target.size
    us = torch.arange(width, dtype=torch.float64)  # float64 on the CPU: MPS has none
    vs = torch.arange(height, dtype=torch.float64)
    v, u = torch.meshgrid(vs, us, indexing="ij")

    # Target pixel -> ray on the unit plane -> where the source lens images that ray
    rays = torch.stack(((u - target.cx) / target.fx, (v - target.cy) / target.fy), dim=-1)
    return distort(source, rays).float().to(device)


@torch.inference_mode()
def remap(frames: ClipArray, pixel_map: PixelMap, *, batch_size: int = 16) -> ClipArray:
    """Resample uint8 `(F, H, W, 3)` frames through `pixel_map` on its device, in batches.

    Bilinear, zeros outside the source image, as `cv2.remap(INTER_LINEAR, BORDER_CONSTANT)`.
    """

    device = pixel_map.device
    source_height, source_width = frames.shape[1:3]

    # Pixel centres to `align_corners=True` coordinates: 0 -> -1 and N - 1 -> 1 per axis
    span = torch.tensor([source_width - 1, source_height - 1], device=device, dtype=pixel_map.dtype)
    grid = rearrange(pixel_map / span * 2 - 1, "h w xy -> 1 h w xy")

    remapped: list[np.ndarray] = []
    for start in range(0, len(frames), batch_size):
        batch = torch.from_numpy(frames[start : start + batch_size]).to(device, non_blocking=True)
        images = rearrange(batch, "f h w c -> f c h w").float()
        sampled = F.grid_sample(
            images,
            grid.expand(len(images), -1, -1, -1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        pixels = sampled.round().clamp(0.0, UINT8_MAX).to(torch.uint8)
        remapped.append(rearrange(pixels, "f c h w -> f h w c").cpu().numpy())
    return np.concatenate(remapped)


def _brown_conrady(coefficients: tuple[float, ...], points: UnitPlanePoints) -> UnitPlanePoints:
    """OpenCV `projectPoints`: rational radial term plus tangential `p1, p2`."""

    k1, k2, p1, p2, k3, k4, k5, k6 = coefficients
    x, y = points.unbind(-1)
    r2 = x * x + y * y
    radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return torch.stack((xd, yd), dim=-1)


def _brown_conrady_inverse(
    coefficients: tuple[float, ...], distorted: UnitPlanePoints
) -> UnitPlanePoints:
    """OpenCV `undistortPoints`: fixed-point iteration `x = (xd - tangential(x)) / radial(x)`."""

    k1, k2, p1, p2, k3, k4, k5, k6 = coefficients
    xd, yd = distorted.unbind(-1)
    x, y = xd, yd
    for _ in range(INVERSION_ITERATIONS):
        r2 = x * x + y * y
        radial = (1 + r2 * (k4 + r2 * (k5 + r2 * k6))) / (1 + r2 * (k1 + r2 * (k2 + r2 * k3)))
        tangential_x = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        tangential_y = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x, y = (xd - tangential_x) * radial, (yd - tangential_y) * radial
    return torch.stack((x, y), dim=-1)


def _kannala_brandt(coefficients: tuple[float, ...], points: UnitPlanePoints) -> UnitPlanePoints:
    """OpenCV `fisheye.projectPoints`: `theta_d = theta (1 + k1 theta^2 + ... + k4 theta^8)`."""

    x, y = points.unbind(-1)
    r = torch.sqrt(x * x + y * y)
    theta = torch.atan(r)
    theta_d = theta * _fisheye_polynomial(coefficients, theta)
    scale = torch.where(r > 0, theta_d / r.clamp_min(torch.finfo(r.dtype).tiny), 1.0)
    return torch.stack((x * scale, y * scale), dim=-1)


def _kannala_brandt_inverse(
    coefficients: tuple[float, ...], distorted: UnitPlanePoints
) -> UnitPlanePoints:
    """OpenCV `fisheye.undistortPoints`: Newton on `theta`, then `r = tan(theta)`."""

    k1, k2, k3, k4 = coefficients
    xd, yd = distorted.unbind(-1)
    theta_d = torch.sqrt(xd * xd + yd * yd)
    theta = theta_d.clone()
    for _ in range(INVERSION_ITERATIONS):
        theta2 = theta * theta
        residual = theta * _fisheye_polynomial(coefficients, theta) - theta_d
        slope = 1 + theta2 * (3 * k1 + theta2 * (5 * k2 + theta2 * (7 * k3 + theta2 * 9 * k4)))
        theta = (theta - residual / slope).clamp(0.0, THETA_MAX)
    scale = torch.where(
        theta_d > 0, torch.tan(theta) / theta_d.clamp_min(torch.finfo(theta_d.dtype).tiny), 1.0
    )
    return torch.stack((xd * scale, yd * scale), dim=-1)


def _fisheye_polynomial(coefficients: tuple[float, ...], theta: torch.Tensor) -> torch.Tensor:
    k1, k2, k3, k4 = coefficients
    theta2 = theta * theta
    return 1 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4)))


def _inner_rectangle(source: CameraModel) -> tuple[float, float, float, float]:
    """`(left, top, right, bottom)` on the unit plane: the borderless view of the source image.

    Each source edge is undistorted; the innermost point of each edge bounds the rectangle,
    which is what OpenCV's `icvGetRectangles` computes for `alpha = 0`.
    """

    width, height = source.size
    us = torch.linspace(0.0, width - 1, BORDER_SAMPLES, dtype=torch.float64)
    vs = torch.linspace(0.0, height - 1, BORDER_SAMPLES, dtype=torch.float64)
    edges = torch.stack(
        (
            torch.stack((torch.zeros_like(vs), vs), dim=-1),  # left edge
            torch.stack((us, torch.zeros_like(us)), dim=-1),  # top edge
            torch.stack((torch.full_like(vs, width - 1), vs), dim=-1),  # right edge
            torch.stack((us, torch.full_like(us, height - 1)), dim=-1),  # bottom edge
        )
    )
    rays = undistort_points(source, edges)  # (4, BORDER_SAMPLES, 2)
    left = rays[0, :, 0].max().item()
    top = rays[1, :, 1].max().item()
    right = rays[2, :, 0].min().item()
    bottom = rays[3, :, 1].min().item()
    return left, top, right, bottom
