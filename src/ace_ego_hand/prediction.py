"""The result contract of the pipeline: hands for every source frame of one video.

`VideoPrediction` is what `HandEstimator.predict` returns and what the pose table, the LeRobot
writer and the visualiser consume; `WeightsInfo` records which weights produced it. Both are
plain numpy so they can be written and read without the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import (
    ClipBetas,
    ClipGlobalOrient,
    ClipHandPose,
    ClipIntrinsics,
    ClipJoints2D,
    ClipJointsCamera,
    ClipProbability,
    ClipTranslation,
    Pixels,
    Points,
)

MIN_DEPTH = 1e-4  # depth floor of the pinhole projection, the same as the inference path


@dataclass(frozen=True, slots=True)
class VideoPrediction:
    """Hands for every source frame of one video, in the source image's pixel frame."""

    global_orient: ClipGlobalOrient
    hand_pose: ClipHandPose
    betas: ClipBetas  # per frame: each window predicts its own shape
    translation: ClipTranslation
    joints_camera: ClipJointsCamera
    joints_2d: ClipJoints2D
    presence: ClipProbability
    visible: ClipProbability
    quality: (
        ClipProbability  # how far to trust the frame: min(presence, visible), halved if synthesised
    )
    intrinsics: ClipIntrinsics  # given, or fitted from the predicted rays
    intrinsics_fitted: bool
    source_size: tuple[int, int]
    encode_size: tuple[int, int]
    fps: float


@dataclass(frozen=True, slots=True)
class WeightsInfo:
    """Where the loaded weights came from, recorded in every output."""

    repo_id: str
    revision: str
    variant: str
    latent_encoder: str


def frame_quality(presence: ClipProbability, visible: ClipProbability) -> ClipProbability:
    """One number per hand and frame to filter on: the weaker of `presence` and `visible`."""

    return np.minimum(presence, visible)


def project_points(points: Points, intrinsics: ClipIntrinsics) -> Pixels:
    """Pinhole projection of camera-frame points `(..., 3)` to pixels `(..., 2)`, numpy."""

    depth = np.clip(points[..., 2:3], MIN_DEPTH, None)
    u = intrinsics[0, 0] * points[..., 0:1] / depth + intrinsics[0, 2]
    v = intrinsics[1, 1] * points[..., 1:2] / depth + intrinsics[1, 2]
    return np.concatenate([u, v], axis=-1).astype(points.dtype)
