"""Optional temporal clean-up of a finished `VideoPrediction`: depth smoothing, one shape per hand,
short presence gaps bridged.

The inference path is parity-exact with upstream and stays so: nothing here runs unless the
caller applies it to a result. The backbone reads hands out in 88-frame windows, and the wrist
depth is by far its noisiest quantity (our ground-truth check: ~20-35 mm articulation error
against 130-170 mm of wrist depth error on wide-angle clips), so the main step regularises the
depth along each frame's camera ray, which keeps the 2D projection of the wrist untouched.
The steps run in this order, each on top of the previous one:

1. Jumps: a wrist that moves faster than `max_wrist_speed` frame widths per second between two
   visible frames is a glitch, hands do not do that. If it comes back within `spike_frames`
   the spike is replaced by the interpolation of its two ends; otherwise the shorter of the
   two visible runs it separates is hidden when it lasts at most `teleport_frames` (a hand that
   re-enters somewhere else after leaving the view is not touched: it was hidden in between).
2. Presence gaps: an absent run of at most `presence_gap` frames between two present frames is
   marked present, with the pose and translation interpolated from its two neighbours.
3. Wrist depth: over every run of present frames the depths minimise
   `sum (d_t - d0_t)^2 + w * sum (d_{t+1} - 2 d_t + d_{t-1})^2`, then each wrist is moved along
   its own ray to the new depth.
4. Betas: every hand gets its presence-weighted mean shape over the clip.
5. The camera joints and their projection are decoded again from the updated parameters.

Everything is numpy (float64 for the solves) plus the caller's MANO layer for the decode.

`interpolate_frames` is the second entry point: it fills the frames the estimator skipped when a
clip was predicted at a reduced frame rate (`InferenceConfig.target_fps`), with the same slerp.
`verified_visibility` and `detector_visibility` are the second opinions on `visible`
(`InferenceConfig.verify`): the first combines a clip's `visible` with the one of a pass on the
mirrored video, so a hand the model hallucinates in one orientation only is hidden; the second
hides a hand that a box detector does not support for a sustained run while it finds hands
elsewhere in those frames (so footage the detector cannot read, gloves for instance, is left
alone). Tuned on a 257-clip pool: the detector rule touches 12 of 26 clips flagged wrong,
4 of 114 flagged fine and none of the ground-truth clips.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np
import torch
from einops import einsum, repeat

from .modeling.mano import ManoLayer
from .prediction import MIN_DEPTH, VideoPrediction, frame_quality, project_points
from .types import (
    ClipBetas,
    ClipGlobalOrient,
    ClipHandMask,
    ClipHandPose,
    ClipJointsCamera,
    ClipProbability,
    ClipTranslation,
    FrameBoxes,
    FrameIndices,
    FrameMask,
    GapFractions,
    HandRotations,
    HandTranslations,
    Quaternions,
    Ratios,
    RotationArray,
    RunDepths,
    WristTrack,
)
from .utils.tensors import to_numpy

# Below this quaternion dot product the two rotations are interpolated linearly (the slerp arc is
# too short for its sine to be well conditioned)
SLERP_LINEAR_DOT = 1.0 - 1e-6
WRIST = 0
UNVERIFIED_VISIBLE = 0.4  # ceiling of `visible` where the mirrored pass puts the hand elsewhere
UNSUPPORTED_VISIBLE = 0.25  # ceiling of `visible` over a run no detector box supports


@dataclass(frozen=True, slots=True)
class PostprocessConfig:
    """Quantities of the post-processing; zero switches a step off."""

    depth_acceleration: float = 0.2  # weight of the depth second difference, 0 disables
    clip_betas: bool = True  # one presence-weighted shape per hand over the clip
    presence_gap: int = 4  # bridge absent runs of at most this many frames, 0 disables
    presence_threshold: float = 0.5  # a hand is present when presence >= this
    max_wrist_speed: float = 6.0  # frame widths per second a visible wrist may move, 0 disables
    max_wrist_speed_3d: float = 4.5  # metres per second in camera space, 0 disables
    max_wrist_turn: float = 1350.0  # degrees per second the hand root may rotate, 0 disables
    spike_frames: int = 3  # a jump that comes back within this many frames is interpolated
    teleport_frames: int = 12  # a jump's shorter side is hidden when it lasts at most this
    visible_threshold: float = 0.5  # a hand is in view when visible >= this


def postprocess(
    prediction: VideoPrediction, config: PostprocessConfig, *, mano: ManoLayer
) -> VideoPrediction:
    """Return a new prediction with the steps of `config` applied, joints decoded again by `mano`."""

    global_orient = prediction.global_orient.copy()
    hand_pose = prediction.hand_pose.copy()
    translation = prediction.translation.copy()
    presence = prediction.presence.copy()
    visible = prediction.visible.copy()
    present: ClipHandMask = presence >= config.presence_threshold

    synthesised: ClipHandMask = np.zeros_like(present)

    if config.max_wrist_speed > 0 or config.max_wrist_speed_3d > 0 or config.max_wrist_turn > 0:
        in_view: ClipHandMask = visible >= config.visible_threshold
        wrist = prediction.joints_2d[:, :, WRIST].copy()
        per_frame = 1.0 / prediction.fps
        for hand in range(present.shape[1]):
            # The views read the arrays `_bridge` edits, so a repaired spike is no longer a jump
            motion = _WristMotion(
                pixels=wrist[:, hand],
                metres=translation[:, hand],
                orient=global_orient[:, hand],
                max_pixels=config.max_wrist_speed * prediction.source_size[0] * per_frame,
                max_metres=config.max_wrist_speed_3d * per_frame,
                max_radians=np.deg2rad(config.max_wrist_turn) * per_frame,
            )
            for start, stop in _spikes(motion, in_view[:, hand], config.spike_frames):
                _bridge(global_orient, hand_pose, translation, hand, start, stop)
                fractions = _fractions(start, stop)
                wrist[start:stop, hand] = wrist[start - 1, hand] + fractions[:, None] * (
                    wrist[stop, hand] - wrist[start - 1, hand]
                )
                synthesised[start:stop, hand] = True
            for start, stop in _teleports(motion, in_view[:, hand], config.teleport_frames):
                visible[start:stop, hand] = np.minimum(
                    visible[start:stop, hand], config.visible_threshold / 2
                )

    # Gaps first, so the bridged frames join their neighbouring runs for the depth solve
    if config.presence_gap > 0:
        for hand in range(present.shape[1]):
            for start, stop in _gaps(present[:, hand], config.presence_gap):
                _bridge(global_orient, hand_pose, translation, hand, start, stop)
                # A bridged frame is never more confident than the weaker of its two neighbours
                presence[start:stop, hand] = min(presence[start - 1, hand], presence[stop, hand])
                present[start:stop, hand] = True
                synthesised[start:stop, hand] = True

    if config.depth_acceleration > 0:
        for hand in range(present.shape[1]):
            for start, stop in _runs(present[:, hand]):
                translation[start:stop, hand] = _smooth_wrist_depth(
                    translation[start:stop, hand], config.depth_acceleration
                )

    betas = _clip_betas(prediction.betas, presence) if config.clip_betas else prediction.betas

    quality = frame_quality(presence, visible)
    quality[synthesised] *= 0.5
    joints_camera = _decode_joints(global_orient, hand_pose, betas, translation, mano)
    joints_2d = project_points(joints_camera, prediction.intrinsics)
    return replace(
        prediction,
        global_orient=global_orient,
        hand_pose=hand_pose,
        betas=betas,
        translation=translation,
        joints_camera=joints_camera,
        joints_2d=joints_2d,
        presence=presence,
        visible=visible,
        quality=quality,
    )


def verified_visibility(
    prediction: VideoPrediction,
    mirrored: VideoPrediction,
    *,
    max_wrist_distance: float = 0.15,
    window: int = 5,
) -> ClipProbability:
    """`visible` both passes agree on: the geometric mean of the two where their wrists land within
    `max_wrist_distance` frame widths of each other, at most 0.4 where they do not; then a
    `window`-frame median so a one-frame dropout of either pass does not cut a hand."""

    distance = np.linalg.norm(
        prediction.joints_2d[:, :, WRIST] - mirrored.joints_2d[:, :, WRIST], axis=-1
    )
    agree = distance <= max_wrist_distance * prediction.source_size[0]
    both = np.sqrt(prediction.visible * mirrored.visible)
    neither = np.minimum(np.minimum(prediction.visible, mirrored.visible), UNVERIFIED_VISIBLE)
    score = np.where(agree, both, neither)
    half = window // 2
    padded = np.pad(score, ((half, half), (0, 0)), mode="edge")
    stacked = np.stack([padded[k : k + len(score)] for k in range(2 * half + 1)])
    return np.median(stacked, axis=0).astype(prediction.visible.dtype)


def detector_visibility(
    prediction: VideoPrediction,
    boxes: list[FrameBoxes],
    *,
    box_margin: float = 0.35,
    support: float = 0.6,
    gap_seconds: float = 1.5,
    active: float = 0.5,
    confirmed: float = 0.3,
    blip_seconds: float = 0.5,
    jump_speed: float = 4.0,
    jump_min_seconds: float = 0.0,
    jump_window_seconds: float = 0.3,
    jump_boxed: float = 0.6,
    visible_threshold: float = 0.5,
) -> ClipProbability:
    """`visible` capped at `UNSUPPORTED_VISIBLE` over every run of at least `gap_seconds` where
    fewer than `support` of a visible hand's joints fall inside a detected box (grown by
    `box_margin` of its longer side) while the detector finds boxes in at least `active` of the
    run's unsupported frames; a supported blip shorter than `blip_seconds` inside such a run (a
    phantom drifting over the other hand's box) does not end it. Only a hand the detector
    confirms in fewer than `confirmed` of its visible frames over the clip is gated this way: a
    real hand loses its box while occluded or in fast motion but has it before and after, a
    phantom or an exoskeleton hand never has one.

    A second rule needs no clip-level guard because a wrist jump supplies the evidence: cut a
    hand's visible track at every jump faster than `jump_speed` frame widths per second, call a
    segment supported when `support` holds in `jump_boxed` of its frames and unsupported when
    no `jump_window_seconds` window inside it is, and hide every unsupported segment of at least
    `jump_min_seconds` that jumps join to a supported one (directly or through other unsupported
    ones): a phantom that teleports around before landing on the real hand."""

    joints = prediction.joints_2d
    supported = np.zeros(joints.shape[:2])
    has_box = np.zeros(len(joints), dtype=bool)
    for index, frame_boxes in enumerate(boxes[: len(joints)]):
        has_box[index] = len(frame_boxes) > 0
        for x0, y0, x1, y1, _score in frame_boxes:
            margin = box_margin * max(x1 - x0, y1 - y0)
            inside = (
                (joints[index, ..., 0] >= x0 - margin)
                & (joints[index, ..., 0] <= x1 + margin)
                & (joints[index, ..., 1] >= y0 - margin)
                & (joints[index, ..., 1] <= y1 + margin)
            )
            supported[index] = np.maximum(supported[index], inside.mean(axis=-1))
    visible = prediction.visible.copy()
    min_run = max(1, round(gap_seconds * prediction.fps))
    max_blip = round(blip_seconds * prediction.fps)
    for hand in range(visible.shape[1]):
        in_view = visible[:, hand] >= visible_threshold
        boxed: FrameMask = supported[:, hand] >= support
        jump_spans = (
            _jump_phantoms(
                prediction.joints_2d[:, hand, WRIST],
                in_view,
                boxed,
                has_box,
                max_jump=jump_speed * prediction.source_size[0] / prediction.fps,
                min_length=max(1, round(jump_min_seconds * prediction.fps)),
                window=max(1, round(jump_window_seconds * prediction.fps)),
                boxed_share=jump_boxed,
                active=active,
            )
            if jump_speed > 0
            else []
        )  # zero switches the jump rule off
        for start, stop in jump_spans:
            visible[start:stop, hand] = np.minimum(visible[start:stop, hand], UNSUPPORTED_VISIBLE)
        if in_view.any() and boxed[in_view].mean() >= confirmed:
            continue
        unsupported = in_view & ~boxed

        # Runs on their own, then the same runs with short supported blips absorbed: a span is
        # hidden when it qualifies as a whole or contains a run that did
        spans = unsupported.copy()
        for start, stop in _runs(~unsupported):
            if start > 0 and stop < len(spans) and stop - start <= max_blip:
                spans[start:stop] = in_view[start:stop]
        strong = [
            (a, b)
            for a, b in _runs(unsupported)
            if _qualifies(a, b, unsupported, has_box, min_run, active)
        ]
        for start, stop in _runs(spans):
            whole = _qualifies(start, stop, unsupported, has_box, min_run, active)
            if whole or any(start <= a and b <= stop for a, b in strong):
                visible[start:stop, hand] = np.minimum(
                    visible[start:stop, hand], UNSUPPORTED_VISIBLE
                )
    return visible


def _jump_phantoms(
    wrist: WristTrack,
    in_view: FrameMask,
    boxed: FrameMask,
    has_box: FrameMask,
    *,
    max_jump: float,
    min_length: int,
    window: int,
    boxed_share: float,
    active: float,
) -> list[tuple[int, int]]:
    """Segments of the visible track between wrist jumps that no box supports, joined by jumps
    to a segment boxes do support: the spans a teleporting phantom occupied."""

    step = np.linalg.norm(np.diff(wrist, axis=0), axis=-1)
    jumps = [int(t) + 1 for t in np.flatnonzero((step > max_jump) & in_view[1:] & in_view[:-1])]
    spans: list[tuple[int, int]] = []
    for start, stop in _runs(in_view):
        cuts = [start, *(t for t in jumps if start < t < stop), stop]
        segments = list(pairwise(cuts))
        if len(segments) < 2:
            continue
        strong = [boxed[a:b].mean() >= boxed_share for a, b in segments]
        weak = [not _sustained(boxed[a:b], window, boxed_share) for a, b in segments]
        if not any(strong):
            continue
        for index, (a, b) in enumerate(segments):
            if not weak[index] or b - a < min_length or has_box[a:b].mean() < active:
                continue
            # Walk over neighbouring weak segments to the nearest strong one on either side
            left = index - 1
            while left >= 0 and weak[left]:
                left -= 1
            right = index + 1
            while right < len(segments) and weak[right]:
                right += 1
            if (left >= 0 and strong[left]) or (right < len(segments) and strong[right]):
                spans.append((a, b))
    return spans


def _sustained(boxed: FrameMask, window: int, active: float) -> bool:
    """Whether some `window` consecutive frames are boxed in at least `active` of them."""

    if len(boxed) < window:
        return bool(boxed.mean() >= active)
    counts = np.convolve(boxed.astype(np.float64), np.ones(window), mode="valid")
    return bool(counts.max() / window >= active)


def _qualifies(
    start: int, stop: int, unsupported: FrameMask, has_box: FrameMask, min_run: int, active: float
) -> bool:
    """A span long enough to hide, whose unsupported frames had boxes elsewhere often enough."""

    hidden = unsupported[start:stop]
    return stop - start >= min_run and bool(has_box[start:stop][hidden].mean() >= active)


def interpolate_frames(
    prediction: VideoPrediction, stride: int, num_frames: int, *, mano: ManoLayer
) -> VideoPrediction:
    """Expand a prediction made on every `stride`-th frame to all `num_frames` source frames.

    Rotations are slerped and translation, shape, presence and visibility interpolated linearly
    between consecutive predicted frames; frames after the last predicted one repeat it. The
    joints are decoded again so every output frame is a valid MANO hand.
    """

    if stride == 1:
        return prediction
    keyframes = np.arange(len(prediction.presence)) * stride
    positions = np.arange(num_frames)
    before = np.clip(np.searchsorted(keyframes, positions, side="right") - 1, 0, len(keyframes) - 1)
    after = np.minimum(before + 1, len(keyframes) - 1)
    fractions = np.where(after > before, (positions - keyframes[before]) / stride, 0.0).astype(
        np.float64
    )

    def lerp(values: np.ndarray) -> np.ndarray:
        shape = (-1, *([1] * (values.ndim - 1)))
        mixed = values[before] + fractions.reshape(shape) * (values[after] - values[before])
        return mixed.astype(values.dtype)

    global_orient = np.empty(
        (num_frames, *prediction.global_orient.shape[1:]), prediction.global_orient.dtype
    )
    hand_pose = np.empty((num_frames, *prediction.hand_pose.shape[1:]), prediction.hand_pose.dtype)
    for key in range(len(keyframes)):
        frames = np.flatnonzero(before == key)
        if frames.size == 0:
            continue
        nxt = min(key + 1, len(keyframes) - 1)
        global_orient[frames] = _slerp_matrices(
            prediction.global_orient[key], prediction.global_orient[nxt], fractions[frames]
        )
        hand_pose[frames] = _slerp_matrices(
            prediction.hand_pose[key], prediction.hand_pose[nxt], fractions[frames]
        )

    translation = lerp(prediction.translation)
    betas = lerp(prediction.betas)
    joints_camera = _decode_joints(global_orient, hand_pose, betas, translation, mano)
    return replace(
        prediction,
        global_orient=global_orient,
        hand_pose=hand_pose,
        betas=betas,
        translation=translation,
        joints_camera=joints_camera,
        joints_2d=project_points(joints_camera, prediction.intrinsics),
        presence=lerp(prediction.presence),
        visible=lerp(prediction.visible),
        quality=lerp(prediction.quality),
        fps=prediction.fps * stride,
    )


def _fractions(start: int, stop: int) -> GapFractions:
    """Interpolation positions of the frames `[start, stop)` between their two outer neighbours."""

    return np.arange(1, stop - start + 1) / (stop - start + 1)


def _bridge(
    global_orient: ClipGlobalOrient,
    hand_pose: ClipHandPose,
    translation: ClipTranslation,
    hand: int,
    start: int,
    stop: int,
) -> None:
    """Overwrite frames `[start, stop)` of one hand with the interpolation of frames `start - 1` and
    `stop`: rotations slerped, translation linear. In place."""

    fractions = _fractions(start, stop)
    before, after = start - 1, stop
    global_orient[start:stop, hand] = _slerp_matrices(
        global_orient[before, hand], global_orient[after, hand], fractions
    )
    hand_pose[start:stop, hand] = _slerp_matrices(
        hand_pose[before, hand], hand_pose[after, hand], fractions
    )
    translation[start:stop, hand] = translation[before, hand] + fractions[:, None] * (
        translation[after, hand] - translation[before, hand]
    )


def _runs(mask: FrameMask) -> list[tuple[int, int]]:
    """Half-open `[start, stop)` spans of consecutive True frames."""

    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop)) for start, stop in zip(edges[::2], edges[1::2], strict=True)]


@dataclass(frozen=True, slots=True)
class _WristMotion:
    """One hand's wrist track in pixels, metres and orientation, with the per-frame limits a real
    hand stays under; a limit of zero switches that quantity off."""

    pixels: WristTrack
    metres: HandTranslations
    orient: HandRotations
    max_pixels: float
    max_metres: float
    max_radians: float

    def ratio(self, before: FrameIndices, after: FrameIndices) -> Ratios:
        """The largest of the three motions from `before` to `after`, relative to its limit."""

        ratio = np.zeros(len(before))
        if self.max_pixels > 0:
            step = np.linalg.norm(self.pixels[after] - self.pixels[before], axis=-1)
            ratio = np.maximum(ratio, step / self.max_pixels)
        if self.max_metres > 0:
            step = np.linalg.norm(self.metres[after] - self.metres[before], axis=-1)
            ratio = np.maximum(ratio, step / self.max_metres)
        if self.max_radians > 0:
            relative = np.swapaxes(self.orient[before], -1, -2) @ self.orient[after]
            cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0
            ratio = np.maximum(ratio, np.arccos(np.clip(cosine, -1.0, 1.0)) / self.max_radians)
        return ratio


def _jumps(motion: _WristMotion, in_view: FrameMask) -> list[int]:
    """Frames `t` whose wrist moved past a limit since `t - 1`, both frames in view."""

    frames = np.arange(1, len(in_view))
    jumped = motion.ratio(frames - 1, frames) > 1.0
    return [int(t) for t in frames[jumped & in_view[1:] & in_view[:-1]]]


def _spikes(motion: _WristMotion, in_view: FrameMask, max_length: int) -> list[tuple[int, int]]:
    """Half-open spans `[start, stop)` of at most `max_length` frames that a jump enters and, within
    `max_length` frames, leaves again close to where it started: the frames to interpolate."""

    spans: list[tuple[int, int]] = []
    for start in _jumps(motion, in_view):
        if spans and start < spans[-1][1]:
            continue
        for stop in range(start + 1, min(start + max_length, len(in_view) - 1) + 1):
            back = motion.ratio(np.array([start - 1]), np.array([stop]))[0]
            if in_view[stop] and back <= 0.5:
                spans.append((start, stop))
                break
    return spans


def _teleports(motion: _WristMotion, in_view: FrameMask, max_length: int) -> list[tuple[int, int]]:
    """The shorter visible run on either side of a jump, when it lasts at most `max_length` frames:
    a hand cannot be in both places, and the brief one is the glitch."""

    spans: list[tuple[int, int]] = []
    runs = _runs(in_view)
    for start in _jumps(motion, in_view):
        run = next((a, b) for a, b in runs if a <= start < b)
        before, after = (run[0], start), (start, run[1])
        shorter = before if before[1] - before[0] <= after[1] - after[0] else after
        if shorter[1] - shorter[0] <= max_length:
            spans.append(shorter)
    return spans


def _gaps(mask: FrameMask, max_length: int) -> list[tuple[int, int]]:
    """Absent spans of at most `max_length` frames that sit between two present frames."""

    return [
        (start, stop)
        for start, stop in _runs(~mask)
        if start > 0 and stop < len(mask) and stop - start <= max_length
    ]


def _smooth_wrist_depth(translation: ClipTranslation, weight: float) -> ClipTranslation:
    """Move the wrists of one run along their own rays to the acceleration-penalised depths."""

    depth = np.clip(translation[:, 2].astype(np.float64), MIN_DEPTH, None)
    rays = translation.astype(np.float64) / depth[:, None]  # z component is exactly 1
    smoothed = _smooth_depths(depth, weight)
    return (rays * smoothed[:, None]).astype(translation.dtype)


def _smooth_depths(depths: RunDepths, weight: float) -> RunDepths:
    """Solve `(I + w D^T D) d = d0` for the second-difference operator D of the run."""

    n = len(depths)
    if n < 3:
        return depths
    second_difference = np.zeros((n - 2, n))
    rows = np.arange(n - 2)
    second_difference[rows, rows] = 1.0
    second_difference[rows, rows + 1] = -2.0
    second_difference[rows, rows + 2] = 1.0
    # Symmetric positive definite and pentadiagonal; dense is fine at a few thousand frames
    system = np.eye(n) + weight * second_difference.T @ second_difference
    return np.linalg.solve(system, depths)


def _clip_betas(betas: ClipBetas, presence: ClipProbability) -> ClipBetas:
    """One presence-weighted mean shape per hand, repeated over the frames."""

    weights = presence.astype(np.float64)
    total = weights.sum(axis=0)
    # A hand that is never present falls back to the plain mean over its frames
    weights[:, total == 0] = 1.0
    weights /= weights.sum(axis=0)
    mean = einsum(weights, betas.astype(np.float64), "f s, f s k -> s k")
    return repeat(mean, "s k -> f s k", f=len(presence)).astype(betas.dtype)


def _decode_joints(
    global_orient: ClipGlobalOrient,
    hand_pose: ClipHandPose,
    betas: ClipBetas,
    translation: ClipTranslation,
    mano: ManoLayer,
) -> ClipJointsCamera:
    """MANO joints of every frame placed in the camera, like the inference path does."""

    device = mano.template.device
    with torch.inference_mode():
        joints = mano(
            torch.from_numpy(global_orient).to(device),
            torch.from_numpy(hand_pose).to(device),
            torch.from_numpy(betas).to(device),
        ).joints
    return (to_numpy(joints) + translation[:, :, None, :]).astype(translation.dtype)


# --- rotation interpolation ---


def _slerp_matrices(
    start: RotationArray, end: RotationArray, fractions: GapFractions
) -> RotationArray:
    """Interpolate every rotation of `start` towards its twin in `end`, one output per fraction."""

    q0 = _matrix_to_quaternion(start.astype(np.float64))
    q1 = _matrix_to_quaternion(end.astype(np.float64))
    # Take the short arc: q and -q are the same rotation
    q1 = np.where(einsum(q0, q1, "... i, ... i -> ...")[..., None] < 0, -q1, q1)
    dot = np.clip(einsum(q0, q1, "... i, ... i -> ..."), -1.0, 1.0)
    angle = np.arccos(dot)
    sine = np.sin(angle)

    # (N, ...) fractions against (...,) angles; the tiny-angle case degrades to a lerp
    t = fractions[(slice(None), *([None] * q0.ndim))]
    safe_sine = np.where(sine > 0, sine, 1.0)[..., None]
    linear = (dot > SLERP_LINEAR_DOT)[..., None]
    weight_start = np.where(linear, 1.0 - t, np.sin((1.0 - t) * angle[..., None]) / safe_sine)
    weight_end = np.where(linear, t, np.sin(t * angle[..., None]) / safe_sine)
    interpolated = weight_start * q0 + weight_end * q1
    interpolated /= np.linalg.norm(interpolated, axis=-1, keepdims=True)
    return _quaternion_to_matrix(interpolated).astype(start.dtype)


def _matrix_to_quaternion(rotation: RotationArray) -> Quaternions:
    """Rotation matrices to unit quaternions `(w, x, y, z)`, reading the best-conditioned row."""

    m = rotation
    # One candidate per component that could be the largest, each scaled by 4 times that component
    candidates = np.stack(
        [
            np.stack(
                [
                    1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2],
                    m[..., 2, 1] - m[..., 1, 2],
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] - m[..., 0, 1],
                ],
                axis=-1,
            ),
            np.stack(
                [
                    m[..., 2, 1] - m[..., 1, 2],
                    1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
                    m[..., 0, 1] + m[..., 1, 0],
                    m[..., 0, 2] + m[..., 2, 0],
                ],
                axis=-1,
            ),
            np.stack(
                [
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 0, 1] + m[..., 1, 0],
                    1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
                    m[..., 1, 2] + m[..., 2, 1],
                ],
                axis=-1,
            ),
            np.stack(
                [
                    m[..., 1, 0] - m[..., 0, 1],
                    m[..., 0, 2] + m[..., 2, 0],
                    m[..., 1, 2] + m[..., 2, 1],
                    1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
                ],
                axis=-1,
            ),
        ],
        axis=-2,
    )  # (..., 4 candidates, 4)

    # The candidate whose squared leading component is largest divides by the largest number
    leading = np.abs(np.diagonal(candidates, axis1=-2, axis2=-1))
    best = np.argmax(leading, axis=-1)
    quaternion = np.take_along_axis(candidates, best[..., None, None], axis=-2)[..., 0, :]
    return quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)


def _quaternion_to_matrix(quaternion: Quaternions) -> RotationArray:
    w, x, y, z = (quaternion[..., i] for i in range(4))
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )
