"""The overlay panel: skeletons or the shaded MANO mesh of the hands over one source frame.

`hand_status` decides what each hand is on a frame from the model's two probabilities, then
the geometry. The model keeps predicting a hand that exists but is out of sight (`presence`
stays ~1 while `visible` drops), so `visible` is what gates the drawing: a hand below the
visible threshold is "hidden" and gets only a dimmed corner note, never a skeleton at its
guessed place. A visible hand whose wrist, or most of whose joints, project outside the image
is "out of frame": its in-image joints are drawn as usual and a blinking outlined arrow at the
border points to where it is. An absent hand (presence below its threshold) gets a grey "no
left hand" note. In-frame hands are labelled at the wrist with both probabilities. The mesh
path rasterises both hands' triangles depth-sorted together (painter's algorithm), so one hand
occludes the other.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from einops import rearrange

from ..types import FrameJoints2D, FrameProbability, Image, Pixels
from .canvas import (
    ABSENT_COLOR,
    DIM_WEIGHT,
    HAND_COLORS,
    HAND_NAMES,
    SUPERSAMPLE,
    Canvas,
    Color,
    HandState,
    HandStatus,
    MeshFrame,
    Point2,
    Style,
    blend,
)

# OpenPose-21 hand skeleton: wrist -> [root..tip] of thumb, index, middle, ring, pinky
SKELETON_EDGES: tuple[tuple[int, int], ...] = tuple(
    (0 if i == 0 else finger + i, finger + i + 1) for finger in (0, 4, 8, 12, 16) for i in range(4)
)
WRIST = 0
MIN_INSIDE_FRACTION = 0.5  # fewer joints than this inside the image and the hand is out of frame
MESH_OPACITY = 0.6
MESH_AMBIENT = 0.35  # Lambert-ish shading: faces seen head-on brightest, grazing ones dimmer
MIN_DEPTH = 1e-3  # metres; anything closer to the camera than this is not drawn
ARROW_LENGTH = 1.6  # the border arrow, in multiples of the font size
ARROW_HALF_WIDTH = 0.7


def render_overlay(
    frame: Image,
    joints_2d: FrameJoints2D,
    status: Sequence[HandStatus],
    *,
    style: Style | None = None,
    mesh: MeshFrame | None = None,
    blink: bool = True,
) -> Image:
    """The visible hands over one frame, with labels and the out-of-frame / hidden / absent notes."""

    height, width = frame.shape[:2]
    canvas = Canvas((width, height), style or Style.at(width))
    drawn = [slot for slot, hand in enumerate(status) if hand.drawn]

    if mesh is None:
        for slot in drawn:
            _draw_skeleton(canvas, joints_2d[slot], HAND_COLORS[slot])
    else:
        for corners, color in _shaded_triangles(mesh, drawn, (width, height)):
            canvas.draw.polygon(corners, fill=color)

    for slot, hand in enumerate(status):
        _draw_marker(canvas, hand, slot, joints_2d[slot], blink)
    return canvas.composite(frame, 1.0 if mesh is None else MESH_OPACITY)


def hand_status(
    joints_2d: FrameJoints2D,
    presence: FrameProbability,
    visible: FrameProbability,
    size: tuple[int, int],
    *,
    presence_threshold: float,
    visible_threshold: float,
) -> tuple[HandStatus, ...]:
    """Each hand's state on the frame: the two probabilities first, then where it projects."""

    inside = _inside(joints_2d, size)
    status: list[HandStatus] = []
    for slot, (p, v) in enumerate(zip(presence.tolist(), visible.tolist(), strict=True)):
        state: HandState
        if p < presence_threshold:
            state = "absent"
        elif v < visible_threshold:
            state = "hidden"
        elif inside[slot, WRIST] and inside[slot].mean() >= MIN_INSIDE_FRACTION:
            state = "in_frame"
        else:
            state = "out_of_frame"
        status.append(HandStatus(state, p, v))
    return tuple(status)


def _inside(points: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    finite = np.isfinite(points).all(axis=-1)
    bounded = (points >= 0).all(axis=-1) & (points < np.array(size)).all(axis=-1)
    return finite & bounded


def _draw_skeleton(canvas: Canvas, joints: Pixels, color: Color) -> None:
    """Bones then dots; PIL clips what falls outside the image, so the in-image part shows."""

    width, height = canvas.size
    if not np.isfinite(joints).all():
        return
    # Far-off projections (a joint near the camera plane) are pulled in so PIL stays cheap
    points = np.clip(joints, (-width, -height), (2 * width, 2 * height))
    for a, b in SKELETON_EDGES:
        canvas.line(_point(points[a]), _point(points[b]), color)
    for point in points:
        canvas.dot(_point(point), color)


def _draw_marker(canvas: Canvas, hand: HandStatus, slot: int, joints: Pixels, blink: bool) -> None:
    """The per-hand annotation: a wrist label, the border arrow, or a corner note."""

    name = HAND_NAMES[slot]
    corner = (canvas.style.margin, canvas.style.margin + slot * canvas.style.font_size * 1.4)
    if hand.state == "absent":
        canvas.text(corner, f"no {name} hand", ABSENT_COLOR)
    elif hand.state == "hidden":
        note = f"{name}: not visible ({hand.visible:.2f})"
        canvas.text(corner, note, blend(HAND_COLORS[slot], ABSENT_COLOR, DIM_WEIGHT))
    elif hand.state == "in_frame":
        wrist = joints[WRIST]
        offset = canvas.style.font_size * 0.6
        canvas.text((wrist[0] + offset, wrist[1] + offset), hand.label(name), HAND_COLORS[slot])
    elif blink:
        _draw_border_arrow(canvas, joints, slot)


def _draw_border_arrow(canvas: Canvas, joints: Pixels, slot: int) -> None:
    """An outlined arrow on the image border pointing at the hand, with its label."""

    width, height = canvas.size
    finite = joints[np.isfinite(joints).all(axis=-1)]
    target = joints[WRIST] if np.isfinite(joints[WRIST]).all() else finite.mean(axis=0)
    if target.shape != (2,) or not np.isfinite(target).all():
        return
    center = np.array([width / 2, height / 2])
    direction = target - center
    norm = float(np.linalg.norm(direction))
    if norm < 1.0:
        return
    direction = direction / norm

    # Where the ray from the centre towards the hand leaves the inset image rectangle
    margin = canvas.style.margin + canvas.style.stroke
    half = np.array([width / 2 - margin, height / 2 - margin])
    reach = np.min(half / np.maximum(np.abs(direction), 1e-6))
    tip = center + min(reach, norm) * direction
    length = ARROW_LENGTH * canvas.style.font_size
    base = tip - length * direction
    side = ARROW_HALF_WIDTH * canvas.style.font_size * np.array([-direction[1], direction[0]])
    canvas.outline([_point(tip), _point(base + side), _point(base - side)], HAND_COLORS[slot])

    label = base - canvas.style.font_size * direction
    canvas.text(_point(label), f"{HAND_NAMES[slot]}: out of frame", HAND_COLORS[slot], align="mm")


def _point(coordinates: np.ndarray) -> Point2:
    return float(coordinates[0]), float(coordinates[1])


def _shaded_triangles(
    mesh: MeshFrame, slots: Sequence[int], size: tuple[int, int]
) -> list[tuple[list[float], Color]]:
    """The drawn hands' visible triangles as flat 2x pixel lists with their shade, far to near."""

    width, height = size
    depths: list[np.ndarray] = []
    corners: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for slot in slots:
        faces = mesh.faces[slot]
        corners_2d = mesh.vertices_2d[slot][faces]  # (M, 3, 2)
        corners_3d = mesh.vertices_camera[slot][faces]  # (M, 3, 3)
        depth = corners_3d[..., 2]

        # Cull per face: non-finite, behind the camera, or entirely outside the image
        keep = (
            np.isfinite(corners_2d).all(axis=(-1, -2))
            & np.isfinite(corners_3d).all(axis=(-1, -2))
            & (depth > MIN_DEPTH).all(axis=-1)
            & (corners_2d[..., 0].max(axis=-1) >= 0)
            & (corners_2d[..., 0].min(axis=-1) < width)
            & (corners_2d[..., 1].max(axis=-1) >= 0)
            & (corners_2d[..., 1].min(axis=-1) < height)
        )
        corners_2d, corners_3d, depth = corners_2d[keep], corners_3d[keep], depth[keep]

        # Shade by how squarely the face looks at the camera
        normal = np.cross(corners_3d[:, 1] - corners_3d[:, 0], corners_3d[:, 2] - corners_3d[:, 0])
        facing = np.abs(normal[:, 2]) / np.clip(np.linalg.norm(normal, axis=-1), 1e-9, None)
        shade = MESH_AMBIENT + (1 - MESH_AMBIENT) * facing
        colors.append((shade[:, None] * np.array(HAND_COLORS[slot], np.float32)).round())
        depths.append(depth.mean(axis=-1))
        corners.append(SUPERSAMPLE * np.clip(corners_2d, -width, 2 * width))
    if not depths:
        return []

    order = np.argsort(-np.concatenate(depths), kind="stable")  # painter's: far first
    flat = rearrange(np.concatenate(corners)[order], "m v c -> m (v c)").tolist()
    rgb = np.concatenate(colors)[order].astype(np.uint8).tolist()
    return [(triangle, (r, g, b)) for triangle, (r, g, b) in zip(flat, rgb, strict=True)]
