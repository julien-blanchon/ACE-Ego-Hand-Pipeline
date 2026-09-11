"""The 3D panel: both hands, the real camera's frustum and a ground grid from a fixed viewpoint.

The virtual camera sits above, behind and to the right of the real one and looks down at where
egocentric hands are, over a 1 m grid; the real camera is drawn as its frustum and axis triad,
and each wrist leaves a fading trail. Every visible hand is drawn here whether or not it is in
the real image, labelled at its wrist; a hidden hand (the model says it exists but is out of
sight) is ghosted, thin and faded, so the reader sees where the model thinks it is. A legend
in the corner names the colours with each hand's presence / visible probabilities and greys
out an absent hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..types import ClipIntrinsics, FrameJointsCamera, Image, Pixels, Points, WristTrail
from .canvas import (
    ABSENT_COLOR,
    BACKGROUND,
    GHOST_WEIGHT,
    HAND_COLORS,
    HAND_NAMES,
    Canvas,
    Color,
    HandStatus,
    Point2,
    Style,
    blend,
)
from .overlay import SKELETON_EDGES, WRIST

type Point3 = tuple[float, float, float]

MIN_DEPTH = 1e-3  # metres; anything closer to the virtual camera than this is not drawn
DOT_RADIUS = 0.75  # joint dots, in multiples of the bone thickness
GHOST_WIDTH = 0.5  # a hidden hand's bones, in multiples of the base stroke

# The fixed virtual camera, in the real camera's frame (x right, y down, z forward)
VIEW_EYE = (0.55, -0.5, -0.55)
VIEW_TARGET = (0.0, 0.3, 0.5)
VIEW_FOCAL = 1.7  # focal length in units of the panel height
WORLD_DOWN = (0.0, 1.0, 0.0)
GRID_DEPTH = 0.5  # the ground-ish plane sits this far below the real camera
GRID_RANGE_X = (-0.5, 0.5)  # a 1 m x 1 m grid in front of the camera
GRID_RANGE_Z = (0.0, 1.0)
GRID_STEP = 0.1
FRUSTUM_DEPTH = 0.1  # the real camera's image plane is drawn this far from its centre
TRIAD_LENGTH = 0.05
GRID_COLOR: Color = (78, 78, 86)
FRUSTUM_COLOR: Color = (205, 205, 210)
TRIAD_COLORS: tuple[Color, Color, Color] = ((225, 70, 70), (70, 200, 70), (80, 130, 245))
LEGEND_LINE = 1.5  # legend row pitch, in multiples of the font size
ABSENT_SWATCH_WEIGHT = 0.35  # how far an absent hand's swatch fades towards the background


@dataclass(frozen=True, slots=True)
class _Stroke:
    """A line (or a dot when `start` is `end`) in the real camera's frame."""

    start: Point3
    end: Point3
    color: Color
    width: float  # in multiples of the base stroke


def render_3d(
    joints_camera: FrameJointsCamera,
    status: Sequence[HandStatus],
    trails: Sequence[WristTrail],
    *,
    size: tuple[int, int],
    intrinsics: ClipIntrinsics,
    image_size: tuple[int, int],
    style: Style | None = None,
) -> Image:
    """The scene from the virtual camera: grid, frustum, hands with labels, trails, legend."""

    width, height = size
    canvas = Canvas(size, style or Style.at(width))
    strokes = [*_grid_strokes(), *_frustum_strokes(intrinsics, image_size), *_triad_strokes()]
    for slot, hand in enumerate(status):
        if hand.state != "absent":
            color, stroke_width = _hand_paint(hand, slot)
            strokes.extend(_skeleton_strokes(joints_camera[slot], color, stroke_width))
    for slot, trail in enumerate(trails):
        strokes.extend(_trail_strokes(trail, HAND_COLORS[slot]))

    # Project both ends through the virtual camera, drop what is behind it, paint far to near
    rotation, eye = _view_pose()
    starts = np.array([stroke.start for stroke in strokes], dtype=np.float32)
    ends = np.array([stroke.end for stroke in strokes], dtype=np.float32)
    pixels_start, depth_start = _project_view(starts, rotation, eye, size)
    pixels_end, depth_end = _project_view(ends, rotation, eye, size)
    visible = (depth_start > MIN_DEPTH) & (depth_end > MIN_DEPTH)
    for index in np.argsort(-(depth_start + depth_end), kind="stable"):
        if not visible[index]:
            continue
        stroke = strokes[index]
        start, end = _point(pixels_start[index]), _point(pixels_end[index])
        if stroke.start == stroke.end:
            canvas.dot(start, stroke.color, DOT_RADIUS * stroke.width)
        else:
            canvas.line(start, end, stroke.color, stroke.width)

    # Name each drawn hand at its wrist, then the legend
    wrists, wrist_depth = _project_view(joints_camera[:, WRIST], rotation, eye, size)
    offset = canvas.style.font_size * 0.6
    for slot, hand in enumerate(status):
        if hand.state == "absent" or wrist_depth[slot] <= MIN_DEPTH:
            continue
        anchor = (float(wrists[slot, 0]) + offset, float(wrists[slot, 1]) + offset)
        name = HAND_NAMES[slot] if hand.drawn else f"{HAND_NAMES[slot]} (not visible)"
        canvas.text(anchor, name, _hand_paint(hand, slot)[0])
    _draw_legend(canvas, status)

    background = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    return canvas.composite(background)


def _hand_paint(hand: HandStatus, slot: int) -> tuple[Color, float]:
    """Colour and stroke width of a hand's skeleton and label: full, or ghosted when hidden."""

    if hand.drawn:
        return HAND_COLORS[slot], 1.0
    return blend(BACKGROUND, HAND_COLORS[slot], GHOST_WEIGHT), GHOST_WIDTH


def _draw_legend(canvas: Canvas, status: Sequence[HandStatus]) -> None:
    """Swatches, names and probabilities in the top-left corner; absent hands dimmed."""

    margin, font_size = canvas.style.margin, canvas.style.font_size
    for slot, (name, hand) in enumerate(zip(HAND_NAMES, status, strict=True)):
        top = margin + slot * LEGEND_LINE * font_size
        if hand.state == "absent":
            swatch = blend(BACKGROUND, HAND_COLORS[slot], ABSENT_SWATCH_WEIGHT)
            label, color = f"no {name} hand", ABSENT_COLOR
        else:
            color, _ = _hand_paint(hand, slot)
            swatch, label = color, hand.label(name)
        canvas.box((margin, top), font_size, swatch)
        canvas.text((margin + 1.5 * font_size, top), label, color)


def _view_pose() -> tuple[np.ndarray, np.ndarray]:
    """Rotation (rows: right, down, forward) and position of the virtual camera, look-at style."""

    eye = np.array(VIEW_EYE, dtype=np.float32)
    forward = np.array(VIEW_TARGET, dtype=np.float32) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(np.array(WORLD_DOWN, dtype=np.float32), forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward]), eye


def _project_view(
    points: Points, rotation: np.ndarray, eye: np.ndarray, size: tuple[int, int]
) -> tuple[Pixels, np.ndarray]:
    """Pinhole projection through the virtual camera; returns pixels and view depths."""

    width, height = size
    in_view = (points - eye) @ rotation.T
    focal = VIEW_FOCAL * height
    depth = in_view[:, 2]
    safe_depth = np.where(depth > MIN_DEPTH, depth, MIN_DEPTH)
    u = focal * in_view[:, 0] / safe_depth + width / 2
    v = focal * in_view[:, 1] / safe_depth + height / 2
    return np.stack([u, v], axis=-1), depth


def _grid_strokes() -> list[_Stroke]:
    xs = np.arange(GRID_RANGE_X[0], GRID_RANGE_X[1] + GRID_STEP / 2, GRID_STEP)
    zs = np.arange(GRID_RANGE_Z[0], GRID_RANGE_Z[1] + GRID_STEP / 2, GRID_STEP)
    along_z = [
        _Stroke((x, GRID_DEPTH, zs[0]), (x, GRID_DEPTH, zs[-1]), GRID_COLOR, 0.5) for x in xs
    ]
    along_x = [
        _Stroke((xs[0], GRID_DEPTH, z), (xs[-1], GRID_DEPTH, z), GRID_COLOR, 0.5) for z in zs
    ]
    return along_z + along_x


def _frustum_strokes(intrinsics: ClipIntrinsics, image_size: tuple[int, int]) -> list[_Stroke]:
    """The real camera: four rays from its centre to the image corners at `FRUSTUM_DEPTH`."""

    width, height = image_size
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    corners = [
        (
            float((u - cx) / fx * FRUSTUM_DEPTH),
            float((v - cy) / fy * FRUSTUM_DEPTH),
            FRUSTUM_DEPTH,
        )
        for u, v in ((0, 0), (width, 0), (width, height), (0, height))
    ]
    origin = (0.0, 0.0, 0.0)
    rays = [_Stroke(origin, corner, FRUSTUM_COLOR, 0.5) for corner in corners]
    rim = [_Stroke(corners[i], corners[(i + 1) % 4], FRUSTUM_COLOR, 0.5) for i in range(4)]
    return rays + rim


def _triad_strokes() -> list[_Stroke]:
    """A small x / y / z triad of the real camera's frame, at its centre."""

    origin = (0.0, 0.0, 0.0)
    tips = (
        (origin[0] + TRIAD_LENGTH, origin[1], origin[2]),
        (origin[0], origin[1] + TRIAD_LENGTH, origin[2]),
        (origin[0], origin[1], origin[2] + TRIAD_LENGTH),
    )
    return [_Stroke(origin, tip, color, 1.0) for tip, color in zip(tips, TRIAD_COLORS, strict=True)]


def _point3(coordinates: np.ndarray) -> Point3:
    return float(coordinates[0]), float(coordinates[1]), float(coordinates[2])


def _point(coordinates: np.ndarray) -> Point2:
    return float(coordinates[0]), float(coordinates[1])


def _skeleton_strokes(joints: np.ndarray, color: Color, width: float) -> list[_Stroke]:
    points = [_point3(joint) for joint in joints]
    bones = [_Stroke(points[a], points[b], color, width) for a, b in SKELETON_EDGES]
    dots = [_Stroke(point, point, color, width) for point in points]
    return bones + dots


def _trail_strokes(trail: WristTrail, color: Color) -> list[_Stroke]:
    """Segments between consecutive wrist positions, fading into the background with age."""

    points = [_point3(position) for position in trail]
    strokes: list[_Stroke] = []
    for i in range(len(points) - 1):
        age = (i + 1) / (len(points) - 1)  # 0 oldest, 1 newest
        strokes.append(_Stroke(points[i], points[i + 1], blend(BACKGROUND, color, age), 0.75))
    return strokes
