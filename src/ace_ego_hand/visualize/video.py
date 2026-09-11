"""Stream a whole clip through the two panels and write the mp4 at the source frame rate.

The overlay is upscaled so its width is at least `min_width` (a 512-wide source becomes 960
wide, wider sources are kept), the 3D panel is rendered to match it, and the two are stacked
vertically (overlay on top) or side by side. Only the pose arrays are held in memory; the
source frames are decoded as they are drawn. The out-of-frame marker blinks on a quarter-second
period at the video's frame rate.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Literal

import numpy as np
import PIL.Image
import torch

from ..data.video import iter_frames, write_video
from ..modeling.mano import ManoLayer
from ..prediction import VideoPrediction, project_points
from ..types import ClipFaces, ClipVertices2D, ClipVerticesCamera, Image, WristTrail
from ..utils.tensors import to_numpy
from .canvas import MeshFrame, Style
from .overlay import hand_status, render_overlay
from .scene import render_3d

logger = logging.getLogger(__name__)

type Panels = Literal["overlay", "3d", "both"]
type Layout = Literal["vertical", "horizontal"]

PANEL_ASPECT = 16 / 9  # width / height of the 3D panel: the overlay's usual shape
TRAIL_LENGTH = 30  # frames of wrist history in the 3D panel
BLINK_PERIOD = 0.25  # seconds the out-of-frame marker stays on, then off
WRIST = 0


def render_video(
    video: Path,
    prediction: VideoPrediction,
    output: Path,
    *,
    panels: Panels = "both",
    layout: Layout = "vertical",
    min_width: int = 960,
    mesh: bool = False,
    mano: ManoLayer | None = None,
    presence_threshold: float = 0.5,
    visible_threshold: float = 0.5,
) -> int:
    """Draw the requested panels for every frame and write the mp4; returns the frame count.

    A hand below `presence_threshold` is absent, one below `visible_threshold` is hidden (the
    model keeps predicting a hand that exists but is out of sight): the overlay skips it and
    the 3D panel ghosts it.
    """

    num_frames = len(prediction.presence)
    surfaces = None
    if mesh:
        if mano is None:
            raise ValueError("mesh rendering needs a ManoLayer")
        surfaces = _mesh_vertices(prediction, mano)

    overlay_size, panel_size = _panel_sizes(prediction.source_size, min_width, layout)
    factor = np.array(overlay_size, dtype=np.float32) / np.array(prediction.source_size)
    style = Style.at(overlay_size[0])
    trails = [deque[np.ndarray](maxlen=TRAIL_LENGTH) for _ in range(prediction.presence.shape[1])]

    def frames() -> Iterator[Image]:
        for index, frame in enumerate(islice(iter_frames(video), num_frames)):
            joints_2d = prediction.joints_2d[index] * factor
            status = hand_status(
                joints_2d,
                prediction.presence[index],
                prediction.visible[index],
                overlay_size,
                presence_threshold=presence_threshold,
                visible_threshold=visible_threshold,
            )
            parts: list[Image] = []
            if panels != "3d":
                parts.append(
                    render_overlay(
                        _upscale(frame, overlay_size),
                        joints_2d,
                        status,
                        style=style,
                        mesh=_mesh_frame(surfaces, index, factor),
                        blink=blink_on(index, prediction.fps),
                    )
                )
            if panels != "overlay":
                for slot, trail in enumerate(trails):
                    if status[slot].drawn:
                        trail.append(prediction.joints_camera[index, slot, WRIST])
                    elif trail:
                        trail.popleft()  # let a vanished or hidden hand's trail fade out
                parts.append(
                    render_3d(
                        prediction.joints_camera[index],
                        status,
                        [_trail_array(trail) for trail in trails],
                        size=panel_size,
                        style=style,
                        intrinsics=prediction.intrinsics,
                        image_size=prediction.source_size,
                    )
                )
            yield parts[0] if len(parts) == 1 else np.concatenate(parts, axis=_stack_axis(layout))

    width, height = _output_size(panels, layout, overlay_size, panel_size)
    count = write_video(output, frames(), fps=prediction.fps, width=width, height=height)
    if count != num_frames:
        logger.warning("%s: %d poses but %d frames rendered", video, num_frames, count)
    logger.info("wrote %d frames (%s, %s) to %s", count, panels, layout, output)
    return count


def blink_on(index: int, fps: float) -> bool:
    """Whether the out-of-frame marker is lit on frame `index`: on, off, on... per `BLINK_PERIOD`."""

    return int(index / fps / BLINK_PERIOD) % 2 == 0


def _panel_sizes(
    source_size: tuple[int, int], min_width: int, layout: Layout
) -> tuple[tuple[int, int], tuple[int, int]]:
    """The overlay grows to `min_width`; the 3D panel matches its width (vertical) or height."""

    source_width, source_height = source_size
    overlay = _even_size(source_width, source_height, max(1.0, min_width / source_width))
    if layout == "vertical":
        return overlay, _even_size(overlay[0], overlay[0] / PANEL_ASPECT, 1.0)
    return overlay, _even_size(overlay[1] * PANEL_ASPECT, overlay[1], 1.0)


def _mesh_frame(
    surfaces: tuple[ClipVertices2D, ClipVerticesCamera, ClipFaces] | None,
    index: int,
    factor: np.ndarray,
) -> MeshFrame | None:
    """Frame `index` of the MANO surfaces, its projection scaled to the overlay's pixels."""

    if surfaces is None:
        return None
    vertices_2d, vertices_camera, faces = surfaces
    return MeshFrame(vertices_2d[index] * factor, vertices_camera[index], faces)


def _even_size(width: float, height: float, factor: float) -> tuple[int, int]:
    """Scale and round both sides to even pixels, which the yuv420p encoder needs."""

    return 2 * round(width * factor / 2), 2 * round(height * factor / 2)


def _stack_axis(layout: Layout) -> int:
    return 0 if layout == "vertical" else 1


def _output_size(
    panels: Panels, layout: Layout, overlay: tuple[int, int], panel: tuple[int, int]
) -> tuple[int, int]:
    if panels == "overlay":
        return overlay
    if panels == "3d":
        return panel
    if layout == "vertical":
        return overlay[0], overlay[1] + panel[1]
    return overlay[0] + panel[0], overlay[1]


def _upscale(frame: Image, size: tuple[int, int]) -> Image:
    if frame.shape[1::-1] == size:
        return frame
    return np.asarray(PIL.Image.fromarray(frame).resize(size, PIL.Image.Resampling.BILINEAR))


def _trail_array(trail: deque[np.ndarray]) -> WristTrail:
    return np.stack(list(trail)) if trail else np.zeros((0, 3), dtype=np.float32)


def _mesh_vertices(
    prediction: VideoPrediction, mano: ManoLayer
) -> tuple[ClipVertices2D, ClipVerticesCamera, ClipFaces]:
    """MANO surfaces of every frame, placed in the camera and projected through K."""

    device = mano.template.device
    with torch.inference_mode():
        output = mano(
            torch.from_numpy(prediction.global_orient).to(device),
            torch.from_numpy(prediction.hand_pose).to(device),
            torch.from_numpy(prediction.betas).to(device),
        )
    vertices = to_numpy(output.vertices) + prediction.translation[:, :, None, :]
    return project_points(vertices, prediction.intrinsics), vertices, mano.faces.cpu().numpy()
