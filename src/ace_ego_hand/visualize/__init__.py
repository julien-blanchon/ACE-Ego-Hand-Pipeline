"""Draw the predicted hands over the source video and in a 3D panel, with PIL only.

`canvas.py` holds the shared drawing plumbing (supersampled layer, sizes that scale with the
output width, labels), `overlay.py` the source-frame panel, `scene.py` the 3D panel and
`video.py` streams a whole clip through both and writes the mp4.
"""

from __future__ import annotations

from .canvas import (
    ABSENT_COLOR,
    BACKGROUND,
    HAND_COLORS,
    HandState,
    HandStatus,
    MeshFrame,
    Style,
)
from .overlay import hand_status, render_overlay
from .scene import render_3d
from .video import Layout, Panels, blink_on, render_video

__all__ = [
    "ABSENT_COLOR",
    "BACKGROUND",
    "HAND_COLORS",
    "HandState",
    "HandStatus",
    "Layout",
    "MeshFrame",
    "Panels",
    "Style",
    "blink_on",
    "hand_status",
    "render_3d",
    "render_overlay",
    "render_video",
]
