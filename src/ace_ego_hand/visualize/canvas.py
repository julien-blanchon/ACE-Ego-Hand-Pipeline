"""Shared drawing plumbing of the two panels: colours, sizes, the supersampled layer, labels.

Everything is drawn at 2x on a transparent RGBA layer and box-filtered down over the panel's
background, which is how PIL gets anti-aliased strokes. Every drawn size (bone thickness, joint
radius, font, margins) is a `Style` derived from the output width, so a panel reads the same at
960 and at 1920 pixels wide. Colours follow the upstream viewer: slot 0 (left) blue, slot 1
(right) pink.
Adapted from https://github.com/ggxxii/ACE-Ego-Hand/blob/main/scripts/viz_preds.py
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import PIL.Image
from PIL import ImageDraw, ImageFont

from ..data.video import UINT8_MAX
from ..types import ClipFaces, FrameVertices2D, FrameVerticesCamera, Image

type Color = tuple[int, int, int]
type Point2 = tuple[float, float]
# absent: the model says the hand does not exist; hidden: it exists but is not in view (the
# model keeps predicting it); out_of_frame: visible but projecting outside the image
type HandState = Literal["absent", "hidden", "out_of_frame", "in_frame"]

HAND_NAMES: tuple[str, str] = ("left", "right")  # slot 0, slot 1
HAND_COLORS: tuple[Color, Color] = ((90, 180, 230), (255, 105, 180))  # RGB: left blue, right pink
ABSENT_COLOR: Color = (150, 150, 155)  # dimmed grey of the "no left hand" markers
DIM_WEIGHT = 0.5  # a hidden hand's note: its colour pulled this far towards the absent grey
GHOST_WEIGHT = 0.4  # a hidden hand in the 3D panel: this much of its colour over the background
LABEL_OUTLINE: Color = (20, 20, 24)  # dark halo behind text so it reads over any frame
BACKGROUND: Color = (28, 28, 32)  # the 3D panel's ground
SUPERSAMPLE = 2  # draw at 2x and box-filter down: PIL strokes are not anti-aliased

# Drawn sizes, in pixels at `REFERENCE_WIDTH`; they scale linearly with the output width
REFERENCE_WIDTH = 960
STROKE_AT_REFERENCE = 3  # bone thickness
FONT_AT_REFERENCE = 18
MARGIN_AT_REFERENCE = 12  # inset of labels, legend and border markers from the panel edge
MIN_FONT = 8


@dataclass(frozen=True, slots=True)
class Style:
    """Drawn sizes for one output width, in output pixels."""

    stroke: int
    font_size: int
    margin: int

    @classmethod
    def at(cls, width: int) -> Style:
        scale = width / REFERENCE_WIDTH
        return cls(
            stroke=max(1, round(STROKE_AT_REFERENCE * scale)),
            font_size=max(MIN_FONT, round(FONT_AT_REFERENCE * scale)),
            margin=max(2, round(MARGIN_AT_REFERENCE * scale)),
        )


@dataclass(frozen=True, slots=True)
class HandStatus:
    """One hand's state on a frame, with the two probabilities that decided it."""

    state: HandState
    presence: float
    visible: float

    @property
    def drawn(self) -> bool:
        """Whether the hand is drawn in full (it is visible; in or out of the image)."""

        return self.state in ("in_frame", "out_of_frame")

    def label(self, name: str) -> str:
        """`left  p 0.99  v 0.12`: the name with its presence and visible probabilities."""

        return f"{name}  p {self.presence:.2f}  v {self.visible:.2f}"


@dataclass(frozen=True, slots=True)
class MeshFrame:
    """One frame's MANO surfaces: projected and camera-space vertices plus the triangles."""

    vertices_2d: FrameVertices2D
    vertices_camera: FrameVerticesCamera
    faces: ClipFaces


class Canvas:
    """A transparent 2x layer over a panel of `size`; coordinates are given in output pixels."""

    def __init__(self, size: tuple[int, int], style: Style) -> None:
        width, height = size
        self.size = size
        self.style = style
        self.layer = PIL.Image.new("RGBA", (SUPERSAMPLE * width, SUPERSAMPLE * height))
        self.draw = ImageDraw.Draw(self.layer)
        self.font = _font(SUPERSAMPLE * style.font_size)

    def line(self, start: Point2, end: Point2, color: Color, width: float = 1.0) -> None:
        """A bone or grid line; `width` is in multiples of the style's stroke."""

        thickness = max(1, round(SUPERSAMPLE * self.style.stroke * width))
        self.draw.line([_scaled(start), _scaled(end)], fill=color, width=thickness)

    def dot(self, center: Point2, color: Color, radius: float = 1.0) -> None:
        """A joint dot; `radius` is in multiples of the style's stroke."""

        x, y = _scaled(center)
        r = SUPERSAMPLE * self.style.stroke * radius
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    def outline(self, points: list[Point2], color: Color) -> None:
        """A closed polygon drawn as an outline with a dark halo (the out-of-frame arrow)."""

        thickness = SUPERSAMPLE * self.style.stroke
        corners = [_scaled(point) for point in points]
        self.draw.polygon(corners, outline=LABEL_OUTLINE, width=2 * thickness)
        self.draw.polygon(corners, outline=color, width=thickness)

    def box(self, corner: Point2, side: float, color: Color) -> None:
        """A filled square with its top-left at `corner` (a legend swatch)."""

        x, y = _scaled(corner)
        self.draw.rectangle([x, y, x + SUPERSAMPLE * side, y + SUPERSAMPLE * side], fill=color)

    def text(self, anchor: Point2, string: str, color: Color, *, align: str = "la") -> None:
        """Outlined text at `anchor`, kept inside the panel; `align` is a PIL anchor code."""

        width, height = self.size
        text_width, text_height = self.text_size(string)
        x, y = anchor
        if align[0] == "m":
            x -= text_width / 2
        if align[1] == "m":
            y -= text_height / 2
        margin = self.style.margin
        x = min(max(x, margin), max(margin, width - margin - text_width))
        y = min(max(y, margin), max(margin, height - margin - text_height))
        self.draw.text(
            _scaled((x, y)),
            string,
            fill=color,
            font=self.font,
            anchor="la",
            stroke_width=max(1, SUPERSAMPLE * self.style.stroke // 2),
            stroke_fill=LABEL_OUTLINE,
        )

    def text_size(self, string: str) -> tuple[float, float]:
        left, top, right, bottom = self.draw.textbbox((0, 0), string, font=self.font, anchor="la")
        return (right - left) / SUPERSAMPLE, (bottom - top) / SUPERSAMPLE

    def composite(self, background: Image, opacity: float = 1.0) -> Image:
        """Box-filter the layer down and blend it over `background` with `opacity`."""

        # `reduce` averages in premultiplied alpha and `paste` blends by the mask, both in C: the
        # layer is four times the frame and dominates the render done in numpy
        small = self.layer.reduce(SUPERSAMPLE)
        alpha = small.getchannel("A")
        if opacity < 1.0:
            alpha = alpha.point(_opacity_table(opacity))
        base = PIL.Image.fromarray(background)
        base.paste(small, mask=alpha)
        return np.asarray(base)


def blend(a: Color, b: Color, weight: float) -> Color:
    """`a` towards `b` by `weight` in [0, 1]."""

    return (
        round(a[0] + weight * (b[0] - a[0])),
        round(a[1] + weight * (b[1] - a[1])),
        round(a[2] + weight * (b[2] - a[2])),
    )


def _scaled(point: Point2) -> tuple[float, float]:
    return SUPERSAMPLE * point[0], SUPERSAMPLE * point[1]


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size)  # Pillow's bundled face: no system font needed


@lru_cache(maxsize=8)
def _opacity_table(opacity: float) -> list[int]:
    return [round(value * opacity) for value in range(int(UINT8_MAX) + 1)]
