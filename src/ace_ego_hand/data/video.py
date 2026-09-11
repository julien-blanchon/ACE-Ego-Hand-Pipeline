"""Decode, resize and encode video with PyAV, the only place the package talks to FFmpeg.

Reading yields RGB frames as uint8 tensors, either a whole file or a frame range of one
episode inside a shared file (LeRobot packs many episodes per mp4). Frames are resized on the
device with an antialiased area-style downscale to the model's encode size. Writing takes RGB
uint8 frames and produces an H.264 mp4.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch
from einops import einsum, rearrange

from ..types import Clip, Frames

ENCODE_GRID = 32  # the latent encoders stride 16 and the DiT patches 2x2 latent cells
UINT8_MAX = 255.0


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """What a decoder learns from a file's header before reading frames."""

    width: int
    height: int
    fps: float
    num_frames: int  # from the stream header; the decode loop is the authority


def probe(path: Path) -> VideoInfo:
    """Read size, frame rate and frame count from the container header."""

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate or Fraction(30, 1)
        frames = stream.frames
        if not frames and stream.duration and stream.time_base:
            frames = round(float(stream.duration * stream.time_base) * float(rate))
        return VideoInfo(
            int(stream.codec_context.width),
            int(stream.codec_context.height),
            float(rate),
            int(frames),
        )


def iter_frames(
    path: Path, *, start_seconds: float = 0.0, num_frames: int | None = None
) -> Iterator[np.ndarray]:
    """Yield RGB `(H, W, 3)` uint8 frames from `start_seconds`, at most `num_frames` of them.

    Seeking lands on the preceding keyframe; frames before `start_seconds` are decoded and
    dropped, which is what makes an episode inside a shared LeRobot file addressable by time.
    """

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if start_seconds > 0.0:
            time_base = stream.time_base or Fraction(1, 90_000)
            container.seek(int(start_seconds / float(time_base)), stream=stream, backward=True)
        yielded = 0
        for frame in container.decode(stream):
            if frame.time < start_seconds - 1e-6:
                continue
            if num_frames is not None and yielded >= num_frames:
                break
            yield frame.to_ndarray(format="rgb24")
            yielded += 1


def read_clip(path: Path, *, start_seconds: float = 0.0, num_frames: int | None = None) -> Clip:
    """Decode frames into one `(F, H, W, 3)` uint8 tensor."""

    frames = list(iter_frames(path, start_seconds=start_seconds, num_frames=num_frames))
    if not frames:
        raise ValueError(f"{path}: no frames decoded")
    return torch.from_numpy(np.stack(frames))


def encode_size(width: int, height: int, encode_width: int) -> tuple[int, int]:
    """Size the model sees: scale the width to `encode_width`, keep the aspect, snap both to the grid."""

    scale = encode_width / width
    snapped_w = max(ENCODE_GRID, round(width * scale / ENCODE_GRID) * ENCODE_GRID)
    snapped_h = max(ENCODE_GRID, round(height * scale / ENCODE_GRID) * ENCODE_GRID)
    return snapped_w, snapped_h


def area_weights(source: int, target: int, device: torch.device) -> torch.Tensor:
    """`(target, source)` resampling matrix of OpenCV's `INTER_AREA`: interval-overlap weights.

    Destination pixel `i` covers the source interval `[i s, (i + 1) s)` with `s = source /
    target`; each source pixel contributes the length of its overlap with that interval,
    normalized by `s`. This is the exact box filter when shrinking and the two-tap "area zoom"
    OpenCV uses when enlarging, which is what the reference pipeline resized with.
    """

    # Built in float64 on the CPU (MPS has no float64) and moved as float32
    scale = source / target
    starts = torch.arange(target, dtype=torch.float64) * scale
    ends = starts + scale
    pixels = torch.arange(source, dtype=torch.float64)
    overlap = torch.minimum(ends[:, None], pixels[None, :] + 1) - torch.maximum(
        starts[:, None], pixels[None, :]
    )
    return (overlap.clamp_min(0.0) / scale).float().to(device)


def frames_to_model_input(clip: Clip, size: tuple[int, int], device: torch.device) -> Frames:
    """uint8 `(F, H, W, 3)` -> float `(1, 3, F, H', W')` in [0, 1], resized on `device`.

    The resize reproduces OpenCV's `INTER_AREA` as two matrix products (separable overlap
    weights) and rounds back to 8 bits, as the reference pipeline's frames were.
    """

    frames = rearrange(clip.to(device, non_blocking=True), "f h w c -> f c h w").float()
    width, height = size
    if (clip.shape[2], clip.shape[1]) != (width, height):
        rows = area_weights(clip.shape[1], height, device)
        cols = area_weights(clip.shape[2], width, device)
        frames = einsum(rows, frames, cols, "hh h, f c h w, ww w -> f c hh ww")
        frames = frames.round().clamp(0.0, UINT8_MAX)
    return rearrange(frames / UINT8_MAX, "f c h w -> 1 c f h w")


def write_video(
    path: Path, frames: Iterator[np.ndarray], *, fps: float, width: int, height: int
) -> int:
    """Encode RGB uint8 `(H, W, 3)` frames as H.264 mp4; returns the number of frames written."""

    rate = Fraction(fps).limit_denominator(1000)
    count = 0
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        for image in frames:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
            count += 1
        for packet in stream.encode():
            container.mux(packet)
    return count
