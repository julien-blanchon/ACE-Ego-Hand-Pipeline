# /// script
# requires-python = ">=3.12"
# dependencies = ["ace-ego-hand @ git+https://github.com/julien-blanchon/ACE-Ego-Hand-Pipeline"]
# ///
"""Resample a fisheye or distorted video to a pinhole view the model can take.

Writes `<stem>.pinhole.mp4` and its `<stem>.pinhole.camera.parquet` sidecar next to the
input, so `ace-ego-hand infer <stem>.pinhole.mp4 --model.variant k` finds the calibration.

    uv run ace-ego-hand undistort fisheye.mp4 --camera fisheye.camera.parquet
    uv run ace-ego-hand undistort fisheye.mp4 --camera 300,300,320,240,kannala_brandt,0.1,-0.02
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import batched
from pathlib import Path

import numpy as np
import torch
import tyro

from ace_ego_hand.data.calibration import (
    CameraModel,
    camera_sidecar_path,
    parse_camera_model,
    read_camera_model,
    write_camera_sidecar,
)
from ace_ego_hand.data.undistort import remap, sampling_map, target_pinhole
from ace_ego_hand.data.video import iter_frames, probe, write_video
from ace_ego_hand.types import Image, PixelMap


@dataclass(frozen=True, slots=True)
class UndistortConfig:
    """Undistort one video to a pinhole view and write the pinhole calibration next to it."""

    video: tyro.conf.Positional[Path]
    camera: str | None = None
    """A `.camera.parquet` / `.csv` sidecar, or inline `fx,fy,cx,cy,<distortion>,k1,k2,...` in
    video pixels (`brown_conrady`: k1,k2,p1,p2,k3,k4,k5,k6; `kannala_brandt`: k1,k2,k3,k4).
    Default: the `<stem>.camera.*` sidecar next to the video."""

    output: Path | None = None  # default `<stem>.pinhole.mp4` next to the video
    fov_scale: float = 1.0  # > 1 widens the pinhole view (black corners), < 1 crops in
    device: str = "cuda"
    batch_size: int = 16  # frames resampled per GPU call


def source_camera(video: Path, spec: str | None) -> CameraModel:
    """The lens of `video`: the CLI value, else its sidecar, rescaled to the video's pixels."""

    info = probe(video)
    size = (info.width, info.height)
    if spec is not None:
        return parse_camera_model(spec, size)
    sidecar = camera_sidecar_path(video)
    if sidecar is None:
        raise SystemExit(f"{video}: no --camera and no <stem>.camera.parquet/.csv sidecar")
    return read_camera_model(sidecar).rescaled(size)


def undistorted_frames(video: Path, pixel_map: PixelMap, batch_size: int) -> Iterator[Image]:
    for batch in batched(iter_frames(video), batch_size):
        yield from remap(np.stack(batch), pixel_map, batch_size=batch_size)


def main(config: UndistortConfig) -> None:
    output = config.output or config.video.with_name(f"{config.video.stem}.pinhole.mp4")
    info = probe(config.video)
    source = source_camera(config.video, config.camera)
    target = target_pinhole(source, source.size, fov_scale=config.fov_scale)
    print(
        f"{config.video.name}: {source.distortion} -> pinhole fx={target.fx:.1f} fy={target.fy:.1f} "
        f"cx={target.cx:.1f} cy={target.cy:.1f} ({target.size[0]}x{target.size[1]})"
    )

    device = torch.device(config.device)
    pixel_map = sampling_map(source, target, device)
    count = write_video(
        output,
        undistorted_frames(config.video, pixel_map, config.batch_size),
        fps=info.fps,
        width=target.size[0],
        height=target.size[1],
    )
    sidecar = write_camera_sidecar(output.with_name(f"{output.stem}.camera.parquet"), target)
    print(f"wrote {count} frames to {output} and the pinhole calibration to {sidecar.name}")


if __name__ == "__main__":
    main(tyro.cli(UndistortConfig))
