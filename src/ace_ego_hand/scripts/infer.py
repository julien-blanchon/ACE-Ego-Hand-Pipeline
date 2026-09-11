# /// script
# requires-python = ">=3.12"
# dependencies = ["ace-ego-hand @ git+https://github.com/julien-blanchon/ACE-Ego-Hand-Pipeline"]
# ///
"""Estimate bimanual 3D hand poses for one or more videos (local files or URLs).

Each video gets a pose table (`<stem>.hands.parquet` by default) and, with `--render`, a
visualisation video (`<stem>.hands.mp4`), next to a local file or in the working directory for
a URL (`--output-dir` overrides both). Any container FFmpeg decodes is accepted.

    uv run ace-ego-hand infer clip.mp4 other.mov --render
    uv run ace-ego-hand infer https://example.org/clip.mp4 --render
    uv run ace-ego-hand infer clip.mp4 --model.variant k --intrinsics 700,700,640,360
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro

from ace_ego_hand.config import InferenceConfig, ModelConfig
from ace_ego_hand.data.calibration import camera_sidecar_path, read_camera_sidecar, read_intrinsics
from ace_ego_hand.data.pose_table import (
    OutputFormat,
    pose_table_path,
    prediction_to_table,
    table_info,
    write_pose_table,
)
from ace_ego_hand.data.sources import fetch_video, output_dir
from ace_ego_hand.data.video import probe
from ace_ego_hand.inference import HandEstimator, load_estimator
from ace_ego_hand.modeling.mano import ManoLayer
from ace_ego_hand.postprocess import PostprocessConfig, postprocess
from ace_ego_hand.types import Intrinsics
from ace_ego_hand.visualize import render_video


@dataclass(frozen=True, slots=True)
class InferConfig:
    """Hand poses for videos, written next to each file (or into --output-dir)."""

    videos: tyro.conf.Positional[tuple[str, ...]]
    """Input videos: paths or http(s) URLs. A `<stem>.camera.parquet` / `.csv` sidecar next to a
    local video is picked up."""

    output_dir: Path | None = None
    """Where the outputs go; default next to each local video, the working directory for URLs."""

    intrinsics: str | None = None
    """`fx,fy,cx,cy` in video pixels, or a `.npy` / `.parquet` / `.csv` calibration file. Required
    by the K variant; with K-free it is only used to project the 2D output."""

    output_format: OutputFormat = "parquet"
    """Pose table format; csv flattens the list columns and writes the metadata to a .json."""

    render: bool = False
    """Also write `<stem>.hands.mp4` with the overlay and the 3D view."""

    panels: Literal["overlay", "3d", "both"] = "both"
    """Panels of the rendered video."""

    mesh: bool = False
    """Draw the shaded MANO mesh instead of the skeleton in the overlay."""

    postprocess: bool = False
    """Repair wrist jumps, smooth the wrist depth, use one hand shape per clip and bridge short
    presence gaps."""

    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def video_intrinsics(video: Path, spec: str | None) -> tuple[str, Intrinsics] | None:
    """Resolve the calibration for one video: the CLI value, else a sidecar next to the video."""

    info = probe(video)
    size = (info.width, info.height)
    if spec is not None:
        return ("cli", read_intrinsics(spec, size))
    sidecar = camera_sidecar_path(video)
    if sidecar is not None:
        return (sidecar.name, read_camera_sidecar(sidecar, size))
    return None


def run_video(
    estimator: HandEstimator, source: str, config: InferConfig, mano: ManoLayer | None
) -> Path:
    video = fetch_video(source)
    outputs = output_dir(source, video, config.output_dir)
    resolved = video_intrinsics(video, config.intrinsics)
    intrinsics = None if resolved is None else resolved[1]
    started = time.perf_counter()
    prediction = estimator.predict(video, intrinsics=intrinsics, config=config.inference)
    if config.postprocess:
        prediction = postprocess(prediction, PostprocessConfig(), mano=estimator.mano)
    elapsed = time.perf_counter() - started

    table = prediction_to_table(prediction, table_info(prediction, video, estimator.weights))
    output = write_pose_table(
        pose_table_path(outputs / video.name, config.output_format), table, config.output_format
    )
    frames = len(prediction.presence)
    camera = (
        "fitted camera"
        if prediction.intrinsics_fitted
        else f"camera from {resolved[0] if resolved else 'cli'}"
    )
    print(
        f"{video.name}: {frames} frames in {elapsed:.1f} s ({frames / elapsed:.1f} fps), {camera} -> {output.name}"
    )

    if config.render:
        rendered = outputs / f"{video.stem}.hands.mp4"
        render_video(video, prediction, rendered, panels=config.panels, mesh=config.mesh, mano=mano)
        print(f"{video.name}: rendered {rendered.name}")
    return output


def main(config: InferConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if config.model.variant == "k":
        for source in config.videos:
            if video_intrinsics(fetch_video(source), config.intrinsics) is None:
                raise SystemExit(f"{source}: the K variant needs --intrinsics or a camera sidecar")

    estimator = load_estimator(config.model)
    mano = estimator.mano if config.render and config.mesh else None
    for index, source in enumerate(config.videos, start=1):
        print(f"[{index}/{len(config.videos)}] {source}")
        run_video(estimator, source, config, mano)


if __name__ == "__main__":
    main(tyro.cli(InferConfig))
