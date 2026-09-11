# /// script
# requires-python = ">=3.12"
# dependencies = ["ace-ego-hand @ git+https://github.com/julien-blanchon/ACE-Ego-Hand-Pipeline"]
# ///
"""Render a pose table over its source video (a local file or a URL).

uv run ace-ego-hand visualize clip.mp4 --mesh
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import tyro

from ace_ego_hand.config import WEIGHTS_REPO
from ace_ego_hand.data.pose_table import pose_table_path, read_pose_table
from ace_ego_hand.data.sources import fetch_video, output_dir
from ace_ego_hand.modeling.mano import ManoLayer
from ace_ego_hand.utils.hub import component_dir
from ace_ego_hand.visualize import Layout, Panels, render_video


@dataclass(frozen=True, slots=True)
class VisualizeConfig:
    """Draw the predicted hands of a video: a 2D overlay, a 3D panel, or both side by side."""

    video: tyro.conf.Positional[str]  # the source clip (path or URL) the poses were predicted from
    poses: Path | None = None  # the pose table; default `<stem>.hands.parquet` next to the video
    output: Path | None = None  # the rendered mp4; default `<stem>.hands.mp4` next to the video
    panels: Panels = "both"  # which panels to draw: the overlay, the 3D view, or both
    layout: Layout = "vertical"  # overlay above the 3D panel, or the two side by side
    min_width: int = 960  # the overlay is upscaled to at least this width for readability
    mesh: bool = False  # shade the MANO surface in the overlay instead of the skeleton
    presence_threshold: float = 0.5  # below this presence probability a hand is absent (not drawn)
    visible_threshold: float = (
        0.5  # below this visible probability a hand is hidden (ghosted in 3D)
    )


def main(config: VisualizeConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    video = fetch_video(config.video)
    outputs = output_dir(config.video, video, None)
    poses = config.poses or pose_table_path(outputs / video.name, "parquet")
    output = config.output or outputs / f"{video.stem}.hands.mp4"

    prediction, info = read_pose_table(poses)
    print(f"{poses}: {info.num_frames} frames of {info.source_video} at {info.fps:.3f} fps")
    mano = (
        ManoLayer.from_pretrained(
            str(component_dir(WEIGHTS_REPO, None, "mano")), map_location="cpu", strict=True
        )
        if config.mesh
        else None
    )

    count = render_video(
        video,
        prediction,
        output,
        panels=config.panels,
        layout=config.layout,
        min_width=config.min_width,
        mesh=config.mesh,
        mano=mano,
        presence_threshold=config.presence_threshold,
        visible_threshold=config.visible_threshold,
    )
    print(f"wrote {count} frames to {output}")


if __name__ == "__main__":
    main(tyro.cli(VisualizeConfig))
