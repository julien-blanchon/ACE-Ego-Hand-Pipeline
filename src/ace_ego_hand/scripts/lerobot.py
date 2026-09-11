# /// script
# requires-python = ">=3.12"
# dependencies = ["ace-ego-hand @ git+https://github.com/julien-blanchon/ACE-Ego-Hand-Pipeline"]
# ///
"""Annotate a LeRobot v3 dataset with bimanual 3D hand poses, episode by episode.

Poses go to a sidecar mirroring the data files (`<root>/hand_pose/data/...`, joined on `index`)
or, with `--write-in-place`, into the data files themselves as `action.hand_pose.*` features.

    uv run ace-ego-hand lerobot /data/egoverse/datasets/aria
    uv run ace-ego-hand lerobot /data/hrdexdb/datasets/human --video-key observation.images.ego_left
    uv run ace-ego-hand lerobot /data/axis/datasets/ego-human-native --episodes 0 1 2
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from ace_ego_hand.config import InferenceConfig, ModelConfig
from ace_ego_hand.data.lerobot import (
    Episode,
    LeRobotDataset,
    episode_frames,
    episode_intrinsics,
)
from ace_ego_hand.data.lerobot_writer import EpisodeResult, write_in_place, write_sidecar
from ace_ego_hand.data.pose_table import table_info
from ace_ego_hand.inference import HandEstimator, load_estimator


@dataclass(frozen=True, slots=True)
class LerobotConfig:
    """Hand poses for every episode of a LeRobot v3 dataset."""

    root: tyro.conf.Positional[Path]
    """Dataset root: the folder holding `meta/`, `data/` and `videos/`."""

    video_key: str | None = None
    """Video stream to run on (`observation.images.<name>`); required when the dataset has several."""

    intrinsics_column: str | None = "observation.hand_pose.camera"
    """Feature holding the camera (4, 6 or 9 values, see docs/lerobot.md); absent or None runs K-free."""

    episodes: tuple[int, ...] | None = None
    """Episode indices to process; the whole dataset by default."""

    write_in_place: bool = False
    """Append the columns to the data files and `meta/info.json` instead of writing the sidecar."""

    model: ModelConfig = field(default_factory=lambda: ModelConfig(compile=True))
    """Compiled by default: an annotation run has enough episodes to amortise the warm-up."""
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def resolve_video_key(dataset: LeRobotDataset, video_key: str | None) -> str:
    """The one stream, or the chosen one; several streams need an explicit choice."""

    if video_key is not None:
        if video_key not in dataset.video_keys:
            raise SystemExit(
                f"{dataset.root}: no video stream {video_key!r}; has {dataset.video_keys}"
            )
        return video_key
    if len(dataset.video_keys) != 1:
        raise SystemExit(
            f"{dataset.root} has {len(dataset.video_keys)} video streams; pick one with --video-key: "
            f"{', '.join(dataset.video_keys)}"
        )
    return dataset.video_keys[0]


def selected_episodes(dataset: LeRobotDataset, subset: tuple[int, ...] | None) -> list[Episode]:
    if subset is None:
        return list(dataset.episodes)
    return [dataset.episode(index) for index in subset]


def run_episodes(
    estimator: HandEstimator,
    dataset: LeRobotDataset,
    episodes: list[Episode],
    config: LerobotConfig,
    video_key: str,
) -> Iterator[EpisodeResult]:
    """Predict one episode at a time; the writer consumes the stream."""

    for index, episode in enumerate(episodes, start=1):
        print(f"[{index}/{len(episodes)}] episode {episode.index} ({episode.length} frames)")
        video, start, num_frames = episode_frames(dataset, episode, video_key)
        intrinsics = episode_intrinsics(dataset, episode, config.intrinsics_column, video_key)
        started = time.perf_counter()
        prediction = estimator.predict(
            video,
            intrinsics=intrinsics,
            config=config.inference,
            start_seconds=start,
            num_frames=num_frames,
        )
        elapsed = time.perf_counter() - started
        print(
            f"episode {episode.index}: {num_frames} frames in {elapsed:.1f} s ({num_frames / elapsed:.1f} fps)"
        )
        yield EpisodeResult(
            episode=episode,
            prediction=prediction,
            info=table_info(prediction, video, estimator.weights),
        )


def main(config: LerobotConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    dataset = LeRobotDataset.open(config.root)
    video_key = resolve_video_key(dataset, config.video_key)
    episodes = selected_episodes(dataset, config.episodes)
    has_intrinsics = (
        config.intrinsics_column is not None and config.intrinsics_column in dataset.features
    )
    if config.model.variant == "k" and not has_intrinsics:
        raise SystemExit(
            f"the K variant needs a calibration column; {config.intrinsics_column!r} is not a "
            f"feature of {config.root} (use --model.variant kfree or --intrinsics-column)"
        )
    if not has_intrinsics:
        print(f"{config.root.name}: no intrinsics column, running K-free")

    estimator = load_estimator(config.model)
    results = run_episodes(estimator, dataset, episodes, config, video_key)
    writer = write_in_place if config.write_in_place else write_sidecar
    written = writer(dataset, results, video_key=video_key, weights=estimator.weights)
    mode = "in place" if config.write_in_place else "sidecar"
    print(f"{config.root.name}: {len(episodes)} episodes -> {len(written)} files ({mode})")


if __name__ == "__main__":
    main(tyro.cli(LerobotConfig))
