"""Read LeRobot v3 datasets episode by episode: the index, the video slices and the calibration.

`docs/lerobot.md` is the specification. A v3 dataset packs many episodes per parquet data file
(`data/chunk-XXX/file-XXX.parquet`, one row group per episode) and per mp4
(`videos/<key>/chunk-XXX/file-XXX.mp4`, episodes back to back); `meta/episodes/` says where each
episode lives and at which video timestamp it starts. `LeRobotDataset.open` reads that index,
`episode_frames` turns it into the `(video, start_seconds, num_frames)` triple `iter_frames`
takes, and `episode_intrinsics` reads a calibration column in one of the layouts seen in the
wild. Writing the poses back is `lerobot_writer.py`; the typed slice of `meta/info.json` both
sides touch is declared here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import numpy as np
import pyarrow.parquet as pq

from ..types import Intrinsics
from .calibration import intrinsics_matrix, rescale

logger = logging.getLogger(__name__)

CAMERA_NAMES = ("fx", "fy", "cx", "cy", "width", "height")
PINHOLE_NAMES = CAMERA_NAMES[:4]
MATRIX_SIZE = 9  # a row-major 3x3 K
_distortion_warned: set[Path] = set()  # datasets already warned about ignored distortion


class LeRobotFeature(TypedDict):
    """One entry of `info.json`'s `features`: LeRobot's own schema for a column."""

    dtype: str
    shape: list[int]
    names: list[str] | None
    info: NotRequired[dict[str, float | int | str]]  # video streams carry fps and size here


class RunMetadata(TypedDict):
    """What produced the pose columns, recorded once per dataset under `ace_ego_hand`."""

    video_key: str
    weights_repo: str
    weights_revision: str
    variant: str
    latent_encoder: str
    joint_order: str
    units: str
    camera_frame: str


class LeRobotInfo(TypedDict):
    """The keys of `meta/info.json` this package reads or writes; the rest passes through."""

    codebase_version: str
    fps: float
    data_path: str
    video_path: str
    features: dict[str, LeRobotFeature]
    ace_ego_hand: NotRequired[RunMetadata]


def read_info(path: Path) -> LeRobotInfo:
    """Parse `meta/info.json`; the JSON blob is LeRobot's, so it is trusted rather than checked."""

    return cast(LeRobotInfo, json.loads(path.read_text()))


@dataclass(frozen=True, slots=True)
class Episode:
    """Where one episode lives: its rows, its data file and its slice of each video stream."""

    index: int
    length: int
    dataset_from: int  # first dataset `index` (inclusive)
    dataset_to: int  # last dataset `index` (exclusive)
    data_file: Path  # relative to the root
    video_files: dict[str, Path]  # video key -> mp4, relative to the root
    video_starts: dict[str, float]  # video key -> seconds into that mp4


@dataclass(frozen=True, slots=True)
class LeRobotDataset:
    """A v3 dataset root with `meta/info.json` and the episode index parsed."""

    root: Path
    info: LeRobotInfo
    episodes: tuple[Episode, ...]

    @classmethod
    def open(cls, root: Path) -> LeRobotDataset:
        info = read_info(root / "meta" / "info.json")
        video_keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]
        episodes = tuple(
            _parse_episode(row, info, video_keys)
            for path in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
            for row in pq.read_table(path, columns=_episode_columns(video_keys)).to_pylist()
        )
        return cls(root=root, info=info, episodes=episodes)

    @property
    def fps(self) -> float:
        return float(self.info["fps"])

    @property
    def features(self) -> dict[str, LeRobotFeature]:
        return self.info["features"]

    @property
    def video_keys(self) -> tuple[str, ...]:
        return tuple(k for k, f in self.features.items() if f["dtype"] == "video")

    def episode(self, index: int) -> Episode:
        for episode in self.episodes:
            if episode.index == index:
                return episode
        raise KeyError(f"{self.root}: no episode {index}")

    def video_size(self, video_key: str) -> tuple[int, int]:
        height, width, _ = self.features[video_key]["shape"]  # LeRobot stores (H, W, C)
        return int(width), int(height)


def _episode_columns(video_keys: list[str]) -> list[str]:
    columns = [
        "episode_index",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        "data/chunk_index",
        "data/file_index",
    ]
    for key in video_keys:
        columns += [f"videos/{key}/{c}" for c in ("chunk_index", "file_index", "from_timestamp")]
    return columns


def _parse_episode(
    row: dict[str, int | float], info: LeRobotInfo, video_keys: list[str]
) -> Episode:
    """One `meta/episodes` row (indices and timestamps only) to an `Episode`."""

    return Episode(
        index=int(row["episode_index"]),
        length=int(row["length"]),
        dataset_from=int(row["dataset_from_index"]),
        dataset_to=int(row["dataset_to_index"]),
        data_file=Path(
            info["data_path"].format(
                chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
            )
        ),
        video_files={
            key: Path(
                info["video_path"].format(
                    video_key=key,
                    chunk_index=row[f"videos/{key}/chunk_index"],
                    file_index=row[f"videos/{key}/file_index"],
                )
            )
            for key in video_keys
        },
        video_starts={key: float(row[f"videos/{key}/from_timestamp"]) for key in video_keys},
    )


def episode_frames(
    dataset: LeRobotDataset, episode: Episode, video_key: str
) -> tuple[Path, float, int]:
    """`(video, start_seconds, num_frames)`: exactly this episode's frames, for `iter_frames`."""

    return (
        dataset.root / episode.video_files[video_key],
        episode.video_starts[video_key],
        episode.length,
    )


def episode_intrinsics(
    dataset: LeRobotDataset, episode: Episode, column: str | None, video_key: str
) -> Intrinsics | None:
    """K in `video_key` pixels from the episode's first row of `column`, or None without one.

    Three layouts are read, told apart by the feature's `names` (falling back on the length):
    `fx, fy, cx, cy` already in video pixels; `fx, fy, cx, cy, width, height` in its own grid,
    rescaled; or a row-major 3x3. Distortion coefficients next to the column are ignored: the
    model is pinhole-only, so a non-zero set gets one warning per dataset.
    """

    if column is None or column not in dataset.features:
        return None
    distortion = _distortion_column(dataset, column)
    table = pq.read_table(
        dataset.root / episode.data_file,
        columns=[column, *([distortion] if distortion else [])],
        filters=[("episode_index", "==", episode.index)],
    )
    if table.num_rows == 0:
        raise ValueError(f"{episode.data_file}: no rows for episode {episode.index}")
    values = np.asarray(table.column(column)[0].as_py(), dtype=np.float32).reshape(-1)

    if distortion and dataset.root not in _distortion_warned:
        coefficients = np.asarray(table.column(distortion)[0].as_py(), dtype=np.float32)
        if np.any(coefficients != 0.0):
            _distortion_warned.add(dataset.root)
            logger.warning(
                "%s: %s is non-zero (%s); the pinhole model ignores distortion",
                dataset.root,
                distortion,
                coefficients.tolist(),
            )
    names = dataset.features[column].get("names")
    return _intrinsics_from_values(values, names, dataset.video_size(video_key))


def _distortion_column(dataset: LeRobotDataset, column: str) -> str | None:
    """`<column>_distortion`, or the sibling `camera_distortion` of a dotted column."""

    for candidate in (f"{column}_distortion", f"{column.rpartition('.')[0]}.camera_distortion"):
        if candidate in dataset.features:
            return candidate
    return None


def _intrinsics_from_values(
    values: np.ndarray, names: list[str] | None, video_size: tuple[int, int]
) -> Intrinsics:
    if len(values) == MATRIX_SIZE:
        k = values.reshape(3, 3)
        return intrinsics_matrix(float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]))
    if len(values) not in (len(PINHOLE_NAMES), len(CAMERA_NAMES)):
        raise ValueError(f"intrinsics column has {len(values)} values; expected 4, 6 or 9")

    # Named layouts are read by name; unnamed ones by position, `width, height` after the four
    named = names is not None and all(n in names for n in PINHOLE_NAMES)
    order: list[str] = list(names) if named and names is not None else list(CAMERA_NAMES)
    lookup = {name: float(values[order.index(name)]) for name in order[: len(values)]}
    k = intrinsics_matrix(lookup["fx"], lookup["fy"], lookup["cx"], lookup["cy"])
    if "width" in lookup and "height" in lookup:
        k = rescale(k, (int(lookup["width"]), int(lookup["height"])), video_size)
    return k
