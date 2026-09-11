"""Write hand poses back into a LeRobot v3 dataset, as a sidecar or in place.

Output is one row per frame, aligned on the dataset `index`, with one column per (hand, quantity)
under LeRobot-style feature names (`action.hand_pose.left.joints_cam`, ...) plus the per-episode
camera. The **sidecar** (`<root>/hand_pose/data/...`) mirrors the data files one to one, so a
loader joins it on `index` or concatenates the columns file by file; **in place** appends the same
columns to the data files and declares them in `meta/info.json`. Both merge: a rerun on a subset
replaces only those episodes' rows and keeps the rest. `docs/lerobot.md` is the specification;
`lerobot.py` reads the dataset these rows go into.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from einops import rearrange

from ..prediction import VideoPrediction, WeightsInfo
from .lerobot import (
    CAMERA_NAMES,
    Episode,
    LeRobotDataset,
    LeRobotFeature,
    LeRobotInfo,
    RunMetadata,
    read_info,
)
from .pose_table import HANDS, LIST_COLUMNS, PoseTableInfo

SIDECAR_DIR = "hand_pose"
FEATURE_PREFIX = "action.hand_pose"
INDEX_COLUMNS = ("index", "episode_index", "frame_index")

CAMERA_FEATURE = f"{FEATURE_PREFIX}.camera"  # fx, fy, cx, cy, width, height of the source video
CAMERA_FITTED_FEATURE = f"{FEATURE_PREFIX}.camera_fitted"  # True when K-free fitted the camera
HAND_QUANTITIES: tuple[tuple[str, int], ...] = (  # (name, width); width 1 is a scalar column
    *((name, width) for name, width, _ in LIST_COLUMNS),
    ("presence", 1),
    ("visible", 1),
)


def _features() -> dict[str, LeRobotFeature]:
    """The added features in LeRobot's schema: dtype, shape and element names per column."""

    features: dict[str, LeRobotFeature] = {
        f"{FEATURE_PREFIX}.{hand}.{name}": {"dtype": "float32", "shape": [width], "names": None}
        for hand in HANDS
        for name, width in HAND_QUANTITIES
    }
    features[CAMERA_FEATURE] = {
        "dtype": "float32",
        "shape": [len(CAMERA_NAMES)],
        "names": list(CAMERA_NAMES),
    }
    features[CAMERA_FITTED_FEATURE] = {"dtype": "bool", "shape": [1], "names": None}
    return features


HAND_POSE_FEATURES = _features()


# --- rows ---


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """One processed episode, ready to be laid out as rows."""

    episode: Episode
    prediction: VideoPrediction
    info: PoseTableInfo


def episode_table(result: EpisodeResult) -> pa.Table:
    """One row per frame of the episode: the three index columns plus every pose feature."""

    episode, prediction = result.episode, result.prediction
    num_frames = len(prediction.presence)
    assert num_frames == episode.length, (
        f"episode {episode.index}: {num_frames} predicted frames for {episode.length} rows"
    )
    columns: dict[str, pa.Array] = {
        "index": pa.array(np.arange(episode.dataset_from, episode.dataset_to, dtype=np.int64)),
        "episode_index": pa.array(np.full(num_frames, episode.index, dtype=np.int64)),
        "frame_index": pa.array(np.arange(num_frames, dtype=np.int64)),
    }
    for slot, hand in enumerate(HANDS):
        for name, _, attribute in LIST_COLUMNS:
            values = rearrange(getattr(prediction, attribute)[:, slot], "f ... -> f (...)")
            columns[f"{FEATURE_PREFIX}.{hand}.{name}"] = _list_array(values)
        for name in ("presence", "visible"):
            values = getattr(prediction, name)[:, slot].astype(np.float32)
            columns[f"{FEATURE_PREFIX}.{hand}.{name}"] = pa.array(values)

    camera = np.array([*result.info.intrinsics, *result.info.source_size], dtype=np.float32)
    columns[CAMERA_FEATURE] = _list_array(np.tile(camera, (num_frames, 1)))
    columns[CAMERA_FITTED_FEATURE] = pa.array(np.full(num_frames, result.info.intrinsics_fitted))
    return pa.table(columns)


def _list_array(values: np.ndarray) -> pa.Array:
    rows, width = values.shape
    flat = pa.array(np.ascontiguousarray(values, dtype=np.float32).reshape(rows * width))
    return pa.FixedSizeListArray.from_arrays(flat, width)


def _blank_column(name: str, num_rows: int) -> pa.Array:
    """The value of a row no episode has filled: NaN, or False for the flag (nulls in fixed-size
    lists do not survive parquet, and NaN is what a loader can stack without special cases)."""

    feature = HAND_POSE_FEATURES[name]
    if feature["dtype"] == "bool":
        return pa.array(np.zeros(num_rows, dtype=bool))
    width = feature["shape"][0]
    if width == 1:
        return pa.array(np.full(num_rows, np.nan, dtype=np.float32))
    return _list_array(np.full((num_rows, width), np.nan, dtype=np.float32))


def overlay_rows(target_index: np.ndarray, sources: Iterable[pa.Table]) -> dict[str, pa.Array]:
    """Place the pose columns of `sources` on the `target_index` rows; later sources win, gaps NaN.

    Every source carries an `index` column, and a source row lands on the target row with the
    same dataset index. This is the one merge both writers use: the existing file first, then
    the fresh episode tables.
    """

    num_rows = len(target_index)
    order = np.argsort(target_index, kind="stable")
    sorted_index = target_index[order]
    combined = {name: _blank_column(name, num_rows) for name in HAND_POSE_FEATURES}
    for source in sources:
        source_index = np.asarray(source.column("index"))
        positions = np.clip(np.searchsorted(sorted_index, source_index), 0, num_rows - 1)
        hit = sorted_index[positions] == source_index
        target_rows = order[positions[hit]]

        # `take` with a null index yields null, so one gather scatters the hits and blanks the rest
        gather = np.full(num_rows, -1, dtype=np.int64)
        gather[target_rows] = np.flatnonzero(hit)
        indices = pa.array(gather, mask=gather < 0)
        replaced = pa.array(gather >= 0)
        for name in HAND_POSE_FEATURES:
            if name in source.column_names:
                placed = source.column(name).combine_chunks().take(indices)
                combined[name] = pc.call_function("if_else", [replaced, placed, combined[name]])
    return combined


# --- writers ---


def write_sidecar(
    dataset: LeRobotDataset,
    results: Iterable[EpisodeResult],
    *,
    video_key: str,
    weights: WeightsInfo,
) -> list[Path]:
    """Write `<root>/hand_pose/data/...` files mirroring the data files of the given episodes.

    Results are consumed as they come and a file is flushed when the stream moves on to the next
    one, so a run over thousands of episodes never holds more than one file's rows. A file that
    already exists is merged: rows of the episodes being written replace the old ones.
    """

    written: list[Path] = []
    for data_file, tables in _group_by_file(results):
        path = dataset.root / SIDECAR_DIR / data_file
        sources = [*([pq.read_table(path)] if path.is_file() else []), *tables]
        index = np.unique(np.concatenate([np.asarray(t.column("index")) for t in sources]))
        episode_index = _lookup(sources, index, "episode_index")
        table = pa.table(
            {
                "index": pa.array(index),
                "episode_index": pa.array(episode_index),
                "frame_index": pa.array(_lookup(sources, index, "frame_index")),
                **overlay_rows(index, sources),
            }
        )
        _write_parquet_atomic(path, table, _run_lengths(episode_index))
        written.append(path)
    if written:
        info: LeRobotInfo = {
            "codebase_version": dataset.info["codebase_version"],
            "fps": dataset.fps,
            "data_path": dataset.info["data_path"],
            "video_path": dataset.info["video_path"],
            "features": {
                **{c: dataset.features[c] for c in INDEX_COLUMNS if c in dataset.features},
                **HAND_POSE_FEATURES,
            },
            "ace_ego_hand": _run_metadata(video_key, weights),
        }
        _write_json_atomic(dataset.root / SIDECAR_DIR / "meta" / "info.json", info)
    return written


def write_in_place(
    dataset: LeRobotDataset,
    results: Iterable[EpisodeResult],
    *,
    video_key: str,
    weights: WeightsInfo,
) -> list[Path]:
    """Append the pose columns to the data files themselves and declare them in `meta/info.json`.

    Each data file is rewritten atomically with its row groups exactly as they were (one per
    episode in the datasets we know, so that property survives). Rows of episodes not in
    `results` keep their previous values, or are NaN when the columns are new to the file.
    """

    written: list[Path] = []
    for data_file, tables in _group_by_file(results):
        path = dataset.root / data_file
        reader = pq.ParquetFile(path)
        row_groups = [reader.metadata.row_group(i).num_rows for i in range(reader.num_row_groups)]
        data = reader.read()
        present = [c for c in HAND_POSE_FEATURES if c in data.column_names]
        existing = data.select(["index", *present])
        table = data.drop_columns(present)
        for name, array in overlay_rows(
            np.asarray(data.column("index")), [existing, *tables]
        ).items():
            table = table.append_column(name, array)
        _write_parquet_atomic(path, table, row_groups)
        written.append(path)
    if written:
        info_path = dataset.root / "meta" / "info.json"
        info = read_info(info_path)
        info["features"] = {**info["features"], **HAND_POSE_FEATURES}
        info["ace_ego_hand"] = _run_metadata(video_key, weights)
        _write_json_atomic(info_path, info)
    return written


def _group_by_file(results: Iterable[EpisodeResult]) -> Iterator[tuple[Path, list[pa.Table]]]:
    """Batch consecutive results by data file; a file reappearing later is simply merged again."""

    current: Path | None = None
    tables: list[pa.Table] = []
    for result in results:
        if current is not None and result.episode.data_file != current:
            yield current, tables
            tables = []
        current = result.episode.data_file
        tables.append(episode_table(result))
    if current is not None:
        yield current, tables


def _lookup(sources: list[pa.Table], index: np.ndarray, column: str) -> np.ndarray:
    """The `column` value of every row of the sorted `index`, from whichever source has the row."""

    values = np.zeros(len(index), dtype=np.int64)
    for source in sources:
        positions = np.searchsorted(index, np.asarray(source.column("index")))
        values[positions] = np.asarray(source.column(column))
    return values


def _run_lengths(episode_index: np.ndarray) -> list[int]:
    """One row group per run of equal `episode_index`, the layout the data files use."""

    boundaries = np.flatnonzero(np.diff(episode_index)) + 1
    return np.diff([0, *boundaries.tolist(), len(episode_index)]).tolist()


def _write_parquet_atomic(path: Path, table: pa.Table, row_groups: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with pq.ParquetWriter(temporary, table.schema) as writer:
            offset = 0
            for size in row_groups:
                writer.write_table(table.slice(offset, size))
                offset += size
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: LeRobotInfo) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=4) + "\n")
    temporary.replace(path)


def _run_metadata(video_key: str, weights: WeightsInfo) -> RunMetadata:
    return {
        "video_key": video_key,
        "weights_repo": weights.repo_id,
        "weights_revision": weights.revision,
        "variant": weights.variant,
        "latent_encoder": weights.latent_encoder,
        "joint_order": "openpose21",
        "units": "metres",
        "camera_frame": "opencv (x right, y down, z forward)",
    }
