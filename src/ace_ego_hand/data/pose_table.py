"""The on-disk pose format: one row per (frame, hand), written as parquet or csv.

`docs/pose_format.md` is the specification. A table is long: every source frame contributes two
rows (left, right), each carrying the MANO parameters, the camera-space joints, their pixel
projection and the two probabilities. Clip-level facts (source video, camera, weights) live in
the parquet key-value metadata, or in a `.json` sidecar for csv, whose flat columns spell the
lists out with an index suffix (`joints_cam_17`). `read_pose_table` inverts either form back into
a `VideoPrediction`, so the visualiser and the LeRobot writer never touch the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
from einops import rearrange

from ..prediction import VideoPrediction, WeightsInfo

HANDS = ("left", "right")
OutputFormat = Literal["parquet", "csv"]

# (column, values per row, slice of a VideoPrediction attribute flattened per hand)
LIST_COLUMNS: tuple[tuple[str, int, str], ...] = (
    ("cam_trans", 3, "translation"),
    ("global_orient", 9, "global_orient"),
    ("hand_pose", 135, "hand_pose"),
    ("betas", 10, "betas"),
    ("joints_cam", 63, "joints_camera"),
    ("joints_2d", 42, "joints_2d"),
)


@dataclass(frozen=True, slots=True)
class PoseTableInfo:
    """Clip-level facts stored next to the rows."""

    source_video: str
    fps: float
    num_frames: int
    source_size: tuple[int, int]
    encode_size: tuple[int, int]
    intrinsics: tuple[float, float, float, float]  # fx, fy, cx, cy in source pixels
    intrinsics_fitted: bool
    weights: WeightsInfo

    def to_metadata(self) -> dict[str, str]:
        fx, fy, cx, cy = self.intrinsics
        return {
            "ace.source_video": self.source_video,
            "ace.fps": repr(self.fps),
            "ace.num_frames": str(self.num_frames),
            "ace.source_width": str(self.source_size[0]),
            "ace.source_height": str(self.source_size[1]),
            "ace.encode_width": str(self.encode_size[0]),
            "ace.encode_height": str(self.encode_size[1]),
            "ace.fx": repr(fx),
            "ace.fy": repr(fy),
            "ace.cx": repr(cx),
            "ace.cy": repr(cy),
            "ace.intrinsics_fitted": str(self.intrinsics_fitted).lower(),
            "ace.weights_repo": self.weights.repo_id,
            "ace.weights_revision": self.weights.revision,
            "ace.variant": self.weights.variant,
            "ace.latent_encoder": self.weights.latent_encoder,
            "ace.joint_order": "openpose21",
            "ace.units": "metres",
            "ace.camera_frame": "opencv (x right, y down, z forward)",
        }

    @classmethod
    def from_metadata(cls, meta: dict[str, str]) -> PoseTableInfo:
        return cls(
            source_video=meta["ace.source_video"],
            fps=float(meta["ace.fps"]),
            num_frames=int(meta["ace.num_frames"]),
            source_size=(int(meta["ace.source_width"]), int(meta["ace.source_height"])),
            encode_size=(int(meta["ace.encode_width"]), int(meta["ace.encode_height"])),
            intrinsics=(
                float(meta["ace.fx"]),
                float(meta["ace.fy"]),
                float(meta["ace.cx"]),
                float(meta["ace.cy"]),
            ),
            intrinsics_fitted=meta["ace.intrinsics_fitted"] == "true",
            weights=WeightsInfo(
                repo_id=meta["ace.weights_repo"],
                revision=meta["ace.weights_revision"],
                variant=meta["ace.variant"],
                latent_encoder=meta["ace.latent_encoder"],
            ),
        )


def table_info(
    prediction: VideoPrediction, source_video: Path, weights: WeightsInfo
) -> PoseTableInfo:
    k = prediction.intrinsics
    return PoseTableInfo(
        source_video=str(source_video),
        fps=prediction.fps,
        num_frames=len(prediction.presence),
        source_size=prediction.source_size,
        encode_size=prediction.encode_size,
        intrinsics=(float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])),
        intrinsics_fitted=prediction.intrinsics_fitted,
        weights=weights,
    )


def prediction_to_table(prediction: VideoPrediction, info: PoseTableInfo) -> pa.Table:
    """Lay the per-frame arrays out as one row per (frame, hand), hands interleaved."""

    num_frames = len(prediction.presence)
    frame_index = np.repeat(np.arange(num_frames, dtype=np.int32), len(HANDS))
    hand = np.tile(np.array(HANDS), num_frames)
    columns: dict[str, pa.Array] = {
        "frame_index": pa.array(frame_index),
        "hand": pa.array(hand),
        "presence": pa.array(rearrange(prediction.presence, "f s -> (f s)").astype(np.float32)),
        "visible": pa.array(rearrange(prediction.visible, "f s -> (f s)").astype(np.float32)),
        "quality": pa.array(rearrange(prediction.quality, "f s -> (f s)").astype(np.float32)),
    }
    for name, width, attribute in LIST_COLUMNS:
        values = rearrange(getattr(prediction, attribute), "f s ... -> (f s) (...)")
        flat = pa.array(np.ascontiguousarray(values, dtype=np.float32).reshape(-1))
        columns[name] = pa.FixedSizeListArray.from_arrays(flat, width)
    return pa.table(columns, metadata=info.to_metadata())


def write_pose_table(path: Path, table: pa.Table, output_format: OutputFormat) -> Path:
    """Write the table as parquet, or as csv with the lists flattened and the metadata in a json sidecar."""

    if output_format == "parquet":
        pq.write_table(table, path)
        return path
    flat = flatten_table(table)
    pa_csv.write_csv(flat, path)
    metadata = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return path


def flatten_table(table: pa.Table) -> pa.Table:
    """Replace every fixed-size list column by `<name>_<i>` scalar columns."""

    columns: dict[str, pa.Array] = {}
    for name in table.column_names:
        column = table.column(name).combine_chunks()
        if pa.types.is_fixed_size_list(column.type):
            width = column.type.list_size
            values = np.asarray(column.values).reshape(-1, width)
            for i in range(width):
                columns[f"{name}_{i}"] = pa.array(values[:, i])
        else:
            columns[name] = column
    return pa.table(columns)


def read_pose_table(path: Path) -> tuple[VideoPrediction, PoseTableInfo]:
    """Read a parquet (or csv + json) pose table back into a `VideoPrediction`."""

    if path.suffix == ".parquet":
        table = pq.read_table(path)
        metadata = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
        arrays = {name: _list_column(table, name, width) for name, width, _ in LIST_COLUMNS}
    else:
        table = pa_csv.read_csv(path)
        metadata = json.loads(path.with_suffix(".json").read_text())
        arrays = {
            name: np.stack([table.column(f"{name}_{i}").to_numpy() for i in range(width)], axis=1)
            for name, width, _ in LIST_COLUMNS
        }
    info = PoseTableInfo.from_metadata(metadata)
    num_frames = info.num_frames
    order = np.argsort(
        table.column("frame_index").to_numpy() * len(HANDS)
        + np.where(table.column("hand").to_numpy(zero_copy_only=False) == "right", 1, 0),
        kind="stable",
    )

    def per_frame(values: np.ndarray, *shape: int) -> np.ndarray:
        return values[order].reshape(num_frames, len(HANDS), *shape).astype(np.float32)

    fx, fy, cx, cy = info.intrinsics
    prediction = VideoPrediction(
        global_orient=per_frame(arrays["global_orient"], 3, 3),
        hand_pose=per_frame(arrays["hand_pose"], 15, 3, 3),
        betas=per_frame(arrays["betas"], 10),
        translation=per_frame(arrays["cam_trans"], 3),
        joints_camera=per_frame(arrays["joints_cam"], 21, 3),
        joints_2d=per_frame(arrays["joints_2d"], 21, 2),
        presence=per_frame(table.column("presence").to_numpy()),
        visible=per_frame(table.column("visible").to_numpy()),
        quality=per_frame(table.column("quality").to_numpy()),
        intrinsics=np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32),
        intrinsics_fitted=info.intrinsics_fitted,
        source_size=info.source_size,
        encode_size=info.encode_size,
        fps=info.fps,
    )
    return prediction, info


def _list_column(table: pa.Table, name: str, width: int) -> np.ndarray:
    column = table.column(name).combine_chunks()
    return np.asarray(column.values).reshape(-1, width)


def pose_table_path(video: Path, output_format: OutputFormat) -> Path:
    """`<stem>.hands.parquet` (or `.csv`) next to the video."""

    return video.with_name(f"{video.stem}.hands.{output_format}")
