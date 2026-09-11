"""Read and write camera calibrations: pinhole intrinsics plus an optional lens distortion.

One value type, `CameraModel`, in the forms callers have:

- `fx,fy,cx,cy` inline (pinhole, in the video's own pixels), optionally followed by
  `,<distortion>,k1,k2,...` for a distorted lens;
- `<stem>.camera.parquet` / `.csv` with columns `fx, fy, cx, cy, width, height` (one row) and
  the optional `distortion` / `coefficients` columns: the values are in the `width x height`
  grid and are rescaled to the video;
- a `.npy` holding a 3x3 matrix in video pixels (pinhole).

The model itself only takes pinhole cameras — the released checkpoints were trained on
undistorted footage — so `read_intrinsics` / `read_camera_sidecar` refuse a distorted model;
`ace-ego-hand undistort` turns such footage into a pinhole video with a pinhole sidecar first.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
import torch

from ..types import Intrinsics

CAMERA_COLUMNS = ("fx", "fy", "cx", "cy", "width", "height")
COEFFICIENT_SEPARATOR = ";"  # the csv form of the list column

type Distortion = Literal["none", "brown_conrady", "kannala_brandt"]
# Brown-Conrady is OpenCV's `(k1, k2, p1, p2, k3, k4, k5, k6)` including the rational terms;
# Kannala-Brandt is OpenCV's fisheye `(k1, k2, k3, k4)`. Shorter tuples are zero-padded.
COEFFICIENT_COUNTS: dict[Distortion, int] = {"none": 0, "brown_conrady": 8, "kannala_brandt": 4}


@dataclass(frozen=True, slots=True)
class CameraModel:
    """Pinhole intrinsics in the pixels of a `size = (width, height)` image, plus the lens model."""

    fx: float
    fy: float
    cx: float
    cy: float
    size: tuple[int, int]
    distortion: Distortion = "none"
    coefficients: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        count = COEFFICIENT_COUNTS[self.distortion]
        if len(self.coefficients) > count:
            raise ValueError(
                f"{self.distortion} takes at most {count} coefficients, got {self.coefficients}"
            )
        padded = tuple(float(c) for c in self.coefficients) + (0.0,) * (
            count - len(self.coefficients)
        )
        object.__setattr__(self, "coefficients", padded)

    @property
    def is_pinhole(self) -> bool:
        return self.distortion == "none"

    def intrinsics(self) -> Intrinsics:
        return intrinsics_matrix(self.fx, self.fy, self.cx, self.cy)

    def rescaled(self, size: tuple[int, int]) -> CameraModel:
        """Move the pinhole part to another pixel grid; the distortion is in unit-plane terms."""

        scale_x, scale_y = size[0] / self.size[0], size[1] / self.size[1]
        return replace(
            self,
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
            size=size,
        )


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> Intrinsics:
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)


def rescale(intrinsics: Intrinsics, source: tuple[int, int], target: tuple[int, int]) -> Intrinsics:
    """Move K between pixel grids: fx and cx follow the width, fy and cy the height."""

    scale = torch.tensor(
        [[target[0] / source[0]] * 3, [target[1] / source[1]] * 3, [1.0, 1.0, 1.0]],
        device=intrinsics.device,
        dtype=intrinsics.dtype,
    )
    return intrinsics * scale


def read_intrinsics(spec: str, video_size: tuple[int, int]) -> Intrinsics:
    """Parse `spec` (inline values or a path) into a pinhole K in `video_size` pixels."""

    path = Path(spec)
    if path.suffix == ".npy" and path.is_file():
        matrix = np.load(path)
        return torch.as_tensor(np.asarray(matrix, dtype=np.float32).reshape(3, 3))
    if path.suffix in (".parquet", ".csv") and path.is_file():
        return read_camera_sidecar(path, video_size)
    values = [float(v) for v in spec.split(",")]
    if len(values) != 4:
        raise ValueError(
            f"intrinsics must be 'fx,fy,cx,cy' or a .npy/.parquet/.csv file, got {spec!r}"
        )
    return intrinsics_matrix(*values)


def read_camera_sidecar(path: Path, video_size: tuple[int, int]) -> Intrinsics:
    """Read the sidecar as a pinhole K rescaled to the video; refuse a distorted model."""

    return _pinhole_only(read_camera_model(path), path).rescaled(video_size).intrinsics()


def read_camera_model(path: Path) -> CameraModel:
    """Read the one-row camera table, distortion included, in its own `width x height` grid."""

    if path.suffix == ".parquet":
        table = pq.read_table(path)
    else:
        # The list column is a `;`-joined string in csv; keep it (and a single value) as text
        column_types = {"distortion": pa.string(), "coefficients": pa.string()}
        table = pa_csv.read_csv(
            path, convert_options=pa_csv.ConvertOptions(column_types=column_types)
        )
    missing = [c for c in CAMERA_COLUMNS if c not in table.column_names]
    if missing:
        raise ValueError(f"{path}: missing camera columns {missing}")
    if table.num_rows < 1:
        raise ValueError(f"{path}: no rows")
    row = {c: float(table.column(c)[0].as_py()) for c in CAMERA_COLUMNS}

    distortion: Distortion = "none"
    coefficients: tuple[float, ...] = ()
    if "distortion" in table.column_names:
        distortion = _distortion(str(table.column("distortion")[0].as_py() or "none"))
    if "coefficients" in table.column_names:
        coefficients = _coefficients(table.column("coefficients")[0].as_py())
    return CameraModel(
        row["fx"],
        row["fy"],
        row["cx"],
        row["cy"],
        (int(row["width"]), int(row["height"])),
        distortion,
        coefficients,
    )


def parse_camera_model(spec: str, video_size: tuple[int, int]) -> CameraModel:
    """`fx,fy,cx,cy[,<distortion>,k1,k2,...]` in video pixels, or a sidecar path rescaled to it."""

    path = Path(spec)
    if path.suffix in (".parquet", ".csv") and path.is_file():
        return read_camera_model(path).rescaled(video_size)
    fields = [v.strip() for v in spec.split(",")]
    if len(fields) < 4:
        raise ValueError(f"camera must be 'fx,fy,cx,cy[,model,k1,...]' or a sidecar, got {spec!r}")
    fx, fy, cx, cy = (float(v) for v in fields[:4])
    distortion = _distortion(fields[4]) if len(fields) > 4 else "none"
    coefficients = tuple(float(v) for v in fields[5:])
    return CameraModel(fx, fy, cx, cy, video_size, distortion, coefficients)


def write_camera_sidecar(path: Path, camera: CameraModel) -> Path:
    """Write the one-row camera table as parquet or csv, by `path`'s suffix."""

    columns: dict[str, list[float | int | str | list[float]]] = {
        "fx": [camera.fx],
        "fy": [camera.fy],
        "cx": [camera.cx],
        "cy": [camera.cy],
        "width": [camera.size[0]],
        "height": [camera.size[1]],
        "distortion": [camera.distortion],
    }
    if path.suffix == ".parquet":
        columns["coefficients"] = [list(camera.coefficients)]
        pq.write_table(pa.table(columns), path)
    else:
        columns["coefficients"] = [COEFFICIENT_SEPARATOR.join(str(c) for c in camera.coefficients)]
        pa_csv.write_csv(pa.table(columns), path)
    return path


def camera_sidecar_path(video: Path) -> Path | None:
    """`<stem>.camera.parquet` or `.csv` next to the video, if one exists."""

    for suffix in ("parquet", "csv"):
        candidate = video.with_name(f"{video.stem}.camera.{suffix}")
        if candidate.is_file():
            return candidate
    return None


def _pinhole_only(camera: CameraModel, path: Path) -> CameraModel:
    if not camera.is_pinhole:
        raise ValueError(
            f"{path}: {camera.distortion} camera; the model is pinhole-only, run "
            "`ace-ego-hand undistort` first and use the .pinhole.mp4 it writes"
        )
    return camera


def _distortion(name: str) -> Distortion:
    match name:
        case "none" | "brown_conrady" | "kannala_brandt":
            return name
        case _:
            raise ValueError(
                f"unknown distortion {name!r}, expected one of {list(COEFFICIENT_COUNTS)}"
            )


def _coefficients(value: object) -> tuple[float, ...]:
    """The list column as stored: a list (parquet), a `;` string (csv), or empty."""

    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(float(v) for v in value.split(COEFFICIENT_SEPARATOR) if v.strip())
    if isinstance(value, list):
        return tuple(float(v) for v in value)
    if isinstance(value, int | float):
        return (float(value),)
    raise ValueError(f"cannot read coefficients from {value!r}")
