# LeRobot datasets

`ace-ego-hand lerobot <root>` annotates a LeRobot **v3** dataset with bimanual hand poses. It
reads `meta/episodes` to find each episode's rows and its slice of the chosen video stream, runs
the estimator on every episode, and writes one row per frame back, aligned on the dataset-wide
`index`. `ace_ego_hand.data.lerobot` reads the dataset, `ace_ego_hand.data.lerobot_writer` writes
the rows.

```
uv run ace-ego-hand lerobot /data/hrdexdb/datasets/human --video-key observation.images.ego_left
```

`--video-key` (`observation.images.<name>`) is required when the dataset has several streams;
`--episodes 0 1 2` selects a subset; `--model.*` and `--inference.*` are as for `infer`.

## Sidecar layout (default)

```
<root>/hand_pose/
  meta/info.json                    the added features in LeRobot's schema + what produced them
  data/chunk-XXX/file-XXX.parquet   mirrors <root>/data/chunk-XXX/file-XXX.parquet
```

Each sidecar file has one row per frame with the same `index` values as the data file it
mirrors, one row group per episode, sorted by `index`, so a loader joins on `index` or
concatenates the columns positionally. Only files with processed episodes are written; an
existing file is merged (the episodes being written replace their rows, the rest is kept), so
subset runs and restarts never lose earlier work. Writes are atomic.

## Columns

| column | type | meaning |
|---|---|---|
| `index`, `episode_index`, `frame_index` | int64 | copied from the data row |
| `action.hand_pose.<hand>.cam_trans` | list<float32>[3] | camera-space wrist translation, metres |
| `action.hand_pose.<hand>.global_orient` | list<float32>[9] | row-major 3x3 root rotation |
| `action.hand_pose.<hand>.hand_pose` | list<float32>[135] | 15 row-major 3x3 finger rotations, MANO order |
| `action.hand_pose.<hand>.betas` | list<float32>[10] | MANO shape |
| `action.hand_pose.<hand>.joints_cam` | list<float32>[63] | 21 OpenPose joints x (x, y, z), camera metres |
| `action.hand_pose.<hand>.joints_2d` | list<float32>[42] | 21 joints x (u, v), pixels of the chosen video |
| `action.hand_pose.<hand>.presence`, `.visible` | float32 | probability the hand exists / is in view |
| `action.hand_pose.camera` | list<float32>[6] | `fx, fy, cx, cy, width, height` used for the episode, video pixels |
| `action.hand_pose.camera_fitted` | bool | `true` when K-free fitted the camera |

`<hand>` is `left` or `right`. Quantities, joint order, units and camera frame are those of
`docs/pose_format.md`; only the layout is wide (one column per hand) instead of long.

## In-place mode

`--write-in-place` appends the same columns to the data files themselves and declares them in
`<root>/meta/info.json`. Each data file is rewritten atomically with its row groups unchanged;
rows of episodes not yet processed hold NaN (`false` for `camera_fitted`) until a later run fills
them in. `meta/stats.json` is not updated. Never use it on a dataset you cannot regenerate.

## Intrinsics

`--intrinsics-column` (default `observation.hand_pose.camera`) names the feature holding the
calibration, read from the episode's first row: 4 values `fx, fy, cx, cy` in video pixels, 6
values `fx, fy, cx, cy, width, height` rescaled from that grid to the video, or a row-major 3x3.
A missing feature or `None` runs K-free; `--model.variant k` requires the column. Distortion
coefficients next to the column are ignored (one warning per dataset): undistort first or run
K-free on fisheye footage.
