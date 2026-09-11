# Pose format

One table per video, written next to it as `<stem>.hands.parquet` (default) or `<stem>.hands.csv`.
The table is **long**: one row per (frame, hand), so a 161-frame video has 322 rows.

## Rows

| column | type | meaning |
|---|---|---|
| `frame_index` | int32 | 0-based source frame |
| `hand` | string | `left` or `right` |
| `presence` | float32 | probability that this hand exists in the frame (the 3D head); draw when `> 0.5` |
| `visible` | float32 | probability that this hand is in view (the 2D head), after the `verify` second opinion |
| `quality` | float32 | one number to filter on: `min(presence, visible)`, halved on frames post-processing synthesised (an interpolated spike, a bridged presence gap) |
| `cam_trans` | list<float32>[3] | camera-space wrist translation, metres |
| `global_orient` | list<float32>[9] | row-major 3x3 rotation of the hand root |
| `hand_pose` | list<float32>[135] | 15 row-major 3x3 rotations of the MANO finger joints, MANO order |
| `betas` | list<float32>[10] | MANO shape; constant within an inference window |
| `joints_cam` | list<float32>[63] | 21 joints x (x, y, z), camera-space metres, `joints_cam = MANO(global_orient, hand_pose, betas) + cam_trans` |
| `joints_2d` | list<float32>[42] | 21 joints x (u, v), `joints_cam` projected through the table's camera, source-video pixels |

In csv the list columns are flattened to `<column>_<i>` scalars and the metadata goes to
`<stem>.hands.json`.

## Conventions

- **Camera frame**: OpenCV, x right, y down, z forward, origin at the camera centre. Metres.
- **Joint order**: OpenPose 21: 0 wrist; 1-4 thumb (CMC, MCP, IP, tip); 5-8 index; 9-12 middle;
  13-16 ring; 17-20 pinky, each root to tip.
- **MANO rotations** are the rotation matrices the model regresses (6D Gram-Schmidt). To rebuild
  the mesh, run them through a MANO layer with `flat_hand_mean=False` semantics (the 15 finger
  rotations receive the MANO pose mean in axis-angle space), as `ace_ego_hand.modeling.mano` does.
- **Hands are slots**: `left` and `right` are the model's two fixed slots, not detections; a
  slot with low `presence` still carries a (meaningless) pose. Filter on `quality` (or on
  `visible`: `presence` is about 1.0 nearly always, the model keeps tracking a hand that left the
  view).

## Metadata (parquet key-value, or the json sidecar)

| key | value |
|---|---|
| `ace.source_video` | path the poses were computed from |
| `ace.fps`, `ace.num_frames` | of the source video |
| `ace.source_width`, `ace.source_height` | source pixel grid the 2D columns and `ace.fx..cy` are in |
| `ace.encode_width`, `ace.encode_height` | size the model saw (width scaled to 832, both axes snapped to 32) |
| `ace.fx`, `ace.fy`, `ace.cx`, `ace.cy` | pinhole camera in source pixels |
| `ace.intrinsics_fitted` | `true` when the camera was fitted from the predicted ray field (K-free), `false` when given |
| `ace.weights_repo`, `ace.weights_revision`, `ace.variant`, `ace.latent_encoder` | what produced the table |
| `ace.joint_order`, `ace.units`, `ace.camera_frame` | the conventions above, spelled out |

## Calibration sidecar

A one-row table `<stem>.camera.parquet` or `.csv` next to a video with columns
`fx, fy, cx, cy, width, height`. `width` and `height` name the pixel grid the four values are
in; they are rescaled to the video automatically.

Two optional columns describe the lens:

| column | type | meaning |
| --- | --- | --- |
| `distortion` | string | `none` (default), `brown_conrady` or `kannala_brandt` |
| `coefficients` | list<float64> (parquet) / `;`-separated string (csv) | `brown_conrady`: OpenCV `k1;k2;p1;p2;k3;k4;k5;k6` (`projectPoints`, rational terms included); `kannala_brandt`: OpenCV fisheye `k1;k2;k3;k4`. Shorter lists are zero-padded. |

The coefficients apply to unit-plane coordinates, so a sidecar stays valid at any video
resolution. The model itself is pinhole-only: `infer` refuses a distorted sidecar. Run
`ace-ego-hand undistort video.mp4 --camera video.camera.parquet` first; it writes
`video.pinhole.mp4` and a pinhole `video.pinhole.camera.parquet` that `infer` picks up.
The inline `--intrinsics fx,fy,cx,cy` form is always pinhole; `undistort --camera` also takes
`fx,fy,cx,cy,<distortion>,k1,k2,...` inline.
