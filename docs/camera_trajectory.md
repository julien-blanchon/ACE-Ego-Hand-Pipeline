# Camera trajectory, and the comparison with MINT

ACE-Ego-Hand predicts hands in the camera frame of each frame and nothing about how the camera
moved. This note records what that costs, what a camera trajectory would buy, how MINT
(wuji-ego-mint) gets one, how MINT compares with this pipeline on our data, and how a camera
head could be trained on top of ACE if it is ever needed.

## What a camera trajectory is and when it matters

A camera trajectory is the pose of the camera in a fixed world frame for every frame: a rotation
and a translation (camera-to-world), usually predicted as relative motion between consecutive
frames and integrated. With it, a camera-frame hand becomes a world-frame hand:
`p_world = R_c2w · p_cam + t_c2w`, and the wrist rotation is left-multiplied by `R_c2w`.

It matters when the hand motion is used as an action signal rather than as a per-frame pose:

- separating hand motion from head motion (in the camera frame a resting hand moves whenever the
  wearer turns);
- retargeting to a robot whose base does not move with the operator's head;
- bridging presence gaps and look-aways in a frame where the table does not move;
- scene-level reasoning across frames or cameras, workspace and reach statistics.

It does not change per-frame hand quality: articulation, grasp, contact and everything measured
in the benchmarks below are unaffected.

For the delivery datasets the recorded device pose (Aria MPS, HoloLens, Vision Pro ARKit,
HRDexDB calibration) already provides a metric trajectory, better than any monocular estimate;
reading that column next to the hand output is the first thing to do. A predicted trajectory only
earns its cost on footage without a recorded pose.

## MINT (wuji-ego-mint)

Paper: arXiv 2609.04958; code `wuji-technology/wuji-ego-mint`; weights `ZZJAsher/mint_v1`
(4.5 GB fp32); dataset `ZZJAsher/wuji_ego_mint` (pseudo-labels only, no videos).

Architecture: 1.14 B parameters. Frames are resized to 378x518 and encoded by a DINOv2
ViT-L/14 (999 tokens per frame); the LingBot-Map geometric aggregator alternates frame attention
and global attention over a 32-frame window, with a camera token, four register tokens and a
scale token prepended to every frame. Four heads read the features: camera extrinsics
(translation + quaternion, four refinement iterations), field of view, both hands' MANO state
(wrist translation in the camera frame, 6D wrist rotation, 15x6D joint rotations, betas; four
learned queries per hand, two refinement iterations), and a per-hand observability logit. Longer
videos are covered by 32-frame windows overlapping by 8 frames; camera poses of adjacent windows
are chained in SE(3) and the hand outputs blended. The command-line path adds an unscented Kalman
smoother on the CPU. Training: 1,021 h of pseudo-labels on Ego4D, EPIC-KITCHENS and EgoDex from
a HaWoR-based pipeline, then a camera-only stage on in-house metric trajectories. World-space
hands are the camera-frame hands composed with the predicted camera pose.

Reported zero-shot numbers: ARCTIC 29.6 mm, HOT3D 29.9 mm MPJPE-p; ACE-Ego-Hand reports 15.3 and
12.9 mm there but trains on both, so the papers are not comparable.

## Benchmark against this pipeline (2026-09-12, one GH200)

Ground truth: the EgoVerse Aria, HoloAssist and HRDexDB clips used in the earlier configuration
study (8 clips, 161 to 481 frames), zero-shot for both models. EgoDex clips were cut too, but
the delivery EgoDex hand joints do not reproject on the video (wrist off by 75 to 140 px, both
models 65 to 100 mm wrist-aligned while Procrustes error is 14 mm), so they are excluded.
Errors in mm over GT-valid joints, coverage-aware (every GT-valid frame counted). Both models
flag the same frames as not visible (recall 0.99).

| family | model | abs MPJPE | wrist-aligned | PA | 2D EPE px | wrist depth pred/GT |
|---|---|---:|---:|---:|---:|---:|
| Aria (4 clips) | ACE K-free | 146 | 26.1 | 9.9 | 31 | 1.35 |
| | MINT | 219 | 23.8 | 9.5 | 21 | 1.47 |
| HoloAssist (2) | ACE K-free | 66 | 29.8 | 11.7 | 17 | 0.91 |
| | MINT | 213 | 24.6 | 11.5 | 18 | 1.57 |
| HRDexDB (2) | ACE K-free | 137 | 26.4 | 9.7 | 23 | 0.78 |
| | MINT | 52 | 26.0 | 10.3 | 17 | 0.97 |
| all 8 | ACE K-free | 124 | 27.1 | 10.3 | | |
| | MINT | 176 | 24.6 | 10.2 | | |

Speed per predicted frame, eager, model loaded, median over 23 clips: ACE 11.1 ms from file to
joints (10.1 compiled, 4.8 with the taehv encoder), MINT 41.5 ms without its smoother and 55.5 ms
with it. Peak GPU memory 12 GiB (ACE) and 7.8 GiB (MINT). Neither model recovers the focal
length (Aria fx 213: ACE 285 to 298, MINT 289 to 341).

Reading: MINT articulates slightly better (2.5 mm wrist-aligned, identical after Procrustes) and
places hands better in 2D; its absolute 3D placement is worse on two of three cameras because of
a larger per-camera depth-scale bias (both models have one); it is 4 to 5x slower. On the hard
Space examples it is no better: it swaps handedness on the gloved hands and keeps a "left" hand
on the partner's hand in the Jenga clip, where ACE's `visible` at least drops it. Its one
advantage is the camera trajectory. Conclusion: keep ACE for hands; a per-camera depth-scale
correction would help either model more than switching.

## Getting a trajectory without training

1. Use the device pose of the dataset where one exists (all delivery families).
2. For footage without a pose, run a feed-forward multi-view model (VGGT, pi3, Depth Anything 3)
   on windows of 16 to 32 frames at 15 fps and ~336 px, chain the windows with an SE(3) fit on
   the overlapping frames, interpolate to the hand frame rate and compose. This is what MINT does
   internally; cost is comparable to MINT on the subsampled frames only, and the scale drifts
   like any monocular estimate unless a metric variant or a known length fixes it.
3. Running MINT for its camera head alone costs its whole trunk (40 ms per frame) for the same
   drift, so it is not a shortcut.

## Training a camera head on top of ACE

The trunk already encodes scene geometry per frame (the ray-field head is what makes the K-free
variant work), so the addition is a head, and possibly a second LoRA.

**Head.** Relative motion between consecutive latent frames (translation and a 6D or quaternion
rotation), integrated into a trajectory; relative motion avoids the origin ambiguity and lets
windows chain. Input: the trunk tokens the projector reads, pooled per latent frame through a
learned camera query or a small temporal transformer. The latent stream is 4 frames per token, so
the native rate is one pose per 4 frames (7.5 Hz at 30 fps), interpolated, or four sub-steps per
latent for full rate. Losses as in MINT: L1 on translation, geodesic on rotation, plus velocity
terms; metric translation supervised directly since the data is metric.

**Data.** The delivery families with a device pose: Aria (19.7 k episodes), Mecka (41.6 k),
HoloAssist (1.2 k), EgoDex (338 k, ARKit), HRDexDB, all with hand ground truth in the same rows,
so hands and camera can be trained jointly without pseudo-labels. Check that each family's pose
reprojects before training on it (the EgoDex hand columns do not).

**Stages.**

1. Head-only probe: run the frozen trunk once over a few thousand episodes, cache the tokens,
   train the head on the cache (hours on one GPU). This answers whether the trunk encodes
   egomotion without touching the hands. Target: relative pose error within about 2x of MINT
   (3 to 5 mm and 0.25 degrees per step on HOT3D/ARCTIC).
2. If close but not there: a second LoRA on the trunk trained jointly with the hand losses, with
   the study's GT clips as the regression gate (the numbers above must not move).
3. Evaluate zero-shot on a held-out family; expect a per-device scale bias like MINT's, and
   drift over long horizons like every monocular model.

Compute is small; the work is data plumbing (the LeRobot reader and the fixture tooling cover
most of it) and evaluation. The recorded device pose remains the reference for the delivery
datasets; the trained head is for footage without one.
