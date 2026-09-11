---
title: ACE-Ego-Hand
emoji: 🤲
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.27.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Bimanual 3D hand motion from egocentric video
models:
- blanchon/ACE-Ego-Hand-Safetensors
---

# ACE-Ego-Hand

Bimanual 3D hand motion (MANO pose, shape and camera-space translation per frame) from an
egocentric video, through occlusion and out-of-view gaps, with no camera calibration (the
K-free model fits its own pinhole camera); a small hand detector only vets `visible`.

Upload a clip, get the rendered overlay + 3D view and a parquet table of the poses.

- Paper: [ACE-Ego-Hand, arXiv:2608.20308](https://arxiv.org/abs/2608.20308) (Liu et al., 2026)
- Upstream code: https://github.com/ggxxii/ACE-Ego-Hand
- This reimplementation: https://github.com/julien-blanchon/ACE-Ego-Hand-Pipeline
- Weights: https://huggingface.co/blanchon/ACE-Ego-Hand-Safetensors (CC BY-NC 4.0)
