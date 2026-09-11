# Architecture

ACE-Ego-Hand reads hand geometry out of a video diffusion transformer instead of denoising with
it: clean video latents go through the first 16 blocks of Wan2.2-5B in one deterministic pass,
and a small readout turns the block-15 hidden state into per-frame MANO hands. This document
walks the forward pass of this implementation with shapes for one 85-frame window at 832x480.

## 1. Video to latents (`data/video.py`, `wan/vae.py` or `wan/taehv.py`)

| step | shape | notes |
|---|---|---|
| decoded frames | `(F, H, W, 3)` uint8 | PyAV, streamed in segments on a background thread |
| resized | `(1, 3, F, 480, 832)` float [0, 1] | width scaled to 832, both axes snapped to 32, antialiased bilinear |
| latents | `(1, 48, G, 30, 52)` float32 | `G = 1 + (F - 1) / 4`; normalized with the VAE's per-channel mean / std |

The Wan2.2 VAE encoder is causal: the first frame is encoded alone, then 4-frame chunks, with a
two-frame convolution cache and the residual down-block's average shortcut computed per chunk.
`start_stream()` exposes exactly that structure so a long video is encoded piecewise with results
identical to one call. TAEHV (`latent_encoder: taehv`) is a 1.5 M-parameter approximation that
emits latents in the same normalized space (relative RMS error ~0.2 against the VAE, 30x faster).

A video is padded to `1 + 4k` frames by repeating its last frame; the copies are dropped on output.

## 2. Windows (`inference.py`)

Latents are cut into windows of `window_latents` (22 → 85 frames), one anchored at the start of
every 81-frame segment (`anchor = 81 i / 4`, clamped to fit), plus an end-anchored window that
writes only the frames the others left uncovered. Overlapping frames take the value of the later
window, as upstream. Windows are stacked along the batch axis (`windows_per_batch`) for the
trunk and the projector; several videos can share batches in the LeRobot script.

## 3. Trunk (`wan/dit.py`, `modeling/backbone.py`)

| step | shape |
|---|---|
| patchify Conv3d(148 → 3072, kernel = stride = (1, 2, 2)), evaluated over the 48 latent channels (the 100 control channels are zero) | `(B, 8580, 3072)` |
| adaLN modulation from timestep 0 (shared MLP, float32) | `(B, 6, 3072)` |
| text context: the constant 27-token caption embedding padded with zeros to 512, projected | `(B, 512, 3072)` |
| 16 blocks: adaLN self-attention with 3D rotary positions over (frame, row, column), cross-attention to the caption, adaLN feed-forward; LoRA rank 64 on every projection | `(B, 8580, 3072)` |
| fold back onto the grid | `(B, 3072, 22, 15, 26)` float32 |

Numerics follow upstream: bf16 projections and attention, float32 norms, modulation, rotary
rotation and residual stream (the stream is promoted by the first gated add).

`trunk_precision: int8` loads `dit/` from the `int8` branch of the weights repository instead:
the same trunk with the 160 block linears quantized by torchao to int8 weight-only (int8 storage
with a per-row scale, dequantized to bf16 at the matmul; LoRA, patch, time and text embeddings
stay bf16). It halves the trunk on disk (5.4 -> 2.8 GB), cuts peak GPU memory by ~2.4 GiB and
costs no speed on CUDA, for a joint error of 0.45-0.8 mm median against the bf16 trunk (the bf16
noise floor is 0.35 mm). It needs the `int8` extra (`torchao>=0.16`); loading rebuilds the
`Int8Tensor` weights from the flattened safetensors and assigns them in place of the bf16
parameters before the adapters are applied.

With `compile: true` each block is compiled in place with `block.compile(dynamic=True)`
(regional compilation): dynamo caches per code object, so the 16 blocks share two compiled
entries (a bf16 stream for block 0, float32 after) traced once, with no graph breaks, and a new
aspect ratio costs 0.4-0.6 s. `torch.compiler.nested_compile_region` around `WanBlock.forward`
with one compiled block loop was measured on torch 2.11 and not adopted: with dynamic shapes it
fails to compile (`DataDependentOutputException` under `invoke_subgraph`), and with static shapes
it recompiles the whole loop per aspect ratio (~21 s each) for the same steady-state time.

## 4. Readout (`modeling/projector.py`)

1. **Ray field**: a 1x1 head on the tap predicts a ray direction per cell; its mean over frames is
   the clip's camera, used for a Fourier ray positional encoding and, in the K-free variant, as
   the bearing source of the translation solve.
2. **Patch tokens**: Linear(3072 → 384) + LayerNorm + learned 16x16 spatial encoding (bilinearly
   resized to 15x26) + ray encoding → `(B·22, 390, 384)`.
3. **Alternating encoder**: 48 special tokens per latent frame (2 hand, 42 joint, 4 register)
   cross-attend the frame's patches, then self-attend across the clip (`(B, 22·48, 384)`) with
   rotary frame positions; 4 rounds.
4. **Joint readout**: refined joint tokens attend the patches once more; the head-averaged
   attention map is a heatmap whose soft-argmax gives the 2D anchor in [0, 1].
5. **Heads** on the hand tokens, interpolated to 85 frames: global orientation (6D), 15 finger
   rotations (6D), camera anchor `(u, v, log z)`, presence, visibility; shape from the hand
   features pooled over the window.
6. **Placement** (`modeling/camera.py`): MANO decodes the canonical joints; `(tx, ty)` is the
   weighted least-squares shift that aligns them to the 2D anchors (bearings from K, or from the
   ray field), accepted when at least 6 joints vote and the residual is small, else the wrist
   anchor is inverse-projected at the predicted depth.

## 5. Output (`inference.py`, `data/pose_table.py`)

For K-free runs one pinhole camera is fitted to the ray field of all windows (least squares of
pixel centre against ray tangent). The MANO joints of every frame are decoded, translated and
projected through that camera (or the given one) into source pixels, and written as the pose
table (`docs/pose_format.md`).

## Weights (`utils/hub.py`, `inference.py`, `lora/adapters.py`)

Everything loads from one repository, `blanchon/ACE-Ego-Hand-Safetensors`, already in this
implementation's own tensor layout; there is no conversion at load time. Each weight-bearing
component (`dit/`, `vae/`, `taehv/`, `mano/`, `<variant>/projector/`) is a `HubModule`: a
`config.json` that rebuilds its frozen config dataclass plus a `model.safetensors` loaded with
`strict=True`. `load_estimator` builds every component on the CPU and moves it to the device
afterwards (the ZeroGPU runtime only supports that path), then:

1. `<variant>/config.json` gives the LoRA rank, alpha and target modules; `apply_lora` swaps the
   targeted `nn.Linear` layers of the trunk for `LoRALinear` in place, so the trunk keeps its
   identity and its `state_dict` names.
2. `<variant>/backbone_delta.safetensors` holds what ACE-Ego-Hand trained inside the trunk: the
   adapter tensors, named after the adapter attributes (`blocks.N.<target>.lora_A.weight` /
   `lora_B.weight`), and the re-trained `patch_embedding`. The adapters are loaded through the
   `LoRAModel` handle and stay unmerged, evaluated live in bf16 beside the frozen projections;
   the patch embedding is copied into the trunk.
3. `caption_embedding.safetensors` is the encoded constant caption `(27, 4096)`; the text encoder
   never runs at inference.

## What is deliberately not here

No denoising head, no sampling, no image-conditioning or reference-frame branches of the
Fun-Control model, no text encoder at inference (the caption embedding is precomputed), no
training code, and none of the upstream projector's ablation variants: the released
configuration is the only path.
