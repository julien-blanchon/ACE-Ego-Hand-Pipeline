"""Frozen dataclass configs: quantities and the two checkpoint choices, never code paths.

`ModelConfig` says which weights to load and how; `InferenceConfig` says how a video is cut into
windows and batches. The scripts nest these into their own CLI configs (`tyro`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

WEIGHTS_REPO = "blanchon/ACE-Ego-Hand-Safetensors"

Variant = Literal["kfree", "k"]  # K-free reads the camera off the image; K takes a calibration
LatentEncoderKind = Literal["wan", "taehv"]  # the Wan2.2 VAE encoder, or its tiny distillation
# The trunk weights as shipped, or the torchao int8 weight-only copy on the Hub branch `int8`
TrunkPrecision = Literal["bf16", "int8"]
# Second opinions on `visible`, the estimator's own being confident on hallucinated hands:
# "detector" (default, 1 ms/frame) hides a hand no box of WiLoR's YOLOv8-m hand detector has
# supported for 1.5 s while boxes are found elsewhere, for hands the detector rarely confirms over
# the clip, and the unsupported stretches a wrist jump joins to a supported one; "mirror" runs the
# clip again flipped left-right and keeps a hand only where both passes agree (a second full
# pass, the only one that tells another person's hand from the wearer's); "both" applies the two.
Verification = Literal["none", "detector", "mirror", "both"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Which weights to load and where to run them."""

    repo_id: str = WEIGHTS_REPO
    revision: str | None = None
    variant: Variant = "kfree"
    latent_encoder: LatentEncoderKind = "wan"
    trunk_precision: TrunkPrecision = "bf16"  # int8 needs the `int8` extra; ~0.5 mm off bf16
    device: str = "cuda"
    compile: bool = False  # regional torch.compile of the trunk blocks; pays off past ~25 clips
    # Fold the LoRA adapters into the bf16 trunk weights at load (one matmul per linear instead
    # of three); the int8 trunk cannot be merged into, so it always keeps the live adapters
    merge_lora: bool = True


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """How a clip is resized, cut into windows and batched through the model."""

    encode_width: int = 832  # frames are scaled to this width, both axes snapped to 32
    target_fps: float | None = (
        None  # predict every n-th frame to approach this rate, interpolate the rest
    )
    window_latents: int = (
        22  # latent frames per window (85 video frames); the clip length runs it whole
    )
    windows_per_batch: int = 4  # windows the trunk and projector see in one forward
    segment_frames: int = 1 + 4 * 8  # frames decoded and encoded at a time (memory, not results)
    verify: Verification = "detector"  # what gates `visible` beyond the model's own head
