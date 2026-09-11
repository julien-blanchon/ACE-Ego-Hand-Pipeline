"""ACE-Ego-Hand on Hugging Face Spaces (ZeroGPU).

One file: the package does the work, this wires it to `spaces.GPU`. Models are loaded on CUDA at
import time, as ZeroGPU requires (CUDA is emulated outside the decorated function), and
`torch.compile` stays off because ZeroGPU does not support it. Only the K-free model is offered:
a visitor is not expected to know the intrinsics of an uploaded clip.
"""

from __future__ import annotations

from pathlib import Path

import spaces  # pyright: ignore[reportMissingImports]

from ace_ego_hand.config import InferenceConfig, ModelConfig
from ace_ego_hand.scripts.demo import EstimatorCache, build_demo, example_clips

GPU_SECONDS = 180

estimators = EstimatorCache(ModelConfig(device="cuda", compile=False))
estimators.get("kfree", "wan")  # the default model, resident before the first request

demo = build_demo(
    estimators,
    InferenceConfig(),
    variants=("kfree",),
    examples=example_clips(Path(__file__).parent / "examples"),
    wrap=spaces.GPU(duration=GPU_SECONDS),
)

if __name__ == "__main__":
    demo.queue().launch()
