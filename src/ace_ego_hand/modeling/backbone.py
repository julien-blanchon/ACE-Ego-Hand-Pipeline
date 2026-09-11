"""The feature extractor: Wan2.2 trunk with its adapters, run in the fixed ACE regime.

`WanBackbone` owns the pruned trunk (`WanDiT`) with the ACE LoRA adapters and trained patch
embedding applied, and the one constant caption context the model was trained against. A forward
takes normalized latents of one or more windows and returns the block-15 features folded onto
the latent grid, in float32, which is what the projector reads. The timestep is always zero and
there is no noise, so the extractor is a deterministic function of the latents.
Ref: ACE-Ego-Hand arXiv:2608.20308
"""

from __future__ import annotations

from typing import Self, override

import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn

from ..types import CaptionEmbedding, TapFeatures, TextContext, VideoLatents
from ..wan.dit import WanDiT

CLEAN_TIMESTEP = 0.0  # the extraction regime: clean latents, sigma 0


class WanBackbone(nn.Module):
    """`latents -> block-15 features`, with the caption context and timestep fixed."""

    context: TextContext

    def __init__(self, trunk: WanDiT, caption: CaptionEmbedding) -> None:
        super().__init__()
        self.trunk = trunk
        # The trunk was trained on a 512-token context, right-padded with zeros
        padded = F.pad(caption, (0, 0, 0, trunk.config.text_length - caption.shape[0]))
        self.register_buffer("context", padded[None].to(trunk.dtype), persistent=False)

    def compile_blocks(self) -> Self:
        """Compile the repeated trunk block in place and return self for chaining."""

        self.trunk.compile_blocks()
        return self

    @override
    def forward(self, latents: VideoLatents) -> TapFeatures:
        batch = latents.shape[0]
        timesteps = torch.full((batch,), CLEAN_TIMESTEP, device=latents.device)
        context = repeat(self.context, "1 l d -> b l d", b=batch)
        tokens = self.trunk(latents, timesteps, context)
        return self.trunk.fold(tokens, self.trunk.grid(latents)).float()
