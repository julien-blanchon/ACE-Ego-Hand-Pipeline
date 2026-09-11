"""Encode clips to Wan 2.2 latents with TAEHV, the Tiny AutoEncoder for video (`taew2_2`).

The Wan 2.2 VAE encoder is the expensive half of feature extraction; TAEHV's 1.5 M-parameter
encoder emits the same 48-channel latents, already in the whitened space the DiT consumes, at a
fraction of the time and memory. Only the encoder is ported: nothing in the inference path ever
decodes. It is one of the two latent encoders behind `latent_encoder` in the pipeline config.

Architecture: a 2D conv stack applied per frame in which every memory block also sees the
previous frame's features (zeros for the first frame), and three temporal pooling stages, two of
which halve the frame count, so 4 frames become one latent frame and the receptive field is
causal. Frames are padded at the end to a multiple of 4 by holding the last frame, so a `1 + 4k`
clip yields `k + 1` latent frames like the Wan VAE.

STREAMING. The pooling groups align with 4-frame boundaries at every stage (4 -> 2 -> 1 -> 1), so
the only state a group leaves behind is the last input frame of each memory block. `TaehvStream`
carries those nine frames, buffers pushed frames until a group is whole, and on `finish` pads the
trailing 1..3 frames like the whole-clip scheme does. The module itself holds no state, so
`encode` is one push and a finish and several streams may share one encoder.

Adapted from https://github.com/madebyollin/taehv/blob/main/taehv.py (MIT).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import nn

from ..types import FrameFeatures, FrameMemory, Frames, VideoLatents
from ..utils.hub import HubModule

IMAGE_CHANNELS = 3
KERNEL_SIZE = 3
SPATIAL_DOWNSAMPLE = 2  # every stage halves height and width once
BLOCKS_PER_STAGE = 3

type StageMemory = list[FrameMemory | None]  # per memory block, `None` before the first frame
type EncoderMemory = list[StageMemory]  # one entry per stage


@dataclass(frozen=True, slots=True)
class TaehvConfig:
    """Architecture of the `taew2_2` encoder; `temporal_downsample` is per stage."""

    latent_channels: int = 48
    width: int = 64
    patch_size: int = 2
    temporal_downsample: tuple[bool, bool, bool] = (True, True, False)

    @property
    def spatial_stride(self) -> int:
        return self.patch_size * SPATIAL_DOWNSAMPLE ** len(self.temporal_downsample)

    @property
    def temporal_stride(self) -> int:
        return 2 ** sum(self.temporal_downsample)


def _conv(n_in: int, n_out: int, *, stride: int = 1, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(n_in, n_out, KERNEL_SIZE, stride=stride, padding=KERNEL_SIZE // 2, bias=bias)


class _MemBlock(nn.Module):
    """Residual block whose input is the frame's features concatenated with the previous frame's."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.conv_in = _conv(width * 2, width)
        self.conv_mid = _conv(width, width)
        self.conv_out = _conv(width, width)

    def forward(self, x: FrameFeatures, past: FrameFeatures) -> FrameFeatures:
        hidden = torch.relu(self.conv_in(torch.cat([x, past], dim=1)))
        hidden = self.conv_out(torch.relu(self.conv_mid(hidden)))
        return torch.relu(hidden + x)


class _TemporalPool(nn.Module):
    """Merge `stride` consecutive frames into one with a 1x1 convolution over their channels."""

    def __init__(self, width: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(width * stride, width, 1, bias=False)

    def forward(self, x: FrameFeatures) -> FrameFeatures:
        # Consecutive frames are stacked along channels, earlier frame first
        return self.conv(rearrange(x, "(bf s) c h w -> bf (s c) h w", s=self.stride))


class _Stage(nn.Module):
    """One temporal pool, one stride-2 conv and three memory blocks."""

    def __init__(self, width: int, *, temporal_downsample: bool) -> None:
        super().__init__()
        self.pool = _TemporalPool(width, 2 if temporal_downsample else 1)
        self.down = _conv(width, width, stride=SPATIAL_DOWNSAMPLE, bias=False)
        self.blocks = nn.ModuleList([_MemBlock(width) for _ in range(BLOCKS_PER_STAGE)])

    def forward(
        self, x: FrameFeatures, batch: int, memory: StageMemory
    ) -> tuple[FrameFeatures, StageMemory]:
        """Run the frames of one or more groups; `memory` is what each block saw last (or `None`)."""

        x = self.down(self.pool(x))

        # Every frame is processed at once; each memory block reads frame t - 1 of its own
        # input, which is the carried frame for the first one (zeros at the start of a clip)
        carried: StageMemory = []
        for block, past_frame in zip(self.blocks, memory, strict=True):
            frames = rearrange(x, "(b f) c h w -> b f c h w", b=batch)
            first = torch.zeros_like(frames[:, 0]) if past_frame is None else past_frame
            past = torch.cat([first[:, None], frames[:, :-1]], dim=1)
            carried.append(frames[:, -1])
            x = block(x, rearrange(past, "b f c h w -> (b f) c h w"))
        return x, carried


class TaehvEncoder(nn.Module, HubModule):
    """The TAEHV encoder; `encode` maps a `[0, 1]` clip to normalized Wan 2.2 latents.

    Stateless: the frame memory lives on a `TaehvStream`, and `encode` is one push and a finish.
    """

    config_class = TaehvConfig

    def __init__(
        self,
        config: TaehvConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.conv_in = _conv(IMAGE_CHANNELS * config.patch_size**2, config.width)
        self.stages = nn.ModuleList(
            [
                _Stage(config.width, temporal_downsample=downsample)
                for downsample in config.temporal_downsample
            ]
        )
        self.conv_out = _conv(config.width, config.latent_channels)
        self.to(device=device, dtype=dtype)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def spatial_stride(self) -> int:
        return self.config.spatial_stride

    @property
    def temporal_stride(self) -> int:
        return self.config.temporal_stride

    def forward(self, frames: Frames, memory: EncoderMemory) -> tuple[VideoLatents, EncoderMemory]:
        """The per-frame stack over whole groups (a multiple of the temporal stride) in the model dtype."""

        batch, patch = frames.shape[0], self.config.patch_size
        # Pixel-unshuffle: each 2x2 pixel block becomes channels, (C, P, Q) order like the upstream
        x = rearrange(frames, "b c f (h p) (w q) -> (b f) (c p q) h w", p=patch, q=patch)

        x = torch.relu(self.conv_in(x))
        carried: EncoderMemory = []
        for stage, stage_memory in zip(self.stages, memory, strict=True):
            x, stage_carried = stage(x, batch, stage_memory)
            carried.append(stage_carried)
        x = self.conv_out(x)
        return rearrange(x, "(b g) c h w -> b c g h w", b=batch), carried

    def encode(self, frames: Frames) -> VideoLatents:
        """Encode a `1 + 4k`-frame clip in `[0, 1]` to `k + 1` normalized latent frames (fp32)."""

        stream = self.start_stream()
        return torch.cat([stream.push(frames), stream.finish()], dim=2)

    def start_stream(self) -> TaehvStream:
        return TaehvStream(self)


class TaehvStream:
    """Feed a clip to `TaehvEncoder` in pieces; whole groups of frames run as soon as they exist.

    Between pushes the stream holds the trailing incomplete group (fewer than `temporal_stride`
    frames) and one frame of features per memory block; `finish` completes the last group by
    repeating its last frame, exactly as the whole-clip scheme pads the end of the clip.
    """

    def __init__(self, encoder: TaehvEncoder) -> None:
        self._encoder: TaehvEncoder | None = encoder
        self._buffer: Frames | None = None  # frames of the incomplete group, in the model dtype
        self._memory: EncoderMemory = [[None] * BLOCKS_PER_STAGE for _ in encoder.stages]
        self._empty: VideoLatents | None = None  # a zero-frame result, shaped by the first group

    def push(self, frames: Frames) -> VideoLatents:
        encoder = self._active_encoder()
        x = frames.to(encoder.dtype)
        x = x if self._buffer is None else torch.cat([self._buffer, x], dim=2)

        # One forward per whole group, so the result is the same whatever the push sizes (conv
        # kernels are chosen per batch shape) and activations never exceed one group's worth;
        # the remainder waits for the next push or the finish
        stride = encoder.config.temporal_stride
        whole = x.shape[2] - x.shape[2] % stride
        self._buffer = x[:, :, whole:].contiguous() if whole < x.shape[2] else None
        if whole == 0:
            return self._empty_latents(frames)
        groups = [x[:, :, start : start + stride] for start in range(0, whole, stride)]
        return torch.cat([self._run(encoder, group) for group in groups], dim=2)

    def finish(self) -> VideoLatents:
        encoder = self._active_encoder()
        self._encoder = None
        if self._buffer is None:
            raise ValueError(
                f"a clip needs 1 + {encoder.config.temporal_stride}k frames; the last group "
                "must be incomplete before finish"
            )

        # Complete the last group by holding the last frame, then run it as the final group
        x, self._buffer = self._buffer, None
        pad = encoder.config.temporal_stride - x.shape[2]
        x = torch.cat([x, x[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
        return self._run(encoder, x)

    def _run(self, encoder: TaehvEncoder, groups: Frames) -> VideoLatents:
        latents, self._memory = encoder(groups, self._memory)
        latents = latents.float()
        self._empty = latents[:, :, :0]
        return latents

    def _empty_latents(self, frames: Frames) -> VideoLatents:
        if self._empty is not None:
            return self._empty
        encoder = self._active_encoder()
        b, _, _, h, w = frames.shape
        stride = encoder.config.spatial_stride
        return frames.new_zeros(
            (b, encoder.config.latent_channels, 0, h // stride, w // stride), dtype=torch.float32
        )

    def _active_encoder(self) -> TaehvEncoder:
        if self._encoder is None:
            raise RuntimeError("this TAEHV stream is finished")
        return self._encoder
