"""Wan2.2 causal video VAE, encoder half: the RGB-to-latent map the diffusion backbone reads.

`encode` takes an RGB clip in `[0, 1]` and returns normalized latents, 16x smaller in space, 4x
in time, with 48 channels. There is no decoder: ACE only ever reads latents, never renders them.

    clip        (B, 3, 1 + 4k, H, W)         RGB in [0, 1], scaled to [-1, 1] on the way in
    unshuffle   (B, 12, 1 + 4k, H/2, W/2)     2x2 pixel-unshuffle: the first spatial halving
    conv_in     160 channels
    stage 0     160 -> 160, spatial /2        two residual blocks, then a stride-2 conv
    stage 1     160 -> 320, spatial /2, time /2
    stage 2     320 -> 640, spatial /2, time /2
    stage 3     640 -> 640                    no downsample
    mid         residual, attention, residual at the bottleneck
    moments     (B, 96, 1 + k, H/16, W/16)    mean and log-variance of a diagonal Gaussian
    latents     (B, 48, 1 + k, H/16, W/16)    the mean, whitened per channel

Each stage is residual in the Wan2.2 sense: alongside the convolutional path, an average-pooling
shortcut folds the pooled pixels into channels and averages channel groups, so the stage output is
`convs(x) + avg_shortcut(x)`. Sampling is never used; like the reference pipelines we take the
posterior mean, so the map is deterministic. The 48 channels have very different scales, so the
`latents_mean` / `latents_std` of the checkpoint config whiten them; the backbone only ever sees
the whitened form.

CAUSALITY. Every temporal operation looks only backwards and the first frame is always its own
group: `1 + 4k` video frames become `1 + k` latent frames with frame 0 mapping to latent frame 0
alone. Hence a single image is `k = 0` on the identical code path, and a prefix of a clip encodes
to a prefix of its latents (`test_vae_is_causal_in_time`).

STREAMING. The reference runs the encoder in temporal chunks -- the first frame alone, then four
frames at a time -- threading a list of cached activations and an integer cursor through every
call. Here the state lives on the module that owns it (`past` on `CausalConv3d` and
`TemporalDownsample`) and a `WanVAEStream` drives the loop: it runs the first frame as soon as
it arrives, then `CALL_FRAMES` frames per call (several chunks at once, which the causal caches
make identical to one call per chunk, at better GPU occupancy), and the remainder at `finish`.
Call sizes follow the clip length alone, so the latents do not depend on how the frames were
pushed, and `encode` is simply one push and a finish. Two reference behaviours depend on the
chunk boundaries and are replicated exactly rather than made chunk-invariant: the temporal
stride-2 convolution skips the first chunk and later reads `[last cached frame, chunk]`, and the
average shortcut zero-pads each chunk at the front to an even frame count, so the lone first
frame is averaged with a zero frame.

# Adapted from https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/vae2_2.py and
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_wan.py
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol, override, runtime_checkable

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import nn

from ..types import Frames, LatentStatistics, VAEFeatures, VideoLatents
from ..utils.hub import HubModule

logger = logging.getLogger(__name__)

RGB_CHANNELS = 3
TIME_KERNEL = 3  # temporal extent of every causal convolution that mixes frames
TEMPORAL_KERNEL_SIZE = (TIME_KERNEL, 1, 1)
CALL_FRAMES = 16  # video frames per encoder call after the first (memory and speed, not results)


@dataclass(frozen=True, slots=True)
class WanVAEConfig:
    """Architecture quantities of the Wan2.2 VAE; defaults are the released TI2V-5B checkpoint."""

    base_dim: int = 160
    z_dim: int = 48
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    temporal_downsample: tuple[bool, ...] = (False, True, True)  # one flag per downsampling stage
    patch_size: int = 2  # pixel-unshuffle factor applied before the first convolution
    latents_mean: tuple[float, ...] = (
        -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
        -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
        -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
        -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
        -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
        0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
    )  # fmt: skip
    latents_std: tuple[float, ...] = (
        0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
        0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
        0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
        0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
        0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
        0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
    )  # fmt: skip

    @property
    def spatial_stride(self) -> int:
        return self.patch_size * 2 ** (len(self.dim_mult) - 1)

    @property
    def temporal_stride(self) -> int:
        return 2 ** sum(self.temporal_downsample)


@runtime_checkable
class Streaming(Protocol):
    """A module that carries state from one temporal chunk to the next.

    Two implementers -- `CausalConv3d` and `TemporalDownsample` -- so `WanVAEEncoder` can reset
    whatever declares itself streaming without knowing the classes.
    """

    def reset_stream(self) -> None: ...


class CausalConv3d(nn.Conv3d):
    """3D convolution that only looks at past frames.

    Spatial padding is symmetric, so height and width survive unchanged. Temporal padding is moved
    entirely to the front, which is what makes the layer causal: output frame `t` depends on input
    frames `t - k + 1 ... t` and never on later ones. The front frames are zeros at the start of a
    clip and the trailing frames of the previous chunk when streaming.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=(0, kernel_size // 2, kernel_size // 2),
            device=device,
            dtype=dtype,
        )
        self.context_frames = kernel_size - 1
        self.past: VAEFeatures | None = None

    def reset_stream(self) -> None:
        self.past = None

    @override
    def forward(self, input: VAEFeatures) -> VAEFeatures:
        if self.context_frames == 0:
            return super().forward(input)

        if self.past is None:
            input = F.pad(input, (0, 0, 0, 0, self.context_frames, 0))
        else:
            input = torch.cat([self.past, input], dim=2)
        # A slice is a view onto the whole chunk; compacting it keeps only the trailing frames alive
        self.past = input[:, :, -self.context_frames :].contiguous()
        return super().forward(input)


class ChannelRMSNorm(nn.Module):
    """RMS normalization over the channel axis of `(B, C, F, H, W)`, scaled by `sqrt(C)` and a gain.

    Per-pixel, so it never mixes frames. Normalized in the working dtype, as the checkpoint's own
    code does; the gain is stored flat as `(C,)`.
    """

    def __init__(
        self,
        dim: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.scale = math.sqrt(dim)
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    @override
    def forward(self, x: VAEFeatures) -> VAEFeatures:
        return F.normalize(x, dim=1) * self.scale * rearrange(self.weight, "c -> c 1 1 1")


class ResidualBlock(nn.Module):
    """Two norm-SiLU-conv steps with a (projected) skip connection."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = ChannelRMSNorm(in_dim, device=device, dtype=dtype)
        self.conv1 = CausalConv3d(in_dim, out_dim, 3, device=device, dtype=dtype)
        self.norm2 = ChannelRMSNorm(out_dim, device=device, dtype=dtype)
        self.conv2 = CausalConv3d(out_dim, out_dim, 3, device=device, dtype=dtype)
        self.shortcut = (
            CausalConv3d(in_dim, out_dim, 1, device=device, dtype=dtype)
            if in_dim != out_dim
            else nn.Identity()
        )

    @override
    def forward(self, x: VAEFeatures) -> VAEFeatures:
        skip = self.shortcut(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + skip


class AttentionBlock(nn.Module):
    """Single-head self-attention over the pixels of each frame, added residually.

    Frames are attended independently -- the batch axis absorbs them -- which keeps this global
    spatial mixing causal for free. It runs only at the bottleneck, where the grid is 1/16 scale.
    """

    def __init__(
        self,
        dim: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.norm = ChannelRMSNorm(dim, device=device, dtype=dtype)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, device=device, dtype=dtype)
        self.output = nn.Conv2d(dim, dim, 1, device=device, dtype=dtype)

    @override
    def forward(self, x: VAEFeatures) -> VAEFeatures:
        b, _, _, h, _ = x.shape
        frames = rearrange(self.norm(x), "b c f h w -> (b f) c h w")
        query, key, value = rearrange(
            self.qkv(frames), "bf (three c) h w -> three bf 1 (h w) c", three=3
        )
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = rearrange(attended, "bf 1 (h w) c -> bf c h w", h=h)
        return x + rearrange(self.output(attended), "(b f) c h w -> b c f h w", b=b)


class MidBlock(nn.Module):
    """Residual block, attention block, residual block at the bottleneck resolution."""

    def __init__(
        self,
        dim: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.block_in = ResidualBlock(dim, dim, device=device, dtype=dtype)
        self.attention = AttentionBlock(dim, device=device, dtype=dtype)
        self.block_out = ResidualBlock(dim, dim, device=device, dtype=dtype)

    @override
    def forward(self, x: VAEFeatures) -> VAEFeatures:
        return self.block_out(self.attention(self.block_in(x)))


class TemporalDownsample(nn.Module):
    """Halve the frame count causally with a stride-2 kernel-3 convolution.

    Chunks arrive as the first frame alone, then even frame counts. The first chunk passes through
    untouched and only seeds the cache; every later chunk is convolved over `[last cached frame,
    chunk]`, so `2m` new frames become `m` and each output summarizes two new frames plus the one
    before them. This is what turns `1 + 4k` frames into `1 + k`.
    """

    def __init__(
        self,
        dim: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            dim, dim, TEMPORAL_KERNEL_SIZE, stride=(2, 1, 1), device=device, dtype=dtype
        )
        self.past: VAEFeatures | None = None

    def reset_stream(self) -> None:
        self.past = None

    @override
    def forward(self, x: VAEFeatures) -> VAEFeatures:
        if self.past is None:
            self.past = x[:, :, -1:].contiguous()
            return x

        stream = torch.cat([self.past, x], dim=2)
        self.past = stream[:, :, -1:].contiguous()
        return self.conv(stream)


class Downsample(nn.Module):
    """Spatial 2x downsampling per frame (pad right and bottom, stride-2 3x3 conv), then optionally
    temporal 2x downsampling."""

    def __init__(
        self,
        dim: int,
        *,
        temporal: bool,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.space = nn.Conv2d(dim, dim, 3, stride=2, device=device, dtype=dtype)
        self.time = (
            TemporalDownsample(dim, device=device, dtype=dtype) if temporal else nn.Identity()
        )

    @override
    def forward(self, x: VAEFeatures) -> VAEFeatures:
        f = x.shape[2]
        frames = rearrange(x, "b c f h w -> (b f) c h w")
        frames = self.space(F.pad(frames, (0, 1, 0, 1)))
        return self.time(rearrange(frames, "(b f) c h w -> b c f h w", f=f))


class AverageDownsample(nn.Module):
    """Parameter-free shortcut of a stage: pool `factor` pixels into channels, then average channel
    groups down to `out_dim`.

    The front temporal zero-padding to a multiple of `factor_t` is applied to whatever chunk is
    passed in, so the lone first frame of a clip is averaged with a zero frame -- a reference
    behaviour the checkpoint was trained with.
    """

    def __init__(self, in_dim: int, out_dim: int, *, factor_t: int, factor_s: int) -> None:
        super().__init__()
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.group_size = in_dim * factor_t * factor_s * factor_s // out_dim

    @override
    def forward(self, x: VAEFeatures) -> VAEFeatures:
        pad_t = -x.shape[2] % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        pooled = rearrange(
            x,
            "b c (t ft) (h fh) (w fw) -> b (c ft fh fw) t h w",
            ft=self.factor_t,
            fh=self.factor_s,
            fw=self.factor_s,
        )
        return reduce(pooled, "b (o g) t h w -> b o t h w", "mean", g=self.group_size)


class ResidualDownBlock(nn.Module):
    """One encoder stage: residual blocks and a downsample, plus the average-pooling shortcut."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        *,
        spatial: bool,
        temporal: bool,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        widths = [in_dim] + [out_dim] * (num_res_blocks - 1)
        self.resnets = nn.ModuleList(
            ResidualBlock(width, out_dim, device=device, dtype=dtype) for width in widths
        )
        self.downsampler = (
            Downsample(out_dim, temporal=temporal, device=device, dtype=dtype)
            if spatial
            else nn.Identity()
        )
        self.avg_shortcut = AverageDownsample(
            in_dim, out_dim, factor_t=2 if temporal else 1, factor_s=2 if spatial else 1
        )

    @override
    def forward(self, x: VAEFeatures) -> VAEFeatures:
        skip = self.avg_shortcut(x)
        for resnet in self.resnets:
            x = resnet(x)
        return self.downsampler(x) + skip


class WanVAEEncoder(nn.Module, HubModule):
    """Wan2.2 VAE encoder: `encode(clip in [0, 1]) -> normalized latents`, whole or streamed.

    The temporal state of the causal layers belongs to exactly one `WanVAEStream` at a time
    (`start_stream`), which resets it when it starts and again when it finishes or fails, so a
    clip's latents never depend on what was encoded before. `encode` is one push and a finish.
    """

    config_class = WanVAEConfig
    latents_mean: LatentStatistics
    latents_std: LatentStatistics

    def __init__(
        self,
        config: WanVAEConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        dims = [config.base_dim * mult for mult in (1, *config.dim_mult)]
        in_channels = RGB_CHANNELS * config.patch_size**2
        self.conv_in = CausalConv3d(in_channels, dims[0], 3, device=device, dtype=dtype)

        # Every stage but the last halves space; the config says which of those also halve time
        last_stage = len(config.dim_mult) - 1
        self.down_blocks = nn.ModuleList(
            ResidualDownBlock(
                in_dim,
                out_dim,
                config.num_res_blocks,
                spatial=stage < last_stage,
                temporal=stage < last_stage and config.temporal_downsample[stage],
                device=device,
                dtype=dtype,
            )
            for stage, (in_dim, out_dim) in enumerate(pairwise(dims))
        )

        self.mid_block = MidBlock(dims[-1], device=device, dtype=dtype)
        self.norm_out = ChannelRMSNorm(dims[-1], device=device, dtype=dtype)
        self.conv_out = CausalConv3d(dims[-1], 2 * config.z_dim, 3, device=device, dtype=dtype)
        self.quant_conv = CausalConv3d(
            2 * config.z_dim, 2 * config.z_dim, 1, device=device, dtype=dtype
        )

        # Per-channel latent statistics are config constants, not checkpoint weights
        mean = torch.tensor(config.latents_mean, device=device)
        std = torch.tensor(config.latents_std, device=device)
        self.register_buffer("latents_mean", rearrange(mean, "c -> 1 c 1 1 1"), persistent=False)
        self.register_buffer("latents_std", rearrange(std, "c -> 1 c 1 1 1"), persistent=False)
        self._streaming_modules = tuple(
            module for module in self.modules() if isinstance(module, Streaming)
        )
        self._stream_active = False  # a plain flag, so it never enters the state dict

    @property
    def dtype(self) -> torch.dtype:
        return self.conv_in.weight.dtype

    @property
    def spatial_stride(self) -> int:
        return self.config.spatial_stride

    @property
    def temporal_stride(self) -> int:
        return self.config.temporal_stride

    def encode(self, frames: Frames) -> VideoLatents:
        """Encode a `1 + 4k`-frame clip in `[0, 1]` to normalized posterior means in float32."""

        stream = self.start_stream()
        return torch.cat([stream.push(frames), stream.finish()], dim=2)

    def start_stream(self) -> WanVAEStream:
        """Begin a clip; the stream owns the temporal state until `finish` or an error."""

        if self._stream_active:
            raise RuntimeError("the Wan VAE encoder is already streaming another clip")
        self._reset_stream()
        self._stream_active = True
        return WanVAEStream(self)

    def _encode_chunk(self, x: VAEFeatures) -> VideoLatents:
        """One causal chunk of unshuffled pixels to its normalized latent frames (fp32)."""

        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        moments = self.quant_conv(self.conv_out(F.silu(self.norm_out(x))))

        mean, _log_variance = moments.float().chunk(2, dim=1)
        return (mean - self.latents_mean) / self.latents_std

    def _reset_stream(self) -> None:
        for module in self._streaming_modules:
            module.reset_stream()
        self._stream_active = False


class WanVAEStream:
    """Feed a clip to `WanVAEEncoder` in pieces; the encoder runs as soon as a call is whole.

    Pushed frames are scaled and pixel-unshuffled once, then held until a call is whole: the
    first frame alone, thereafter `CALL_FRAMES` frames each; `finish` runs whatever remains.
    Only the trailing incomplete call is buffered between pushes, next to the causal layers'
    own two-frame caches.
    """

    def __init__(self, encoder: WanVAEEncoder) -> None:
        self._encoder: WanVAEEncoder | None = encoder
        self._buffer: VAEFeatures | None = None  # unshuffled frames of the incomplete call
        self._call_frames = 1  # the first call is the lone first frame
        self._empty: VideoLatents | None = None  # a zero-frame result, shaped by the first call

    def push(self, frames: Frames) -> VideoLatents:
        encoder = self._active_encoder()
        try:
            # Scale in the input precision and round once, then pixel-unshuffle into channels
            x = (frames * 2 - 1).to(encoder.dtype)
            x = rearrange(
                x,
                "b c f (h q) (w r) -> b (c r q) f h w",
                q=encoder.config.patch_size,
                r=encoder.config.patch_size,
            )
            x = x if self._buffer is None else torch.cat([self._buffer, x], dim=2)

            # Run every call the buffer now completes, in order, keeping the causal state alive
            completed = []
            while x.shape[2] >= self._call_frames:
                completed.append(encoder._encode_chunk(x[:, :, : self._call_frames]))
                x = x[:, :, self._call_frames :]
                self._call_frames = CALL_FRAMES
            self._buffer = x.contiguous() if x.shape[2] else None
        except Exception:
            self._release()
            raise

        # The first push always completes the first call, so `_empty` exists from then on
        if not completed:
            return self._empty_latents()
        latents = torch.cat(completed, dim=2)
        self._empty = latents[:, :, :0]
        return latents

    def finish(self) -> VideoLatents:
        encoder = self._active_encoder()
        try:
            if self._call_frames == 1:
                raise ValueError("no frames were pushed before finish")
            if self._buffer is None:
                return self._empty_latents()
            stride = encoder.config.temporal_stride
            if self._buffer.shape[2] % stride:
                raise ValueError(
                    f"a clip needs 1 + {stride}k frames for the causal VAE; "
                    f"{self._buffer.shape[2] % stride} frames are left over"
                )
            return encoder._encode_chunk(self._buffer)
        finally:
            self._release()

    def _empty_latents(self) -> VideoLatents:
        assert self._empty is not None, "no frames were pushed"
        return self._empty

    def _active_encoder(self) -> WanVAEEncoder:
        if self._encoder is None:
            raise RuntimeError("this Wan VAE stream is finished")
        return self._encoder

    def _release(self) -> None:
        if self._encoder is not None:
            self._encoder._reset_stream()
        self._encoder = None
        self._buffer = None
