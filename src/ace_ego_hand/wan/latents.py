"""The contract shared by the two latent encoders: whole-clip `encode` and a push/finish stream.

The pipeline chooses between the Wan VAE encoder and TAEHV (`latent_encoder` in the config) and
otherwise never tells them apart, so both satisfy `LatentEncoder`. A stream lets a long video be
encoded as it is decoded, without holding every frame: `push` accepts any number of consecutive
frames and returns the latent frames those frames completed, `finish` flushes the rest. For a
clip of `1 + 4k` frames the concatenated push and finish results equal `encode` of the whole clip
whatever the partition into pushes; `encode` itself is one push followed by `finish`, so there is
a single code path. Both encoders map `1 + 4k` frames to `1 + k` latent frames at 1/16 scale.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Frames, VideoLatents


@runtime_checkable
class LatentStream(Protocol):
    """One clip being encoded incrementally; done after `finish` (or after an error)."""

    def push(self, frames: Frames) -> VideoLatents:
        """Add `f >= 1` frames; return the latent frames they completed (possibly none)."""
        ...

    def finish(self) -> VideoLatents:
        """Flush the buffered frames into the final latent frames (possibly none)."""
        ...


@runtime_checkable
class LatentEncoder[StreamT: LatentStream](Protocol):
    """RGB clip in `[0, 1]` to normalized Wan 2.2 latents, whole or streamed.

    Generic over the concrete stream an encoder hands out, so `start_stream` keeps its type.
    """

    @property
    def spatial_stride(self) -> int: ...

    @property
    def temporal_stride(self) -> int: ...

    def encode(self, frames: Frames) -> VideoLatents: ...

    def start_stream(self) -> StreamT: ...
