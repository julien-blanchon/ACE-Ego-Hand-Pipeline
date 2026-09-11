"""The single inference path: a video file in, per-frame bimanual MANO hands out.

`load_estimator` assembles the four components from the weights repository (latent encoder,
trunk with adapters and caption, projector, MANO). `HandEstimator.predict` then runs one video:

1. decode and resize frames in segments on a background thread, stream-encode them to latents;
2. cut the latent clip into fixed windows (`InferenceConfig.window_latents`, upstream's tiling:
   anchored every 81 video frames, an end-anchored window covering the tail) and run the trunk
   and projector on batches of windows; later windows overwrite the overlap of earlier ones;
3. fit one pinhole camera to the predicted ray field when no calibration was given;
4. decode the MANO joints of every frame and project them through the camera.

Frames beyond the last `1 + 4k` boundary are covered by repeating the last frame, so every
source frame gets a pose; the copies are dropped again on output.
Ref: ACE-Ego-Hand arXiv:2608.20308
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import torch
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import load_file

from .config import InferenceConfig, ModelConfig
from .data.calibration import rescale
from .data.video import VideoInfo, encode_size, frames_to_model_input, iter_frames, probe
from .lora.adapters import LoRAConfig, apply_lora, is_lora_parameter_name
from .modeling.backbone import WanBackbone
from .modeling.camera import fit_pinhole, intrinsics_batch
from .modeling.hand_detector import HandDetector
from .modeling.mano import ManoLayer
from .modeling.projector import HandPrediction, HandProjector
from .postprocess import detector_visibility, interpolate_frames, verified_visibility
from .prediction import VideoPrediction, WeightsInfo, frame_quality
from .types import ClipProbability, FrameBoxes, Intrinsics, RayMap, VideoLatents
from .utils.hub import component_dir, config_from_json, no_init, repo_file
from .utils.tensors import to_numpy
from .wan.dit import WanDiT, WanDiTConfig
from .wan.latents import LatentEncoder, LatentStream
from .wan.taehv import TaehvEncoder
from .wan.vae import WanVAEEncoder

logger = logging.getLogger(__name__)

CAPTION_FILE = "caption_embedding.safetensors"
INT8_REVISION = "int8"  # the Hub branch whose `dit/` is the torchao int8 weight-only trunk
SEGMENT_FRAMES = 81  # upstream's training window; windows are anchored per segment
TEMPORAL_STRIDE = 4
MIN_WINDOW_LATENTS = SEGMENT_FRAMES // TEMPORAL_STRIDE + 1  # a window must span its segment
DETECT_BATCH = 32  # frames per hand-detector forward
PREFETCH_SEGMENTS = 2


class HandEstimator:
    """The loaded model: `predict(video)` is the only entry point."""

    def __init__(
        self,
        *,
        latent_encoder: LatentEncoder[LatentStream],
        backbone: WanBackbone,
        projector: HandProjector,
        mano: ManoLayer,
        detector: HandDetector,
        weights: WeightsInfo,
        device: torch.device,
    ) -> None:
        self.latent_encoder = latent_encoder
        self.backbone = backbone
        self.projector = projector
        self.mano = mano
        self.detector = detector
        self.weights = weights
        self.device = device

    @property
    def uses_calibration(self) -> bool:
        return self.weights.variant == "k"

    def compile(self) -> HandEstimator:
        """Regionally compile the trunk blocks."""

        self.backbone.compile_blocks()
        return self

    @torch.inference_mode()
    def predict(
        self,
        video: Path,
        *,
        intrinsics: Intrinsics | None,
        config: InferenceConfig,
        start_seconds: float = 0.0,
        num_frames: int | None = None,
    ) -> VideoPrediction:
        """Run the whole pipeline on one video, or on the frame range of an episode inside it.

        `intrinsics` is a 3x3 K in source pixels; `start_seconds` and `num_frames` address one
        episode of a file that packs several (LeRobot), otherwise the whole file is used.
        """

        if self.uses_calibration and intrinsics is None:
            raise ValueError("the K variant needs intrinsics; pass a calibration or use kfree")
        info = probe(video)
        detect = _BoxCollector(self.detector) if config.verify in ("detector", "both") else None
        prediction = self._predict_pass(
            video, info, intrinsics, config, start_seconds, num_frames, tap=detect
        )
        if detect is not None:
            detect.flush()
            prediction = _with_visible(prediction, detector_visibility(prediction, detect.boxes))
        if config.verify in ("mirror", "both"):
            mirrored = self._predict_pass(
                video, info, intrinsics, config, start_seconds, num_frames, mirror=True
            )
            prediction = _with_visible(prediction, verified_visibility(prediction, mirrored))
        return prediction

    def _predict_pass(
        self,
        video: Path,
        info: VideoInfo,
        intrinsics: Intrinsics | None,
        config: InferenceConfig,
        start_seconds: float,
        num_frames: int | None,
        *,
        mirror: bool = False,
        tap: Callable[[np.ndarray], None] | None = None,
    ) -> VideoPrediction:
        """One pass over the frames, or over their left-right mirror image whose result is mirrored
        back (hands swapped, x negated) so both passes describe the same video. `tap` sees every
        decoded frame as it streams by."""

        size = encode_size(info.width, info.height, config.encode_width)
        stride = 1 if config.target_fps is None else max(1, round(info.fps / config.target_fps))
        decoded = _Counted(
            iter_frames(video, start_seconds=start_seconds, num_frames=num_frames), tap=tap
        )
        frames = (frame for index, frame in enumerate(decoded) if index % stride == 0)
        if mirror:
            frames = (np.ascontiguousarray(frame[:, ::-1]) for frame in frames)
            if intrinsics is not None:
                intrinsics = _mirror_intrinsics(intrinsics, info.width)
        latents, num_source = self._encode_video(frames, size, config.segment_frames)
        num_frames = TEMPORAL_STRIDE * (latents.shape[2] - 1) + 1

        # The model only ever sees the encode grid, so K moves with the frames
        model_intrinsics = None
        if intrinsics is not None:
            model_intrinsics = rescale(intrinsics, (info.width, info.height), size).to(self.device)
        prediction, ray_map = self._predict_windows(
            latents, num_frames, size, model_intrinsics if self.uses_calibration else None, config
        )

        fitted = model_intrinsics is None
        camera = model_intrinsics if model_intrinsics is not None else fit_pinhole(ray_map, size)
        source_camera = rescale(camera, size, (info.width, info.height))

        # MANO joints of every frame, placed in the camera and projected through K
        joints = self.mano(prediction.global_orient, prediction.hand_pose, prediction.betas).joints
        joints_camera = joints + prediction.translation[..., None, :]
        joints_2d = _project(joints_camera, source_camera)

        keep = slice(0, num_source)  # drop the frames added to complete the last chunk
        predicted = VideoPrediction(
            global_orient=to_numpy(prediction.global_orient[0])[keep],
            hand_pose=to_numpy(prediction.hand_pose[0])[keep],
            betas=to_numpy(prediction.betas[0])[keep],
            translation=to_numpy(prediction.translation[0])[keep],
            joints_camera=to_numpy(joints_camera[0])[keep],
            joints_2d=to_numpy(joints_2d[0])[keep],
            presence=to_numpy(prediction.presence[0])[keep],
            visible=to_numpy(prediction.visible[0])[keep],
            quality=frame_quality(
                to_numpy(prediction.presence[0])[keep], to_numpy(prediction.visible[0])[keep]
            ),
            intrinsics=to_numpy(source_camera),
            intrinsics_fitted=fitted,
            source_size=(info.width, info.height),
            encode_size=size,
            fps=info.fps / stride,
        )
        if mirror:
            predicted = _unmirror(predicted)
        return interpolate_frames(predicted, stride, decoded.count, mano=self.mano)

    def _encode_video(
        self, source: Iterator[np.ndarray], size: tuple[int, int], segment_frames: int
    ) -> tuple[VideoLatents, int]:
        """Stream decoded frames through the latent encoder; returns `(latents, source frame count)`."""

        stream = self.latent_encoder.start_stream()
        pieces: list[VideoLatents] = []
        count = 0
        last: torch.Tensor | None = None
        for segment in _prefetch(_segments(source, segment_frames), PREFETCH_SEGMENTS):
            frames = frames_to_model_input(torch.from_numpy(segment), size, self.device)
            pieces.append(stream.push(frames))
            count += frames.shape[2]
            last = frames[:, :, -1:]
        if last is None:
            raise ValueError("no frames decoded")

        # Complete the last causal chunk by holding the final frame
        padding = -(count - 1) % TEMPORAL_STRIDE
        if padding:
            pieces.append(stream.push(last.expand(-1, -1, padding, -1, -1)))
        pieces.append(stream.finish())
        return torch.cat([p for p in pieces if p.shape[2] > 0], dim=2), count

    def _predict_windows(
        self,
        latents: VideoLatents,
        num_frames: int,
        size: tuple[int, int],
        intrinsics: Intrinsics | None,
        config: InferenceConfig,
    ) -> tuple[HandPrediction, RayMap]:
        """Run every window through the trunk and projector, batched, and merge them in time."""

        num_latents = latents.shape[2]
        width = min(config.window_latents, num_latents)
        plan = window_plan(num_latents, width)
        window_frames = TEMPORAL_STRIDE * (width - 1) + 1

        merged: dict[str, torch.Tensor] = {}
        ray_maps: list[RayMap] = []
        for start in range(0, len(plan), config.windows_per_batch):
            batch = plan[start : start + config.windows_per_batch]
            batch_latents = torch.cat([latents[:, :, a : a + width] for a, _ in batch], dim=0)
            camera = None
            if intrinsics is not None:
                camera = intrinsics_batch(intrinsics, size, len(batch))
            features = self.backbone(batch_latents)
            prediction = self.projector(
                features,
                num_frames=window_frames,
                mano=self.mano,
                image_size=size,
                intrinsics=camera,
            )
            ray_maps.append(prediction.ray_map)
            for index, (anchor, write_from) in enumerate(batch):
                low = TEMPORAL_STRIDE * anchor
                high = min(low + window_frames, num_frames)
                if write_from is not None:
                    low = max(low, write_from)
                for name in _PER_FRAME:
                    value = getattr(prediction, name)[index : index + 1]
                    if name not in merged:
                        merged[name] = value.new_zeros((1, num_frames, *value.shape[2:]))
                    offset = low - TEMPORAL_STRIDE * anchor
                    merged[name][:, low:high] = value[:, offset : offset + high - low]
                betas = prediction.betas[index : index + 1]
                if "betas" not in merged:
                    merged["betas"] = betas.new_zeros((1, num_frames, *betas.shape[1:]))
                merged["betas"][:, low:high] = betas[:, None]

        ray_map = torch.cat(ray_maps, dim=0)
        ray_map = rearrange(ray_map, "b c g h w -> 1 c (b g) h w")
        return (
            HandPrediction(
                global_orient=merged["global_orient"],
                hand_pose=merged["hand_pose"],
                betas=merged["betas"],
                translation=merged["translation"],
                joints_2d=merged["joints_2d"],
                presence=merged["presence"],
                visible=merged["visible"],
                ray_map=ray_map,
            ),
            ray_map,
        )


_PER_FRAME = ("global_orient", "hand_pose", "translation", "joints_2d", "presence", "visible")


def window_plan(num_latents: int, width: int) -> list[tuple[int, int | None]]:
    """Window anchors (in latent frames) and, for the tail window, the first video frame it writes.

    One window per 81-frame segment, anchored at the segment start and clamped so it fits; if the
    last segment window leaves frames uncovered, an end-anchored window writes only that gap, so
    every earlier frame keeps its value.
    """

    if width < min(MIN_WINDOW_LATENTS, num_latents):
        raise ValueError(
            f"window_latents must be at least {MIN_WINDOW_LATENTS} so consecutive segment "
            f"windows overlap, got {width}"
        )
    num_frames = TEMPORAL_STRIDE * (num_latents - 1) + 1
    num_segments = max(num_frames // SEGMENT_FRAMES, 1)
    last_start = num_latents - width
    anchors: list[tuple[int, int | None]] = [
        (min(SEGMENT_FRAMES * i // TEMPORAL_STRIDE, last_start), None) for i in range(num_segments)
    ]
    covered = TEMPORAL_STRIDE * anchors[-1][0] + TEMPORAL_STRIDE * (width - 1) + 1
    if covered < num_frames:
        anchors.append((last_start, covered))
    return anchors


class _Counted[T]:
    """Pass an iterator through, remember how many items it yielded, show each to `tap`."""

    def __init__(self, items: Iterator[T], tap: Callable[[T], None] | None = None) -> None:
        self.items = items
        self.tap = tap
        self.count = 0

    def __iter__(self) -> Iterator[T]:
        for item in self.items:
            self.count += 1
            if self.tap is not None:
                self.tap(item)
            yield item


class _BoxCollector:
    """Tap on the decoded frames: runs the detector on them in batches of `DETECT_BATCH` while the
    trunk pass streams, so the second opinion costs no second decode."""

    def __init__(self, detector: HandDetector) -> None:
        self.detector = detector
        self.boxes: list[FrameBoxes] = []
        self._batch: list[np.ndarray] = []

    def __call__(self, frame: np.ndarray) -> None:
        self._batch.append(frame)
        if len(self._batch) == DETECT_BATCH:
            self.flush()

    def flush(self) -> None:
        if self._batch:
            self.boxes.extend(self.detector.detect(self._batch))
            self._batch = []


def _segments(frames: Iterator[np.ndarray], segment_frames: int) -> Iterator[np.ndarray]:
    """Group decoded frames into `(f, H, W, 3)` arrays of at most `segment_frames`."""

    buffer: list[np.ndarray] = []
    for frame in frames:
        buffer.append(frame)
        if len(buffer) == segment_frames:
            yield np.stack(buffer)
            buffer = []
    if buffer:
        yield np.stack(buffer)


def _prefetch[T](items: Iterable[T], depth: int) -> Iterator[T]:
    """Produce `items` on a background thread, at most `depth` ahead of the consumer."""

    done = object()
    pending: queue.Queue[object] = queue.Queue(maxsize=depth)

    def producer() -> None:
        try:
            for item in items:
                pending.put(item)
        except BaseException as error:  # re-raised on the consumer side
            pending.put(error)
        finally:
            pending.put(done)

    threading.Thread(target=producer, daemon=True).start()
    while (item := pending.get()) is not done:
        if isinstance(item, BaseException):
            raise item
        yield cast(T, item)


def _mirror_intrinsics(intrinsics: Intrinsics, width: int) -> Intrinsics:
    """K of the left-right mirrored image: the principal point moves to `width - cx`."""

    mirrored = intrinsics.clone()
    mirrored[0, 2] = width - intrinsics[0, 2]
    return mirrored


def _unmirror(prediction: VideoPrediction) -> VideoPrediction:
    """Map a prediction made on the mirrored video back onto the original: a mirrored left hand
    is a right hand, so the slots swap; x coordinates flip and rotations are conjugated by the
    reflection `diag(-1, 1, 1)`."""

    flip = np.diag([-1.0, 1.0, 1.0]).astype(prediction.global_orient.dtype)
    swap = slice(None, None, -1)
    width = prediction.source_size[0]
    joints_2d = prediction.joints_2d[:, swap].copy()
    joints_2d[..., 0] = width - joints_2d[..., 0]
    intrinsics = prediction.intrinsics.copy()
    intrinsics[0, 2] = width - intrinsics[0, 2]
    return replace(
        prediction,
        global_orient=flip @ prediction.global_orient[:, swap] @ flip,
        hand_pose=flip @ prediction.hand_pose[:, swap] @ flip,
        betas=np.ascontiguousarray(prediction.betas[:, swap]),
        translation=prediction.translation[:, swap] * flip.diagonal(),
        joints_camera=prediction.joints_camera[:, swap] * flip.diagonal(),
        joints_2d=joints_2d,
        presence=np.ascontiguousarray(prediction.presence[:, swap]),
        visible=np.ascontiguousarray(prediction.visible[:, swap]),
        quality=np.ascontiguousarray(prediction.quality[:, swap]),
        intrinsics=intrinsics,
    )


def _with_visible(prediction: VideoPrediction, visible: ClipProbability) -> VideoPrediction:
    """The prediction with a second opinion on `visible`, `quality` following it."""

    return replace(prediction, visible=visible, quality=frame_quality(prediction.presence, visible))


def _project(points: torch.Tensor, intrinsics: Intrinsics) -> torch.Tensor:
    """Pinhole projection of camera-frame points `(..., 3)` to pixels `(..., 2)`."""

    depth = points[..., 2:3].clamp_min(1e-4)
    u = intrinsics[0, 0] * points[..., 0:1] / depth + intrinsics[0, 2]
    v = intrinsics[1, 1] * points[..., 1:2] / depth + intrinsics[1, 2]
    return torch.cat([u, v], dim=-1)


# --- loading ---


def _apply_backbone_delta(trunk: WanDiT, variant_dir: Path, *, merge: bool) -> None:
    """Inject the variant's LoRA adapters into the trunk and load them with the patch embedding.

    `backbone_delta.safetensors` holds everything ACE-Ego-Hand trained in the trunk: the adapter
    tensors, named after the adapter attributes, and the re-trained `patch_embedding`. The
    tensors are cast to the trunk's dtype (bf16), the rounding upstream applied under autocast.
    With `merge`, the adapters are then folded into the base weights and the plain linears
    are put back.
    """

    lora = apply_lora(
        trunk, config_from_json(LoRAConfig, json.loads((variant_dir / "config.json").read_text()))
    )
    delta = load_file(str(variant_dir / "backbone_delta.safetensors"))
    lora.load_lora_state_dict(
        {name: value for name, value in delta.items() if is_lora_parameter_name(name)}
    )

    trained = {name: value for name, value in delta.items() if not is_lora_parameter_name(name)}
    unexpected = trunk.load_state_dict(trained, strict=False).unexpected_keys
    if unexpected:
        raise KeyError(f"delta tensors with no place in the trunk: {sorted(unexpected)[:3]}")
    if merge:
        lora.merge()


def _load_int8_trunk(dit_dir: Path, device: torch.device) -> WanDiT:
    """Build the trunk and load its int8 weight-only variant (the Hub branch `int8`).

    The file is the trunk's own state dict with the 160 block linears quantized by torchao
    (`Int8WeightOnlyConfig`: int8 storage and a per-row scale, dequantized to bf16 at the
    matmul) and flattened for safetensors: each such weight is split into `._weight_qdata` and
    `._weight_scale` tensors plus JSON metadata in the header. Unflattening rebuilds the
    `Int8Tensor` subclass weights and `assign=True` puts them in place of the bf16 parameters;
    the LoRA adapters wrap these layers afterwards exactly as they wrap the bf16 ones.
    """

    try:
        from torchao.prototype.safetensors.safetensors_support import (
            unflatten_tensor_state_dict,
        )
    except ImportError as error:
        raise ImportError(
            "the int8 trunk needs torchao: install the `int8` extra (`ace-ego-hand[int8]`)"
        ) from error

    dit_config = config_from_json(WanDiTConfig, json.loads((dit_dir / "config.json").read_text()))
    with no_init():
        trunk = WanDiT(dit_config, device=device, dtype=torch.bfloat16)

    with safe_open(str(dit_dir / "model.safetensors"), framework="pt", device="cpu") as file:
        names: list[str] = file.keys()
        flat = {name: file.get_tensor(name) for name in names}
        state, leftover = unflatten_tensor_state_dict(flat, file.metadata())
    if leftover:
        raise KeyError(f"int8 trunk tensors with no owner: {sorted(leftover)[:3]}")
    trunk.load_state_dict(state, strict=True, assign=True)
    return trunk


def load_estimator(config: ModelConfig) -> HandEstimator:
    """Assemble the pipeline from the weights repository for one variant and latent encoder."""

    device = torch.device(config.device)
    repo, revision = config.repo_id, config.revision
    # Every component is built and loaded on the CPU and then moved: the ZeroGPU runtime of
    # Hugging Face Spaces emulates CUDA at import time and only supports the `.to(device)` path.
    cpu = torch.device("cpu")

    if config.latent_encoder == "wan":
        encoder: LatentEncoder[LatentStream] = WanVAEEncoder.from_pretrained(
            str(component_dir(repo, revision, "vae")),
            map_location="cpu",
            strict=True,
            device=cpu,
            dtype=torch.bfloat16,
        ).to(device)
    else:
        encoder = TaehvEncoder.from_pretrained(
            str(component_dir(repo, revision, "taehv")),
            map_location="cpu",
            strict=True,
            device=cpu,
            dtype=torch.bfloat16,
        ).to(device)

    if config.trunk_precision == "int8":
        trunk = _load_int8_trunk(component_dir(repo, INT8_REVISION, "dit"), cpu)
    else:
        trunk = WanDiT.from_pretrained(
            str(component_dir(repo, revision, "dit")),
            map_location="cpu",
            strict=True,
            device=cpu,
            dtype=torch.bfloat16,
        )
    variant_dir = component_dir(repo, revision, config.variant)
    # `Int8Tensor` weights cannot be added into, so the int8 trunk keeps the live adapters
    merge = config.merge_lora and config.trunk_precision != "int8"
    _apply_backbone_delta(trunk, variant_dir, merge=merge)

    caption_path = repo_file(repo, revision, CAPTION_FILE)
    caption = load_file(str(caption_path))["caption_embedding"]
    backbone = WanBackbone(trunk, caption).to(device).eval()

    projector = HandProjector.from_pretrained(
        str(variant_dir / "projector"), map_location="cpu", strict=True, device=cpu
    ).to(device)
    detector = HandDetector.from_pretrained(
        str(component_dir(repo, revision, "detector")), map_location="cpu", strict=True
    ).to(device)
    detector.eval()
    mano = ManoLayer.from_pretrained(
        str(component_dir(repo, revision, "mano")), map_location="cpu", strict=True, device=cpu
    ).to(device)
    weights = WeightsInfo(
        repo_id=repo,
        revision=variant_dir.parent.name,
        variant=config.variant,
        latent_encoder=config.latent_encoder,
    )
    logger.info(
        "loaded %s (%s, %s encoder, %s trunk) on %s",
        repo,
        config.variant,
        config.latent_encoder,
        config.trunk_precision,
        device,
    )
    estimator = HandEstimator(
        latent_encoder=encoder,
        backbone=backbone,
        projector=projector,
        mano=mano,
        detector=detector,
        weights=weights,
        device=device,
    )
    return estimator.compile() if config.compile else estimator
