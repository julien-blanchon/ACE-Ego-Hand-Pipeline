"""A YOLOv8 hand detector, used as an independent second opinion on `visible`.

The estimator sometimes draws a hand with `visible` near 1.0 where there is none (another
person's hand, a hand hallucinated on empty background); its own signals cannot tell. A box
detector trained on a different corpus can: a visible hand whose joints sit under no detected
box for a sustained run, while the detector is finding hands elsewhere in those frames, is
hidden by `detector_visibility` (postprocess). Weights are the hand detector shipped with WiLoR
(Potamias et al., 2024; `rolpotamias/WiLoR` `detector.pt`, a YOLOv8-m pose model whose box
and left/right class branches are kept and keypoint branch dropped), converted to safetensors
under `detector/` of the weights repo; it sees gloved, tool-holding and exoskeleton hands the
lighter detectors miss. A box scores as the best of its class scores; the class itself is not
used, the apparent side of a hand being unreliable.

Architecture: YOLOv8 CSP backbone of `Conv`/`C2f`/`SPPF`, a PAN neck, an anchor-free head
that regresses each box side as a 16-bin distribution (DFL). Scale m: width 0.75, depth 0.67,
deepest stage capped at 768 channels. Ref: Jocher et al., Ultralytics YOLOv8,
https://github.com/ultralytics/ultralytics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import nn
from torchvision.ops import batched_nms

from ..types import (
    Boxes,
    BoxScores,
    DetectorInput,
    DetectorLogits,
    FeatureMap,
    FrameBoxes,
    Image,
)
from ..utils.hub import HubModule

REG_MAX = 16  # bins of the distance distribution per box side
STRIDES = (8, 16, 32)
PAD_MULTIPLE = 32
PAD_VALUE = 114 / 255


@dataclass(frozen=True, slots=True)
class HandDetectorConfig:
    width: float = 0.75  # channel multiplier (0.5 at scale s, 0.75 at m)
    depth: float = 0.67  # bottleneck-count multiplier (0.33 at s, 0.67 at m)
    max_channels: int = 768  # cap on the deepest stage before the multiplier (1024 at s, 768 at m)
    num_classes: int = 2  # score outputs per location; a box scores as the best of them
    input_size: int = 512  # longer side the frames are letterboxed to
    score_threshold: float = 0.25
    iou_threshold: float = 0.45


class ConvBlock(nn.Module):
    """Conv2d (no bias) + BatchNorm + SiLU, `same` padding."""

    def __init__(self, cin: int, cout: int, kernel: int = 1, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, stride, kernel // 2, bias=False)
        self.norm = nn.BatchNorm2d(cout, eps=1e-3, momentum=0.03)

    def forward(self, x: FeatureMap) -> FeatureMap:
        return F.silu(self.norm(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, channels: int, shortcut: bool) -> None:
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, 3)
        self.conv2 = ConvBlock(channels, channels, 3)
        self.shortcut = shortcut

    def forward(self, x: FeatureMap) -> FeatureMap:
        y = self.conv2(self.conv1(x))
        return x + y if self.shortcut else y


class C2f(nn.Module):
    """Cross-stage partial block: split, `n` bottlenecks chained, every stage concatenated."""

    def __init__(self, cin: int, cout: int, n: int, shortcut: bool) -> None:
        super().__init__()
        self.hidden = cout // 2
        self.conv_in = ConvBlock(cin, 2 * self.hidden)
        self.blocks = nn.ModuleList(Bottleneck(self.hidden, shortcut) for _ in range(n))
        self.conv_out = ConvBlock((2 + n) * self.hidden, cout)

    def forward(self, x: FeatureMap) -> FeatureMap:
        stages = list(self.conv_in(x).split(self.hidden, dim=1))
        for block in self.blocks:
            stages.append(block(stages[-1]))
        return self.conv_out(torch.cat(stages, dim=1))


class SPPF(nn.Module):
    """Spatial pyramid pooling: three chained 5x5 max-pools, all four scales concatenated."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = channels // 2
        self.conv_in = ConvBlock(channels, hidden)
        self.conv_out = ConvBlock(hidden * 4, channels)

    def forward(self, x: FeatureMap) -> FeatureMap:
        y = [self.conv_in(x)]
        for _ in range(3):
            y.append(F.max_pool2d(y[-1], 5, 1, 2))
        return self.conv_out(torch.cat(y, dim=1))


class DetectHead(nn.Module):
    """Per scale: a box branch (4 sides x 16 bins) and a score branch (`num_classes` outputs)."""

    bins: torch.Tensor

    def __init__(self, channels: tuple[int, ...], num_classes: int) -> None:
        super().__init__()
        box_hidden = max(16, channels[0] // 4, REG_MAX * 4)
        score_hidden = max(channels[0], min(num_classes, 100))
        self.box = nn.ModuleList(
            nn.Sequential(
                ConvBlock(c, box_hidden, 3),
                ConvBlock(box_hidden, box_hidden, 3),
                nn.Conv2d(box_hidden, 4 * REG_MAX, 1),
            )
            for c in channels
        )
        self.score = nn.ModuleList(
            nn.Sequential(
                ConvBlock(c, score_hidden, 3),
                ConvBlock(score_hidden, score_hidden, 3),
                nn.Conv2d(score_hidden, num_classes, 1),
            )
            for c in channels
        )
        self.register_buffer("bins", torch.arange(REG_MAX, dtype=torch.float32))

    def forward(self, features: list[FeatureMap]) -> tuple[Boxes, BoxScores]:
        boxes, scores = [], []
        for feature, box_branch, score_branch, stride in zip(
            features, self.box, self.score, STRIDES, strict=True
        ):
            logits: DetectorLogits = rearrange(
                box_branch(feature), "b (s r) h w -> b h w s r", r=REG_MAX
            )
            distances = reduce(logits.softmax(dim=-1) * self.bins, "b h w s r -> b h w s", "sum")
            h, w = feature.shape[-2:]
            ys, xs = torch.meshgrid(
                torch.arange(h, device=feature.device, dtype=distances.dtype) + 0.5,
                torch.arange(w, device=feature.device, dtype=distances.dtype) + 0.5,
                indexing="ij",
            )
            centers = torch.stack([xs, ys], dim=-1)  # (h, w, 2) in cells
            top_left = centers - distances[..., :2]
            bottom_right = centers + distances[..., 2:]
            boxes.append(
                rearrange(
                    torch.cat([top_left, bottom_right], dim=-1) * stride, "b h w c -> b (h w) c"
                )
            )
            class_scores = rearrange(score_branch(feature), "b c h w -> b (h w) c").sigmoid()
            scores.append(reduce(class_scores, "b n c -> b n", "max"))
        return torch.cat(boxes, dim=1), torch.cat(scores, dim=1)


class HandDetector(nn.Module, HubModule):
    """YOLOv8-s, one class; `detect` takes uint8 RGB frames and returns boxes in frame pixels."""

    config_class: ClassVar[type] = HandDetectorConfig

    def __init__(self, config: HandDetectorConfig) -> None:
        super().__init__()
        self.config = config
        w = config.width
        c = tuple(int(min(x, config.max_channels) * w) for x in (64, 128, 256, 512, 1024))
        n = max(round(3 * config.depth), 1)  # 1 at s
        n2 = max(round(6 * config.depth), 1)  # 2 at s
        self.stem = nn.Sequential(
            ConvBlock(3, c[0], 3, 2), ConvBlock(c[0], c[1], 3, 2), C2f(c[1], c[1], n, True)
        )
        self.stage2 = nn.Sequential(
            ConvBlock(c[1], c[2], 3, 2), C2f(c[2], c[2], n2, True)
        )  # stride 8
        self.stage3 = nn.Sequential(
            ConvBlock(c[2], c[3], 3, 2), C2f(c[3], c[3], n2, True)
        )  # stride 16
        self.stage4 = nn.Sequential(
            ConvBlock(c[3], c[4], 3, 2), C2f(c[4], c[4], n, True), SPPF(c[4])
        )
        self.up_high = C2f(c[4] + c[3], c[3], n, False)  # stride 16 after the first upsample
        self.up_low = C2f(c[3] + c[2], c[2], n, False)  # stride 8
        self.down_mid = ConvBlock(c[2], c[2], 3, 2)
        self.merge_mid = C2f(c[2] + c[3], c[3], n, False)  # stride 16
        self.down_high = ConvBlock(c[3], c[3], 3, 2)
        self.merge_high = C2f(c[3] + c[4], c[4], n, False)  # stride 32
        self.head = DetectHead((c[2], c[3], c[4]), config.num_classes)

    def forward(self, x: DetectorInput) -> tuple[Boxes, BoxScores]:
        p3 = self.stage2(self.stem(x))
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        n4 = self.up_high(
            torch.cat([F.interpolate(p5, scale_factor=2.0, mode="nearest"), p4], dim=1)
        )
        n3 = self.up_low(
            torch.cat([F.interpolate(n4, scale_factor=2.0, mode="nearest"), p3], dim=1)
        )
        m4 = self.merge_mid(torch.cat([self.down_mid(n3), n4], dim=1))
        m5 = self.merge_high(torch.cat([self.down_high(m4), p5], dim=1))
        return self.head([n3, m4, m5])

    @torch.inference_mode()
    def detect(self, frames: list[Image]) -> list[FrameBoxes]:
        """Boxes `(x0, y0, x1, y1, score)` in source pixels for every frame, after NMS."""

        if not frames:
            return []
        device = next(self.parameters()).device
        batch = torch.from_numpy(np.stack(frames)).to(device)
        batch = rearrange(batch, "b h w c -> b c h w").float() / 255
        height, width = batch.shape[-2:]
        scale = self.config.input_size / max(height, width)
        new_h, new_w = round(height * scale), round(width * scale)
        resized = F.interpolate(batch, size=(new_h, new_w), mode="bilinear", align_corners=False)
        pad_h = -new_h % PAD_MULTIPLE
        pad_w = -new_w % PAD_MULTIPLE
        top, left = pad_h // 2, pad_w // 2
        padded = F.pad(resized, (left, pad_w - left, top, pad_h - top), value=PAD_VALUE)
        boxes, scores = self(padded)
        boxes = (boxes - boxes.new_tensor([left, top, left, top])) / scale
        keep = scores > self.config.score_threshold
        frame_index = torch.arange(len(frames), device=device)[:, None].expand_as(scores)[keep]
        candidates, candidate_scores = boxes[keep], scores[keep]
        kept = batched_nms(candidates, candidate_scores, frame_index, self.config.iou_threshold)
        detections = (
            torch.cat([candidates[kept], candidate_scores[kept, None]], dim=1).cpu().numpy()
        )
        owner = frame_index[kept].cpu().numpy()
        return [detections[owner == index] for index in range(len(frames))]
