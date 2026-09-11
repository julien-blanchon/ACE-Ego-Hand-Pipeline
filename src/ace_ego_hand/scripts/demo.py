# /// script
# requires-python = ">=3.12"
# dependencies = ["ace-ego-hand[gradio] @ git+https://github.com/julien-blanchon/ACE-Ego-Hand-Pipeline"]
# ///
"""Gradio demo: upload an egocentric clip, get the rendered hands and the pose table.

    uv run ace-ego-hand demo --port 7860

Needs the `gradio` extra (`uv sync --extra gradio`). The Space at
https://huggingface.co/spaces/blanchon/ACE-Ego-Hand wraps `build_demo` for ZeroGPU with the
K-free model only.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import tyro
from huggingface_hub import snapshot_download

from ace_ego_hand.config import (
    InferenceConfig,
    LatentEncoderKind,
    ModelConfig,
    Variant,
    Verification,
)
from ace_ego_hand.data.calibration import read_intrinsics
from ace_ego_hand.data.pose_table import prediction_to_table, table_info, write_pose_table
from ace_ego_hand.data.video import probe
from ace_ego_hand.inference import HandEstimator, load_estimator
from ace_ego_hand.postprocess import PostprocessConfig, postprocess
from ace_ego_hand.visualize import render_video

if TYPE_CHECKING:
    import gradio as gr

SPACE_REPO = "blanchon/ACE-Ego-Hand"
MAX_FRAMES_DEFAULT = 301
MAX_FRAMES_LIMIT = 4 * 300 + 1
TEMPORAL_STRIDE = 4
ENCODE_WIDTHS = (832, 640, 512)
FRAME_RATES = ("source", "15", "10")
VERIFICATIONS = ("none", "detector", "mirror", "both")
STEPS = PostprocessConfig()  # slider defaults
OUTPUT_HEIGHT = 540

type Wrap = Callable[[Callable[..., tuple[str, str, str]]], Callable[..., tuple[str, str, str]]]

GUIDE = """
# ACE-Ego-Hand

Bimanual 3D hand motion from egocentric video: MANO pose, shape and camera-space translation for
every frame, through occlusion and out-of-view gaps, with no hand detector.
[Paper](https://arxiv.org/abs/2608.20308) ·
[Upstream code](https://github.com/ggxxii/ACE-Ego-Hand) ·
[This reimplementation](https://github.com/julien-blanchon/ACE-Ego-Hand-Pipeline) ·
[Weights](https://huggingface.co/blanchon/ACE-Ego-Hand-Safetensors)

Upload a first-person clip or pick an example at the bottom, then press *Estimate hands*. The
model is pinhole-only: undistort fisheye footage first (`ace-ego-hand undistort`).
"""

SETTINGS_GUIDE = """
**K-free or K.** *K-free* needs nothing: the model predicts a ray field, fits its own pinhole
camera and places the hands in it. *K* uses the real intrinsics of the video (`fx,fy,cx,cy` in
pixels of the uploaded file) and gives placement consistent with that calibration.

**Latent encoder.** *wan* is the VAE the model was trained with, the reference result. *taehv*
encodes 30x faster with a few millimetres of extra placement error: previews of long clips.

**Encode width.** Frames are resized to this width before the model. 832 is the training width;
lower values are faster but move the 3D placement by 40 to 100 mm.

**Frame rate.** *source* predicts every frame. 15 or 10 fps predicts every 2nd or 3rd frame of a
30 fps clip and interpolates the rest: 1.3 to 1.8x faster, visibly smoother, a few mm away from
the full-rate result.

**Verify.** A second opinion on `visible`, the model's own being confident on hands that are
not there (another person's hand, a hand drawn on empty background). *detector* (default) runs
WiLoR's YOLO hand detector on every frame (a few percent of the runtime) and hides a hand no
box has supported for 1.5 s while the detector was finding hands elsewhere, but only a hand
the detector rarely confirms over the whole clip, so an occluded hand the model places from
context keeps its box history and is left alone; it also hides the stretches a wrist jump
joins to a box-supported one (a phantom teleporting before it lands on the real hand).
*mirror* runs the clip a second time flipped left-right and keeps a hand only where both
passes see it at the same place: the only check that tells another person's hand from the
wearer's, at twice the compute, and it can hide a real hand the mirrored pass misses (gloves).
*both* applies the two. *none* leaves the model's `visible` as is.

**Post-processing.** Off, the output is exactly the model's. On, wrist jumps no hand can make
are repaired (*max wrist speed* in frame widths per second, *3D* in metres per second, *turn*
in degrees per second of the hand root; a jump that comes back within *spike frames* is
interpolated, a teleport whose shorter side lasts at most *teleport frames* has that side
hidden), the wrist depth is smoothed along its camera ray (*depth weight*, 0 disables), one
hand shape is used for the clip and presence gaps of up to *presence gap* frames are bridged.
The `quality` column of the table is halved on synthesised frames.
"""


@dataclass(frozen=True, slots=True)
class DemoConfig:
    """Local Gradio demo of the pipeline."""

    port: int = 7860
    share: bool = False
    """Create a public Gradio share link."""

    variants: tuple[Variant, ...] = ("kfree", "k")
    """Variants offered; with only `kfree` the intrinsics field is hidden."""

    examples: Path | None = None
    """Folder of example clips; default: the examples of the Space, downloaded once."""

    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


class EstimatorCache:
    """Estimators loaded on demand, one per (variant, latent encoder)."""

    def __init__(self, base: ModelConfig) -> None:
        self.base = base
        self.loaded: dict[tuple[Variant, LatentEncoderKind], HandEstimator] = {}

    def get(self, variant: Variant, latent_encoder: LatentEncoderKind) -> HandEstimator:
        key = (variant, latent_encoder)
        if key not in self.loaded:
            self.loaded[key] = load_estimator(
                replace(self.base, variant=variant, latent_encoder=latent_encoder)
            )
        return self.loaded[key]


@dataclass(frozen=True, slots=True)
class DemoRequest:
    """One request of the interface, already parsed from the widget values."""

    video: Path
    variant: Variant
    latent_encoder: LatentEncoderKind
    intrinsics: str
    encode_width: int
    target_fps: float | None
    max_frames: int
    mesh: bool
    verify: Verification
    postprocess: PostprocessConfig | None


def run_demo(
    estimators: EstimatorCache, inference: InferenceConfig, request: DemoRequest
) -> tuple[str, str, str]:
    """Returns the rendered video path, the parquet path and a summary line."""

    info = probe(request.video)
    size = (info.width, info.height)
    camera = read_intrinsics(request.intrinsics, size) if request.intrinsics.strip() else None
    if request.variant == "k" and camera is None:
        raise ValueError("the K variant needs intrinsics: enter fx,fy,cx,cy in video pixels")
    estimator = estimators.get(request.variant, request.latent_encoder)

    num_frames = min(request.max_frames, info.num_frames) if info.num_frames else request.max_frames
    prediction = estimator.predict(
        request.video,
        intrinsics=camera,
        config=replace(
            inference,
            encode_width=request.encode_width,
            target_fps=request.target_fps,
            verify=request.verify,
        ),
        num_frames=num_frames,
    )
    if request.postprocess is not None:
        prediction = postprocess(prediction, request.postprocess, mano=estimator.mano)

    workdir = Path(tempfile.mkdtemp(prefix="ace-ego-hand-"))
    table = prediction_to_table(
        prediction, table_info(prediction, request.video, estimator.weights)
    )
    poses = write_pose_table(workdir / f"{request.video.stem}.hands.parquet", table, "parquet")
    rendered = workdir / f"{request.video.stem}.hands.mp4"
    render_video(
        request.video,
        prediction,
        rendered,
        panels="both",
        layout="vertical",
        mesh=request.mesh,
        mano=estimator.mano if request.mesh else None,
    )

    k = prediction.intrinsics
    how = "fitted" if prediction.intrinsics_fitted else "given"
    visible = (prediction.visible >= 0.5).mean(axis=0)
    mean_p, mean_v = prediction.presence.mean(axis=0), prediction.visible.mean(axis=0)
    hands = ", ".join(
        f"{name} visible {visible[slot]:.0%} of frames (mean p {mean_p[slot]:.2f} v {mean_v[slot]:.2f})"
        for slot, name in enumerate(("left", "right"))
    )
    summary = (
        f"{len(prediction.presence)} frames at {prediction.fps:.2f} fps, encoded at "
        f"{prediction.encode_size[0]}x{prediction.encode_size[1]}; camera {how}: "
        f"fx {k[0, 0]:.1f} fy {k[1, 1]:.1f} cx {k[0, 2]:.1f} cy {k[1, 2]:.1f}; {hands}"
    )
    return str(rendered), str(poses), summary


def example_clips(folder: Path | None) -> tuple[Path, ...]:
    """The example clips: a local folder, else the `examples/` of the Space repository."""

    if folder is None:
        folder = (
            Path(
                snapshot_download(SPACE_REPO, repo_type="space", allow_patterns=["examples/*.mp4"])
            )
            / "examples"
        )
    return tuple(sorted(folder.glob("*.mp4")))


def build_demo(
    estimators: EstimatorCache,
    inference: InferenceConfig,
    *,
    variants: tuple[Variant, ...] = ("kfree", "k"),
    examples: tuple[Path, ...] = (),
    wrap: Wrap | None = None,
) -> gr.Blocks:
    """The Gradio Blocks app; `wrap` decorates the predict function (the Space passes `spaces.GPU`)."""

    import gradio as gr

    def predict(
        video: str,
        variant: str,
        encoder: str,
        intrinsics: str,
        encode_width: float,
        frame_rate: str,
        max_frames: float,
        mesh: bool,
        verify: str,
        post: bool,
        max_wrist_speed: float,
        max_wrist_speed_3d: float,
        max_wrist_turn: float,
        spike_frames: float,
        teleport_frames: float,
        depth_acceleration: float,
        presence_gap: float,
        clip_betas: bool,
    ) -> tuple[str, str, str]:
        steps = PostprocessConfig(
            max_wrist_speed=max_wrist_speed,
            max_wrist_speed_3d=max_wrist_speed_3d,
            max_wrist_turn=max_wrist_turn,
            spike_frames=int(spike_frames),
            teleport_frames=int(teleport_frames),
            depth_acceleration=depth_acceleration,
            presence_gap=int(presence_gap),
            clip_betas=clip_betas,
        )
        request = DemoRequest(
            video=Path(video),
            variant=cast(Variant, variant),
            latent_encoder=cast(LatentEncoderKind, encoder),
            intrinsics=intrinsics,
            encode_width=int(encode_width),
            target_fps=None if frame_rate == "source" else float(frame_rate),
            max_frames=int(max_frames),
            mesh=mesh,
            verify=cast(Verification, verify),
            postprocess=steps if post else None,
        )
        return run_demo(estimators, inference, request)

    predict_fn = wrap(predict) if wrap is not None else predict
    with gr.Blocks(title="ACE-Ego-Hand") as demo:
        gr.Markdown(GUIDE)
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="Egocentric video", sources=["upload"])
                run = gr.Button("Estimate hands", variant="primary")
                with gr.Accordion("Settings", open=False):
                    variant = gr.Radio(
                        list(variants),
                        value=variants[0],
                        label="Variant",
                        visible=len(variants) > 1,
                    )
                    intrinsics = gr.Textbox(
                        label="Intrinsics fx,fy,cx,cy in video pixels (K only)",
                        value="",
                        visible="k" in variants,
                    )
                    encoder = gr.Radio(["wan", "taehv"], value="wan", label="Latent encoder")
                    encode_width = gr.Radio(
                        list(ENCODE_WIDTHS), value=inference.encode_width, label="Encode width"
                    )
                    frame_rate = gr.Radio(
                        list(FRAME_RATES), value="source", label="Predicted frame rate"
                    )
                    max_frames = gr.Slider(
                        5,
                        MAX_FRAMES_LIMIT,
                        value=MAX_FRAMES_DEFAULT,
                        step=TEMPORAL_STRIDE,
                        label="Max frames (from the start of the clip)",
                    )
                    mesh = gr.Checkbox(value=True, label="Draw the MANO mesh (else the skeleton)")
                    verify = gr.Radio(
                        list(VERIFICATIONS), value=inference.verify, label="Verify `visible`"
                    )
                    post = gr.Checkbox(value=False, label="Post-processing")
                    max_wrist_speed = gr.Slider(
                        0, 12, value=STEPS.max_wrist_speed, step=0.5, label="Max wrist speed"
                    )
                    max_wrist_speed_3d = gr.Slider(
                        0,
                        10,
                        value=STEPS.max_wrist_speed_3d,
                        step=0.5,
                        label="Max wrist speed 3D (m/s)",
                    )
                    max_wrist_turn = gr.Slider(
                        0, 3000, value=STEPS.max_wrist_turn, step=50, label="Max wrist turn (deg/s)"
                    )
                    spike_frames = gr.Slider(
                        1, 8, value=STEPS.spike_frames, step=1, label="Spike frames"
                    )
                    teleport_frames = gr.Slider(
                        0, 30, value=STEPS.teleport_frames, step=1, label="Teleport frames"
                    )
                    depth_acceleration = gr.Slider(
                        0, 1, value=STEPS.depth_acceleration, step=0.05, label="Depth weight"
                    )
                    presence_gap = gr.Slider(
                        0, 12, value=STEPS.presence_gap, step=1, label="Presence gap (frames)"
                    )
                    clip_betas = gr.Checkbox(
                        value=STEPS.clip_betas, label="One hand shape per clip"
                    )
                    gr.Markdown(SETTINGS_GUIDE)
            with gr.Column(scale=1):
                rendered = gr.Video(label="Overlay and 3D view", height=OUTPUT_HEIGHT)
        with gr.Row():
            summary = gr.Textbox(label="Summary", interactive=False, scale=3, lines=1)
            poses = gr.File(label="Pose table (parquet)", scale=1, height=80)
        if examples:
            gr.Examples(
                examples=[[str(path)] for path in examples],
                inputs=[video],
                label="Examples (10 s clips)",
                examples_per_page=len(examples),
            )
        run.click(
            predict_fn,
            [
                video,
                variant,
                encoder,
                intrinsics,
                encode_width,
                frame_rate,
                max_frames,
                mesh,
                verify,
                post,
                max_wrist_speed,
                max_wrist_speed_3d,
                max_wrist_turn,
                spike_frames,
                teleport_frames,
                depth_acceleration,
                presence_gap,
                clip_betas,
            ],
            [rendered, poses, summary],
        )
    return demo


def main(config: DemoConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    estimators = EstimatorCache(config.model)
    demo = build_demo(
        estimators,
        config.inference,
        variants=config.variants,
        examples=example_clips(config.examples),
    )
    demo.launch(server_port=config.port, share=config.share)


if __name__ == "__main__":
    main(tyro.cli(DemoConfig))
