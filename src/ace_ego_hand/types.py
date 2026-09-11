"""Jaxtyping aliases and the axis glossary shared by every module in the package."""

from __future__ import annotations

import numpy as np
from jaxtyping import Bool, Float, Int, UInt8
from torch import Tensor

# --- axis glossary ---
# B batch (windows)   H height (pixels or latent)   N tokens / sequence length
# C channels          W width  (pixels or latent)   D model / head / embed dim
# F frames (video)    G latent frames               L text tokens
# S hands per frame (2: left, right)                A attention heads
# J hand joints (21, OpenPose order)                K special tokens per latent frame
# P MANO pose rotations (16: global + 15 joints)    V MANO vertices (778)
# M MANO mesh triangles (1538)                      R LoRA rank
# Q joint queries per frame (S * J = 42)         T trail length (recent frames)

# --- video ---
type Clip = UInt8[Tensor, "F H W 3"]  # decoded RGB frames as stored by the decoder
type Frames = Float[Tensor, "B 3 F H W"]  # RGB in [0, 1], the latent encoders' input
type Image = UInt8[np.ndarray, "H W 3"]  # one RGB frame at the drawing boundary

# --- latent encoders ---
type VideoLatents = Float[Tensor, "B C G H W"]  # normalized latent video (C = 48)
type VAEFeatures = Float[Tensor, "B C G H W"]  # internal causal VAE feature maps
type LatentStatistics = Float[Tensor, "1 C 1 1 1"]  # per-channel latent mean or std
type FrameFeatures = Float[Tensor, "BF C H W"]  # per-frame 2D feature maps (TAEHV)
type FrameMemory = Float[Tensor, "B C H W"]  # the last frame a TAEHV memory block saw

# --- diffusion transformer ---
type Tokens = Float[Tensor, "B N D"]  # DiT token stream in (frame, height, width) raster order
type AttentionHeads = Float[Tensor, "B N A D"]  # projected query / key / value heads
type TextContext = Float[Tensor, "B L D"]  # text embeddings the DiT cross-attends to
type CaptionEmbedding = Float[Tensor, "L D"]  # one unpadded umT5 encoding of the caption
type TokenIds = Int[Tensor, "B L"]  # tokenizer ids, right-padded with the pad id
type TokenMask = Bool[Tensor, "B L"]  # True on real tokens, False on padding
type RelativePositionBuckets = Int[Tensor, "L L"]  # T5 relative-position bucket indices
type AttentionMaskBias = Float[Tensor, "B 1 1 L"]  # additive padding mask for attention
type Timesteps = Float[Tensor, "B"]  # diffusion timestep per sample, on the 0..1000 scale
type TimeEmbedding = Float[Tensor, "B D"]  # embedded timestep before the adaLN projection
type Modulation = Float[Tensor, "B 6 D"]  # adaLN shift / scale / gate vectors of one block
type RotaryTable = Float[Tensor, "N D"]  # per-token cos or sin of the rotary angles
type LatentGrid = tuple[int, int, int]  # (frames, height, width) of the DiT token raster
type TapFeatures = Float[Tensor, "B D G H W"]  # block-15 tokens folded back onto the grid

# --- hand detector (YOLOv8) ---
type DetectorInput = Float[Tensor, "B 3 H W"]  # letterboxed RGB in [0, 1]
type FeatureMap = Float[Tensor, "B C H W"]
type DetectorLogits = Float[Tensor, "B H W 4 R"]  # per-side distance bins (R = 16)
type Boxes = Float[Tensor, "B N 4"]  # x0 y0 x1 y1 in input pixels, every anchor
type BoxScores = Float[Tensor, "B N"]
type FrameBoxes = Float[np.ndarray, "K 5"]  # kept boxes of one frame: x0 y0 x1 y1 score

# --- LoRA ---
type WeightTensor = Float[Tensor, "..."]  # a parameter of any layout
type Activations = Float[Tensor, "..."]  # activations of an arbitrary wrapped layer

# --- projector ---
type PatchTokens = Float[Tensor, "BG HW D"]  # projected latent patches, one frame per row
type SpecialTokens = Float[Tensor, "BG K D"]  # hand, joint and register tokens per frame
type ClipSpecialTokens = Float[Tensor, "B GK D"]  # the same, unrolled over frames for time
type HandFeatures = Float[Tensor, "B F S D"]  # per-frame per-hand features (video frames)
type LatentHandFeatures = Float[Tensor, "B G S D"]  # the same at latent-frame resolution
type ShapeToken = Float[Tensor, "B D"]  # clip-level shape summary (register mean)
type JointHeatmaps = Float[Tensor, "BG Q HW"]  # readout attention over the patch grid
type RayMap = Float[Tensor, "B 3 G H W"]  # predicted camera-ray directions per latent cell
type RayField = Float[Tensor, "B 3 H W"]  # the ray map pooled over latent frames
type RayEncoding = Float[Tensor, "B HW D"]  # ray-direction positional encoding per patch
type SpatialEncoding = Float[Tensor, "1 HW D"]  # learned spatial positional encoding

# --- hand predictions (video-frame resolution, S = 2 slots: 0 left, 1 right) ---
type Rotation6D = Float[Tensor, "... 6"]
type RotationMatrices = Float[Tensor, "... 3 3"]
type GlobalOrient = Float[Tensor, "B F S 3 3"]
type HandPose = Float[Tensor, "B F S 15 3 3"]
type HandBetas = Float[Tensor, "B S 10"]  # one shape per hand per clip
type CameraRaw = Float[Tensor, "B F S 3"]  # head output: (u_norm, v_norm, log depth)
type HandTranslation = Float[Tensor, "B F S 3"]  # camera-space wrist translation, metres
type HandProbability = Float[Tensor, "B F S"]  # presence / visibility probabilities
type Joints2D = Float[Tensor, "B F S J 2"]  # normalized image coordinates in [0, 1]
type JointsRootRelative = Float[Tensor, "B F S J 3"]  # metres, wrist at the origin
type JointsCamera = Float[Tensor, "B F S J 3"]  # metres in the camera frame
type JointMask = Bool[Tensor, "B F S J"]
type HandMask = Bool[Tensor, "B F S"]

# --- MANO ---
type AxisAngle = Float[Tensor, "... P 3"]
type ManoJoints = Float[Tensor, "... J 3"]  # 21 OpenPose-ordered joints, metres
type ManoNativeJoints = Float[Tensor, "... P 3"]  # the 16 regressed MANO joints, metres
type ManoVertices = Float[Tensor, "... V 3"]
type ManoFaces = Int[Tensor, "*S M 3"]
# MANO model tensors: one hand as loaded from its pickle, or both hands stacked on a leading S
type ManoTemplate = Float[Tensor, "*S V 3"]  # mean-shape rest mesh
type ManoShapeDirs = Float[Tensor, "*S V 3 10"]  # shape blend shapes, one per beta
type ManoPoseDirs = Float[Tensor, "*S V 3 135"]  # pose blend shapes over the 15 (R - I) matrices
type ManoJointRegressor = Float[Tensor, "*S P V"]  # rest joints from the shaped mesh
type ManoSkinningWeights = Float[Tensor, "*S V P"]
type ManoParents = Int[Tensor, "P"]  # kinematic tree, -1 at the root
type RigidTransforms = Float[Tensor, "... P 4 4"]  # homogeneous per-joint transforms

# --- camera ---
type Intrinsics = Float[Tensor, "3 3"]  # pinhole K in pixels of a stated image size
type IntrinsicsBatch = Float[Tensor, "B 6"]  # (fx, fy, cx, cy, width, height) per window
type Points2D = Float[Tensor, "... 2"]  # pixel coordinates
type Points3D = Float[Tensor, "... 3"]  # camera-frame metres
type UnitPlanePoints = Float[Tensor, "... 2"]  # (x, y) on the z = 1 plane, before the lens
type PixelMap = Float[Tensor, "H W 2"]  # per target pixel, the source (u, v) it samples, pixels
type ClipArray = UInt8[np.ndarray, "F H W 3"]  # decoded RGB frames at the numpy boundary

# --- per-video results, before the parquet boundary (numpy, F = source frames) ---
type ClipGlobalOrient = Float[np.ndarray, "F S 3 3"]
type ClipHandPose = Float[np.ndarray, "F S 15 3 3"]
type ClipBetas = Float[np.ndarray, "S 10"]
type ClipTranslation = Float[np.ndarray, "F S 3"]
type ClipJointsCamera = Float[np.ndarray, "F S J 3"]
type ClipJoints2D = Float[np.ndarray, "F S J 2"]  # source-video pixels
type ClipProbability = Float[np.ndarray, "F S"]
type ClipIntrinsics = Float[np.ndarray, "3 3"]

# --- visualisation (numpy, one frame at a time at the drawing boundary) ---
type FrameJoints2D = Float[np.ndarray, "S J 2"]  # one frame's joints in image pixels
type FrameJointsCamera = Float[np.ndarray, "S J 3"]  # one frame's joints, camera metres
type FrameProbability = Float[np.ndarray, "S"]  # one frame's presence or visible per hand
type FrameVertices2D = Float[np.ndarray, "S V 2"]  # projected MANO vertices, pixels
type FrameVerticesCamera = Float[np.ndarray, "S V 3"]  # MANO vertices, camera metres
type ClipVertices2D = Float[np.ndarray, "F S V 2"]
type ClipVerticesCamera = Float[np.ndarray, "F S V 3"]
type ClipFaces = Int[np.ndarray, "S M 3"]  # MANO triangles of both hands
type WristTrail = Float[np.ndarray, "T 3"]  # recent wrist positions of one hand, oldest first
type Points = Float[np.ndarray, "N 3"]  # a batch of camera-frame points to project
type Pixels = Float[np.ndarray, "N 2"]  # their projections

# --- post-processing (numpy, after the per-video boundary) ---
type ClipHandMask = Bool[np.ndarray, "F S"]  # hands above the presence threshold, per frame
type FrameMask = Bool[np.ndarray, "F"]  # one hand's mask over the frames of a clip
type WristTrack = Float[np.ndarray, "F 2"]  # one hand's wrist in source pixels over a clip
type HandTranslations = Float[np.ndarray, "F 3"]  # one hand's wrist in camera metres over a clip
type HandRotations = Float[np.ndarray, "F 3 3"]  # one hand's root rotation over a clip
type FrameIndices = Int[np.ndarray, "N"]  # frame numbers into a clip
type Ratios = Float[np.ndarray, "N"]  # motions relative to their limit, 1 at the limit
type RunDepths = Float[np.ndarray, "N"]  # wrist depths along the camera ray over a run of frames
type Quaternions = Float[np.ndarray, "... 4"]  # unit quaternions (w, x, y, z)
type RotationArray = Float[np.ndarray, "... 3 3"]  # rotation matrices at the numpy boundary
type GapFractions = Float[np.ndarray, "N"]  # interpolation positions in (0, 1) across a gap
