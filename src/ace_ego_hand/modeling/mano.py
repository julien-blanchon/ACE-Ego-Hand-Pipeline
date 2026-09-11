"""Decode MANO hand meshes and joints from the projector's rotation-matrix pose parameters.

A pure-torch port of `smplx.create(..., model_type="mano", use_pca=False, flat_hand_mean=False)`
as ACE-Ego-Hand calls it: the camera solve needs the canonical (camera-less) joints of every
frame and the visualiser needs the mesh, so this is the one place the MANO parameters are read.
Both hands are stored as one `HubModule` (`mano/` in the weights repository: the official MANO
tensors stacked on a leading slot axis) and run through a single batched linear blend skinning.

Conventions, all inherited from upstream so the numbers match:
- `flat_hand_mean=False`: the 15 finger rotations are converted to axis-angle, `hands_mean` is
  added, and the sum goes back through Rodrigues; the global orientation has no mean.
- Slot S = 2: slot 0 is decoded by the LEFT model, slot 1 by the RIGHT model.
- Joints: the 16 regressed MANO joints plus the five fingertip vertices, reordered to the
  21-joint OpenPose layout (wrist, thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20).
- Units are metres in MANO's own frame with the wrist near the origin; no translation is applied,
  the caller adds the camera-space wrist translation afterwards.

Adapted from https://github.com/ggxxii/ACE-Ego-Hand/blob/main/ace_ego_hand/mano_utils.py and
https://github.com/vchoutas/smplx/blob/main/smplx/lbs.py
Ref: MANO, Romero et al., SIGGRAPH Asia 2017.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, override

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import nn

from ..types import (
    AxisAngle,
    HandBetas,
    ManoFaces,
    ManoJointRegressor,
    ManoJoints,
    ManoNativeJoints,
    ManoParents,
    ManoPoseDirs,
    ManoShapeDirs,
    ManoSkinningWeights,
    ManoTemplate,
    ManoVertices,
    RigidTransforms,
    RotationMatrices,
)
from ..utils.hub import HubModule

NUM_HANDS = 2  # slot 0 left, slot 1 right
NUM_FACES = 1538
# MANO's kinematic tree: the parent of each of the 16 joints, -1 at the wrist (root). The chain
# loop needs python ints, so it reads the tree from here; the `parents` buffer holds the same.
MANO_PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14)
# Fingertip vertices in OpenPose finger order (thumb, index, middle, ring, pinky) and the map from
# [16 MANO joints, 5 tips] to the 21-joint OpenPose order.
FINGERTIP_VERTICES = (744, 320, 443, 554, 671)
MANO_TO_OPENPOSE = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20)
# Below this sine a rotation is treated as the identity (its axis is undefined)
SMALL_ANGLE_SINE = 1e-6


@dataclass(frozen=True, slots=True)
class ManoConfig:
    """Sizes of the MANO model; the defaults are the official release."""

    num_betas: int = 10
    num_joints: int = 16
    num_vertices: int = 778
    flat_hand_mean: bool = False  # False: `hands_mean` is added to the finger rotations


@dataclass(frozen=True, slots=True)
class ManoOutput:
    joints: ManoJoints  # (..., S, 21, 3) OpenPose order, metres, MANO frame
    vertices: ManoVertices  # (..., S, 778, 3)


def rotation_matrix_to_axis_angle(rotation: RotationMatrices) -> AxisAngle:
    """Invert Rodrigues for a batch of rotation matrices without NaNs at 0 or pi.

    The angle comes from atan2 of the skew part's norm and the trace, exact on [0, pi]. Up to a
    quarter turn the axis is the normalized skew part `2 sin(angle) a`; beyond it that part shrinks
    towards pi, so the axis is read from `(R + R^T) / 2 - cos(angle) I = (1 - cos(angle)) a a^T`
    instead, with the sign of the skew part.
    """

    skew = torch.stack(
        [
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ],
        dim=-1,
    )
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    cosine = 0.5 * (trace - 1)
    angle = torch.atan2(sine, cosine)

    # Lower half: axis * angle = skew * angle / (2 sin(angle)), and angle / sin(angle) -> 1 at 0
    small = sine <= SMALL_ANGLE_SINE
    scale = torch.where(small, torch.ones_like(sine), angle / torch.where(small, 1.0, sine))
    from_skew = 0.5 * skew * scale[..., None]

    # Upper half: dominant column of the outer product, oriented like the skew part
    identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
    outer = 0.5 * (rotation + rotation.mT) - cosine[..., None, None] * identity
    column = torch.argmax(torch.linalg.vector_norm(outer, dim=-2), dim=-1)
    axis = torch.gather(outer, -1, column[..., None, None].expand(*column.shape, 3, 1))[..., 0]
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp(min=1e-12)
    sign = torch.where(einsum(axis, skew, "... i, ... i -> ...") < 0, -1.0, 1.0)
    from_outer = axis * (sign * angle)[..., None]

    return torch.where((cosine < 0)[..., None], from_outer, from_skew)


def axis_angle_to_rotation_matrix(axis_angle: AxisAngle) -> RotationMatrices:
    """Rodrigues' formula, batched; the zero vector maps to the identity."""

    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / angle.clamp(min=1e-12)
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    cross = rearrange(
        torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1), "... (i j) -> ... i j", i=3
    )

    identity = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    sine, cosine = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
    return identity + sine * cross + (1 - cosine) * (cross @ cross)


class ManoLayer(nn.Module, HubModule):
    """Both MANO hands as fixed buffers, decoded together by one batched linear blend skinning.

    Every buffer carries a leading slot axis S = 2 (left, right) so the two hands broadcast
    against the projector's `(..., S, ...)` predictions; the kinematic tree is the same for both.
    The buffers are zero until `from_pretrained` fills them from `mano/model.safetensors`.
    """

    config_class: ClassVar[type] = ManoConfig

    template: ManoTemplate
    shapedirs: ManoShapeDirs
    posedirs: ManoPoseDirs
    joint_regressor: ManoJointRegressor
    skinning_weights: ManoSkinningWeights
    hands_mean: AxisAngle  # (S, 15, 3)
    faces: ManoFaces
    parents: ManoParents

    def __init__(
        self,
        config: ManoConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.config = config
        S, V, P, K = NUM_HANDS, config.num_vertices, config.num_joints, config.num_betas
        num_pose_features = 9 * (P - 1)  # the flattened (R - I) of every non-root joint

        def zeros(*shape: int) -> torch.Tensor:
            return torch.zeros(shape, device=device, dtype=dtype)

        self.register_buffer("template", zeros(S, V, 3))
        self.register_buffer("shapedirs", zeros(S, V, 3, K))
        self.register_buffer("posedirs", zeros(S, V, 3, num_pose_features))
        self.register_buffer("joint_regressor", zeros(S, P, V))
        self.register_buffer("skinning_weights", zeros(S, V, P))
        self.register_buffer("hands_mean", zeros(S, P - 1, 3))
        self.register_buffer(
            "faces", torch.zeros((S, NUM_FACES, 3), device=device, dtype=torch.int64)
        )
        self.register_buffer(
            "parents", torch.tensor(MANO_PARENTS, device=device, dtype=torch.int64)
        )

    @override
    def forward(
        self,
        global_orient: RotationMatrices,
        hand_pose: RotationMatrices,
        betas: HandBetas,
    ) -> ManoOutput:
        """Decode `(..., S, 3, 3)`, `(..., S, 15, 3, 3)`, `(..., S, 10)` into joints and vertices.

        All three inputs share their leading dims; the caller repeats clip-level betas over frames.
        """

        dtype = self.template.dtype
        global_orient, hand_pose, betas = (
            global_orient.to(dtype),
            hand_pose.to(dtype),
            betas.to(dtype),
        )

        # Shape blend shapes, then the rest-pose joints regressed from the shaped mesh
        shaped: ManoVertices = self.template + einsum(
            betas, self.shapedirs, "... s k, s v c k -> ... s v c"
        )
        rest_joints: ManoNativeJoints = einsum(
            self.joint_regressor, shaped, "s p v, ... s v c -> ... s p c"
        )

        # flat_hand_mean=False: the mean pose is added in axis-angle space, so the finger rotation
        # matrices must go through axis-angle and back; the global orientation has no mean
        finger_rotations = hand_pose
        if not self.config.flat_hand_mean:
            finger_axis_angle: AxisAngle = (
                rotation_matrix_to_axis_angle(hand_pose) + self.hands_mean
            )
            finger_rotations = axis_angle_to_rotation_matrix(finger_axis_angle)
        rotations = torch.cat([global_orient[..., None, :, :], finger_rotations], dim=-3)

        # Pose blend shapes on the flattened (R - I) of the 15 finger joints
        identity = torch.eye(3, device=shaped.device, dtype=dtype)
        pose_feature = rearrange(finger_rotations - identity, "... s p i j -> ... s (p i j)")
        posed: ManoVertices = shaped + einsum(
            pose_feature, self.posedirs, "... s q, s v c q -> ... s v c"
        )

        joints, transforms = self._rigid_transforms(rotations, rest_joints)

        # Linear blend skinning: each vertex is moved by its weighted mix of joint transforms
        vertex_transforms = einsum(
            self.skinning_weights, transforms, "s v p, ... s p i j -> ... s v i j"
        )
        vertices: ManoVertices = (
            einsum(vertex_transforms[..., :3, :3], posed, "... s v i j, ... s v j -> ... s v i")
            + vertex_transforms[..., :3, 3]
        )

        # 16 MANO joints + 5 fingertip vertices, reordered to OpenPose
        joints_21: ManoJoints = torch.cat(
            [joints, vertices[..., list(FINGERTIP_VERTICES), :]], dim=-2
        )[..., list(MANO_TO_OPENPOSE), :]
        return ManoOutput(joints=joints_21, vertices=vertices)

    def _rigid_transforms(
        self, rotations: RotationMatrices, rest_joints: ManoNativeJoints
    ) -> tuple[ManoNativeJoints, RigidTransforms]:
        """Chain the per-joint rotations down the kinematic tree (smplx `batch_rigid_transform`)."""

        # Local transform of each joint: its rotation about its own rest position, expressed as an
        # offset from the parent joint
        offsets = rest_joints.clone()  # the root's offset is its own rest position
        offsets[..., 1:, :] = rest_joints[..., 1:, :] - rest_joints[..., self.parents[1:], :]
        local = torch.cat(
            [F.pad(rotations, (0, 0, 0, 1)), F.pad(offsets[..., None], (0, 0, 0, 1), value=1.0)],
            dim=-1,
        )

        # Compose parent-to-child; the loop is over 16 joints, not over the batch
        chain = [local[..., 0, :, :]]
        for joint in range(1, len(MANO_PARENTS)):
            chain.append(chain[MANO_PARENTS[joint]] @ local[..., joint, :, :])
        world: RigidTransforms = torch.stack(chain, dim=-3)
        posed_joints: ManoNativeJoints = world[..., :3, 3]

        # Skinning transforms move a vertex from the rest pose, so subtract each joint's rest
        # position rotated into its posed frame
        rest_shift = einsum(world[..., :3, :3], rest_joints, "... p i j, ... p j -> ... p i")
        relative = world.clone()
        relative[..., :3, 3] = posed_joints - rest_shift
        return posed_joints, relative
