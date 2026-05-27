# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

import json
from dataclasses import dataclass
from vggt.utils.pose_enc import extri_intri_to_pose_encoding
from train_utils.general import check_and_fix_inf_nan
from math import ceil, floor, pi


@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    """
    Multi-task loss module that combines different loss types for VGGT.
    
    Supports:
    - Camera loss
    - Depth loss 
    - Point loss
    - Tracking loss (not cleaned yet, dirty code is at the bottom of this file)
    """
    def __init__(
        self,
        camera=None,
        depth=None,
        point=None,
        track=None,
        object_point=None,
        object_mask=None,
        object_srt=None,
        object_presence=None,
        debug_force_model_output_to_ground_truth=False,
        **kwargs,
    ):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.camera = camera
        self.depth = depth
        self.point = point
        self.track = track
        self.object_point = object_point
        self.object_mask = object_mask
        self.object_srt = object_srt
        self.object_presence = object_presence
        self.debug_force_model_output_to_ground_truth = bool(debug_force_model_output_to_ground_truth)

    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Compute the total multi-task loss.
        
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks
            
        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}
        
        # Camera pose loss - if pose encodings are predicted
        if "pose_enc_list" in predictions:
            camera_loss_dict = compute_camera_loss(predictions, batch, **self.camera)   
            camera_loss = camera_loss_dict["loss_camera"] * self.camera["weight"]   
            total_loss = total_loss + camera_loss
            loss_dict.update(camera_loss_dict)
        
        # Depth estimation loss - if depth maps are predicted
        if "depth" in predictions:
            depth_loss_dict = compute_depth_loss(predictions, batch, **self.depth)
            depth_loss = depth_loss_dict["loss_conf_depth"] + depth_loss_dict["loss_reg_depth"] + depth_loss_dict["loss_grad_depth"]
            depth_loss = depth_loss * self.depth["weight"]
            total_loss = total_loss + depth_loss
            loss_dict.update(depth_loss_dict)

        # 3D point reconstruction loss - if world points are predicted
        if "world_points" in predictions:
            point_loss_dict = compute_point_loss(predictions, batch, **self.point)
            point_loss = point_loss_dict["loss_conf_point"] + point_loss_dict["loss_reg_point"] + point_loss_dict["loss_grad_point"]
            point_loss = point_loss * self.point["weight"]
            total_loss = total_loss + point_loss
            loss_dict.update(point_loss_dict)

        # Object-space point reconstruction loss
        if "object_points" in predictions and self.object_point is not None:
            object_point_loss_dict = compute_object_point_loss(
                predictions,
                batch,
                debug_force_model_output_to_ground_truth=self.debug_force_model_output_to_ground_truth,
                **self.object_point,
            )
            object_point_loss = (
                object_point_loss_dict["loss_conf_object_point"]
                + object_point_loss_dict["loss_reg_object_point"]
                + object_point_loss_dict["loss_grad_object_point"]
            )
            object_point_loss = object_point_loss * self.object_point["weight"]
            total_loss = total_loss + object_point_loss
            loss_dict.update(object_point_loss_dict)

        if "object_mask_logits" in predictions and self.object_mask is not None:
            object_mask_loss_dict = compute_object_mask_loss(
                predictions,
                batch,
                **self.object_mask,
            )
            total_loss = total_loss + object_mask_loss_dict["loss_object_mask"] * self.object_mask["weight"]
            loss_dict.update(object_mask_loss_dict)

        # Tracking loss - not cleaned yet, dirty code is at the bottom of this file
        if "track" in predictions:
            raise NotImplementedError("Track loss is not cleaned up yet")

        # 6D object SRT loss
        if "object_pose" in predictions and self.object_srt is not None:
            object_srt_loss_dict = compute_object_srt_loss(
                predictions,
                batch,
                debug_force_model_output_to_ground_truth=self.debug_force_model_output_to_ground_truth,
                **self.object_srt,
            )
            total_loss = total_loss + object_srt_loss_dict["loss_object_srt"] * self.object_srt["weight"]
            loss_dict.update(object_srt_loss_dict)

        if "object_presence_logits" in predictions and self.object_presence is not None:
            object_presence_loss_dict = compute_object_presence_loss(
                predictions,
                batch,
                **self.object_presence,
            )
            total_loss = (
                total_loss
                + object_presence_loss_dict["loss_object_presence"]
                * self.object_presence["weight"]
            )
            loss_dict.update(object_presence_loss_dict)
        
        loss_dict["objective"] = total_loss

        return loss_dict


def _rotation_matrix_to_rot6d(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix (..., 3, 3) to 6D rotation representation (..., 6)
    using the first two columns.
    """
    return rotation_matrix[..., :, :2].reshape(*rotation_matrix.shape[:-2], 6)


def _rot6d_to_rotation_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt: (..., 6) -> (..., 3, 3) with orthonormal columns.

    Mirrors the row-major rot6d convention used in `_rotation_matrix_to_rot6d`
    (first two columns of R flattened from shape (..., 3, 2)).
    """
    rot6d = rot6d.reshape(*rot6d.shape[:-1], 3, 2)
    a1 = rot6d[..., :, 0]
    a2 = rot6d[..., :, 1]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2 - proj, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    # Stack as columns to get shape (..., 3, 3).
    return torch.stack([b1, b2, b3], dim=-1)


def _vector_loss(pred: torch.Tensor, gt: torch.Tensor, loss_type: str = "l1") -> torch.Tensor:
    if loss_type == "l1":
        return (pred - gt).abs().mean()
    if loss_type == "l2":
        return ((pred - gt) ** 2).mean()
    raise ValueError(f"Unknown loss_type: {loss_type}")


def _vector_loss_per_candidate(
    pred: torch.Tensor,
    gt_candidates: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    """Return one scalar loss for each candidate target.

    pred: (6,), gt_candidates: (K, 6) -> (K,)
    """
    if loss_type == "l1":
        return (pred.unsqueeze(0) - gt_candidates).abs().mean(dim=-1)
    if loss_type == "l2":
        return ((pred.unsqueeze(0) - gt_candidates) ** 2).mean(dim=-1)
    raise ValueError(f"Unknown loss_type: {loss_type}")


def _axis_angle_to_matrix(axis, angle: float) -> torch.Tensor:
    axis = torch.tensor(axis, dtype=torch.float32)
    axis = axis / axis.norm().clamp_min(1e-8)
    x, y, z = axis.unbind()
    angle = torch.as_tensor(angle, dtype=torch.float32)
    c = torch.cos(angle)
    s = torch.sin(angle)
    one_c = 1.0 - c
    return torch.stack(
        [
            torch.stack([c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s]),
            torch.stack([y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s]),
            torch.stack([z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c]),
        ]
    )


def _load_symmetry_candidates_cpu(
    symmetry_info_path: str,
    continuous_steps: int,
) -> dict[str, torch.Tensor]:
    cache_key = (str(symmetry_info_path), int(continuous_steps))
    cache = getattr(_load_symmetry_candidates_cpu, "_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    identity = torch.eye(3, dtype=torch.float32)
    with open(symmetry_info_path, "r") as f:
        models_info = json.load(f)

    candidates_by_object_id = {}
    steps = max(1, int(continuous_steps))
    for object_id_str, info in models_info.items():
        base_syms = [identity]
        for symmetry in info.get("symmetries_discrete", []) or []:
            mat = torch.tensor(symmetry, dtype=torch.float32).reshape(4, 4)
            base_syms.append(mat[:3, :3])

        continuous_syms = [identity]
        for symmetry in info.get("symmetries_continuous", []) or []:
            axis = symmetry.get("axis", None)
            if axis is None:
                continue
            continuous_syms.extend(
                _axis_angle_to_matrix(axis, 2.0 * pi * step / steps)
                for step in range(1, steps)
            )

        candidates = []
        for base_sym in base_syms:
            for continuous_sym in continuous_syms:
                candidates.append(base_sym @ continuous_sym)
        candidates_by_object_id[str(object_id_str)] = torch.stack(candidates, dim=0)

    cache[cache_key] = candidates_by_object_id
    _load_symmetry_candidates_cpu._cache = cache
    return candidates_by_object_id


def _object_ids_to_list(object_ids) -> list[str]:
    if torch.is_tensor(object_ids):
        return [str(int(x)) for x in object_ids.detach().cpu().reshape(-1).tolist()]
    if isinstance(object_ids, (list, tuple)):
        values = []
        for item in object_ids:
            if torch.is_tensor(item):
                values.extend(str(int(x)) for x in item.detach().cpu().reshape(-1).tolist())
            else:
                values.append(str(item))
        return values
    return [str(object_ids)]


def _filter_object_ids(object_ids, valid_mask: torch.Tensor):
    if object_ids is None:
        return None
    if torch.is_tensor(object_ids):
        return object_ids[valid_mask.to(device=object_ids.device)]
    ids = _object_ids_to_list(object_ids)
    keep = valid_mask.detach().cpu().reshape(-1).tolist()
    return [object_id for object_id, is_valid in zip(ids, keep) if is_valid]


def _symmetric_rot6d_loss(
    pred_rot6d: torch.Tensor,
    gt_R: torch.Tensor,
    object_ids,
    loss_type: str,
    symmetry_info_path: str,
    symmetry_continuous_steps: int,
) -> torch.Tensor:
    if not symmetry_info_path or object_ids is None:
        gt_rot6d = _rotation_matrix_to_rot6d(gt_R)
        return _vector_loss(pred_rot6d, gt_rot6d, loss_type=loss_type)

    ids = _object_ids_to_list(object_ids)
    if len(ids) != pred_rot6d.shape[0]:
        raise ValueError(
            f"object_id count ({len(ids)}) does not match batch size ({pred_rot6d.shape[0]})."
        )

    candidates_cpu = _load_symmetry_candidates_cpu(symmetry_info_path, symmetry_continuous_steps)
    losses = []
    for sample_idx, object_id in enumerate(ids):
        symmetries = candidates_cpu.get(str(object_id))
        if symmetries is None:
            symmetries = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3)
        symmetries = symmetries.to(device=gt_R.device, dtype=gt_R.dtype)
        gt_equiv_R = gt_R[sample_idx].unsqueeze(0) @ symmetries
        gt_equiv_rot6d = _rotation_matrix_to_rot6d(gt_equiv_R).to(dtype=pred_rot6d.dtype)
        candidate_losses = _vector_loss_per_candidate(
            pred_rot6d[sample_idx],
            gt_equiv_rot6d,
            loss_type=loss_type,
        )
        losses.append(candidate_losses.min())
    return torch.stack(losses).mean()


def _rotation_loss(
    pred_rot6d: torch.Tensor,
    gt_R: torch.Tensor,
    pose_rep: str,
    loss_type: str,
    object_ids=None,
    symmetry_info_path=None,
    symmetry_continuous_steps=36,
) -> torch.Tensor:
    """Rotation loss with selectable representation.

    pose_rep:
      - "rot6d":     vector loss (L1/L2) directly on the 6D representation.
                     Cheap and stable, but not a true SO(3) distance — model
                     can game it by scaling output columns. This preserves the
                     legacy training behavior and does not call
                     `_rot6d_to_rotation_matrix`.
      - "symmetric_rot6d":
                     Same 6D vector loss style as "rot6d", but the GT target is
                     expanded with object symmetries and the minimum candidate
                     loss is used per sample.
      - "frobenius": ||R_pred - R_gt||_F^2 after Gram-Schmidt projection.
                     Equals 4(1 - cos θ); smooth, monotone in geodesic angle,
                     no arccos singularity. Recommended over geodesic.
      - "cosine":    (1 - cos θ).mean(); equivalent to frobenius up to a
                     constant, kept here for logging/A-B convenience.
      - "geodesic":  arccos((tr(R_pred R_gt^T) - 1) / 2). Has gradient
                     instability near θ = π; only use for fine-tuning after
                     the model is already close to GT.
    """
    if pose_rep == "rot6d":
        gt_rot6d = _rotation_matrix_to_rot6d(gt_R)
        return _vector_loss(pred_rot6d, gt_rot6d, loss_type=loss_type)
    if pose_rep == "symmetric_rot6d":
        return _symmetric_rot6d_loss(
            pred_rot6d,
            gt_R,
            object_ids=object_ids,
            loss_type=loss_type,
            symmetry_info_path=symmetry_info_path,
            symmetry_continuous_steps=symmetry_continuous_steps,
        )

    R_pred = _rot6d_to_rotation_matrix(pred_rot6d)
    if pose_rep == "frobenius":
        return ((R_pred - gt_R) ** 2).sum(dim=(-1, -2)).mean()

    # cos(theta) = (trace(R_pred R_gt^T) - 1) / 2
    trace = (R_pred * gt_R).sum(dim=(-1, -2))
    cos_theta = (trace - 1.0) * 0.5

    if pose_rep == "cosine":
        return (1.0 - cos_theta).mean()
    if pose_rep == "geodesic":
        cos_theta = cos_theta.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return torch.arccos(cos_theta).mean()

    raise ValueError(
        f"Unknown pose_rep '{pose_rep}'. Expected one of: "
        "rot6d, symmetric_rot6d, frobenius, cosine, geodesic."
    )


def compute_object_srt_loss(
    predictions,
    batch,
    loss_type="l1",
    pose_rep="rot6d",
    use_symmetric_rot6d=False,
    symmetry_info_path=None,
    symmetry_continuous_steps=36,
    weight_pose=1.0,
    weight_translation=1.0,
    init_w=1.0,
    debug_force_model_output_to_ground_truth=False,
    **kwargs,
):
    """
    Compute 6D object SRT loss.

    Required prediction keys:
      - object_pose: (B, 6)
      - object_translation: (B, 3)
    Required batch keys:
      - object_rotation: (B, 3, 3)
      - object_translation: (B, 3)
    Optional batch key:
      - has_object: (B,) bool mask

    pose_rep: rotation loss representation. See `_rotation_loss`.
              "rot6d" trains the predicted 6D vector against the first two
              columns of the GT rotation. Set use_symmetric_rot6d=True to use
              symmetry-aware rot6d candidates without changing pose_rep.
    loss_type: applies to translation always, and to rotation only when
               pose_rep == "rot6d".
    """
    pred_pose = predictions["object_pose"]
    pred_translation = predictions["object_translation"]
    # pred_pose_0 = predictions.get("pred_pose_0", None)

    gt_R = batch["object_rotation"]
    gt_translation = batch["object_translation"]
    object_ids = batch.get("symmetry_object_id", batch.get("object_id", None))
    if use_symmetric_rot6d:
        pose_rep = "symmetric_rot6d"

    if debug_force_model_output_to_ground_truth and not getattr(
        compute_object_srt_loss, "_dtype_logged_once", False
    ):
        print(
            "[DebugDType][object_srt] "
            f"pred_pose={pred_pose.dtype}, gt_R={gt_R.dtype}, "
            f"pred_translation={pred_translation.dtype}, gt_translation={gt_translation.dtype}, "
            f"pose_rep={pose_rep}",
            flush=True,
        )
        compute_object_srt_loss._dtype_logged_once = True

    has_object = batch.get("has_object", None)
    if has_object is not None:
        valid_mask = has_object.bool()
        if valid_mask.sum() == 0:
            dummy = (pred_pose * 0).mean() + (pred_translation * 0).mean()
            return {
                "loss_object_srt": dummy,
                "loss_object_pose": dummy,
                "loss_object_translation": dummy,
                # "loss_object_pose_init": dummy,
            }
        pred_pose = pred_pose[valid_mask]
        pred_translation = pred_translation[valid_mask]
        gt_R = gt_R[valid_mask]
        gt_translation = gt_translation[valid_mask]
        object_ids = _filter_object_ids(object_ids, valid_mask)
        # if pred_pose_0 is not None:
        #     pred_pose_0 = pred_pose_0[valid_mask]

    loss_pose = _rotation_loss(
        pred_pose,
        gt_R,
        pose_rep=pose_rep,
        loss_type=loss_type,
        object_ids=object_ids,
        symmetry_info_path=symmetry_info_path,
        symmetry_continuous_steps=symmetry_continuous_steps,
    )
    loss_translation = _vector_loss(pred_translation, gt_translation, loss_type=loss_type)

    # if pred_pose_0 is not None:
    #     loss_pose_init = _vector_loss(pred_pose_0, gt_pose, loss_type=loss_type)
    # else:
    #     loss_pose_init = (pred_pose * 0).mean()

    total = (
        weight_pose * loss_pose
        + weight_translation * loss_translation
        # + float(init_w) * loss_pose_init
    )

    return {
        "loss_object_srt": total,
        "loss_object_pose": loss_pose,
        "loss_object_translation": loss_translation,
        # "loss_object_pose_init": loss_pose_init,
    }


def compute_object_mask_loss(
    predictions,
    batch,
    bce_weight=1.0,
    dice_weight=1.0,
    pos_weight=1.0,
    eps=1e-6,
    **kwargs,
):
    pred_logits = predictions["object_mask_logits"]
    gt_mask = batch.get("object_masks", batch.get("point_masks"))
    if gt_mask is None:
        dummy = (pred_logits * 0).mean()
        return {
            "loss_object_mask": dummy,
            "loss_object_mask_bce": dummy,
            "loss_object_mask_dice": dummy,
        }

    gt_mask = gt_mask.to(dtype=pred_logits.dtype)
    if gt_mask.shape != pred_logits.shape:
        raise ValueError(
            f"object mask shape mismatch: pred={tuple(pred_logits.shape)} gt={tuple(gt_mask.shape)}"
        )

    has_object = batch.get("has_object", None)
    if has_object is not None:
        valid_mask = has_object.bool()
        if valid_mask.sum() == 0:
            dummy = (pred_logits * 0).mean()
            return {
                "loss_object_mask": dummy,
                "loss_object_mask_bce": dummy,
                "loss_object_mask_dice": dummy,
            }
        pred_logits = pred_logits[valid_mask]
        gt_mask = gt_mask[valid_mask]

    pos_weight_tensor = torch.as_tensor(float(pos_weight), device=pred_logits.device, dtype=pred_logits.dtype)
    loss_bce = F.binary_cross_entropy_with_logits(pred_logits, gt_mask, pos_weight=pos_weight_tensor)

    pred_prob = torch.sigmoid(pred_logits)
    reduce_dims = tuple(range(1, pred_prob.dim()))
    intersection = (pred_prob * gt_mask).sum(dim=reduce_dims)
    union = pred_prob.sum(dim=reduce_dims) + gt_mask.sum(dim=reduce_dims)
    loss_dice = 1.0 - ((2.0 * intersection + eps) / (union + eps))
    loss_dice = loss_dice.mean()

    total = float(bce_weight) * loss_bce + float(dice_weight) * loss_dice
    return {
        "loss_object_mask": total,
        "loss_object_mask_bce": loss_bce,
        "loss_object_mask_dice": loss_dice,
    }


def compute_object_presence_loss(
    predictions,
    batch,
    pos_weight=None,
    **kwargs,
):
    if "has_object" not in batch:
        raise KeyError("object_presence loss requires batch['has_object']")

    logits = predictions["object_presence_logits"].float().reshape(-1)
    targets = batch["has_object"].float().reshape(-1)
    bce_kwargs = {}
    if pos_weight is not None:
        bce_kwargs["pos_weight"] = torch.as_tensor(
            float(pos_weight),
            device=logits.device,
            dtype=logits.dtype,
        )
    loss_presence = F.binary_cross_entropy_with_logits(logits, targets, **bce_kwargs)
    pred_binary = (torch.sigmoid(logits) >= 0.5).to(dtype=targets.dtype)
    acc = (pred_binary == targets).float().mean()
    return {
        "loss_object_presence": loss_presence,
        "acc_object_presence": acc,
    }


def compute_camera_loss(
    pred_dict,              # predictions dict, contains pose encodings
    batch_data,             # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    gamma=0.6,              # temporal decay weight for multi-stage training
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,       # weight for translation loss
    weight_rot=1.0,         # weight for rotation loss
    weight_focal=0.5,       # weight for focal length loss
    **kwargs
):
    # List of predicted pose encodings per stage
    pred_pose_encodings = pred_dict['pose_enc_list']
    # Binary mask for valid points per frame (B, N, H, W)
    point_masks = batch_data['point_masks']
    # Only consider frames with enough valid points (>100)
    valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100
    # Number of prediction stages
    n_stages = len(pred_pose_encodings)

    # Get ground truth camera extrinsics and intrinsics
    gt_extrinsics = batch_data['extrinsics']
    gt_intrinsics = batch_data['intrinsics']
    image_hw = batch_data['images'].shape[-2:]

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    # Initialize loss accumulators for translation, rotation, focal length
    total_loss_T = total_loss_R = total_loss_FL = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]

        if valid_frame_mask.sum() == 0:
            # If no valid frames, set losses to zero to avoid gradient issues
            loss_T_stage = (pred_pose_stage * 0).mean()
            loss_R_stage = (pred_pose_stage * 0).mean()
            loss_FL_stage = (pred_pose_stage * 0).mean()
        else:
            # Only consider valid frames for loss computation
            loss_T_stage, loss_R_stage, loss_FL_stage = camera_loss_single(
                pred_pose_stage[valid_frame_mask].clone(),
                gt_pose_encoding[valid_frame_mask].clone(),
                loss_type=loss_type
            )
        # Accumulate weighted losses across stages
        total_loss_T += loss_T_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight
        total_loss_FL += loss_FL_stage * stage_weight

    # Average over all stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_FL = total_loss_FL / n_stages

    # Compute total weighted camera loss
    total_camera_loss = (
        avg_loss_T * weight_trans +
        avg_loss_R * weight_rot +
        avg_loss_FL * weight_focal
    )

    # Return loss dictionary with individual components
    return {
        "loss_camera": total_camera_loss,
        "loss_T": avg_loss_T,
        "loss_R": avg_loss_R,
        "loss_FL": avg_loss_FL
    }

def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.
    
    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)
    
    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL


def compute_point_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute point loss.
    
    Args:
        predictions: Dict containing 'world_points' and 'world_points_conf'
        batch: Dict containing ground truth 'world_points' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_points = predictions['world_points']
    pred_points_conf = predictions['world_points_conf']
    gt_points = batch['world_points']
    gt_points_mask = batch['point_masks']
    
    gt_points = check_and_fix_inf_nan(gt_points, "gt_points")
    
    if gt_points_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_points).mean()
        loss_dict = {f"loss_conf_point": dummy_loss,
                    f"loss_reg_point": dummy_loss,
                    f"loss_grad_point": dummy_loss,}
        return loss_dict
    
    # Compute confidence-weighted regression loss with optional gradient loss
    loss_conf, loss_grad, loss_reg = regression_loss(pred_points, gt_points, gt_points_mask, conf=pred_points_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)
    
    loss_dict = {
        f"loss_conf_point": loss_conf,
        f"loss_reg_point": loss_reg,
        f"loss_grad_point": loss_grad,
    }
    
    return loss_dict


def compute_depth_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute depth loss.
    
    Args:
        predictions: Dict containing 'depth' and 'depth_conf'
        batch: Dict containing ground truth 'depths' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_depth = predictions['depth']
    pred_depth_conf = predictions['depth_conf']

    gt_depth = batch['depths']
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    gt_depth = gt_depth[..., None]              # (B, H, W, 1)
    gt_depth_mask = batch['point_masks'].clone()   # 3D points derived from depth map, so we use the same mask

    if gt_depth_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_depth).mean()
        loss_dict = {f"loss_conf_depth": dummy_loss,
                    f"loss_reg_depth": dummy_loss,
                    f"loss_grad_depth": dummy_loss,}
        return loss_dict

    # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
    # this is hacky, but very easier to implement
    loss_conf, loss_grad, loss_reg = regression_loss(pred_depth, gt_depth, gt_depth_mask, conf=pred_depth_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)

    loss_dict = {
        f"loss_conf_depth": loss_conf,
        f"loss_reg_depth": loss_reg,    
        f"loss_grad_depth": loss_grad,
    }

    return loss_dict


def compute_object_point_loss(
    predictions,
    batch,
    gamma=1.0,
    alpha=0.2,
    gradient_loss_fn=None,
    valid_range=-1,
    debug_force_model_output_to_ground_truth=False,
    **kwargs,
):
    """
    Compute object-space point loss.

    Same logic as compute_point_loss, but uses object prediction keys:
      - predictions['object_points']
      - predictions['object_points_conf']
    """
    pred_points = predictions["object_points"]
    pred_points_conf = predictions["object_points_conf"]
    gt_points_raw = batch["world_points"]
    gt_points_mask = batch["point_masks"]

    gt_points = check_and_fix_inf_nan(gt_points_raw, "gt_object_points", 1000)

    if debug_force_model_output_to_ground_truth and not getattr(
        compute_object_point_loss, "_dtype_logged_once", False
    ):
        print(
            "[DebugDType][object_point] "
            f"pred_points={pred_points.dtype}, gt_points={gt_points.dtype}, "
            f"pred_points_conf={pred_points_conf.dtype}, mask={gt_points_mask.dtype}",
            flush=True,
        )
        compute_object_point_loss._dtype_logged_once = True

    # Debug diagnostics for GT-vs-pred mismatch after check_and_fix_inf_nan().

    if gt_points_mask.sum() < 100:
        dummy_loss = (0.0 * pred_points).mean()
        return {
            "loss_conf_object_point": dummy_loss,
            "loss_reg_object_point": dummy_loss,
            "loss_grad_object_point": dummy_loss,
        }

    # Keep the exact same regression path as compute_point_loss.
    loss_conf, loss_grad, loss_reg = regression_loss(
        pred_points,
        gt_points,
        gt_points_mask,
        conf=pred_points_conf,
        gradient_loss_fn=gradient_loss_fn,
        gamma=gamma,
        alpha=alpha,
        valid_range=valid_range,
    )

    return {
        "loss_conf_object_point": loss_conf,
        "loss_reg_object_point": loss_reg,
        "loss_grad_object_point": loss_grad,
    }


def regression_loss(pred, gt, mask, conf=None, gradient_loss_fn=None, gamma=1.0, alpha=0.2, valid_range=-1):
    """
    Core regression loss function with confidence weighting and optional gradient loss.
    
    Computes:
    1. gamma * ||pred - gt||^2 * conf - alpha * log(conf)
    2. Optional gradient loss
    
    Args:
        pred: (B, S, H, W, C) predicted values
        gt: (B, S, H, W, C) ground truth values
        mask: (B, S, H, W) valid pixel mask
        conf: (B, S, H, W) confidence weights (optional)
        gradient_loss_fn: Type of gradient loss ("normal", "grad", etc.)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        valid_range: Quantile range for outlier filtering
    
    Returns:
        loss_conf: Confidence-weighted loss
        loss_grad: Gradient loss (0 if not specified)
        loss_reg: Regular L2 loss
    """
    bb, ss, hh, ww, nc = pred.shape

    # Compute L2 distance between predicted and ground truth points
    loss_reg = torch.norm(gt[mask] - pred[mask], dim=-1)
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
    # This encourages the model to be confident on easy examples and less confident on hard ones
    loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(conf[mask])
    loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")
        
    # Initialize gradient loss
    loss_grad = 0

    # Prepare confidence for gradient loss if needed
    if "conf" in gradient_loss_fn:
        to_feed_conf = conf.reshape(bb*ss, hh, ww)
    else:
        to_feed_conf = None

    # Compute gradient loss if specified for spatial smoothness
    if "normal" in gradient_loss_fn:
        # Surface normal-based gradient loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=to_feed_conf,
        )
    elif "grad" in gradient_loss_fn:
        # Standard gradient-based loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=gradient_loss,
            conf=to_feed_conf,
        )

    # Process confidence-weighted loss
    if loss_conf.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_conf = filter_by_quantile(loss_conf, valid_range)

        loss_conf = check_and_fix_inf_nan(loss_conf, f"loss_conf_depth")
        loss_conf = loss_conf.mean()
    else:
        loss_conf = (0.0 * pred).mean()

    # Process regular regression loss
    if loss_reg.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_reg = filter_by_quantile(loss_reg, valid_range)

        loss_reg = check_and_fix_inf_nan(loss_reg, f"loss_reg_depth")
        loss_reg = loss_reg.mean()
    else:
        loss_reg = (0.0 * pred).mean()

    return loss_conf, loss_grad, loss_reg


def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values  
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None, gamma=1.0, alpha=0.2):
    """
    Surface normal-based loss for geometric consistency.
    
    Computes surface normals from 3D point maps using cross products of neighboring points,
    then measures the angle between predicted and ground truth normals.
    
    Args:
        prediction: (B, H, W, 3) predicted 3D coordinates/points
        target: (B, H, W, 3) ground-truth 3D coordinates/points
        mask: (B, H, W) valid pixel mask
        cos_eps: Epsilon for numerical stability in cosine computation
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Convert point maps to surface normals using cross products
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = point_map_to_normal(target,     mask, eps=cos_eps)

    # Only consider regions where both predicted and GT normals are valid
    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    # Extract valid normals
    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    dot = torch.sum(pred_normals * gt_normals, dim=-1)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            # Apply confidence weighting
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return loss.mean()
        else:
            return loss.mean()


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Sum gradients and normalize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Convert 3D point map to surface normal vectors using cross products.
    
    Computes normals by taking cross products of neighboring point differences.
    Uses 4 different cross-product directions for robustness.
    
    Args:
        point_map: (B, H, W, 3) 3D points laid out in a 2D grid
        mask: (B, H, W) valid pixels (bool)
        eps: Epsilon for numerical stability in normalization
    
    Returns:
        normals: (4, B, H, W, 3) normal vectors for each of the 4 cross-product directions
        valids: (4, B, H, W) corresponding valid masks
    """
    with torch.cuda.amp.autocast(enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

        # Get neighboring points for each pixel
        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2 , :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        # Compute direction vectors from center to neighbors
        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        # Compute four cross products for different normal directions
        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

        # Validity masks - require both direction pixels to be valid
        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        # Stack normals and validity masks
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize normal vectors
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.
    
    This helps remove outliers that could destabilize training.
    
    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss
    
    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= min_elements:
        # Too few elements, just return as-is
        return loss_tensor

    # Randomly sample if tensor is too large to avoid memory issues
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values to prevent extreme outliers
    loss_tensor = loss_tensor.clamp(max=hard_max)

    # Compute quantile threshold
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out


########################################################################################
########################################################################################

# Dirty code for tracking loss:

########################################################################################
########################################################################################

'''
def _compute_losses(self, coord_preds, vis_scores, conf_scores, batch):
    """Compute tracking losses using sequence_loss"""
    gt_tracks = batch["tracks"]  # B, S, N, 2
    gt_track_vis_mask = batch["track_vis_mask"]  # B, S, N

    # if self.training and hasattr(self, "train_query_points"):
    train_query_points = coord_preds[-1].shape[2]
    gt_tracks = gt_tracks[:, :, :train_query_points]
    gt_tracks = check_and_fix_inf_nan(gt_tracks, "gt_tracks", hard_max=None)

    gt_track_vis_mask = gt_track_vis_mask[:, :, :train_query_points]

    # Create validity mask that filters out tracks not visible in first frame
    valids = torch.ones_like(gt_track_vis_mask)
    mask = gt_track_vis_mask[:, 0, :] == True
    valids = valids * mask.unsqueeze(1)



    if not valids.any():
        print("No valid tracks found in first frame")
        print("seq_name: ", batch["seq_name"])
        print("ids: ", batch["ids"])
        print("time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        dummy_coord = coord_preds[0].mean() * 0          # keeps graph & grads
        dummy_vis = vis_scores.mean() * 0
        if conf_scores is not None:
            dummy_conf = conf_scores.mean() * 0
        else:
            dummy_conf = 0
        return dummy_coord, dummy_vis, dummy_conf                # three scalar zeros


    # Compute tracking loss using sequence_loss
    track_loss = sequence_loss(
        flow_preds=coord_preds,
        flow_gt=gt_tracks,
        vis=gt_track_vis_mask,
        valids=valids,
        **self.loss_kwargs
    )

    vis_loss = F.binary_cross_entropy_with_logits(vis_scores[valids], gt_track_vis_mask[valids].float())

    vis_loss = check_and_fix_inf_nan(vis_loss, "vis_loss", hard_max=None)


    # within 3 pixels
    if conf_scores is not None:
        gt_conf_mask = (gt_tracks - coord_preds[-1]).norm(dim=-1) < 3
        conf_loss = F.binary_cross_entropy_with_logits(conf_scores[valids], gt_conf_mask[valids].float())
        conf_loss = check_and_fix_inf_nan(conf_loss, "conf_loss", hard_max=None)
    else:
        conf_loss = 0

    return track_loss, vis_loss, conf_loss



def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    for a, b in zip(x.size(), mask.size()):
        assert a == b
    prod = x * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / denom.clamp(min=1)
    mean = torch.where(denom > 0,
                       mean,
                       torch.zeros_like(mean))
    return mean


def sequence_loss(flow_preds, flow_gt, vis, valids, gamma=0.8, vis_aware=False, huber=False, delta=10, vis_aware_w=0.1, **kwargs):
    """Loss function defined over sequence of flow predictions"""
    B, S, N, D = flow_gt.shape
    assert D == 2
    B, S1, N = vis.shape
    B, S2, N = valids.shape
    assert S == S1
    assert S == S2
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow_pred = flow_preds[i]

        i_loss = (flow_pred - flow_gt).abs()  # B, S, N, 2
        i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_{i}", hard_max=None)

        i_loss = torch.mean(i_loss, dim=3) # B, S, N

        # Combine valids and vis for per-frame valid masking.
        combined_mask = torch.logical_and(valids, vis)

        num_valid_points = combined_mask.sum()

        if vis_aware:
            combined_mask = combined_mask.float() * (1.0 + vis_aware_w)  # Add, don't add to the mask itself.
            flow_loss += i_weight * reduce_masked_mean(i_loss, combined_mask)
        else:
            if num_valid_points > 2:
                i_loss = i_loss[combined_mask]
                flow_loss += i_weight * i_loss.mean()
            else:
                i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_safe_check_{i}", hard_max=None)
                flow_loss += 0 * i_loss.mean()

    # Avoid division by zero if n_predictions is 0 (though it shouldn't be).
    if n_predictions > 0:
        flow_loss = flow_loss / n_predictions

    return flow_loss
'''
