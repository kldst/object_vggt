"""Verify that GT-as-prediction gives zero object rotation loss.

This script intentionally goes through the same helpers used by training/loss.py:

  gt_R -> _rotation_matrix_to_rot6d(gt_R) -> compute_object_srt_loss(...)

It is useful when changing pose_rep between "rot6d" and matrix-based losses such
as "frobenius".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "training"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from training.loss import _rotation_matrix_to_rot6d, compute_object_srt_loss


DEFAULT_SCENE_GT = (
    "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d/oo3d9dmulti/"
    "oo3d-5rb631pa708yxz9p8agclj0ty8i1wzin/scene_gt.json"
)
DEFAULT_SYMMETRY_INFO = (
    "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d/models_info_with_symmetry.json"
)


def load_first_gt_pose(scene_gt_path: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    path = Path(scene_gt_path)
    data = json.loads(path.read_text())
    frame_id = sorted(data.keys(), key=lambda x: int(x))[0]
    gt = data[frame_id][0]
    rotation = torch.tensor(gt["cam_R_m2c"], dtype=torch.float32).reshape(1, 3, 3)
    translation = torch.tensor(gt["cam_t_m2c"], dtype=torch.float32).reshape(1, 3)
    meta = {
        "scene_gt_path": str(path),
        "frame_id": frame_id,
        "obj_id": gt.get("obj_id"),
    }
    return rotation, translation, meta


def run_case(
    gt_R: torch.Tensor,
    gt_t: torch.Tensor,
    object_id: int,
    pose_rep: str,
    loss_type: str,
) -> dict[str, float]:
    pred_pose = _rotation_matrix_to_rot6d(gt_R)
    predictions = {
        "object_pose": pred_pose.clone(),
        "object_translation": gt_t.clone(),
    }
    batch = {
        "object_rotation": gt_R.clone(),
        "object_translation": gt_t.clone(),
        "has_object": torch.ones(gt_R.shape[0], dtype=torch.float32),
        "object_id": torch.tensor([object_id], dtype=torch.int64),
    }
    losses = compute_object_srt_loss(
        predictions,
        batch,
        pose_rep=pose_rep,
        loss_type=loss_type,
        symmetry_info_path=DEFAULT_SYMMETRY_INFO,
        symmetry_continuous_steps=36,
        weight_pose=1.0,
        weight_translation=1.0,
    )
    return {key: float(value.detach().cpu()) for key, value in losses.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-gt", default=DEFAULT_SCENE_GT)
    parser.add_argument("--loss-type", default="l1", choices=["l1", "l2"])
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    gt_R, gt_t, meta = load_first_gt_pose(args.scene_gt)

    print("Loaded one OV9D GT pose:")
    for key, value in meta.items():
        print(f"  {key}: {value}")
    print(f"  gt_R shape: {tuple(gt_R.shape)}")
    print(f"  gt_t shape: {tuple(gt_t.shape)}")

    failed = False
    for pose_rep in ("rot6d", "symmetric_rot6d", "frobenius"):
        losses = run_case(
            gt_R,
            gt_t,
            object_id=int(meta["obj_id"]),
            pose_rep=pose_rep,
            loss_type=args.loss_type,
        )
        print(f"\npose_rep={pose_rep}")
        for key, value in losses.items():
            print(f"  {key}: {value:.12g}")
        if abs(losses["loss_object_pose"]) > args.tolerance:
            failed = True
            print(
                f"  ERROR: loss_object_pose={losses['loss_object_pose']:.12g} "
                f"> tolerance={args.tolerance:.12g}"
            )

    if failed:
        raise SystemExit(1)
    print("\nOK: GT-as-prediction rotation losses are within tolerance.")


if __name__ == "__main__":
    main()
