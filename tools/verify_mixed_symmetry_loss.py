"""Verify mixed dataset symmetry-aware rot6d loss wiring.

The core check constructs a prediction that is deliberately different from the
GT rotation by one object symmetry:

    R_pred = R_gt @ S

For a symmetric object, normal rot6d loss should be non-zero, while
symmetry-aware rot6d loss should be close to zero.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"
for import_root in (REPO_ROOT, TRAINING_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from training.loss import (  # noqa: E402
    _load_symmetry_candidates_cpu,
    _rotation_matrix_to_rot6d,
    compute_object_srt_loss,
)


DEFAULT_SYMMETRY_INFO = REPO_ROOT / "mixed_symmetry_info.json"


def find_non_identity_symmetry(symmetry_info_path: Path, continuous_steps: int):
    with symmetry_info_path.open("r", encoding="utf-8") as f:
        all_info = json.load(f)

    for key in sorted(all_info):
        info = all_info[key]
        if not info.get("symmetries_discrete") and not info.get("symmetries_continuous"):
            continue
        with tempfile.TemporaryDirectory(prefix="sym_loss_verify_") as tmpdir:
            subset_path = Path(tmpdir) / "one_key_symmetry.json"
            subset_path.write_text(json.dumps({key: info}), encoding="utf-8")
            candidates_by_key = _load_symmetry_candidates_cpu(str(subset_path), continuous_steps)
            identity = torch.eye(3, dtype=torch.float32)
            candidates = candidates_by_key[key]
            for idx, symmetry in enumerate(candidates):
                if not torch.allclose(symmetry, identity, atol=1e-6):
                    return key, idx, symmetry, subset_path.read_text(encoding="utf-8")
    raise RuntimeError(f"No non-identity symmetry candidate found in {symmetry_info_path}")


def loss_values(
    pred_rotation: torch.Tensor,
    gt_rotation: torch.Tensor,
    symmetry_key: str,
    symmetry_info_path: Path,
    args,
):
    zero_t = torch.zeros(1, 3, dtype=torch.float32)
    predictions = {
        "object_pose": _rotation_matrix_to_rot6d(pred_rotation),
        "object_translation": zero_t.clone(),
    }
    batch = {
        "object_rotation": gt_rotation.clone(),
        "object_translation": zero_t.clone(),
        "has_object": torch.ones(1, dtype=torch.bool),
        "symmetry_object_id": [symmetry_key],
    }
    common = {
        "loss_type": args.loss_type,
        "weight_pose": 1.0,
        "weight_translation": 1.0,
    }
    plain = compute_object_srt_loss(
        predictions,
        batch,
        pose_rep="rot6d",
        use_symmetric_rot6d=False,
        **common,
    )
    symmetric = compute_object_srt_loss(
        predictions,
        batch,
        pose_rep="rot6d",
        use_symmetric_rot6d=True,
        symmetry_info_path=str(symmetry_info_path),
        symmetry_continuous_steps=args.continuous_steps,
        **common,
    )
    return (
        {key: float(value.detach().cpu()) for key, value in plain.items()},
        {key: float(value.detach().cpu()) for key, value in symmetric.items()},
    )


def run_core_check(args) -> None:
    symmetry_key, candidate_idx, symmetry, subset_json = find_non_identity_symmetry(
        args.symmetry_info,
        args.continuous_steps,
    )
    gt_rotation = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3)
    pred_rotation = symmetry.reshape(1, 3, 3)

    with tempfile.TemporaryDirectory(prefix="sym_loss_verify_") as tmpdir:
        subset_path = Path(tmpdir) / "one_key_symmetry.json"
        subset_path.write_text(subset_json, encoding="utf-8")
        plain, symmetric = loss_values(pred_rotation, gt_rotation, symmetry_key, subset_path, args)

    print("Core symmetry loss check")
    print(f"  symmetry_info: {args.symmetry_info}")
    print(f"  symmetry_key: {symmetry_key}")
    print(f"  candidate_idx: {candidate_idx}")
    print(f"  plain_rot6d_pose_loss: {plain['loss_object_pose']:.12g}")
    print(f"  symmetric_rot6d_pose_loss: {symmetric['loss_object_pose']:.12g}")

    if symmetric["loss_object_pose"] > args.tolerance:
        raise SystemExit(
            "ERROR: symmetric rot6d loss should be near zero for an equivalent symmetry pose."
        )
    if plain["loss_object_pose"] <= args.tolerance:
        raise SystemExit(
            "ERROR: plain rot6d loss unexpectedly near zero; picked symmetry may not change rot6d."
        )
    print("  OK: symmetric loss accepts the equivalent rotation while plain rot6d penalizes it.")


def make_common_conf(img_size: int):
    return SimpleNamespace(
        debug=False,
        training=True,
        inside_random=False,
        img_size=img_size,
        patch_size=14,
        map_xyz_bfloat16=False,
        rescale=True,
        rescale_aug=True,
        landscape_check=False,
        augs=SimpleNamespace(scales=[0.8, 1.2]),
    )


def run_dataset_key_check(args) -> None:
    from data.datasets.mixed_real_pose_normalize import (  # noqa: WPS433
        HouseCat6DMultiPoseNormalizeDataset,
        Real275MultiPoseNormalizeDataset,
        YCBVMultiPoseNormalizeDataset,
    )
    from data.datasets.ov9d_single_pose_normalize import OV9DSinglePoseNormalizeDataset  # noqa: WPS433

    with args.symmetry_info.open("r", encoding="utf-8") as f:
        symmetry_info = json.load(f)
    root = args.freepose_root
    common = make_common_conf(args.img_size)
    common_kwargs = dict(
        num_scene_views=4,
        num_object_views=4,
        fixed_object_view_ids=[1, 5, 10, 15],
        min_view_gap=5,
        load_point_map=False,
        scale_by_points=True,
        negative_object_prob=0.0,
        verify_files=args.verify_files,
    )
    factories = [
        (
            "oo9d",
            lambda: OV9DSinglePoseNormalizeDataset(
                common,
                split="train",
                DATA_ROOT=str(root / "ov9d/ov9d"),
                OBJECT_IMAGE_ROOT=str(root / "ov9d/ov9d_around_image"),
                SPLIT_JSON=str(root / "baseline_0503/splits_ov9d_unseen_category_generalization/single/train.json"),
                strict_fixed_object_view_ids=True,
                **common_kwargs,
            ),
        ),
        (
            "real275",
            lambda: Real275MultiPoseNormalizeDataset(
                common,
                split="train",
                DATA_ROOT=str(root / "real275"),
                SPLIT_ROOT=str(root / "real275/real_train"),
                GT_ROOT=str(root / "real275/gts/real_train_umeyama"),
                OBJECT_IMAGE_ROOT=str(root / "real275/real275_aligned_object_refs"),
                ALIGN_JSON=str(root / "baseline_0503/dataset_align.json"),
                strict_fixed_object_view_ids=True,
                max_records=args.max_records,
                **common_kwargs,
            ),
        ),
        (
            "ycbv",
            lambda: YCBVMultiPoseNormalizeDataset(
                common,
                split="train_real",
                DATA_ROOT=str(root / "datasets_real/ycbv"),
                SPLIT_ROOT=str(root / "datasets_real/ycbv/train_real"),
                OBJECT_IMAGE_ROOT=str(root / "datasets_real/ycbv/ycbv_aligned_object_refs"),
                ALIGN_JSON=str(root / "baseline_0503/dataset_align.json"),
                strict_fixed_object_view_ids=True,
                max_records=args.max_records,
                **common_kwargs,
            ),
        ),
        (
            "housecat6d",
            lambda: HouseCat6DMultiPoseNormalizeDataset(
                common,
                split="train",
                DATA_ROOT=str(root / "housecat6d"),
                OBJECT_IMAGE_ROOT=str(root / "housecat6d/housecat6d_aligned_object_refs"),
                ALIGN_JSON=str(root / "baseline_0503/dataset_align.json"),
                strict_fixed_object_view_ids=True,
                max_records=args.max_records,
                **common_kwargs,
            ),
        ),
    ]

    print("\nDataset sample symmetry key check")
    random.seed(args.seed)
    for name, make_dataset in factories:
        dataset = make_dataset()
        indices = random.sample(
            range(dataset.sequence_list_len),
            min(args.samples_per_dataset, dataset.sequence_list_len),
        )
        for index in indices:
            batch = dataset.get_data(seq_index=index, img_per_seq=4, aspect_ratio=1.0)
            symmetry_key = str(batch.get("symmetry_object_id", ""))
            status = "FOUND" if symmetry_key in symmetry_info else "MISSING(identity fallback)"
            print(
                f"  {name}: index={index} object={batch.get('object_name', '')} "
                f"symmetry_object_id={symmetry_key} {status}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify mixed symmetry-aware rot6d loss.")
    parser.add_argument("--symmetry-info", type=Path, default=DEFAULT_SYMMETRY_INFO)
    parser.add_argument("--continuous-steps", type=int, default=72)
    parser.add_argument("--loss-type", default="l1", choices=["l1", "l2"])
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--check-dataset-keys", action="store_true")
    parser.add_argument("--freepose-root", type=Path, default=Path("/mnt/train-data-4-hdd/yian/freepose"))
    parser.add_argument("--samples-per-dataset", type=int, default=3)
    parser.add_argument("--max-records", type=int, default=50)
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-files", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    run_core_check(args)
    if args.check_dataset_keys:
        run_dataset_key_check(args)


if __name__ == "__main__":
    main()
