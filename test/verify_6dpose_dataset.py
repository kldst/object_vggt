#!/usr/bin/env python3
import argparse
import importlib
import os.path as osp
import random
import sys
from types import SimpleNamespace

import numpy as np


def build_common_conf():
    return SimpleNamespace(
        img_size=518,
        patch_size=14,
        augs=SimpleNamespace(scales=[0.8, 1.2]),
        rescale=True,
        rescale_aug=True,
        landscape_check=False,
        debug=False,
        training=False,
        inside_random=False,
    )


def resolve_image_path(image_obj_dir: str, cam_idx: int) -> str:
    primary = osp.join(image_obj_dir, f"Main_Camera_({cam_idx}).jpg")
    if osp.isfile(primary):
        return primary
    if int(cam_idx) == 0:
        fallback = osp.join(image_obj_dir, "Main_Camera.jpg")
        if osp.isfile(fallback):
            return fallback
    return primary


def verify_one_sample(dataset, sample_index: int, verbose: bool = False, print_compare: bool = False):
    rec = dataset.records[sample_index]
    data = dataset.get_data(seq_index=sample_index, img_per_seq=None, aspect_ratio=1.0)

    run_name = rec["run_name"]
    input_name = rec["input_name"]
    object_name = rec["object_name"]
    camera_indices = rec["camera_indices"]

    image_obj_dir = osp.join(dataset.image_root, run_name, input_name, object_name)
    pc_obj_dir = osp.join(dataset.pc_root, run_name, input_name, object_name)

    # 1) Camera order and ids
    data_ids = list(data["ids"].tolist())
    data_cam_indices = list(data["camera_indices"].tolist())
    if print_compare:
        print(f"[sample {sample_index}] camera_order(gt)   : {camera_indices}")
        print(f"[sample {sample_index}] camera_order(data) : {data_ids}")
        print(f"[sample {sample_index}] camera_indices(data): {data_cam_indices}")

    if data_ids != camera_indices:
        raise AssertionError(
            f"[sample {sample_index}] ids mismatch: data={data['ids'].tolist()} gt={camera_indices}"
        )

    if data_cam_indices != camera_indices:
        raise AssertionError(
            f"[sample {sample_index}] camera_indices mismatch: data={data['camera_indices'].tolist()} gt={camera_indices}"
        )

    # 2) Per-view image/point map/value checks (in exact order)
    from data.dataset_util import read_image_cv2

    for view_i, cam_idx in enumerate(camera_indices):
        image_path = resolve_image_path(image_obj_dir, cam_idx)
        pc_path = osp.join(pc_obj_dir, f"cam{cam_idx}_decoded_map.npz")

        raw_img = read_image_cv2(image_path)
        raw_map = np.load(pc_path)["map_xyz"].astype(np.float32)

        img_equal = np.array_equal(data["images"][view_i], raw_img)
        map_equal = np.array_equal(data["world_points"][view_i], raw_map)
        cam_map_equal = np.array_equal(data["cam_points"][view_i], raw_map)
        if print_compare:
            print(
                f"[sample {sample_index}] view={view_i} cam={cam_idx} "
                f"image_equal={img_equal} world_points_equal={map_equal} cam_points_equal={cam_map_equal}"
            )

        if not img_equal:
            raise AssertionError(
                f"[sample {sample_index}] image mismatch at view#{view_i} cam{cam_idx}: {image_path}"
            )

        if not map_equal:
            raise AssertionError(
                f"[sample {sample_index}] world_points mismatch at view#{view_i} cam{cam_idx}: {pc_path}"
            )

        if not cam_map_equal:
            raise AssertionError(
                f"[sample {sample_index}] cam_points mismatch at view#{view_i} cam{cam_idx}: {pc_path}"
            )

        expected_mask = np.isfinite(raw_map).all(axis=-1) & (np.linalg.norm(raw_map, axis=-1) > 0)
        expected_depth = raw_map[..., 2].copy()
        expected_depth[~expected_mask] = 0.0

        mask_equal = np.array_equal(data["point_masks"][view_i], expected_mask)
        depth_equal = np.array_equal(data["depths"][view_i], expected_depth)
        if print_compare:
            print(
                f"[sample {sample_index}] view={view_i} cam={cam_idx} "
                f"mask_equal={mask_equal} depth_equal={depth_equal}"
            )

        if not mask_equal:
            raise AssertionError(
                f"[sample {sample_index}] point_masks mismatch at view#{view_i} cam{cam_idx}"
            )

        if not depth_equal:
            raise AssertionError(
                f"[sample {sample_index}] depths mismatch at view#{view_i} cam{cam_idx}"
            )

    # 3) S/R/T checks
    srt_gt = rec["srt"]
    s_gt = np.array([np.float32(srt_gt["s"])], dtype=np.float32)
    t_gt = np.array(srt_gt["t"], dtype=np.float32)
    r_gt = np.array(srt_gt["R_flat9_row_major"], dtype=np.float32).reshape(3, 3)
    srt_vec_gt = np.concatenate([s_gt, r_gt.reshape(-1), t_gt], axis=0).astype(np.float32)

    if print_compare:
        print(f"[sample {sample_index}] s(gt)    : {s_gt}")
        print(f"[sample {sample_index}] s(data)  : {data['object_scale']}")
        print(f"[sample {sample_index}] t(gt)    : {t_gt}")
        print(f"[sample {sample_index}] t(data)  : {data['object_translation']}")
        print(f"[sample {sample_index}] R(gt)    :\n{r_gt}")
        print(f"[sample {sample_index}] R(data)  :\n{data['object_rotation']}")

    if not np.allclose(data["object_scale"], s_gt, atol=0.0, rtol=0.0):
        raise AssertionError(f"[sample {sample_index}] object_scale mismatch")
    if not np.allclose(data["object_rotation"], r_gt, atol=0.0, rtol=0.0):
        raise AssertionError(f"[sample {sample_index}] object_rotation mismatch")
    if not np.allclose(data["object_translation"], t_gt, atol=0.0, rtol=0.0):
        raise AssertionError(f"[sample {sample_index}] object_translation mismatch")
    if not np.allclose(data["object_srt"], srt_vec_gt, atol=0.0, rtol=0.0):
        raise AssertionError(f"[sample {sample_index}] object_srt mismatch")

    # 4) skip-normalization control flag
    if data.get("skip_normalization", None) is not True:
        raise AssertionError(f"[sample {sample_index}] skip_normalization should be True")

    if verbose:
        print(
            f"[PASS] sample={sample_index} seq={data['seq_name']} "
            f"frames={data['frame_num']} order={camera_indices}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Verify SixDPoseDataset outputs against automatic_dataset/train source files."
    )
    parser.add_argument(
        "--repo-root",
        default="/mnt/train-data-4-hdd/yian/6dpose_obj/vggt",
        help="VGGT repo root (contains training/).",
    )
    parser.add_argument(
        "--automatic-dataset-dir",
        default="/mnt/train-data-4-hdd/yian/6dpose_obj/automatic_dataset",
        help="Path to automatic_dataset root.",
    )
    parser.add_argument("--split", default="train", choices=["train", "test", "val"])
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="If set, verify this fixed dataset index only.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="How many random samples to verify (used when --sample-index is not set).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verify-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to verify image/point-map files exist while indexing dataset records.",
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--print-compare",
        action="store_true",
        default=False,
        help="Print detailed GT-vs-dataset comparison for each checked sample/view.",
    )
    args = parser.parse_args()

    training_dir = osp.join(args.repo_root, "training")
    if training_dir not in sys.path:
        sys.path.insert(0, training_dir)

    # Import module with numeric file name via importlib.
    sixd_mod = importlib.import_module("data.datasets.6dpose")
    SixDPoseDataset = sixd_mod.SixDPoseDataset

    common_conf = build_common_conf()
    ds = SixDPoseDataset(
        common_conf=common_conf,
        split=args.split,
        AUTOMATIC_DATASET_DIR=args.automatic_dataset_dir,
        verify_files=args.verify_files,
    )

    total = ds.sequence_list_len
    print(f"Indexed samples: {total}")

    if args.sample_index is not None:
        indices = [args.sample_index]
    else:
        random.seed(args.seed)
        n = min(args.num_samples, total)
        indices = random.sample(range(total), n)

    for idx in indices:
        verify_one_sample(ds, idx, verbose=args.verbose, print_compare=args.print_compare)

    print(f"All checks passed for {len(indices)} sample(s).")


if __name__ == "__main__":
    main()
