#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
TRAINING_ROOT = ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.datasets.mixed_real_pose_normalize import (
    HouseCat6DMultiPoseNormalizeDataset,
    Real275MultiPoseNormalizeDataset,
    YCBVMultiPoseNormalizeDataset,
)
from data.datasets.ov9d_single_pose_normalize import OV9DSinglePoseNormalizeDataset


def make_common_conf(img_size: int, training: bool = True):
    return SimpleNamespace(
        debug=False,
        training=training,
        inside_random=False,
        img_size=img_size,
        patch_size=14,
        map_xyz_bfloat16=False,
        rescale=True,
        rescale_aug=training,
        landscape_check=False,
        augs=SimpleNamespace(scales=[0.8, 1.2] if training else None),
    )


def print_sample(name, dataset, index: int, img_per_seq: int, output_dir: Path):
    batch = dataset.get_data(seq_index=index, img_per_seq=img_per_seq, aspect_ratio=1.0)
    overlay_path = draw_pose_bbox_overlay(name, batch, index, output_dir)
    print("=" * 100)
    print(f"[{name}] index={index} records={dataset.sequence_list_len}")
    print(f"seq_name: {batch['seq_name']}")
    print(f"scene_frame_ids: {batch['ids'].tolist()}")
    print(f"object_frame_ids: {batch['object_cam_indices'].tolist()}")
    print(f"object_name: {batch.get('object_name', '')}")
    print(f"symmetry_object_id: {batch.get('symmetry_object_id', '')}")
    print(f"category: {batch.get('category', '')}")
    print(f"has_object: {float(batch['has_object']) > 0.5}")
    print("scene_rgb_paths:")
    for path in batch.get("scene_rgb_paths", _infer_scene_paths(dataset, batch)):
        print(f"  {path}")
    print("scene_depth_paths:")
    for path in batch.get("scene_depth_paths", _infer_scene_depth_paths(dataset, batch)):
        print(f"  {path}")
    print("object_rgb_paths:")
    for path in batch.get("object_rgb_paths", _infer_object_paths(dataset, batch)):
        print(f"  {path}")
    print(
        "loaded_shapes: "
        f"images={len(batch['images'])}x{batch['images'][0].shape}, "
        f"object_images={len(batch['object_images'])}x{batch['object_images'][0].shape}, "
        f"extrinsics={batch['extrinsics'].shape}, object_srt={batch['object_srt'].shape}"
    )
    print(f"overlay_path: {overlay_path}")


def _infer_scene_paths(dataset, batch):
    scene_name = batch.get("scene_name")
    if hasattr(dataset, "multi_root") and scene_name:
        scene_dir = Path(dataset.multi_root) / str(scene_name)
    elif hasattr(dataset, "single_root") and scene_name:
        scene_dir = Path(dataset.single_root) / str(scene_name)
    else:
        rec = dataset.records[0]
        scene_dir = rec.get("scene_dir")
    if scene_dir is None:
        return []
    return [str(Path(scene_dir) / "rgb" / f"{int(i):06d}.png") for i in batch["ids"].tolist()]


def _infer_scene_depth_paths(dataset, batch):
    scene_name = batch.get("scene_name")
    if hasattr(dataset, "multi_root") and scene_name:
        scene_dir = Path(dataset.multi_root) / str(scene_name)
        return [str(scene_dir / "depth" / f"{int(i):06d}.png") for i in batch["ids"].tolist()]
    if hasattr(dataset, "single_root") and scene_name:
        scene_dir = Path(dataset.single_root) / str(scene_name)
        return [str(scene_dir / "depth" / f"{int(i):06d}.png") for i in batch["ids"].tolist()]
    return []


def _infer_object_paths(dataset, batch):
    if hasattr(dataset, "object_image_root"):
        object_id = int(batch.get("object_id", 0))
        object_name = batch.get("object_reference_name") or batch.get("object_name") or f"obj_{object_id:06d}"
        object_scene_name = batch.get("object_reference_scene_name")
        if object_scene_name:
            object_name = object_scene_name
        object_dir = Path(dataset.object_image_root) / str(object_name)
        if not object_dir.is_dir() and object_id > 0:
            object_dir = Path(dataset.object_image_root) / f"obj_{object_id:06d}"
        return [str(object_dir / "rgb" / f"{int(i):06d}.png") for i in batch["object_cam_indices"].tolist()]
    if hasattr(dataset, "single_root"):
        scene_name = batch.get("object_reference_scene_name") or batch.get("scene_name")
        object_dir = Path(dataset.single_root) / str(scene_name)
        return [str(object_dir / "rgb" / f"{int(i):06d}.png") for i in batch["object_cam_indices"].tolist()]
    return []


def project_points(points_cam: np.ndarray, intrinsic: np.ndarray):
    pixels = np.full((len(points_cam), 2), np.nan, dtype=np.float32)
    valid = points_cam[:, 2] > 1e-6
    if np.any(valid):
        projected = (intrinsic @ points_cam[valid].T).T
        pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels, valid


def draw_line_if_visible(draw, pixels, valid, i, j, color, width):
    if valid[i] and valid[j] and np.all(np.isfinite(pixels[[i, j]])):
        draw.line([tuple(pixels[i]), tuple(pixels[j])], fill=color, width=width)


def draw_pose_bbox_overlay(dataset_name: str, batch: dict, index: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = np.asarray(batch["images"][0], dtype=np.uint8)
    pil = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(pil)

    bbox_xyxy = None
    if "object_masks" in batch and len(batch["object_masks"]) > 0:
        mask = np.asarray(batch["object_masks"][0], dtype=bool)
        ys, xs = np.where(mask)
        if len(xs) > 0:
            bbox_xyxy = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            draw.rectangle(bbox_xyxy, outline=(255, 220, 0), width=3)

    rotation = np.asarray(batch["object_rotation"], dtype=np.float32).reshape(3, 3)
    translation = np.asarray(batch["object_translation"], dtype=np.float32).reshape(3)
    intrinsic = np.asarray(batch["intrinsics"][0], dtype=np.float32).reshape(3, 3)
    axis_len = max(float(np.linalg.norm(translation)) * 0.08, 0.05)
    axes_obj = np.asarray(
        [[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]],
        dtype=np.float32,
    )
    axes_cam = (rotation @ axes_obj.T).T + translation[None]
    axis_pixels, axis_valid = project_points(axes_cam, intrinsic)
    draw_line_if_visible(draw, axis_pixels, axis_valid, 0, 1, (255, 0, 0), 4)
    draw_line_if_visible(draw, axis_pixels, axis_valid, 0, 2, (0, 220, 0), 4)
    draw_line_if_visible(draw, axis_pixels, axis_valid, 0, 3, (0, 90, 255), 4)

    stem = f"{dataset_name}_{index:06d}_{batch.get('scene_name', 'scene')}_{batch.get('object_name', 'object')}"
    stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem)
    image_path = output_dir / f"{stem}_pose_bbox.png"
    meta_path = output_dir / f"{stem}_pose_bbox.json"
    pil.save(image_path)
    meta_path.write_text(
        json.dumps(
            {
                "seq_name": str(batch["seq_name"]),
                "scene_frame_id": int(np.asarray(batch["ids"])[0]),
                "object_name": str(batch.get("object_name", "")),
                "category": str(batch.get("category", "")),
                "bbox_xyxy": bbox_xyxy,
                "object_rotation": rotation.tolist(),
                "object_translation": translation.tolist(),
                "normalization_scale": np.asarray(batch.get("normalization_scale", [])).tolist(),
                "axis_pixels": axis_pixels.tolist(),
                "axis_valid": axis_valid.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return image_path


def main():
    parser = argparse.ArgumentParser(description="Print sampled scene/object RGB paths from the mixed multi-view datasets.")
    parser.add_argument("--freepose-root", default="/mnt/train-data-4-hdd/yian/freepose")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--img-per-seq", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=100, help="Cap scanned records for the real datasets during this path check.")
    parser.add_argument("--output-dir", default="baseline_0503/debug_mixed_dataset_overlays")
    parser.add_argument("--verify-files", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    root = Path(args.freepose_root)
    common = make_common_conf(args.img_size, training=True)
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

    dataset_factories = [
        (
            "oo9d",
            lambda: OV9DSinglePoseNormalizeDataset(
                common,
                split="train",
                DATA_ROOT=str(root / "ov9d/ov9d"),
                OBJECT_IMAGE_ROOT=str(root / "ov9d/ov9d_around_image"),
                SPLIT_JSON=str(root / "baseline_0503/splits_ov9d_unseen_category_generalization/single/train.json"),
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

    for name, make_dataset in dataset_factories:
        print(f"building {name}...", flush=True)
        dataset = make_dataset()
        sample_count = min(args.samples, dataset.sequence_list_len)
        for index in random.sample(range(dataset.sequence_list_len), sample_count):
            print_sample(name, dataset, index, args.img_per_seq, Path(args.output_dir))


if __name__ == "__main__":
    main()
