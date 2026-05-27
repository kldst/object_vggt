import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

TRAINING_ROOT = Path(__file__).resolve().parents[2]
OBJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
for import_root in (PROJECT_ROOT, OBJECT_ROOT, TRAINING_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from data.base_dataset import BaseDataset
from data.datasets.ov9d_pose_normalize import OV9DPoseNormalizeDataset


class OV9DSinglePoseNormalizeDataset(OV9DPoseNormalizeDataset):
    """OV9D single-object dataset using the category-filtered single_split manifests.

    Each sample is one oo3d9dsingle scene.  Scene images and object reference
    images are both drawn from the same scene's ``rgb/`` folder; the object
    images have the background replaced with white using the ``mask/`` masks.

    The negative object sampling (``negative_object_prob``) always selects a
    scene from a *different* category so the model learns to detect absence.
    """

    DEFAULT_DATA_ROOT = "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d"
    DEFAULT_OBJECT_IMAGE_ROOT = "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d_around_image"
    DEFAULT_SPLIT_ROOT = "/mnt/train-data-4-hdd/yian/freepose/baseline/single_split"
    DEFAULT_NUM_SCENE_VIEWS = 4
    DEFAULT_NUM_OBJECT_VIEWS = 4

    def __init__(
        self,
        common_conf,
        split: str = "train",
        DATA_ROOT: str = DEFAULT_DATA_ROOT,
        OBJECT_IMAGE_ROOT: str = DEFAULT_OBJECT_IMAGE_ROOT,
        SPLIT_JSON: Optional[str] = None,
        len_train: Optional[int] = None,
        len_test: Optional[int] = None,
        verify_files: bool = True,
        num_scene_views: int = DEFAULT_NUM_SCENE_VIEWS,
        num_object_views: int = DEFAULT_NUM_OBJECT_VIEWS,
        min_view_gap: int = 5,
        object_view_min_gap: int = 6,
        object_view_max_gap: int = 9,
        fixed_object_view_ids: Optional[List[int]] = None,
        strict_fixed_object_view_ids: bool = True,
        load_point_map: bool = False,
        scale_by_points: bool = True,
        negative_object_prob: float = 0.0,
        max_records: Optional[int] = None,
        only_scene_name: str = "",
        only_scene_names: Optional[List[str]] = None,
    ):
        # Bypass OV9DPoseNormalizeDataset.__init__ and call BaseDataset directly
        # so we can set our own defaults before _build_records.
        BaseDataset.__init__(self, common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random

        self.split = str(split)
        self.data_root = Path(DATA_ROOT)
        self.single_root = self.data_root / "oo3d9dsingle"
        self.object_image_root = Path(OBJECT_IMAGE_ROOT)
        self.split_json = (
            Path(SPLIT_JSON)
            if SPLIT_JSON
            else Path(self.DEFAULT_SPLIT_ROOT) / f"{self.split}.json"
        )
        self.models_info = self.load_json(self.data_root / "models_info.json")

        self.verify_files = bool(verify_files)
        self.num_scene_views = int(num_scene_views)
        self.num_object_views = int(num_object_views)
        self.min_view_gap = int(min_view_gap)
        self.object_view_min_gap = int(object_view_min_gap)
        self.object_view_max_gap = int(object_view_max_gap)
        self.fixed_object_view_ids = (
            tuple(int(x) for x in fixed_object_view_ids)
            if fixed_object_view_ids is not None
            else None
        )
        if self.fixed_object_view_ids is not None and len(self.fixed_object_view_ids) != self.num_object_views:
            raise ValueError(
                "fixed_object_view_ids length must match num_object_views "
                f"({len(self.fixed_object_view_ids)} != {self.num_object_views})"
            )
        self.strict_fixed_object_view_ids = bool(strict_fixed_object_view_ids)
        self.object_index = 0  # single-object scenes always have one object
        self.load_point_map = bool(load_point_map)
        self.scale_by_points = bool(scale_by_points)
        self.negative_object_prob = float(negative_object_prob)
        self.max_records = int(max_records) if max_records is not None else None

        self.only_scene_names = [x.strip() for x in (only_scene_names or []) if str(x).strip()]
        if only_scene_name:
            self.only_scene_names = [only_scene_name.strip()]

        self.records, self.records_by_category = self._build_records_with_category_index()
        if self.max_records is not None:
            self.records = self.records[: self.max_records]
            limited_by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for rec in self.records:
                limited_by_category[rec["category"]].append(rec)
            self.records_by_category = dict(limited_by_category)
        self.sequence_list_len = len(self.records)
        self.records_by_frame_num = {
            frame_num: list(range(self.sequence_list_len)) for frame_num in range(4, 7)
        }
        if self.sequence_list_len == 0:
            raise RuntimeError(
                "No valid OV9D single-object normalized pose samples found. "
                f"split_json={self.split_json}, single_root={self.single_root}"
            )

        if self.split == "train":
            self.len_train = int(len_train) if len_train is not None else self.sequence_list_len
        elif self.split in {"test", "val", "test1", "test2"}:
            self.len_train = int(len_test) if len_test is not None else self.sequence_list_len
        else:
            raise ValueError(f"Invalid split: {split}")

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: OV9D single normalized pose sequence count: {self.sequence_list_len}")
        logging.info(f"{status}: OV9D single normalized pose dataset length: {len(self)}")

    # ------------------------------------------------------------------
    # Record building
    # ------------------------------------------------------------------

    def _build_records_with_category_index(
        self,
    ) -> tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        if not self.split_json.is_file():
            raise FileNotFoundError(f"Split JSON not found: {self.split_json}")
        payload = self.load_json(self.split_json)
        scenes = payload.get("scenes", [])
        if self.debug:
            scenes = scenes[:1]

        only = set(self.only_scene_names)
        records: List[Dict[str, Any]] = []
        by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for item in scenes:
            scene_name = str(item["scene_name"])
            if only and scene_name not in only:
                continue
            scene_dir = self.single_root / scene_name
            if self.verify_files and not self._verify_scene_files(scene_dir):
                continue
            rec = {
                "scene_name": scene_name,
                "scene_dir": scene_dir,
                "category": item.get("category", ""),
                "object_instance": item.get("object_instance", ""),
                "object_id": int(item["object_id"]),
            }
            records.append(rec)
            by_category[rec["category"]].append(rec)

        return records, dict(by_category)

    # Override the parent's _build_records so it won't be called accidentally.
    def _build_records(self) -> List[Dict[str, Any]]:
        records, self.records_by_category = self._build_records_with_category_index()
        return records

    # ------------------------------------------------------------------
    # Object image loading — use ``mask/`` instead of ``mask_visib/``
    # ------------------------------------------------------------------

    def _object_reference_dir(self, object_id: int) -> Path:
        return self.object_image_root / f"obj_{int(object_id):06d}"

    def _load_object_image(self, object_id: int, image_id: int) -> np.ndarray:
        return self.read_rgb(self._object_reference_dir(object_id) / "rgb" / f"{image_id:06d}.png")

    def _load_scene_object_mask(self, scene_dir: Path, image_id: int, image_shape: tuple[int, int]) -> np.ndarray:
        mask_path = scene_dir / "mask" / f"{image_id:06d}_{self.object_index:06d}.png"
        if not mask_path.is_file():
            return np.zeros(image_shape, dtype=np.float32)
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32)
        if mask.shape != image_shape:
            raise ValueError(
                f"Object mask shape {mask.shape} does not match image shape {image_shape} for {mask_path}"
            )
        return (mask > 0).astype(np.float32)

    def _process_scene_object_mask(
        self,
        mask: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        aspect_ratio: float,
        filepath: str,
    ) -> np.ndarray:
        dummy_image = np.repeat((mask[..., None] * 255.0).astype(np.uint8), 3, axis=2)
        original_size = np.array(mask.shape[:2], dtype=np.int32)
        target_image_shape = self.get_target_shape(aspect_ratio)
        _, processed_mask, *_ = self.process_one_image(
            image=dummy_image,
            depth_map=mask.astype(np.float32),
            extri_opencv=extrinsic,
            intri_opencv=intrinsic,
            original_size=original_size,
            target_image_shape=target_image_shape,
            filepath=filepath,
        )
        return (processed_mask > 0.5).astype(bool)

    # ------------------------------------------------------------------
    # Object view sampling with gap range (mirrors multi dataset)
    # ------------------------------------------------------------------

    @staticmethod
    def _ids_with_gap_range(
        ids: List[int],
        count: int,
        min_gap: int,
        max_gap: int,
        rng: random.Random,
    ) -> Optional[List[int]]:
        if len(ids) < count:
            return None
        if count <= 1:
            return [rng.choice(ids)]
        ids = sorted(ids)
        candidates = []

        def dfs(path: List[int], start_idx: int) -> None:
            if len(path) == count:
                candidates.append(path.copy())
                return
            prev = path[-1]
            for idx in range(start_idx, len(ids)):
                curr = ids[idx]
                gap = curr - prev
                if gap <= min_gap:
                    continue
                if gap >= max_gap:
                    break
                path.append(curr)
                dfs(path, idx + 1)
                path.pop()

        for start_idx, first in enumerate(ids):
            dfs([first], start_idx + 1)

        if not candidates:
            return None
        return list(rng.choice(candidates))

    def _sample_object_image_ids(
        self, available_ids: List[int], count: int, rng: random.Random
    ) -> List[int]:
        if self.fixed_object_view_ids is not None:
            available_set = {int(x) for x in available_ids}
            missing = [image_id for image_id in self.fixed_object_view_ids if image_id not in available_set]
            if missing and self.strict_fixed_object_view_ids:
                raise FileNotFoundError(
                    f"Fixed object view ids {missing} are not available. "
                    f"Available ids include: {sorted(available_set)[:20]}"
                )
            selected = [image_id for image_id in self.fixed_object_view_ids if image_id in available_set]
            if len(selected) < self.num_object_views:
                fallback = [image_id for image_id in sorted(available_set) if image_id not in selected]
                selected.extend(fallback[: self.num_object_views - len(selected)])
            return selected[: self.num_object_views]

        count = min(int(count), len(available_ids))
        ranged = self._ids_with_gap_range(
            available_ids,
            count,
            self.object_view_min_gap,
            self.object_view_max_gap,
            rng,
        )
        if ranged is not None:
            return ranged
        return self._sample_image_ids(available_ids, count, rng)

    # ------------------------------------------------------------------
    # Negative sampling — must be from a different category
    # ------------------------------------------------------------------

    def _sample_negative_record(
        self, current_scene_name: str, current_category: str, rng: random.Random
    ) -> Optional[Dict[str, Any]]:
        other_categories = [
            cat for cat in self.records_by_category if cat != current_category
        ]
        if not other_categories:
            return None
        cat = rng.choice(other_categories)
        return rng.choice(self.records_by_category[cat])

    # ------------------------------------------------------------------
    # Main data loading
    # ------------------------------------------------------------------

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        del seq_name, ids

        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        if seq_index is None:
            seq_index = 0

        rec = self.records[seq_index % self.sequence_list_len]
        scene_dir: Path = rec["scene_dir"]
        scene_name = rec["scene_name"]
        scene_gt = self.load_json(scene_dir / "scene_gt.json")
        scene_camera = self.load_json(scene_dir / "scene_camera.json")
        available_ids = self._available_image_ids(scene_gt)

        frame_num = int(img_per_seq) if img_per_seq is not None else self.num_scene_views
        frame_num = max(1, min(frame_num, len(available_ids)))
        scene_ids = self._sample_image_ids(available_ids, frame_num, random)

        # Decide whether to use a negative (different-category) object reference.
        use_negative_object = (
            self.negative_object_prob > 0.0
            and random.random() < self.negative_object_prob
        )
        object_rec = rec
        if use_negative_object:
            neg = self._sample_negative_record(scene_name, rec["category"], random)
            if neg is not None:
                object_rec = neg
            else:
                use_negative_object = False

        object_id = int(object_rec["object_id"])
        object_scene_dir = self._object_reference_dir(object_id)
        object_available_ids = sorted(
            int(path.stem)
            for path in (object_scene_dir / "rgb").glob("*.png")
            if path.stem.isdigit()
        )
        object_ids = self._sample_object_image_ids(object_available_ids, self.num_object_views, random)

        # --- scene views ---
        scene_images = []
        raw_depths = []
        raw_cam_points = []
        raw_world_points = []
        point_masks = []
        object_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []
        for image_id in scene_ids:
            image, depth, extrinsic, intrinsic, world_points, cam_points, point_mask = self._load_scene_view(
                scene_dir,
                image_id,
                scene_camera[str(image_id)],
            )
            object_mask = self._load_scene_object_mask(scene_dir, image_id, image.shape[:2])
            raw_extrinsic = extrinsic.copy()
            raw_intrinsic = intrinsic.copy()
            process_rng_state = np.random.get_state()
            image, depth, extrinsic, intrinsic, world_points, cam_points, point_mask, _ = self._process_scene_view(
                image=image,
                depth=depth,
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                aspect_ratio=aspect_ratio,
                filepath=str(scene_dir / "rgb" / f"{image_id:06d}.png"),
            )
            np.random.set_state(process_rng_state)
            object_mask = self._process_scene_object_mask(
                mask=object_mask,
                extrinsic=raw_extrinsic,
                intrinsic=raw_intrinsic,
                aspect_ratio=aspect_ratio,
                filepath=str(scene_dir / "mask" / f"{image_id:06d}_{self.object_index:06d}.png"),
            )
            scene_images.append(image)
            raw_depths.append(depth)
            raw_cam_points.append(cam_points)
            raw_world_points.append(world_points)
            point_masks.append(point_mask)
            object_masks.append(object_mask)
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
            original_sizes.append(np.array(image.shape[:2], dtype=np.int32))

        extrinsics_np = np.stack(extrinsics).astype(np.float32)
        normalized_extrinsics, avg_scale, normalized_world_points = self.normalize_extrinsics_and_world_points(
            extrinsics=extrinsics_np,
            world_points=raw_world_points,
            point_masks=point_masks,
            scale_by_points=self.scale_by_points,
        )

        # --- GT pose (always object_index=0 for single-object scenes) ---
        first_id = scene_ids[0]
        first_camera = scene_camera[str(first_id)]
        first_gt = scene_gt[str(first_id)][self.object_index]
        object_rotation_m2w, object_translation_m2w = self.model_to_world_from_bop(first_camera, first_gt)
        normalized_object_rotation, normalized_object_translation = self.normalize_object_pose_to_first_camera(
            first_extrinsic=extrinsics_np[0],
            object_rotation_m2w=object_rotation_m2w,
            object_translation_m2w=object_translation_m2w,
            avg_scale=avg_scale,
            scale_by_points=self.scale_by_points,
        )
        object_info = self.models_info.get(str(int(rec["object_id"])), {})
        object_size = None
        if all(k in object_info for k in ("size_x", "size_y", "size_z")):
            scale = avg_scale if self.scale_by_points else 1.0
            object_size = (
                np.array(
                    [object_info["size_x"], object_info["size_y"], object_info["size_z"]],
                    dtype=np.float32,
                )
                / float(scale)
            ).astype(np.float32)

        # --- object reference images (masked, white background) ---
        object_images = []
        object_original_sizes = []
        for image_id in object_ids:
            object_image = self._load_object_image(object_id, image_id)
            object_image = self._process_object_image(
                image=object_image,
                aspect_ratio=aspect_ratio,
                filepath=str(object_scene_dir / "rgb" / f"{image_id:06d}.png"),
            )
            object_images.append(object_image)
            object_original_sizes.append(np.array(object_image.shape[:2], dtype=np.int32))

        batch = {
            "seq_name": f"ov9d_single_pose_normalize_{self.split}/{scene_name}",
            "ids": np.array(scene_ids, dtype=np.int64),
            "frame_num": len(scene_ids),
            "images": scene_images,
            "object_images": object_images,
            "extrinsics": normalized_extrinsics,
            "intrinsics": intrinsics,
            "original_sizes": original_sizes,
            "object_original_sizes": object_original_sizes,
            "camera_indices": np.array(scene_ids, dtype=np.int64),
            "object_cam_indices": np.array(object_ids, dtype=np.int64),
            "object_name": rec.get("object_instance") or scene_name,
            "object_reference_name": object_rec.get("object_instance") or object_rec["scene_name"],
            "object_reference_scene_name": f"obj_{object_id:06d}",
            "object_id": np.array(int(rec["object_id"]), dtype=np.int64),
            "object_reference_id": np.array(object_id, dtype=np.int64),
            "symmetry_object_id": f"OO9DSingleCameraPose:{int(rec['object_id'])}",
            "category": rec.get("category", ""),
            "object_reference_category": object_rec.get("category", ""),
            "scene_name": scene_name,
            "run_name": scene_name,
            "skip_normalization": True,
            "has_object": np.array(0.0 if use_negative_object else 1.0, dtype=np.float32),
            "object_rotation": normalized_object_rotation,
            "object_translation": normalized_object_translation,
            "object_masks": object_masks,
            "object_srt": np.concatenate(
                [normalized_object_rotation.reshape(-1), normalized_object_translation], axis=0
            ).astype(np.float32),
            "normalization_scale": np.array([avg_scale], dtype=np.float32),
        }
        if object_size is not None:
            batch["object_size"] = object_size

        if self.load_point_map:
            scale = avg_scale if self.scale_by_points else 1.0
            batch.update(
                {
                    "depths": [(depth / scale).astype(np.float32) for depth in raw_depths],
                    "cam_points": [(points / scale).astype(np.float32) for points in raw_cam_points],
                    "world_points": [points.astype(np.float32) for points in normalized_world_points],
                    "point_masks": point_masks,
                }
            )

        return batch


def _make_main_common_conf(debug: bool):
    from types import SimpleNamespace

    return SimpleNamespace(
        debug=debug,
        training=True,
        inside_random=False,
        img_size=518,
        patch_size=14,
        augs=SimpleNamespace(scales=[0.8, 1.2]),
        rescale=True,
        rescale_aug=True,
        landscape_check=True,
    )


def _format_paths(paths) -> str:
    return "\n".join(f"      {p}" for p in paths)


def _print_sample_paths(dataset: OV9DSinglePoseNormalizeDataset, seq_index: int, img_per_seq: int) -> None:
    batch = dataset.get_data(seq_index=seq_index, img_per_seq=img_per_seq)
    rec = dataset.records[seq_index % dataset.sequence_list_len]
    scene_dir: Path = rec["scene_dir"]

    has_object = bool(float(batch["has_object"]) > 0.5)
    scene_rgb_paths = [scene_dir / "rgb" / f"{i:06d}.png" for i in batch["ids"].tolist()]
    scene_mask_paths = [scene_dir / "mask" / f"{i:06d}_000000.png" for i in batch["ids"].tolist()]
    scene_depth_paths = [scene_dir / "depth" / f"{i:06d}.png" for i in batch["ids"].tolist()]

    obj_scene_dir = dataset.object_image_root / batch["object_reference_scene_name"]
    object_rgb_paths = [obj_scene_dir / "rgb" / f"{i:06d}.png" for i in batch["object_cam_indices"].tolist()]

    print(f"[sample seq_index={seq_index}]")
    print(f"  seq_name: {batch['seq_name']}")
    print(f"  has_object: {str(has_object).lower()}")
    print(f"  category: {batch.get('category', '')}")
    print(f"  object_reference_category: {batch.get('object_reference_category', '')}")
    print(f"  scene_dir: {scene_dir}")
    print("  scene_rgb_paths:")
    print(_format_paths(scene_rgb_paths))
    print("  scene_mask_paths:")
    print(_format_paths(scene_mask_paths))
    print("  scene_depth_paths:")
    print(_format_paths(scene_depth_paths))
    print(f"  object_reference_scene_dir: {obj_scene_dir}")
    print("  object_rgb_paths:")
    print(_format_paths(object_rgb_paths))
    print(
        "  loaded_shapes: "
        f"images={len(batch['images'])}x{batch['images'][0].shape}, "
        f"object_images={len(batch['object_images'])}x{batch['object_images'][0].shape}, "
        f"extrinsics={batch['extrinsics'].shape}, object_srt={batch['object_srt'].shape}"
    )
    print()


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sample OV9D single-object dataset records and print file paths.")
    parser.add_argument("--data-root", default=OV9DSinglePoseNormalizeDataset.DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--img-per-seq", type=int, default=4)
    parser.add_argument("--num-object-views", type=int, default=4)
    parser.add_argument("--min-view-gap", type=int, default=5)
    parser.add_argument("--object-view-min-gap", type=int, default=6)
    parser.add_argument("--object-view-max-gap", type=int, default=9)
    parser.add_argument("--negative-object-prob", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-verify-files", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = OV9DSinglePoseNormalizeDataset(
        common_conf=_make_main_common_conf(debug=args.debug),
        split=args.split,
        DATA_ROOT=args.data_root,
        SPLIT_JSON=args.split_json,
        verify_files=not args.no_verify_files,
        num_scene_views=args.img_per_seq,
        num_object_views=args.num_object_views,
        min_view_gap=args.min_view_gap,
        object_view_min_gap=args.object_view_min_gap,
        object_view_max_gap=args.object_view_max_gap,
        load_point_map=False,
        scale_by_points=True,
        negative_object_prob=args.negative_object_prob,
    )

    sample_count = min(args.samples, dataset.sequence_list_len)
    sample_indices = random.sample(range(dataset.sequence_list_len), sample_count)
    print(f"dataset_records: {dataset.sequence_list_len}")
    print(f"categories: {len(dataset.records_by_category)}")
    print(f"negative_object_prob: {args.negative_object_prob}")
    print(f"object_view_gap_range: ({args.object_view_min_gap}, {args.object_view_max_gap}) exclusive")
    print(f"sample_indices: {sample_indices}")
    print()

    for seq_index in sample_indices:
        _print_sample_paths(dataset, seq_index=seq_index, img_per_seq=args.img_per_seq)


if __name__ == "__main__":
    _main()
