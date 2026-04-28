import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

TRAINING_ROOT = Path(__file__).resolve().parents[2]
OBJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
for import_root in (PROJECT_ROOT, OBJECT_ROOT, TRAINING_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from data.base_dataset import BaseDataset
from data.datasets.ov9d_pose_normalize import OV9DPoseNormalizeDataset


class OV9DMultiPoseNormalizeDataset(OV9DPoseNormalizeDataset):
    """OV9D multi-object scene dataset with one target object per sample.

    The split manifest is expected to contain ``scenes`` entries with
    ``scene_name`` and ``eligible_object_ids``. Each eligible object becomes an
    object-conditioned record while RGB/depth/camera data are read from
    ``oo3d9dmulti``.
    """

    DEFAULT_DATA_ROOT = "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d"
    DEFAULT_NUM_SCENE_VIEWS = 4
    DEFAULT_NUM_OBJECT_VIEWS = 4

    def __init__(
        self,
        common_conf,
        split: str = "train",
        DATA_ROOT: str = DEFAULT_DATA_ROOT,
        SPLIT_JSON: Optional[str] = None,
        NAME_TO_OID_JSON: Optional[str] = None,
        len_train: Optional[int] = None,
        len_test: Optional[int] = None,
        verify_files: bool = True,
        num_scene_views: int = DEFAULT_NUM_SCENE_VIEWS,
        num_object_views: int = DEFAULT_NUM_OBJECT_VIEWS,
        min_view_gap: int = 5,
        object_view_min_gap: int = 6,
        object_view_max_gap: int = 9,
        load_point_map: bool = False,
        scale_by_points: bool = True,
        negative_object_prob: float = 0.0,
        only_scene_name: str = "",
        only_scene_names: Optional[List[str]] = None,
        only_object_id: Optional[int] = None,
        only_object_ids: Optional[List[int]] = None,
    ):
        BaseDataset.__init__(self, common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random

        self.split = str(split)
        self.data_root = Path(DATA_ROOT)
        self.multi_root = self.data_root / "oo3d9dmulti"
        self.single_root = self.data_root / "oo3d9dsingle"
        self.split_json = Path(SPLIT_JSON) if SPLIT_JSON else self.data_root / "splits_multi" / f"{self.split}.json"
        self.name_to_oid_json = Path(NAME_TO_OID_JSON) if NAME_TO_OID_JSON else self.data_root / "name2oid.json"

        self.verify_files = bool(verify_files)
        self.num_scene_views = int(num_scene_views)
        self.num_object_views = int(num_object_views)
        self.min_view_gap = int(min_view_gap)
        self.object_view_min_gap = int(object_view_min_gap)
        self.object_view_max_gap = int(object_view_max_gap)
        self.load_point_map = bool(load_point_map)
        self.scale_by_points = bool(scale_by_points)
        self.negative_object_prob = float(negative_object_prob)

        self.only_scene_names = [x.strip() for x in (only_scene_names or []) if str(x).strip()]
        if only_scene_name:
            self.only_scene_names = [only_scene_name.strip()]

        object_ids = [int(x) for x in (only_object_ids or [])]
        if only_object_id is not None:
            object_ids = [int(only_object_id)]
        self.only_object_ids = set(object_ids)

        self.single_records_by_object_id = self._build_single_records_by_object_id()
        self.records = self._build_records()
        self.sequence_list_len = len(self.records)
        self.records_by_frame_num = {
            frame_num: list(range(self.sequence_list_len)) for frame_num in range(4, 7)
        }
        if self.sequence_list_len == 0:
            raise RuntimeError(
                "No valid OV9D multi-object normalized pose samples found. "
                f"Checked split_json={self.split_json}, multi_root={self.multi_root}, single_root={self.single_root}"
            )

        if self.split == "train":
            self.len_train = int(len_train) if len_train is not None else self.sequence_list_len
        elif self.split in {"test", "val", "test1", "test2", "test3"}:
            self.len_train = int(len_test) if len_test is not None else self.sequence_list_len
        else:
            raise ValueError(f"Invalid split: {split}")

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: OV9D multi normalized pose sequence count: {self.sequence_list_len}")
        logging.info(f"{status}: OV9D multi normalized pose dataset length: {len(self)}")

    @staticmethod
    def _category_for_object(item: Dict[str, Any], object_id: int) -> str:
        object_to_category = item.get("object_id_to_category_id", {})
        category_id = object_to_category.get(str(object_id), object_to_category.get(object_id))
        category_ids = item.get("category_ids", [])
        categories = item.get("categories", [])
        if category_id in category_ids:
            idx = category_ids.index(category_id)
            if idx < len(categories):
                return str(categories[idx])
        return str(category_id) if category_id is not None else ""

    @staticmethod
    def _parse_single_scene_name(scene_name: str) -> tuple[str, str]:
        parts = scene_name.split("_")
        if len(parts) < 3:
            raise ValueError(f"Unexpected single-scene folder name: {scene_name}")
        category = "_".join(parts[:-2])
        object_instance = "_".join(parts[:-1])
        return category, object_instance

    @staticmethod
    def _object_index_for_id(gts: List[Dict[str, Any]], object_id: int) -> Optional[int]:
        for idx, gt in enumerate(gts):
            if int(gt.get("obj_id", -1)) == int(object_id):
                return idx
        return None

    @classmethod
    def _image_ids_with_object(cls, scene_gt: Dict[str, Any], object_id: int) -> List[int]:
        ids = []
        for image_id_str, gts in scene_gt.items():
            if cls._object_index_for_id(gts, object_id) is not None:
                ids.append(int(image_id_str))
        return sorted(ids)

    def _build_records(self) -> List[Dict[str, Any]]:
        if not self.split_json.is_file():
            raise FileNotFoundError(f"Split JSON not found: {self.split_json}")
        payload = self.load_json(self.split_json)
        scenes = payload.get("scenes", [])
        if self.debug:
            scenes = scenes[:1]

        only_scenes = set(self.only_scene_names)
        records = []
        for item in scenes:
            scene_name = str(item["scene_name"])
            if only_scenes and scene_name not in only_scenes:
                continue

            scene_dir = self.multi_root / scene_name
            if self.verify_files and not self._verify_scene_files(scene_dir):
                continue

            eligible_object_ids = [int(x) for x in item.get("eligible_object_ids", [])]
            if self.only_object_ids:
                eligible_object_ids = [x for x in eligible_object_ids if x in self.only_object_ids]
            if not eligible_object_ids:
                continue

            scene_gt = self.load_json(scene_dir / "scene_gt.json")
            scene_object_ids = {int(x) for x in item.get("object_ids", [])}
            scene_categories = {str(x) for x in item.get("categories", [])}
            if not scene_object_ids:
                for gts in scene_gt.values():
                    for gt in gts:
                        scene_object_ids.add(int(gt["obj_id"]))
            for object_id in eligible_object_ids:
                if object_id not in self.single_records_by_object_id:
                    continue
                object_image_ids = self._image_ids_with_object(scene_gt, object_id)
                if self.verify_files and len(object_image_ids) == 0:
                    continue
                records.append(
                    {
                        "scene_name": scene_name,
                        "scene_dir": scene_dir,
                        "target_object_id": object_id,
                        "object_image_ids": object_image_ids,
                        "scene_object_ids": sorted(scene_object_ids),
                        "scene_categories": sorted(scene_categories),
                        "category": self._category_for_object(item, object_id),
                        "category_id": item.get("object_id_to_category_id", {}).get(str(object_id), ""),
                    }
                )
        return records

    def _build_single_records_by_object_id(self) -> Dict[int, List[Dict[str, Any]]]:
        if not self.single_root.is_dir():
            raise FileNotFoundError(f"OV9D single-object root not found: {self.single_root}")
        if not self.name_to_oid_json.is_file():
            raise FileNotFoundError(f"OV9D name-to-object-id JSON not found: {self.name_to_oid_json}")

        name_to_oid = self.load_json(self.name_to_oid_json)
        records_by_object_id: Dict[int, List[Dict[str, Any]]] = {}
        for scene_dir in sorted(p for p in self.single_root.iterdir() if p.is_dir()):
            if self.verify_files and not self._verify_scene_files(scene_dir):
                continue

            parts = scene_dir.name.split("_")
            if len(parts) < 3:
                continue
            category, object_instance = self._parse_single_scene_name(scene_dir.name)
            if object_instance not in name_to_oid:
                continue

            object_id = int(name_to_oid[object_instance])
            object_image_ids = []
            for rgb_path in sorted((scene_dir / "rgb").glob("*.png")):
                image_id = int(rgb_path.stem)
                mask_path = scene_dir / "mask_visib" / f"{image_id:06d}_000000.png"
                if not self.verify_files or mask_path.is_file():
                    object_image_ids.append(image_id)
            if len(object_image_ids) < self.num_object_views:
                continue

            records_by_object_id.setdefault(object_id, []).append(
                {
                    "scene_name": scene_dir.name,
                    "scene_dir": scene_dir,
                    "target_object_id": object_id,
                    "object_image_ids": object_image_ids,
                    "category": category,
                    "object_instance": object_instance,
                }
            )

        return records_by_object_id

    def _sample_single_record_for_object(self, object_id: int, rng: random.Random) -> Dict[str, Any]:
        candidates = self.single_records_by_object_id.get(int(object_id), [])
        if not candidates:
            raise KeyError(f"No oo3d9dsingle reference scene found for object id {object_id}")
        return rng.choice(candidates)

    def _sample_scene_image_ids(
        self,
        all_image_ids: List[int],
        object_image_ids: List[int],
        count: int,
        rng: random.Random,
    ) -> List[int]:
        object_count = min(int(count), len(object_image_ids))
        selected_object_ids = self._sample_image_ids(object_image_ids, object_count, rng) if object_count > 0 else []
        if len(selected_object_ids) >= count:
            return selected_object_ids

        selected_set = set(selected_object_ids)
        filler_candidates = [image_id for image_id in all_image_ids if image_id not in selected_set]
        filler_count = min(int(count) - len(selected_object_ids), len(filler_candidates))
        filler_ids = self._sample_image_ids(filler_candidates, filler_count, rng) if filler_count > 0 else []
        return selected_object_ids + filler_ids

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

    def _sample_object_image_ids(self, available_ids: List[int], count: int, rng: random.Random) -> List[int]:
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

    def _load_object_image(self, scene_dir: Path, image_id: int, object_index: int) -> np.ndarray:
        rgb = self.read_rgb(scene_dir / "rgb" / f"{image_id:06d}.png")
        mask_path = scene_dir / "mask_visib" / f"{image_id:06d}_{object_index:06d}.png"
        return self.apply_mask_white_background(rgb, mask_path)

    def _process_scene_view(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        aspect_ratio: float,
        filepath: str,
    ):
        original_size = np.array(image.shape[:2], dtype=np.int32)
        target_image_shape = self.get_target_shape(aspect_ratio)
        return self.process_one_image(
            image=image,
            depth_map=depth,
            extri_opencv=extrinsic,
            intri_opencv=intrinsic,
            original_size=original_size,
            target_image_shape=target_image_shape,
            filepath=filepath,
        )

    def _process_object_image(
        self,
        image: np.ndarray,
        aspect_ratio: float,
        filepath: str,
    ) -> np.ndarray:
        original_size = np.array(image.shape[:2], dtype=np.int32)
        target_image_shape = self.get_target_shape(aspect_ratio)
        dummy_depth = np.zeros(image.shape[:2], dtype=np.float32)
        dummy_extrinsic = np.concatenate(
            [np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)],
            axis=1,
        )
        dummy_intrinsic = np.array(
            [
                [1.0, 0.0, image.shape[1] / 2.0],
                [0.0, 1.0, image.shape[0] / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        processed_image, *_ = self.process_one_image(
            image=image,
            depth_map=dummy_depth,
            extri_opencv=dummy_extrinsic,
            intri_opencv=dummy_intrinsic,
            original_size=original_size,
            target_image_shape=target_image_shape,
            filepath=filepath,
        )
        return processed_image

    def _sample_negative_record(
        self,
        current_scene_object_ids: List[int],
        current_scene_categories: List[str],
        rng: random.Random,
    ):
        current_scene_object_id_set = {int(x) for x in current_scene_object_ids}
        current_scene_category_set = {str(x) for x in current_scene_categories}
        candidate_object_ids = [
            object_id for object_id, single_records in self.single_records_by_object_id.items()
            if object_id not in current_scene_object_id_set
            and single_records
            and str(single_records[0].get("category", "")) not in current_scene_category_set
        ]
        if not candidate_object_ids:
            return None
        negative_object_id = rng.choice(candidate_object_ids)
        return self._sample_single_record_for_object(negative_object_id, rng)

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
        target_object_id = int(rec["target_object_id"])
        scene_gt = self.load_json(scene_dir / "scene_gt.json")
        scene_camera = self.load_json(scene_dir / "scene_camera.json")
        all_available_ids = self._available_image_ids(scene_gt)
        object_available_ids = rec["object_image_ids"] or self._image_ids_with_object(scene_gt, target_object_id)

        frame_num = int(img_per_seq) if img_per_seq is not None else self.num_scene_views
        frame_num = max(1, min(frame_num, len(all_available_ids)))
        scene_ids = self._sample_scene_image_ids(all_available_ids, object_available_ids, frame_num, random)

        use_negative_object = (
            self.negative_object_prob > 0.0
            and random.random() < self.negative_object_prob
        )
        object_reference_id = target_object_id
        object_reference_category = rec.get("category", "")
        object_single_rec = self._sample_single_record_for_object(object_reference_id, random)
        if use_negative_object:
            negative_single_rec = self._sample_negative_record(
                rec["scene_object_ids"],
                rec.get("scene_categories", []),
                random,
            )
            if negative_single_rec is not None:
                object_single_rec = negative_single_rec
                object_reference_id = int(object_single_rec["target_object_id"])
                object_reference_category = object_single_rec.get("category", "")
            else:
                use_negative_object = False

        if not use_negative_object:
            object_reference_id = target_object_id
            object_reference_category = rec.get("category", "")
        object_scene_dir: Path = object_single_rec["scene_dir"]
        object_scene_gt = self.load_json(object_scene_dir / "scene_gt.json")
        object_available_ids = object_single_rec["object_image_ids"]
        object_ids = self._sample_object_image_ids(object_available_ids, self.num_object_views, random)

        scene_images = []
        raw_depths = []
        raw_cam_points = []
        raw_world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []
        for image_id in scene_ids:
            image, depth, extrinsic, intrinsic, world_points, cam_points, point_mask = self._load_scene_view(
                scene_dir,
                image_id,
                scene_camera[str(image_id)],
            )
            image, depth, extrinsic, intrinsic, world_points, cam_points, point_mask, _ = self._process_scene_view(
                image=image,
                depth=depth,
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                aspect_ratio=aspect_ratio,
                filepath=str(scene_dir / "rgb" / f"{image_id:06d}.png"),
            )
            scene_images.append(image)
            raw_depths.append(depth)
            raw_cam_points.append(cam_points)
            raw_world_points.append(world_points)
            point_masks.append(point_mask)
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

        first_id = scene_ids[0]
        first_camera = scene_camera[str(first_id)]
        first_object_index = self._object_index_for_id(scene_gt[str(first_id)], target_object_id)
        if first_object_index is None:
            raise KeyError(f"Object {target_object_id} not found in {scene_name} frame {first_id}")
        first_gt = scene_gt[str(first_id)][first_object_index]
        object_rotation_m2w, object_translation_m2w = self.model_to_world_from_bop(first_camera, first_gt)
        normalized_object_rotation, normalized_object_translation = self.normalize_object_pose_to_first_camera(
            first_extrinsic=extrinsics_np[0],
            object_rotation_m2w=object_rotation_m2w,
            object_translation_m2w=object_translation_m2w,
            avg_scale=avg_scale,
            scale_by_points=self.scale_by_points,
        )

        object_images = []
        object_original_sizes = []
        for image_id in object_ids:
            object_index = self._object_index_for_id(object_scene_gt[str(image_id)], object_reference_id)
            if object_index is None:
                raise KeyError(f"Object {object_reference_id} not found in {object_single_rec['scene_name']} frame {image_id}")
            object_image = self._load_object_image(object_scene_dir, image_id, object_index)
            object_image = self._process_object_image(
                image=object_image,
                aspect_ratio=aspect_ratio,
                filepath=str(object_scene_dir / "rgb" / f"{image_id:06d}.png"),
            )
            object_images.append(object_image)
            object_original_sizes.append(np.array(object_image.shape[:2], dtype=np.int32))

        object_name = f"obj_{target_object_id:06d}"
        object_reference_name = f"obj_{object_reference_id:06d}"
        batch = {
            "seq_name": f"ov9d_multi_pose_normalize_{self.split}/{scene_name}/{object_name}",
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
            "object_name": object_name,
            "object_reference_name": object_reference_name,
            "object_reference_scene_name": object_single_rec["scene_name"],
            "object_id": np.array(target_object_id, dtype=np.int64),
            "object_reference_id": np.array(object_reference_id, dtype=np.int64),
            "category": rec.get("category", ""),
            "object_reference_category": object_reference_category,
            "scene_name": scene_name,
            "run_name": scene_name,
            "skip_normalization": True,
            "has_object": np.array(0.0 if use_negative_object else 1.0, dtype=np.float32),
            "object_rotation": normalized_object_rotation,
            "object_translation": normalized_object_translation,
            "object_srt": np.concatenate(
                [normalized_object_rotation.reshape(-1), normalized_object_translation], axis=0
            ).astype(np.float32),
            "normalization_scale": np.array([avg_scale], dtype=np.float32),
        }

        if self.load_point_map:
            scale = avg_scale if self.scale_by_points else 1.0
            batch.update(
                {
                    "depths": [(depth / scale).astype(np.float32) for depth in raw_depths],
                    "cam_points": [(points / scale).astype(np.float32) for points in raw_cam_points],
                    "world_points": [points.astype(np.float32) for points in normalized_world_points],
                    "point_masks": point_masks,
                    "object_masks": point_masks,
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


def _format_paths(paths: List[Any]) -> str:
    return "\n".join(f"      {path}" for path in paths)


def _print_sample_paths(dataset: OV9DMultiPoseNormalizeDataset, seq_index: int, img_per_seq: int) -> None:
    batch = dataset.get_data(seq_index=seq_index, img_per_seq=img_per_seq)
    rec = dataset.records[seq_index % dataset.sequence_list_len]
    scene_dir: Path = rec["scene_dir"]
    scene_gt = dataset.load_json(scene_dir / "scene_gt.json")
    target_object_id = int(batch["object_id"])
    has_object = bool(float(batch["has_object"]) > 0.5)

    scene_rgb_paths = []
    scene_depth_paths = []
    scene_mask_paths = []
    for image_id in batch["ids"].tolist():
        object_index = dataset._object_index_for_id(scene_gt[str(image_id)], target_object_id)
        scene_rgb_paths.append(scene_dir / "rgb" / f"{image_id:06d}.png")
        scene_depth_paths.append(scene_dir / "depth" / f"{image_id:06d}.png")
        if object_index is None:
            scene_mask_paths.append(f"None for frame {image_id:06d} (target object not visible)")
        else:
            scene_mask_paths.append(scene_dir / "mask_visib" / f"{image_id:06d}_{object_index:06d}.png")

    object_scene_dir = dataset.single_root / batch["object_reference_scene_name"]
    object_scene_gt = dataset.load_json(object_scene_dir / "scene_gt.json")
    object_target_id = int(batch["object_reference_id"])
    object_rgb_paths = []
    object_mask_paths = []
    for image_id in batch["object_cam_indices"].tolist():
        object_index = dataset._object_index_for_id(object_scene_gt[str(image_id)], object_target_id)
        object_rgb_paths.append(object_scene_dir / "rgb" / f"{image_id:06d}.png")
        object_mask_paths.append(object_scene_dir / "mask_visib" / f"{image_id:06d}_{object_index:06d}.png")

    print(f"[sample seq_index={seq_index}]")
    print(f"  seq_name: {batch['seq_name']}")
    print(f"  has_object: {str(has_object).lower()}")
    if not has_object:
        print("  false")
    print(f"  target_object_id: {target_object_id}")
    print(f"  category: {batch.get('category', '')}")
    print(f"  object_reference_id: {int(batch['object_reference_id'])}")
    print(f"  object_reference_category: {batch.get('object_reference_category', '')}")
    print(f"  multi_scene_dir: {scene_dir}")
    print("  scene_rgb_paths:")
    print(_format_paths(scene_rgb_paths))
    print("  scene_depth_paths:")
    print(_format_paths(scene_depth_paths))
    print("  scene_target_mask_paths:")
    print(_format_paths(scene_mask_paths))
    print(f"  object_reference_scene_dir: {object_scene_dir}")
    print("  object_rgb_paths:")
    print(_format_paths(object_rgb_paths))
    print("  object_mask_paths:")
    print(_format_paths(object_mask_paths))
    print(
        "  loaded_shapes: "
        f"images={len(batch['images'])}x{batch['images'][0].shape}, "
        f"object_images={len(batch['object_images'])}x{batch['object_images'][0].shape}, "
        f"extrinsics={batch['extrinsics'].shape}, object_srt={batch['object_srt'].shape}"
    )
    print()


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sample OV9D multi-object dataset records and print file paths.")
    parser.add_argument("--data-root", default=OV9DMultiPoseNormalizeDataset.DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--name-to-oid-json", default=None)
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

    dataset = OV9DMultiPoseNormalizeDataset(
        common_conf=_make_main_common_conf(debug=args.debug),
        split=args.split,
        DATA_ROOT=args.data_root,
        SPLIT_JSON=args.split_json,
        NAME_TO_OID_JSON=args.name_to_oid_json,
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
    print(f"single_reference_object_ids: {len(dataset.single_records_by_object_id)}")
    print(f"negative_object_prob: {args.negative_object_prob}")
    print(f"object_view_gap_range: ({args.object_view_min_gap}, {args.object_view_max_gap}) exclusive")
    print(f"sample_indices: {sample_indices}")
    print()

    for seq_index in sample_indices:
        _print_sample_paths(dataset, seq_index=seq_index, img_per_seq=args.img_per_seq)


if __name__ == "__main__":
    _main()
