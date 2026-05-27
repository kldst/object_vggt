import json
import logging
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from data.base_dataset import BaseDataset
from data.datasets.ov9d_pose_normalize import OV9DPoseNormalizeDataset

logger = logging.getLogger(__name__)


class MixedPoseNormalizeBase(OV9DPoseNormalizeDataset):
    """Common baseline-compatible multi-view object pose dataset logic.

    The model receives RGB scene views plus fixed object reference RGBs. Depth
    is loaded only inside the dataset to estimate the point-cloud scale used for
    normalized pose targets.
    """

    DEFAULT_NUM_SCENE_VIEWS = 4
    DEFAULT_NUM_OBJECT_VIEWS = 4
    DEFAULT_OBJECT_VIEW_IDS = (1, 5, 10, 15)

    def __init__(
        self,
        common_conf,
        split: str = "train",
        len_train: Optional[int] = None,
        len_test: Optional[int] = None,
        verify_files: bool = True,
        num_scene_views: int = DEFAULT_NUM_SCENE_VIEWS,
        num_object_views: int = DEFAULT_NUM_OBJECT_VIEWS,
        min_view_gap: int = 5,
        fixed_object_view_ids: Optional[List[int]] = None,
        strict_fixed_object_view_ids: bool = True,
        load_point_map: bool = False,
        scale_by_points: bool = True,
        negative_object_prob: float = 0.0,
        print_sample_paths: bool = False,
        print_sample_paths_limit: int = 5,
        max_records: Optional[int] = None,
    ):
        BaseDataset.__init__(self, common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.split = str(split)
        self.verify_files = bool(verify_files)
        self.num_scene_views = int(num_scene_views)
        self.num_object_views = int(num_object_views)
        self.min_view_gap = int(min_view_gap)
        self.fixed_object_view_ids = tuple(
            int(x) for x in (fixed_object_view_ids or self.DEFAULT_OBJECT_VIEW_IDS)
        )
        if len(self.fixed_object_view_ids) != self.num_object_views:
            raise ValueError("fixed_object_view_ids length must match num_object_views")
        self.strict_fixed_object_view_ids = bool(strict_fixed_object_view_ids)
        self.load_point_map = bool(load_point_map)
        self.scale_by_points = bool(scale_by_points)
        self.negative_object_prob = float(negative_object_prob)
        self.print_sample_paths = bool(print_sample_paths)
        self.print_sample_paths_limit = int(print_sample_paths_limit)
        self.max_records = int(max_records) if max_records is not None else None
        self._printed_sample_paths = 0

        self.object_records = self._build_object_records()
        self.records = self._build_records()
        if self.max_records is not None:
            self.records = self.records[: self.max_records]
        if self.debug:
            self.records = self.records[:1]
        self.sequence_list_len = len(self.records)
        self.records_by_frame_num = {
            frame_num: list(range(self.sequence_list_len)) for frame_num in range(4, 7)
        }
        if self.sequence_list_len == 0:
            raise RuntimeError(f"No {type(self).__name__} samples found")

        if self.split in {"train", "train_real", "train_pbr"}:
            self.len_train = int(len_train) if len_train is not None else self.sequence_list_len
        elif self.split in {"test", "val", "validation", "test1", "test2", "test3"}:
            self.len_train = int(len_test) if len_test is not None else self.sequence_list_len
        else:
            raise ValueError(f"Invalid split: {split}")

        logger.info(
            "%s initialized: records=%d objects=%d",
            type(self).__name__,
            len(self.records),
            len(self.object_records),
        )

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_pickle(path: Path) -> Dict[str, Any]:
        with path.open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

    @staticmethod
    def _read_mask(
        path: Optional[Path],
        image_shape: Tuple[int, int],
        inst_id: Optional[int] = None,
    ) -> np.ndarray:
        if path is None or not Path(path).is_file():
            return np.zeros(image_shape, dtype=np.float32)
        raw = np.asarray(Image.open(path), dtype=np.uint8)
        if raw.ndim == 3:
            raw = raw[..., 0]
        if inst_id is None:
            values = np.unique(raw)
            if values.size > 1:
                return (raw == values[-1]).astype(np.float32)
            return (raw > 0).astype(np.float32)
        return (raw == int(inst_id)).astype(np.float32)

    @staticmethod
    def _available_ids(frame_recs: List[Dict[str, Any]]) -> List[int]:
        return sorted(int(rec["image_id"]) for rec in frame_recs)

    def _max_records_ready(self, groups: Dict[Any, Dict[str, Any]]) -> bool:
        if self.max_records is None or len(groups) < self.max_records:
            return False
        ready = sum(
            1 for group in groups.values()
            if len(group.get("frames", [])) >= self.num_scene_views
        )
        return ready >= self.max_records

    @staticmethod
    def _ids_with_gap(
        ids: List[int],
        count: int,
        min_gap: int,
        rng: random.Random,
    ) -> Optional[List[int]]:
        if len(ids) < count:
            return None
        shuffled = list(ids)
        rng.shuffle(shuffled)
        for first in shuffled:
            selected = [first]
            candidates = [x for x in ids if x != first]
            rng.shuffle(candidates)
            for cand in candidates:
                if all(abs(cand - prev) >= min_gap for prev in selected):
                    selected.append(cand)
                    if len(selected) == count:
                        return sorted(selected)
        return None

    def _sample_ids(self, ids: List[int], count: int, rng: random.Random) -> List[int]:
        count = min(int(count), len(ids))
        if count <= 0:
            return []
        spaced = self._ids_with_gap(ids, count, self.min_view_gap, rng)
        if spaced is not None:
            return spaced
        return sorted(rng.sample(ids, count))

    def _sample_object_ids(self, available_ids: List[int]) -> List[int]:
        available = {int(x) for x in available_ids}
        fixed = list(self.fixed_object_view_ids)
        missing = [x for x in fixed if x not in available]
        if missing and self.strict_fixed_object_view_ids:
            raise FileNotFoundError(f"Missing fixed object views {missing}")
        selected = [x for x in fixed if x in available]
        if len(selected) < self.num_object_views:
            fallback = [x for x in sorted(available) if x not in selected]
            selected.extend(fallback[: self.num_object_views - len(selected)])
        return selected[: self.num_object_views]

    def _process_scene_mask(
        self,
        mask: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        aspect_ratio: float,
        filepath: str,
    ) -> np.ndarray:
        dummy = np.repeat((mask[..., None] * 255).astype(np.uint8), 3, axis=2)
        dummy_depth = mask.astype(np.float32)
        target_image_shape = self.get_target_shape(aspect_ratio)
        _, processed_mask, *_ = self.process_one_image(
            image=dummy,
            depth_map=dummy_depth,
            extri_opencv=extrinsic,
            intri_opencv=intrinsic,
            original_size=np.array(mask.shape[:2], dtype=np.int32),
            target_image_shape=target_image_shape,
            filepath=filepath,
        )
        return (processed_mask > 0.5).astype(bool)

    def _process_object_image(
        self,
        image: np.ndarray,
        aspect_ratio: float,
        filepath: str,
    ) -> np.ndarray:
        h, w = image.shape[:2]
        dummy_depth = np.zeros((h, w), dtype=np.float32)
        dummy_extrinsic = np.concatenate(
            [np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)],
            axis=1,
        )
        dummy_intrinsic = np.array(
            [[1.0, 0.0, w / 2.0], [0.0, 1.0, h / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        processed, *_ = self.process_one_image(
            image=image,
            depth_map=dummy_depth,
            extri_opencv=dummy_extrinsic,
            intri_opencv=dummy_intrinsic,
            original_size=np.array([h, w], dtype=np.int32),
            target_image_shape=self.get_target_shape(aspect_ratio),
            filepath=filepath,
        )
        return processed

    def _process_scene_view(self, frame_rec: Dict[str, Any], aspect_ratio: float):
        image = self._read_rgb(frame_rec["rgb_path"])
        depth = self._read_depth(frame_rec)
        extrinsic, intrinsic = self._camera_matrices(frame_rec)
        mask = self._read_mask(
            frame_rec.get("mask_path"),
            image.shape[:2],
            frame_rec.get("inst_id"),
        )
        state = np.random.get_state()
        image, depth, extrinsic, intrinsic, world_points, cam_points, point_mask, _ = (
            self.process_one_image(
                image=image,
                depth_map=depth,
                extri_opencv=extrinsic,
                intri_opencv=intrinsic,
                original_size=np.array(image.shape[:2], dtype=np.int32),
                target_image_shape=self.get_target_shape(aspect_ratio),
                filepath=str(frame_rec["rgb_path"]),
            )
        )
        np.random.set_state(state)
        raw_extrinsic, raw_intrinsic = self._camera_matrices(frame_rec)
        object_mask = self._process_scene_mask(
            mask,
            raw_extrinsic,
            raw_intrinsic,
            aspect_ratio,
            str(frame_rec.get("mask_path", "")),
        )
        return image, depth, extrinsic, intrinsic, world_points, cam_points, point_mask, object_mask

    def _load_object_images(
        self,
        object_key: Any,
        aspect_ratio: float,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[int]]:
        object_rec = self.object_records[object_key]
        image_ids = self._sample_object_ids(object_rec["image_ids"])
        images, sizes, paths = [], [], []
        for image_id in image_ids:
            image_path = object_rec["object_dir"] / "rgb" / f"{int(image_id):06d}.png"
            image = self._read_rgb(image_path)
            sizes.append(np.array(image.shape[:2], dtype=np.int32))
            images.append(self._process_object_image(image, aspect_ratio, str(image_path)))
            paths.append(str(image_path))
        return images, sizes, paths, image_ids

    def _sample_negative_object_key(self, positive_key: Any, rng: random.Random) -> Optional[Any]:
        candidates = [key for key in self.object_records.keys() if key != positive_key]
        return rng.choice(candidates) if candidates else None

    def _object_size_metric(self, object_key: Any, rec: Dict[str, Any]) -> Optional[np.ndarray]:
        del object_key, rec
        return None

    def get_data(
        self,
        seq_index=None,
        img_per_seq=None,
        seq_name=None,
        ids=None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        del seq_name, ids
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        if seq_index is None:
            seq_index = 0
        rec = self.records[int(seq_index) % self.sequence_list_len]
        frame_recs = rec["frames"]
        by_id = {int(x["image_id"]): x for x in frame_recs}
        frame_num = int(img_per_seq) if img_per_seq is not None else self.num_scene_views
        scene_ids = self._sample_ids(self._available_ids(frame_recs), frame_num, random)

        use_negative = (
            self.training
            and self.negative_object_prob > 0
            and random.random() < self.negative_object_prob
        )
        object_key = rec["object_key"]
        if use_negative:
            negative_key = self._sample_negative_object_key(object_key, random)
            if negative_key is None:
                use_negative = False
            else:
                object_key = negative_key

        scene_images, raw_depths, raw_cam_points, raw_world_points = [], [], [], []
        point_masks, object_masks, extrinsics, intrinsics, original_sizes = [], [], [], [], []
        for image_id in scene_ids:
            frame_rec = dict(by_id[int(image_id)])
            if use_negative:
                frame_rec["mask_path"] = None
                frame_rec["inst_id"] = None
            image, depth, extrinsic, intrinsic, world_points, cam_points, point_mask, object_mask = (
                self._process_scene_view(frame_rec, aspect_ratio)
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
        normalized_extrinsics, avg_scale, normalized_world_points = (
            self.normalize_extrinsics_and_world_points(
                extrinsics=extrinsics_np,
                world_points=raw_world_points,
                point_masks=point_masks,
                scale_by_points=self.scale_by_points,
            )
        )
        first_frame = by_id[int(scene_ids[0])]
        if use_negative:
            object_rotation = np.eye(3, dtype=np.float32)
            object_translation = np.zeros(3, dtype=np.float32)
        else:
            r_m2w, t_m2w = self._object_model_to_world(first_frame, rec)
            object_rotation, object_translation = self.normalize_object_pose_to_first_camera(
                first_extrinsic=extrinsics_np[0],
                object_rotation_m2w=r_m2w,
                object_translation_m2w=t_m2w,
                avg_scale=avg_scale,
                scale_by_points=self.scale_by_points,
            )

        object_images, object_sizes, object_paths, object_image_ids = self._load_object_images(object_key, aspect_ratio)
        object_meta = self.object_records[object_key]
        object_size_metric = self._object_size_metric(object_key, rec)
        object_size = None
        if object_size_metric is not None:
            scale = avg_scale if self.scale_by_points else 1.0
            object_size = (np.asarray(object_size_metric, dtype=np.float32) / float(scale)).astype(np.float32)
        batch = {
            "seq_name": f"{self.dataset_name}/{self.split}/{rec['scene_name']}/{rec['object_name']}",
            "ids": np.array(scene_ids, dtype=np.int64),
            "frame_num": len(scene_ids),
            "images": scene_images,
            "object_images": object_images,
            "extrinsics": normalized_extrinsics,
            "intrinsics": intrinsics,
            "original_sizes": original_sizes,
            "object_original_sizes": object_sizes,
            "camera_indices": np.array(scene_ids, dtype=np.int64),
            "object_cam_indices": np.array(object_image_ids, dtype=np.int64),
            "object_name": str(rec["object_name"]),
            "object_reference_name": str(object_meta["object_name"]),
            "object_id": np.array(int(object_meta["object_id"]), dtype=np.int64),
            "symmetry_object_id": f"{self.symmetry_dataset_name}:{int(object_meta['object_id'])}",
            "category": str(rec.get("category", "")),
            "object_reference_category": str(object_meta.get("category", "")),
            "scene_name": str(rec["scene_name"]),
            "run_name": str(rec["scene_name"]),
            "skip_normalization": True,
            "has_object": np.array(0.0 if use_negative else 1.0, dtype=np.float32),
            "object_rotation": object_rotation,
            "object_translation": object_translation,
            "object_masks": object_masks,
            "object_srt": np.concatenate(
                [object_rotation.reshape(-1), object_translation],
                axis=0,
            ).astype(np.float32),
            "normalization_scale": np.array([avg_scale], dtype=np.float32),
            "scene_rgb_paths": [str(by_id[int(i)]["rgb_path"]) for i in scene_ids],
            "scene_depth_paths": [str(by_id[int(i)]["depth_path"]) for i in scene_ids],
            "object_rgb_paths": object_paths,
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
        self._maybe_print_paths(batch, int(seq_index))
        return batch

    def _maybe_print_paths(self, batch: Dict[str, Any], seq_index: int) -> None:
        if not self.print_sample_paths:
            return
        if self.print_sample_paths_limit >= 0 and self._printed_sample_paths >= self.print_sample_paths_limit:
            return
        self._printed_sample_paths += 1
        print(
            f"[{self.dataset_name} sample {self._printed_sample_paths}/{self.print_sample_paths_limit}] "
            f"seq_index={seq_index}",
            flush=True,
        )
        print(f"  seq_name: {batch['seq_name']}", flush=True)
        print(f"  scene_frame_ids: {batch['ids'].tolist()}", flush=True)
        print("  scene_rgb_paths:", flush=True)
        print(format_paths(batch["scene_rgb_paths"]), flush=True)
        print("  scene_depth_paths:", flush=True)
        print(format_paths(batch["scene_depth_paths"]), flush=True)
        print("  object_rgb_paths:", flush=True)
        print(format_paths(batch["object_rgb_paths"]), flush=True)
        print("", flush=True)


def format_paths(paths: List[Any]) -> str:
    return "\n".join(f"      {path}" for path in paths)
