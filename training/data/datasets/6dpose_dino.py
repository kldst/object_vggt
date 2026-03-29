# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import os.path as osp
import random
from typing import Dict, List, Optional

import cv2
import numpy as np

from data.base_dataset import BaseDataset
from data.dataset_util import read_image_cv2


class SixDPoseDataset(BaseDataset):
    """Multi-view object-conditioned 6D pose dataset with fixed object-space views."""

    FIXED_VIEWS = (1, 3, 8, 12, 15, 18)
    FIXED_OBJECT_VIEWS = (1, 5, 10, 15)

    def __init__(
        self,
        common_conf,
        split: str = "train",
        DATA_ROOT: str = "/mnt/train-data-4-hdd/yian/6dpose_obj/0303_fixedCam_1k",
        OBJECT_INPUT_ROOT: Optional[str] = None,
        OBJECT_IMAGE_ROOT: Optional[str] = None,
        MASK_ROOT: Optional[str] = None,
        len_train: int = 3000, # 資料量
        len_test: int = 300,    # 資料量
        verify_files: bool = True,
        only_run_name: str = "",
        only_run_names: Optional[List[str]] = None,
        only_run_start: Optional[str] = "run_0000",
        only_run_end: Optional[str] = "run_0010",
        selected_views: Optional[List[int]] = None,
        object_input_views: Optional[List[int]] = None,
        load_point_map: bool = True,
        num_object_views: int = 4,
    ):
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.num_object_views = int(num_object_views)
        if self.num_object_views != len(self.FIXED_OBJECT_VIEWS):
            raise ValueError(
                f"num_object_views must be {len(self.FIXED_OBJECT_VIEWS)} "
                f"to match FIXED_OBJECT_VIEWS={self.FIXED_OBJECT_VIEWS}, got {self.num_object_views}"
            )

        self.split = split
        self.data_root = DATA_ROOT
        self.scene_root = osp.join(DATA_ROOT, "out_image")
        object_root = OBJECT_INPUT_ROOT or OBJECT_IMAGE_ROOT
        if not object_root:
            object_root = "/mnt/train-data-4-hdd/yian/6dpose_obj/0315_fixedCam_1k/object_space_rgb"
        self.object_root = object_root
        self.map_root = osp.join(DATA_ROOT, "cam_decoded_map")
        self.pose_root = osp.join(DATA_ROOT, "out_pose")
        self.mask_root = MASK_ROOT

        self.verify_files = bool(verify_files)
        self.load_point_map = bool(load_point_map)
        selected_view_list = selected_views or object_input_views or self.FIXED_VIEWS
        self.selected_views = tuple(int(v) for v in selected_view_list)
        self.only_run_name = (only_run_name or "").strip()
        self.only_run_names = [x.strip() for x in (only_run_names or []) if str(x).strip()]
        if self.only_run_name:
            self.only_run_names = [self.only_run_name]
        self.only_run_start_idx = self._parse_run_index(only_run_start)
        self.only_run_end_idx = self._parse_run_index(only_run_end)
        if (
            self.only_run_start_idx is not None
            and self.only_run_end_idx is not None
            and self.only_run_start_idx > self.only_run_end_idx
        ):
            raise ValueError(
                f"Invalid run range: only_run_start={only_run_start} > only_run_end={only_run_end}"
            )

        if split == "train":
            self.len_train = len_train
        elif split in ["test", "val"]:
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.records = self._build_records()
        self.sequence_list_len = len(self.records)
        self.records_by_frame_num = {len(self.selected_views): list(range(self.sequence_list_len))}

        if self.sequence_list_len == 0:
            checked_roots = [f"scene={self.scene_root}", f"object={self.object_root}", f"pose={self.pose_root}"]
            if self.load_point_map:
                checked_roots.append(f"map={self.map_root}")
            if self.mask_root is not None:
                checked_roots.append(f"mask={self.mask_root}")
            raise RuntimeError(
                "No valid 6dpose_dino samples found. "
                f"Checked roots: {', '.join(checked_roots)}"
            )

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: 6DPose DINO sequence count: {self.sequence_list_len}")
        logging.info(f"{status}: 6DPose DINO dataset length: {len(self)}")

    def _build_records(self) -> List[Dict]:
        records: List[Dict] = []
        run_names = self._list_run_names()
        if self.debug:
            run_names = run_names[:1]

        missing_count = 0
        for run_name in run_names:
            scene_dir = osp.join(self.scene_root, run_name)
            pose_path = osp.join(self.pose_root, f"{run_name}.npz")

            if not (osp.isdir(scene_dir) and osp.isfile(pose_path)):
                missing_count += 1
                continue
            if self.load_point_map and not osp.isdir(osp.join(self.map_root, run_name)):
                missing_count += 1
                continue

            pose_lookup = self._load_pose_lookup(pose_path)
            if not pose_lookup:
                missing_count += 1
                continue

            object_names = sorted(pose_lookup.keys())

            for object_name in object_names:
                obj_dir = osp.join(self.object_root, object_name)
                if not osp.isdir(obj_dir):
                    continue
                object_cam_indices = list(self.FIXED_OBJECT_VIEWS)
                if not self._has_required_object_views(obj_dir, object_cam_indices):
                    continue
                if self.verify_files and not self._verify_record_files(run_name, object_name):
                    continue
                records.append(
                    {
                        "run_name": run_name,
                        "object_name": object_name,
                        "camera_indices": list(self.selected_views),
                        "object_cam_indices": object_cam_indices,
                        "pose": pose_lookup[object_name],
                    }
                )

        logging.info(
            f"6DPose DINO indexing done: {len(records)} valid object samples, {missing_count} missing/invalid runs"
        )
        return records

    def _list_run_names(self) -> List[str]:
        if not osp.isdir(self.scene_root):
            raise FileNotFoundError(f"Scene root not found: {self.scene_root}")

        run_names = sorted([d for d in os.listdir(self.scene_root) if d.startswith("run_")])
        if self.only_run_names:
            only_set = set(self.only_run_names)
            run_names = [r for r in run_names if r in only_set]
        if self.only_run_start_idx is not None or self.only_run_end_idx is not None:
            filtered = []
            for run_name in run_names:
                run_idx = self._parse_run_index(run_name)
                if run_idx is None:
                    continue
                if self.only_run_start_idx is not None and run_idx < self.only_run_start_idx:
                    continue
                if self.only_run_end_idx is not None and run_idx > self.only_run_end_idx:
                    continue
                filtered.append(run_name)
            run_names = filtered
        return run_names

    @staticmethod
    def _list_object_cam_indices(obj_dir: str) -> List[int]:
        """Scan object folder and return all available cam indices from Main_Camera_({N})_rgb.png files."""
        import re
        pattern = re.compile(r"^Main_Camera_\((\d+)\)_rgb\.png$")
        indices = []
        for fname in os.listdir(obj_dir):
            m = pattern.match(fname)
            if m:
                indices.append(int(m.group(1)))
        return sorted(indices)

    def _verify_record_files(self, run_name: str, object_name: str) -> bool:
        for cam_idx in self.selected_views:
            if not osp.isfile(self._resolve_scene_image_path(run_name, cam_idx)):
                return False
            if self.mask_root is not None and not osp.isfile(self._resolve_mask_path(run_name, object_name, cam_idx)):
                return False
            if self.load_point_map and not osp.isfile(self._resolve_map_path(run_name, object_name, cam_idx)):
                return False
        for obj_cam_idx in self.FIXED_OBJECT_VIEWS:
            if not osp.isfile(self._resolve_object_image_path(run_name, object_name, obj_cam_idx)):
                return False
        return True

    def _has_required_object_views(self, obj_dir: str, required_views: List[int]) -> bool:
        return all(osp.isfile(osp.join(obj_dir, f"Main_Camera_({cam_idx})_rgb.png")) for cam_idx in required_views)

    @staticmethod
    def _parse_run_index(run_name: Optional[str]) -> Optional[int]:
        if run_name is None:
            return None
        s = str(run_name).strip()
        if not s:
            return None
        if s.isdigit():
            return int(s)
        if s.startswith("run_") and s[4:].isdigit():
            return int(s[4:])
        raise ValueError(f"Invalid run spec '{run_name}'. Expected e.g. 'run_0000' or '0'.")

    @staticmethod
    def _quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = [float(v) for v in quat_wxyz]
        n = np.sqrt(w * w + x * x + y * y + z * z)
        if n < 1e-8:
            return np.eye(3, dtype=np.float32)
        w, x, y, z = w / n, x / n, y / n, z / n
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )

    def _load_pose_lookup(self, pose_path: str) -> Dict[str, Dict[str, np.ndarray]]:
        pose_data = np.load(pose_path, allow_pickle=False)
        names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in pose_data["names"]]
        positions = pose_data["positions"].astype(np.float32)
        quats = pose_data["rot_quat_wxyz"].astype(np.float32)

        pose_lookup = {}
        for idx, object_name in enumerate(names):
            translation = positions[idx]
            rotation = self._quat_wxyz_to_rotmat(quats[idx])
            pose_lookup[object_name] = {
                "object_rotation": rotation,
                "object_translation": translation.astype(np.float32),
                "object_srt": np.concatenate([rotation.reshape(-1), translation.astype(np.float32)], axis=0),
            }
        return pose_lookup

    def _resolve_scene_image_path(self, run_name: str, cam_idx: int) -> str:
        return osp.join(self.scene_root, run_name, f"Main_Camera_({cam_idx}).jpg")

    def _resolve_object_image_path(self, run_name: str, object_name: str, cam_idx: int) -> str:
        del run_name
        return osp.join(
            self.object_root,
            object_name,
            f"Main_Camera_({cam_idx})_rgb.png",
        )

    def _resolve_map_path(self, run_name: str, object_name: str, cam_idx: int) -> str:
        return osp.join(self.map_root, run_name, object_name, f"cam{cam_idx}_decoded_map.npz")

    def _resolve_mask_path(self, run_name: str, object_name: str, cam_idx: int) -> str:
        return osp.join(
            self.mask_root,
            run_name,
            object_name,
            f"Main_Camera_({cam_idx})_{object_name}_depth.png",
        )

    @staticmethod
    def _read_mask(mask_path: str) -> np.ndarray:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask image: {mask_path}")
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask > 0

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        del img_per_seq, seq_name, ids, aspect_ratio

        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        if seq_index is None:
            seq_index = 0

        rec = self.records[seq_index % self.sequence_list_len]
        run_name = rec["run_name"]
        object_name = rec["object_name"]
        # scene/mask: always use the configured fixed views in fixed order
        camera_indices = list(rec["camera_indices"])
        # object: always use the fixed object reference views
        object_cam_indices = list(rec["object_cam_indices"])

        scene_images = []
        depths = [] if self.load_point_map else None
        cam_points = [] if self.load_point_map else None
        world_points = [] if self.load_point_map else None
        point_masks = [] if self.load_point_map else None
        object_masks = [] if self.mask_root is not None else None
        extrinsics = []
        intrinsics = []
        original_sizes = []

        for cam_idx in camera_indices:
            scene_image_path = self._resolve_scene_image_path(run_name, cam_idx)
            scene_image = read_image_cv2(scene_image_path)
            if scene_image is None:
                raise FileNotFoundError(f"Failed to read scene image: {scene_image_path}")
            object_mask = None
            if self.mask_root is not None:
                object_mask_path = self._resolve_mask_path(run_name, object_name, cam_idx)
                object_mask = self._read_mask(object_mask_path)

            if self.load_point_map:
                map_path = self._resolve_map_path(run_name, object_name, cam_idx)
                if not osp.isfile(map_path):
                    raise FileNotFoundError(f"Point-map file not found: {map_path}")

                npz_data = np.load(map_path)
                if "map_xyz" not in npz_data:
                    raise KeyError(f"Key 'map_xyz' not found in: {map_path}")
                map_xyz = npz_data["map_xyz"].astype(np.float32)

                if scene_image.shape[:2] != map_xyz.shape[:2]:
                    raise ValueError(
                        f"Scene image/point-map shape mismatch for {scene_image_path}: "
                        f"image={scene_image.shape[:2]}, point_map={map_xyz.shape[:2]}"
                    )

                h_map, w_map = map_xyz.shape[:2]
                valid_mask = np.isfinite(map_xyz).all(axis=-1) & (np.linalg.norm(map_xyz, axis=-1) > 0)
                depth = map_xyz[..., 2].copy()
                depth[~valid_mask] = 0.0
            else:
                h_map, w_map = scene_image.shape[:2]

            if object_mask is not None and scene_image.shape[:2] != object_mask.shape[:2]:
                raise ValueError(
                    f"Scene image/mask shape mismatch for {scene_image_path}: "
                    f"image={scene_image.shape[:2]}, mask={object_mask.shape[:2]}"
                )

            extri = np.concatenate([np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)], axis=1)
            intri = np.array(
                [[1.0, 0.0, w_map / 2.0], [0.0, 1.0, h_map / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32
            )

            scene_images.append(scene_image)
            if object_masks is not None:
                object_masks.append(object_mask)
            if self.load_point_map:
                depths.append(depth)
                cam_points.append(map_xyz)
                world_points.append(map_xyz)
                point_masks.append(valid_mask)
            extrinsics.append(extri)
            intrinsics.append(intri)
            original_sizes.append(np.array(scene_image.shape[:2], dtype=np.int32))

        object_images = []
        object_original_sizes = []
        for obj_cam_idx in object_cam_indices:
            object_image_path = self._resolve_object_image_path(run_name, object_name, obj_cam_idx)
            object_image = read_image_cv2(object_image_path)
            if object_image is None:
                raise FileNotFoundError(f"Failed to read object image: {object_image_path}")
            object_images.append(object_image)
            object_original_sizes.append(np.array(object_image.shape[:2], dtype=np.int32))

        batch = {
            "seq_name": f"6dpose_dino_{run_name}/{object_name}",
            "ids": np.array(camera_indices, dtype=np.int64),
            "frame_num": len(camera_indices),
            "images": scene_images,
            "object_images": object_images,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "original_sizes": original_sizes,
            "object_original_sizes": object_original_sizes,
            "camera_indices": np.array(camera_indices, dtype=np.int64),
            "object_cam_indices": np.array(object_cam_indices, dtype=np.int64),
            "object_name": object_name,
            "run_name": run_name,
            "skip_normalization": True,
        }
        if object_masks is not None:
            batch["object_masks"] = object_masks
        if self.load_point_map:
            batch.update({
                "depths": depths,
                "cam_points": cam_points,
                "world_points": world_points,
                "point_masks": point_masks,
            })
        batch.update(rec["pose"])
        return batch
