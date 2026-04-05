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

import numpy as np

from data.base_dataset import BaseDataset
from data.dataset_util import read_image_cv2


class SixDPoseDataset(BaseDataset):
    """Multi-view object-conditioned dataset with per-view camera-space pose GT."""

    FIXED_VIEWS = (1, 3, 8, 12, 15, 18)
    FIXED_OBJECT_VIEWS = (1, 3, 8, 12)

    def __init__(
        self,
        common_conf,
        split: str = "train",
        DATA_ROOT: str = "/mnt/train-data-4-hdd/yian/6dpose_obj/0327_fixedCam_1k",
        OBJECT_INPUT_ROOT: Optional[str] = None,
        OBJECT_IMAGE_ROOT: Optional[str] = None,
        CAMERA_POSE_ROOT: Optional[str] = None,
        len_train: int = 2514,
        len_test: int = 2514,
        verify_files: bool = True,
        only_run_name: str = "",
        only_run_names: Optional[List[str]] = None,
        only_run_start: Optional[str] = None,
        only_run_end: Optional[str] = None,
        selected_views: Optional[List[int]] = None,
        object_input_views: Optional[List[int]] = None,
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
            object_root = osp.join(DATA_ROOT, "object_space_rgb")
        self.object_root = object_root
        self.pose_root = osp.join(DATA_ROOT, "out_pose")
        self.camera_pose_root = CAMERA_POSE_ROOT or osp.join(DATA_ROOT, "camera_pose")
        self._camera_pose_name_cache: Dict[str, List[str]] = {}

        self.verify_files = bool(verify_files)
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
            raise RuntimeError(
                "No valid 6dpose_camerapose samples found. "
                f"Checked roots: scene={self.scene_root}, object={self.object_root}, "
                f"pose={self.pose_root}, camera_pose={self.camera_pose_root}"
            )

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: 6DPose CameraPose sequence count: {self.sequence_list_len}")
        logging.info(f"{status}: 6DPose CameraPose dataset length: {len(self)}")

    def _build_records(self) -> List[Dict]:
        records: List[Dict] = []
        run_names = self._list_run_names()
        if self.debug:
            run_names = run_names[:1]

        missing_count = 0
        for run_name in run_names:
            scene_dir = osp.join(self.scene_root, run_name)
            pose_path = osp.join(self.pose_root, f"{run_name}.npz")
            camera_pose_dir = osp.join(self.camera_pose_root, run_name)
            if not (osp.isdir(scene_dir) and osp.isfile(pose_path) and osp.isdir(camera_pose_dir)):
                missing_count += 1
                continue

            object_names = self._load_run_object_names(pose_path)
            if not object_names:
                missing_count += 1
                continue

            for object_name in object_names:
                obj_dir = osp.join(self.object_root, object_name)
                object_cam_indices = list(self.FIXED_OBJECT_VIEWS)
                if not osp.isdir(obj_dir):
                    continue
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
                    }
                )

        logging.info(
            f"6DPose CameraPose indexing done: {len(records)} valid object samples, {missing_count} missing/invalid runs"
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
    def _decode_names(raw_names: np.ndarray) -> List[str]:
        names: List[str] = []
        for x in raw_names:
            if isinstance(x, (bytes, np.bytes_)):
                names.append(x.decode("utf-8"))
            else:
                names.append(str(x))
        return names

    def _load_run_object_names(self, pose_path: str) -> List[str]:
        with np.load(pose_path, allow_pickle=False) as pose_data:
            return self._decode_names(pose_data["names"])

    @staticmethod
    def _camera_pose_filename(cam_idx: int) -> str:
        if int(cam_idx) == 0:
            return "camera_Main_Camera.npz"
        return f"camera_Main_Camera_({int(cam_idx)}).npz"

    def _load_camera_pose_names(self, camera_pose_path: str) -> List[str]:
        cached = self._camera_pose_name_cache.get(camera_pose_path)
        if cached is not None:
            return cached
        with np.load(camera_pose_path, allow_pickle=False) as pose_data:
            names = self._decode_names(pose_data["names"])
        self._camera_pose_name_cache[camera_pose_path] = names
        return names

    def _has_required_object_views(self, obj_dir: str, required_views: List[int]) -> bool:
        return all(osp.isfile(osp.join(obj_dir, f"Main_Camera_({cam_idx})_rgb.png")) for cam_idx in required_views)

    def _verify_record_files(self, run_name: str, object_name: str) -> bool:
        camera_pose_dir = osp.join(self.camera_pose_root, run_name)
        for cam_idx in self.selected_views:
            if not osp.isfile(self._resolve_scene_image_path(run_name, cam_idx)):
                return False
            pose_path = osp.join(camera_pose_dir, self._camera_pose_filename(cam_idx))
            if not osp.isfile(pose_path):
                return False
            names = self._load_camera_pose_names(pose_path)
            if object_name not in names:
                return False
        for obj_cam_idx in self.FIXED_OBJECT_VIEWS:
            if not osp.isfile(self._resolve_object_image_path(run_name, object_name, obj_cam_idx)):
                return False
        return True

    def _load_camera_pose_for_object(
        self,
        run_name: str,
        object_name: str,
        camera_indices: List[int],
    ) -> Dict[str, np.ndarray]:
        camera_pose_dir = osp.join(self.camera_pose_root, run_name)
        rotations: List[np.ndarray] = []
        translations: List[np.ndarray] = []

        for cam_idx in camera_indices:
            camera_pose_path = osp.join(camera_pose_dir, self._camera_pose_filename(int(cam_idx)))
            if not osp.isfile(camera_pose_path):
                raise FileNotFoundError(f"Camera-pose file not found: {camera_pose_path}")

            with np.load(camera_pose_path, allow_pickle=False) as pose_data:
                names = self._decode_names(pose_data["names"])
                if object_name not in names:
                    raise KeyError(
                        f"Object '{object_name}' not found in camera-pose file {camera_pose_path}. "
                        f"Available names: {names}"
                    )
                obj_idx = names.index(object_name)
                rot = np.asarray(pose_data["rotation_matrices_flat9"][obj_idx], dtype=np.float32).reshape(3, 3)
                trans = np.asarray(pose_data["positions"][obj_idx], dtype=np.float32).reshape(3)
            rotations.append(rot)
            translations.append(trans)

        object_rotation = np.stack(rotations, axis=0).astype(np.float32)
        object_translation = np.stack(translations, axis=0).astype(np.float32)
        object_srt = np.concatenate(
            [object_rotation.reshape(len(camera_indices), 9), object_translation],
            axis=1,
        ).astype(np.float32)

        return {
            "object_rotation": object_rotation,
            "object_translation": object_translation,
            "object_srt": object_srt,
        }

    def _resolve_scene_image_path(self, run_name: str, cam_idx: int) -> str:
        return osp.join(self.scene_root, run_name, f"Main_Camera_({cam_idx}).jpg")

    def _resolve_object_image_path(self, run_name: str, object_name: str, cam_idx: int) -> str:
        del run_name
        return osp.join(self.object_root, object_name, f"Main_Camera_({cam_idx})_rgb.png")

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
        camera_indices = list(rec["camera_indices"])
        object_cam_indices = list(rec["object_cam_indices"])

        scene_images = []
        extrinsics = []
        intrinsics = []
        original_sizes = []

        for cam_idx in camera_indices:
            scene_image_path = self._resolve_scene_image_path(run_name, cam_idx)
            scene_image = read_image_cv2(scene_image_path)
            if scene_image is None:
                raise FileNotFoundError(f"Failed to read scene image: {scene_image_path}")

            h, w = scene_image.shape[:2]
            extri = np.concatenate([np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)], axis=1)
            intri = np.array(
                [[1.0, 0.0, w / 2.0], [0.0, 1.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32
            )

            scene_images.append(scene_image)
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

        camera_pose_dict = self._load_camera_pose_for_object(run_name, object_name, camera_indices)

        batch = {
            "seq_name": f"6dpose_camerapose_{run_name}/{object_name}",
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
            "object_scale": np.array([1.0], dtype=np.float32),
        }
        batch.update(camera_pose_dict)
        return batch
