import logging
import os
import os.path as osp
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from data.base_dataset import BaseDataset
from data.dataset_util import depth_to_world_coords_points, read_image_cv2
from vggt.utils.geometry import closed_form_inverse_se3

try:
    import cv2
except ImportError:  # pragma: no cover - fallback for environments without cv2
    cv2 = None


class SixDPoseNormalizeDataset(BaseDataset):
    """Object-conditioned 6D pose dataset with pose normalization done inside the dataset."""

    FIXED_VIEWS = tuple(range(20))
    FIXED_OBJECT_VIEWS = (1, 3, 8, 12)

    def __init__(
        self,
        common_conf,
        split: str = "train",
        DATA_ROOT: str = "/mnt/train-data-4-hdd/yian/freepose/0407_fixedCam_diffpose_15k_google",
        OBJECT_IMAGE_ROOT: str = "/mnt/train-data-4-hdd/yian/freepose/object_space_renders_all",
        len_train: Optional[int] = None,
        len_test: Optional[int] = None,
        verify_files: bool = True,
        only_run_name: str = "",
        only_run_names: Optional[List[str]] = None,
        only_run_start: Optional[str] = None,
        only_run_end: Optional[str] = None,
        selected_views: Optional[List[int]] = None,
        object_input_views: Optional[List[int]] = None,
        load_point_map: bool = False,
        num_object_views: int = 4,
        scale_by_points: bool = True,
    ):
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random

        self.split = split
        self.data_root = DATA_ROOT
        self.scene_root = osp.join(DATA_ROOT, "out_image")
        self.camera_root = osp.join(DATA_ROOT, "out_cam_param")
        self.depth_root = osp.join(DATA_ROOT, "out_depth")
        self.pose_root = osp.join(DATA_ROOT, "out_pose")
        self.object_root = OBJECT_IMAGE_ROOT

        self.verify_files = bool(verify_files)
        self.load_point_map = bool(load_point_map)
        self.scale_by_points = bool(scale_by_points)

        self.selected_views = tuple(int(v) for v in (selected_views or self.FIXED_VIEWS))
        self.object_input_views = tuple(int(v) for v in (object_input_views or self.FIXED_OBJECT_VIEWS))
        self.num_object_views = int(num_object_views)
        if self.num_object_views != len(self.object_input_views):
            raise ValueError(
                f"num_object_views must match object_input_views length "
                f"({len(self.object_input_views)}), got {self.num_object_views}"
            )

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

        self.records = self._build_records()
        self.sequence_list_len = len(self.records)
        self.records_by_frame_num = {
            frame_num: list(range(self.sequence_list_len)) for frame_num in range(4, 7)
        }
        if self.sequence_list_len == 0:
            raise RuntimeError(
                "No valid normalized 6D pose samples found. "
                f"Checked scene={self.scene_root}, camera={self.camera_root}, "
                f"depth={self.depth_root}, pose={self.pose_root}, object={self.object_root}"
            )

        if split == "train":
            self.len_train = int(len_train) if len_train is not None else self.sequence_list_len
        elif split in ["test", "val"]:
            self.len_train = int(len_test) if len_test is not None else self.sequence_list_len
        else:
            raise ValueError(f"Invalid split: {split}")

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: normalized 6D pose sequence count: {self.sequence_list_len}")
        logging.info(f"{status}: normalized 6D pose dataset length: {len(self)}")

    @staticmethod
    def read_encoded_depth(depth_path: str) -> np.ndarray:
        """
        Decode an R16 PNG whose uint16 pixels store float16 depth bit patterns.
        """
        if cv2 is not None:
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                raise FileNotFoundError(f"Failed to read depth image: {depth_path}")
            if depth_raw.dtype != np.uint16:
                raise ValueError(f"Expected uint16 R16 depth image, got {depth_raw.dtype} for {depth_path}")
            depthmap = depth_raw.view(np.float16).astype(np.float32)
        else:
            with Image.open(depth_path) as depth_pil:
                depth_raw = np.array(depth_pil, dtype=np.uint16)
            depthmap = depth_raw.view(np.float16).astype(np.float32)

        depthmap[~np.isfinite(depthmap)] = 0.0
        depthmap[depthmap < 0] = 0.0
        return depthmap

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
    def quaternion_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def object_name_to_object_id(object_name: str) -> str:
        if object_name.startswith("google_") and object_name.endswith("_meshes_model"):
            return f"google/{object_name[len('google_'):-len('_meshes_model')]}/meshes"

        prefix_map = {
            "freepose_obj_ycbv_": "ycbv",
            "freepose_obj_handal_": "handal",
            "freepose_obj_hope_": "hope",
            "freepose_obj_rupac_": "rupac",
        }
        for prefix, dataset_name in prefix_map.items():
            if object_name.startswith(prefix):
                return f"{dataset_name}/{object_name[len(prefix):]}"

        raise KeyError(f"Unsupported object name: {object_name}")

    @staticmethod
    def object_id_to_render_dirname(object_id: str) -> str:
        return object_id.replace("/", "__")

    @classmethod
    def object_name_to_render_dirname(cls, object_name: str) -> str:
        return cls.object_id_to_render_dirname(cls.object_name_to_object_id(object_name))

    @classmethod
    def normalize_extrinsics_and_world_points(
        cls,
        extrinsics: np.ndarray,
        world_points: List[np.ndarray],
        point_masks: List[np.ndarray],
        scale_by_points: bool = True,
    ) -> Tuple[np.ndarray, float, List[np.ndarray]]:
        extrinsics = np.asarray(extrinsics, dtype=np.float32)

        bsz = 1
        seq_len = extrinsics.shape[0]
        extrinsics_homog = np.concatenate(
            [
                extrinsics[None, ...],
                np.zeros((bsz, seq_len, 1, 4), dtype=np.float32),
            ],
            axis=-2,
        )
        extrinsics_homog[:, :, 3, 3] = 1.0

        first_cam_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])[0]
        normalized_extrinsics = np.matmul(extrinsics_homog[0], first_cam_inv[None, ...])[:, :3]

        first_R = extrinsics[0, :3, :3]
        first_t = extrinsics[0, :3, 3]
        transformed_world_points = [
            (points @ first_R.T) + first_t.reshape(1, 1, 3) for points in world_points
        ]

        avg_scale = 1.0
        if scale_by_points:
            dist_sum = 0.0
            valid_count = 0.0
            for points, mask in zip(transformed_world_points, point_masks):
                dist = np.linalg.norm(points, axis=-1)
                mask_f = mask.astype(np.float32)
                dist_sum += float((dist * mask_f).sum())
                valid_count += float(mask_f.sum())
            avg_scale = np.clip(dist_sum / (valid_count + 1e-3), 1e-6, 1e6)
            transformed_world_points = [points / avg_scale for points in transformed_world_points]
            normalized_extrinsics[:, :3, 3] = normalized_extrinsics[:, :3, 3] / avg_scale

        return normalized_extrinsics.astype(np.float32), float(avg_scale), transformed_world_points

    @classmethod
    def normalize_extrinsics_and_object_pose(
        cls,
        extrinsics: np.ndarray,
        world_points: List[np.ndarray],
        point_masks: List[np.ndarray],
        object_rotation: np.ndarray,
        object_translation: np.ndarray,
        scale_by_points: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, List[np.ndarray]]:
        extrinsics = np.asarray(extrinsics, dtype=np.float32)
        object_rotation = np.asarray(object_rotation, dtype=np.float32)
        object_translation = np.asarray(object_translation, dtype=np.float32)

        normalized_extrinsics, avg_scale, transformed_world_points = cls.normalize_extrinsics_and_world_points(
            extrinsics=extrinsics,
            world_points=world_points,
            point_masks=point_masks,
            scale_by_points=scale_by_points,
        )

        first_R = extrinsics[0, :3, :3]
        first_t = extrinsics[0, :3, 3]
        normalized_object_rotation = first_R @ object_rotation
        normalized_object_translation = (object_translation @ first_R.T) + first_t
        if scale_by_points:
            normalized_object_translation = normalized_object_translation / avg_scale

        return (
            normalized_extrinsics.astype(np.float32),
            normalized_object_rotation.astype(np.float32),
            normalized_object_translation.astype(np.float32),
            float(avg_scale),
            transformed_world_points,
        )

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

    def _load_pose_lookup(self, pose_path: str) -> Dict[str, Dict[str, np.ndarray]]:
        pose_data = np.load(pose_path, allow_pickle=False)
        names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in pose_data["names"]]
        positions = pose_data["positions"].astype(np.float32)
        quats = pose_data["rot_quat_wxyz"].astype(np.float32)

        pose_lookup: Dict[str, Dict[str, np.ndarray]] = {}
        for idx, object_name in enumerate(names):
            translation = positions[idx]
            rotation = self.quaternion_wxyz_to_rotmat(quats[idx])
            pose_lookup[object_name] = {
                "object_rotation": rotation,
                "object_translation": translation.astype(np.float32),
            }
        return pose_lookup

    def _resolve_scene_image_path(self, run_name: str, cam_idx: int) -> str:
        if int(cam_idx) == 0:
            return osp.join(self.scene_root, run_name, "Main_Camera.jpg")
        return osp.join(self.scene_root, run_name, f"Main_Camera_({int(cam_idx)}).jpg")

    def _resolve_depth_path(self, run_name: str, cam_idx: int) -> str:
        if int(cam_idx) == 0:
            return osp.join(self.depth_root, run_name, "Main_Camera_depth.png")
        return osp.join(self.depth_root, run_name, f"Main_Camera_({int(cam_idx)})_depth.png")

    def _resolve_camera_path(self, run_name: str, cam_idx: int) -> str:
        if int(cam_idx) == 0:
            return osp.join(self.camera_root, run_name, "camera_Main_Camera.npz")
        return osp.join(self.camera_root, run_name, f"camera_Main_Camera_({int(cam_idx)}).npz")

    def _resolve_object_image_path(self, object_name: str, cam_idx: int) -> str:
        render_dir = self.object_name_to_render_dirname(object_name)
        return osp.join(self.object_root, render_dir, f"Main_Camera_({cam_idx})_rgb.png")

    def _load_camera(self, camera_path: str) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(camera_path, allow_pickle=False)
        extrinsic = np.asarray(data["extrinsics.opencv.worldToCamera16"], dtype=np.float32).reshape(4, 4)[:3]
        intrinsic = np.asarray(data["intrinsics.K_flat9"], dtype=np.float32).reshape(3, 3)
        return extrinsic, intrinsic

    def _verify_record_files(self, run_name: str, object_name: str) -> bool:
        for cam_idx in self.selected_views:
            if not osp.isfile(self._resolve_scene_image_path(run_name, cam_idx)):
                return False
            if not osp.isfile(self._resolve_depth_path(run_name, cam_idx)):
                return False
            if not osp.isfile(self._resolve_camera_path(run_name, cam_idx)):
                return False
        for obj_cam_idx in self.object_input_views:
            if not osp.isfile(self._resolve_object_image_path(object_name, obj_cam_idx)):
                return False
        return True

    def _build_records(self) -> List[Dict]:
        records: List[Dict] = []
        run_names = self._list_run_names()
        if self.debug:
            run_names = run_names[:1]

        missing_count = 0
        for run_name in run_names:
            scene_dir = osp.join(self.scene_root, run_name)
            pose_path = osp.join(self.pose_root, f"{run_name}.npz")
            camera_dir = osp.join(self.camera_root, run_name)
            depth_dir = osp.join(self.depth_root, run_name)

            if not (osp.isdir(scene_dir) and osp.isfile(pose_path) and osp.isdir(camera_dir) and osp.isdir(depth_dir)):
                missing_count += 1
                continue

            pose_lookup = self._load_pose_lookup(pose_path)
            if not pose_lookup:
                missing_count += 1
                continue

            for object_name in sorted(pose_lookup.keys()):
                try:
                    self.object_name_to_render_dirname(object_name)
                except KeyError:
                    continue
                if self.verify_files and not self._verify_record_files(run_name, object_name):
                    continue
                records.append(
                    {
                        "run_name": run_name,
                        "object_name": object_name,
                        "camera_indices": list(self.selected_views),
                        "object_cam_indices": list(self.object_input_views),
                        "pose": pose_lookup[object_name],
                    }
                )

        logging.info(
            f"Normalized 6D pose indexing done: {len(records)} valid object samples, "
            f"{missing_count} missing/invalid runs"
        )
        return records

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
        available_camera_indices = list(rec["camera_indices"])
        target_frame_num = int(img_per_seq) if img_per_seq is not None else 6
        target_frame_num = max(4, min(6, target_frame_num))
        if len(available_camera_indices) < target_frame_num:
            raise ValueError(
                f"Requested {target_frame_num} views but only {len(available_camera_indices)} are available"
            )
        if self.training or img_per_seq is not None:
            camera_indices = sorted(random.sample(available_camera_indices, target_frame_num))
        else:
            camera_indices = list(available_camera_indices[:target_frame_num])
        object_cam_indices = list(rec["object_cam_indices"])

        scene_images = []
        raw_depths = [] if self.load_point_map else None
        raw_cam_points = [] if self.load_point_map else None
        raw_world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []

        for cam_idx in camera_indices:
            scene_image_path = self._resolve_scene_image_path(run_name, cam_idx)
            depth_path = self._resolve_depth_path(run_name, cam_idx)
            camera_path = self._resolve_camera_path(run_name, cam_idx)

            scene_image = read_image_cv2(scene_image_path)
            if scene_image is None:
                raise FileNotFoundError(f"Failed to read scene image: {scene_image_path}")

            depth_map = self.read_encoded_depth(depth_path).astype(np.float32)
            extri, intri = self._load_camera(camera_path)
            world_coords_points, cam_coords_points, point_mask = depth_to_world_coords_points(
                depth_map, extri, intri
            )

            scene_images.append(scene_image)
            if self.load_point_map:
                raw_depths.append(depth_map)
                raw_cam_points.append(cam_coords_points.astype(np.float32))
            raw_world_points.append(world_coords_points.astype(np.float32))
            point_masks.append(point_mask.astype(bool))
            extrinsics.append(extri.astype(np.float32))
            intrinsics.append(intri.astype(np.float32))
            original_sizes.append(np.array(scene_image.shape[:2], dtype=np.int32))

        extrinsics_np = np.stack(extrinsics).astype(np.float32)
        (
            normalized_extrinsics,
            normalized_object_rotation,
            normalized_object_translation,
            avg_scale,
            normalized_world_points,
        ) = self.normalize_extrinsics_and_object_pose(
            extrinsics=extrinsics_np,
            world_points=raw_world_points,
            point_masks=point_masks,
            object_rotation=rec["pose"]["object_rotation"],
            object_translation=rec["pose"]["object_translation"],
            scale_by_points=self.scale_by_points,
        )

        object_images = []
        object_original_sizes = []
        for obj_cam_idx in object_cam_indices:
            object_image_path = self._resolve_object_image_path(object_name, obj_cam_idx)
            object_image = read_image_cv2(object_image_path)
            if object_image is None:
                raise FileNotFoundError(f"Failed to read object image: {object_image_path}")
            object_images.append(object_image)
            object_original_sizes.append(np.array(object_image.shape[:2], dtype=np.int32))

        batch = {
            "seq_name": f"6d_pose_normalize_{run_name}/{object_name}",
            "ids": np.array(camera_indices, dtype=np.int64),
            "frame_num": len(camera_indices),
            "images": scene_images,
            "object_images": object_images,
            "extrinsics": normalized_extrinsics,
            "intrinsics": intrinsics,
            "original_sizes": original_sizes,
            "object_original_sizes": object_original_sizes,
            "camera_indices": np.array(camera_indices, dtype=np.int64),
            "object_cam_indices": np.array(object_cam_indices, dtype=np.int64),
            "object_name": object_name,
            "object_id": self.object_name_to_object_id(object_name),
            "run_name": run_name,
            "skip_normalization": True,
            "object_rotation": normalized_object_rotation,
            "object_translation": normalized_object_translation,
            "object_srt": np.concatenate(
                [normalized_object_rotation.reshape(-1), normalized_object_translation], axis=0
            ).astype(np.float32),
            "normalization_scale": np.array([avg_scale], dtype=np.float32),
        }

        if self.load_point_map:
            scale = avg_scale if self.scale_by_points else 1.0
            normalized_cam_points = [
                (points / scale).astype(np.float32) for points in raw_cam_points
            ]
            normalized_depths = [
                (depth / scale).astype(np.float32) for depth in raw_depths
            ]
            batch.update(
                {
                    "depths": normalized_depths,
                    "cam_points": normalized_cam_points,
                    "world_points": [points.astype(np.float32) for points in normalized_world_points],
                    "point_masks": point_masks,
                    "object_masks": point_masks,
                }
            )

        return batch
