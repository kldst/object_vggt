# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import os.path as osp
import random
from typing import Dict, List, Optional

import numpy as np

from data.base_dataset import BaseDataset
from data.dataset_util import read_image_cv2


class SixDPoseDataset(BaseDataset):
    """
    Dataset for 6D object pose + object-space point map supervision.

    Expected directory layout under AUTOMATIC_DATASET_DIR/<split>:
      - black_object_input/run_xxxx/input_images_xxx/<object_name>/Main_Camera_(k).jpg
      - calculate_output_srt/run_xxxx/input_images_xxx/obj_to_vggt_srt.json
      - object_pc/run_xxxx/input_images_xxx/<object_name>/cam{k}_decoded_map.npz
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        AUTOMATIC_DATASET_DIR: str = "/mnt/train-data-4-hdd/yian/6dpose_obj/automatic_dataset",
        len_train: int = 100000,
        len_test: int = 10000,
        verify_files: bool = True,
        min_num_frames: int = 1,
        only_run_name: str = "",
        only_run_names: Optional[List[str]] = None,
        only_run_start: Optional[str] = None,
        only_run_end: Optional[str] = None,
    ):
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random

        self.split = split
        self.dataset_root = osp.join(AUTOMATIC_DATASET_DIR, split)
        self.image_root = osp.join(self.dataset_root, "black_object_input")
        self.srt_root = osp.join(self.dataset_root, "calculate_output_srt")
        self.pc_root = osp.join(self.dataset_root, "object_pc")

        self.verify_files = verify_files
        self.min_num_frames = min_num_frames
        self.only_run_name = (only_run_name or "").strip()
        self.only_run_names = [x.strip() for x in (only_run_names or []) if str(x).strip()]
        if self.only_run_name:
            # Keep backward-friendly single-run flag while allowing multi-run list.
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
        self.records_by_frame_num = self._build_records_by_frame_num(self.records)

        if self.sequence_list_len == 0:
            raise RuntimeError(
                "No valid 6dpose samples found. "
                f"Checked roots: image={self.image_root}, srt={self.srt_root}, pc={self.pc_root}"
            )

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: 6DPose sequence count: {self.sequence_list_len}")
        logging.info(f"{status}: 6DPose dataset length: {len(self)}")

    def _build_records(self) -> List[Dict]:
        records: List[Dict] = []

        if not osp.isdir(self.image_root):
            raise FileNotFoundError(f"Image root not found: {self.image_root}")

        run_names = sorted([d for d in os.listdir(self.image_root) if d.startswith("run_")])
        if self.only_run_names:
            only_set = set(self.only_run_names)
            run_names = [r for r in run_names if r in only_set]
            if len(run_names) == 0:
                raise RuntimeError(
                    f"No runs matched only_run_names={sorted(only_set)} under image_root={self.image_root}"
                )
        if self.only_run_start_idx is not None or self.only_run_end_idx is not None:
            run_names_in_range = []
            for run_name in run_names:
                run_idx = self._parse_run_index(run_name)
                if run_idx is None:
                    continue
                if self.only_run_start_idx is not None and run_idx < self.only_run_start_idx:
                    continue
                if self.only_run_end_idx is not None and run_idx > self.only_run_end_idx:
                    continue
                run_names_in_range.append(run_name)
            run_names = run_names_in_range
            if len(run_names) == 0:
                raise RuntimeError(
                    "No runs matched range filter "
                    f"[{self.only_run_start_idx}, {self.only_run_end_idx}] under image_root={self.image_root}"
                )
        if self.debug:
            run_names = run_names[:1]

        missing_count = 0

        for run_name in run_names:
            run_image_dir = osp.join(self.image_root, run_name)
            if not osp.isdir(run_image_dir):
                continue

            input_names = sorted([d for d in os.listdir(run_image_dir) if d.startswith("input_images_")])

            for input_name in input_names:
                srt_json_path = osp.join(self.srt_root, run_name, input_name, "obj_to_vggt_srt.json")
                if not osp.isfile(srt_json_path):
                    missing_count += 1
                    continue

                try:
                    with open(srt_json_path, "r", encoding="utf-8") as f:
                        srt_data = json.load(f)
                except Exception:
                    missing_count += 1
                    continue

                camera_indices = srt_data.get("camera_order", {}).get("camera_indices", [])
                if not isinstance(camera_indices, list) or len(camera_indices) < self.min_num_frames:
                    continue

                object_list = srt_data.get("objects", [])
                if not isinstance(object_list, list):
                    continue

                for obj in object_list:
                    object_name = obj.get("object_name", None)
                    obj_srt = obj.get("obj_to_vggt_cam1", None)
                    if object_name is None or obj_srt is None:
                        continue

                    image_obj_dir = osp.join(self.image_root, run_name, input_name, object_name)
                    pc_obj_dir = osp.join(self.pc_root, run_name, input_name, object_name)

                    if not osp.isdir(image_obj_dir) or not osp.isdir(pc_obj_dir):
                        continue

                    if self.verify_files:
                        valid = True
                        for cam_idx in camera_indices:
                            image_path = self._resolve_image_path(image_obj_dir, int(cam_idx))
                            pc_path = osp.join(pc_obj_dir, f"cam{int(cam_idx)}_decoded_map.npz")
                            if not (osp.isfile(image_path) and osp.isfile(pc_path)):
                                valid = False
                                break
                        if not valid:
                            continue

                    records.append(
                        {
                            "run_name": run_name,
                            "input_name": input_name,
                            "object_name": object_name,
                            "camera_indices": [int(x) for x in camera_indices],
                            "srt": obj_srt,
                        }
                    )

        logging.info(
            f"6DPose indexing done: {len(records)} valid object samples, {missing_count} missing/invalid input folders"
        )
        return records

    @staticmethod
    def _parse_run_index(run_name: Optional[str]) -> Optional[int]:
        if run_name is None:
            return None
        s = str(run_name).strip()
        if not s:
            return None
        if s.isdigit():
            return int(s)
        if s.startswith("run_"):
            tail = s[4:]
            if tail.isdigit():
                return int(tail)
        raise ValueError(f"Invalid run spec '{run_name}'. Expected e.g. 'run_0000' or '0'.")

    @staticmethod
    def _build_records_by_frame_num(records: List[Dict]) -> Dict[int, List[int]]:
        records_by_len: Dict[int, List[int]] = {}
        for idx, rec in enumerate(records):
            n = len(rec.get("camera_indices", []))
            records_by_len.setdefault(int(n), []).append(idx)
        return records_by_len

    @staticmethod
    def _decode_obj_srt(obj_srt: Dict) -> Dict[str, np.ndarray]:
        # Keep NumPy side in float32 for compatibility (some NumPy builds do not support bfloat16).
        # Cast to torch.bfloat16 later in composed_dataset.py.
        s = np.float32(obj_srt["s"])
        t = np.array(obj_srt["t"], dtype=np.float32)
        r = np.array(obj_srt["R_flat9_row_major"], dtype=np.float32).reshape(3, 3)

        srt_vec = np.concatenate(
            [np.array([s], dtype=np.float32), r.reshape(-1), t.reshape(-1)], axis=0
        ).astype(np.float32)

        return {
            "object_scale": np.array([s], dtype=np.float32),
            "object_rotation": r,
            "object_translation": t,
            "object_srt": srt_vec,
        }

    @staticmethod
    def _resolve_image_path(image_obj_dir: str, cam_idx: int) -> str:
        """
        Resolve camera image path with compatibility fallback.
        Some samples use Main_Camera.jpg for camera index 0.
        """
        primary = osp.join(image_obj_dir, f"Main_Camera_({cam_idx}).jpg")
        if osp.isfile(primary):
            return primary

        if int(cam_idx) == 0:
            fallback = osp.join(image_obj_dir, "Main_Camera.jpg")
            if osp.isfile(fallback):
                return fallback

        return primary

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        del seq_name, aspect_ratio

        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_index is None:
            seq_index = 0

        rec = self.records[seq_index % self.sequence_list_len]

        # Keep sequence length unchanged: if img_per_seq is requested, select a record
        # whose camera_indices length exactly matches img_per_seq.
        if ids is None and img_per_seq is not None and int(img_per_seq) > 0:
            target_len = int(img_per_seq)
            matched_indices = self.records_by_frame_num.get(target_len, [])
            if len(matched_indices) == 0:
                available = sorted(self.records_by_frame_num.keys())
                raise ValueError(
                    f"No 6dpose samples with frame_num={target_len}. Available frame_num set: {available}"
                )
            if self.training:
                rec = self.records[random.choice(matched_indices)]
            else:
                rec = self.records[matched_indices[seq_index % len(matched_indices)]]

        run_name = rec["run_name"]
        input_name = rec["input_name"]
        object_name = rec["object_name"]
        if ids is not None:
            camera_indices = [int(x) for x in ids]
        else:
            camera_indices = rec["camera_indices"]

        image_obj_dir = osp.join(self.image_root, run_name, input_name, object_name)
        pc_obj_dir = osp.join(self.pc_root, run_name, input_name, object_name)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []

        for cam_idx in camera_indices:
            image_path = self._resolve_image_path(image_obj_dir, cam_idx)
            pc_path = osp.join(pc_obj_dir, f"cam{cam_idx}_decoded_map.npz")

            image = read_image_cv2(image_path)
            if image is None:
                raise FileNotFoundError(f"Failed to read image: {image_path}")

            if not osp.isfile(pc_path):
                raise FileNotFoundError(f"Point-map file not found: {pc_path}")

            npz_data = np.load(pc_path)
            if "map_xyz" not in npz_data:
                raise KeyError(f"Key 'map_xyz' not found in: {pc_path}")
            map_xyz = npz_data["map_xyz"].astype(np.float32)

            if image.shape[:2] != map_xyz.shape[:2]:
                raise ValueError(
                    f"Image/point-map shape mismatch for {image_path}: "
                    f"image={image.shape[:2]}, point_map={map_xyz.shape[:2]}"
                )

            h, w = image.shape[:2]

            valid_mask = np.isfinite(map_xyz).all(axis=-1) & (np.linalg.norm(map_xyz, axis=-1) > 0)
            depth = map_xyz[..., 2].copy()
            depth[~valid_mask] = 0.0

            extri = np.concatenate([np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)], axis=1)
            intri = np.array(
                [[1.0, 0.0, w / 2.0], [0.0, 1.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32
            )

            images.append(image)
            depths.append(depth)
            cam_points.append(map_xyz)
            world_points.append(map_xyz)
            point_masks.append(valid_mask)
            extrinsics.append(extri)
            intrinsics.append(intri)
            original_sizes.append(np.array([h, w], dtype=np.int32))

        srt_dict = self._decode_obj_srt(rec["srt"])

        batch = {
            "seq_name": f"6dpose_{run_name}/{input_name}/{object_name}",
            "ids": np.array(camera_indices, dtype=np.int64),
            "frame_num": len(camera_indices),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "object_masks": point_masks,
            "original_sizes": original_sizes,
            "camera_indices": np.array(camera_indices, dtype=np.int64),
            "object_name": object_name,
            "run_name": run_name,
            "input_name": input_name,
            "skip_normalization": True,
        }
        batch.update(srt_dict)
        return batch
