from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from data.datasets.mixed_real_pose_normalize.base import MixedPoseNormalizeBase


class YCBVMultiPoseNormalizeDataset(MixedPoseNormalizeBase):
    dataset_name = "ycbv"
    symmetry_dataset_name = "YCBVCameraPose"

    def __init__(
        self,
        common_conf,
        DATA_ROOT: str = "/mnt/train-data-4-hdd/yian/freepose/datasets_real/ycbv",
        SPLIT_ROOT: Optional[str] = None,
        OBJECT_IMAGE_ROOT: str = "/mnt/train-data-4-hdd/yian/freepose/datasets_real/ycbv/ycbv_aligned_object_refs",
        ALIGN_JSON: str = "/mnt/train-data-4-hdd/yian/freepose/baseline_0503/dataset_align.json",
        MODELS_INFO_PATH: Optional[str] = None,
        split: str = "train_real",
        **kwargs,
    ):
        self.data_root = Path(DATA_ROOT)
        self.split_root = Path(SPLIT_ROOT) if SPLIT_ROOT else self.data_root / split
        self.object_image_root = Path(OBJECT_IMAGE_ROOT)
        align = self._load_json(Path(ALIGN_JSON))["datasets"]["ycbv"]
        self.obj_id_to_category = {int(k): str(v) for k, v in align["obj_id_to_category"].items()}
        self.global_r_align = np.asarray(align["global_R_align"], dtype=np.float32).reshape(3, 3)
        self.r_align_overrides = {
            int(k): np.asarray(v, dtype=np.float32).reshape(3, 3)
            for k, v in align.get("per_object_overrides", {}).items()
        }
        models_info_path = Path(MODELS_INFO_PATH) if MODELS_INFO_PATH else self.data_root / "models" / "models_info.json"
        self.models_info = self._load_json(models_info_path) if models_info_path.is_file() else {}
        super().__init__(common_conf=common_conf, split=split, **kwargs)

    def _build_object_records(self):
        records = {}
        for object_dir in sorted(self.object_image_root.glob("obj_*")):
            if not object_dir.is_dir():
                continue
            object_id = int(object_dir.name.removeprefix("obj_"))
            ids = sorted(int(p.stem) for p in (object_dir / "rgb").glob("*.png") if p.stem.isdigit())
            if ids:
                records[object_id] = {
                    "object_id": object_id,
                    "object_name": f"obj_{object_id:06d}",
                    "object_dir": object_dir,
                    "image_ids": ids,
                    "category": self.obj_id_to_category.get(object_id, ""),
                }
        return records

    def _rgb_path(self, scene_dir: Path, image_id: int) -> Path:
        png = scene_dir / "rgb" / f"{image_id:06d}.png"
        jpg = scene_dir / "rgb" / f"{image_id:06d}.jpg"
        return png if png.is_file() or not jpg.is_file() else jpg

    def _build_records(self):
        groups = {}
        for scene_dir in sorted(p for p in self.split_root.iterdir() if p.is_dir()):
            gt_path = scene_dir / "scene_gt.json"
            cam_path = scene_dir / "scene_camera.json"
            if not gt_path.is_file() or not cam_path.is_file():
                continue
            scene_gt = self._load_json(gt_path)
            for image_id_str, gts in scene_gt.items():
                image_id = int(image_id_str)
                for object_index, gt in enumerate(gts):
                    object_id = int(gt["obj_id"])
                    if object_id not in self.object_records:
                        continue
                    rgb_path = self._rgb_path(scene_dir, image_id)
                    depth_path = scene_dir / "depth" / f"{image_id:06d}.png"
                    mask_path = scene_dir / "mask_visib" / f"{image_id:06d}_{object_index:06d}.png"
                    if self.verify_files and (not rgb_path.is_file() or not depth_path.is_file()):
                        continue
                    key = (scene_dir.name, object_id)
                    groups.setdefault(
                        key,
                        {
                            "scene_name": scene_dir.name,
                            "object_key": object_id,
                            "object_name": f"obj_{object_id:06d}",
                            "category": self.obj_id_to_category.get(object_id, ""),
                            "frames": [],
                        },
                    )
                    groups[key]["frames"].append(
                        {
                            "image_id": image_id,
                            "rgb_path": rgb_path,
                            "depth_path": depth_path,
                            "mask_path": mask_path if mask_path.is_file() else None,
                            "scene_camera_path": cam_path,
                            "scene_gt_path": gt_path,
                            "object_index": object_index,
                            "object_id": object_id,
                        }
                    )
                    if self._max_records_ready(groups):
                        return [v for v in groups.values() if len(v["frames"]) >= self.num_scene_views]
        return [v for v in groups.values() if len(v["frames"]) >= self.num_scene_views]

    def _read_depth(self, frame_rec):
        cam = self._load_json(frame_rec["scene_camera_path"])[str(int(frame_rec["image_id"]))]
        depth = np.asarray(Image.open(frame_rec["depth_path"]), dtype=np.float32)
        depth = depth * float(cam.get("depth_scale", 1.0)) / 1000.0
        depth[~np.isfinite(depth)] = 0.0
        return depth.astype(np.float32)

    def _camera_matrices(self, frame_rec):
        cam = self._load_json(frame_rec["scene_camera_path"])[str(int(frame_rec["image_id"]))]
        r = np.asarray(cam["cam_R_w2c"], dtype=np.float32).reshape(3, 3)
        t = np.asarray(cam["cam_t_w2c"], dtype=np.float32).reshape(3) / 1000.0
        k = np.asarray(cam["cam_K"], dtype=np.float32).reshape(3, 3)
        return np.concatenate([r, t[:, None]], axis=1).astype(np.float32), k

    def _object_model_to_world(self, frame_rec, rec):
        scene_gt = self._load_json(frame_rec["scene_gt_path"])
        gt = scene_gt[str(int(frame_rec["image_id"]))][int(frame_rec["object_index"])]
        cam = self._load_json(frame_rec["scene_camera_path"])[str(int(frame_rec["image_id"]))]
        object_id = int(gt["obj_id"])
        r_align = self.r_align_overrides.get(object_id, self.global_r_align)
        r_m2c = np.asarray(gt["cam_R_m2c"], dtype=np.float32).reshape(3, 3) @ r_align.T
        t_m2c = np.asarray(gt["cam_t_m2c"], dtype=np.float32).reshape(3) / 1000.0
        r_w2c = np.asarray(cam["cam_R_w2c"], dtype=np.float32).reshape(3, 3)
        t_w2c = np.asarray(cam["cam_t_w2c"], dtype=np.float32).reshape(3) / 1000.0
        r_c2w = r_w2c.T
        t_c2w = -(r_c2w @ t_w2c)
        return (r_c2w @ r_m2c).astype(np.float32), (r_c2w @ t_m2c + t_c2w).astype(np.float32)

    def _object_size_metric(self, object_key, rec):
        del rec
        object_id = int(object_key)
        info = self.models_info.get(str(object_id), {})
        if not all(k in info for k in ("size_x", "size_y", "size_z")):
            return None
        size = np.array([info["size_x"], info["size_y"], info["size_z"]], dtype=np.float32) / 1000.0
        r_align = self.r_align_overrides.get(object_id, self.global_r_align)
        return (np.abs(r_align) @ size).astype(np.float32)
