from pathlib import Path

import numpy as np
from PIL import Image

from data.datasets.mixed_real_pose_normalize.base import MixedPoseNormalizeBase


class Real275MultiPoseNormalizeDataset(MixedPoseNormalizeBase):
    dataset_name = "real275"
    symmetry_dataset_name = "Real275CameraPose"
    K_REAL = np.array(
        [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    def __init__(
        self,
        common_conf,
        DATA_ROOT: str = "/mnt/train-data-4-hdd/yian/freepose/real275",
        SPLIT_ROOT: str = "/mnt/train-data-4-hdd/yian/freepose/real275/real_train",
        GT_ROOT: str = "/mnt/train-data-4-hdd/yian/freepose/real275/gts/real_train_umeyama",
        OBJECT_IMAGE_ROOT: str = "/mnt/train-data-4-hdd/yian/freepose/real275/real275_aligned_object_refs",
        ALIGN_JSON: str = "/mnt/train-data-4-hdd/yian/freepose/baseline_0503/dataset_align.json",
        split: str = "train",
        **kwargs,
    ):
        self.data_root = Path(DATA_ROOT)
        self.split_root = Path(SPLIT_ROOT)
        self.gt_root = Path(GT_ROOT)
        self.object_image_root = Path(OBJECT_IMAGE_ROOT)
        align = self._load_json(Path(ALIGN_JSON))["datasets"]["real275"]
        self.class_id_to_name = {int(k): str(v) for k, v in align["class_id_to_name"].items()}
        self.r_align_by_class_id = {
            int(k): np.asarray(v["R_align"], dtype=np.float32).reshape(3, 3)
            for k, v in align["classes"].items()
        }
        super().__init__(common_conf=common_conf, split=split, **kwargs)

    def _build_object_records(self):
        records = {}
        for idx, object_dir in enumerate(sorted(p for p in self.object_image_root.iterdir() if p.is_dir()), start=1):
            ids = sorted(int(p.stem) for p in (object_dir / "rgb").glob("*.png") if p.stem.isdigit())
            if ids:
                meta = self._load_json(object_dir / "metadata.json") if (object_dir / "metadata.json").is_file() else {}
                records[object_dir.name] = {
                    "object_id": idx,
                    "object_name": object_dir.name,
                    "object_dir": object_dir,
                    "image_ids": ids,
                    "category": str(meta.get("class_name", object_dir.name.split("_", 1)[0])),
                    "class_id": int(meta.get("class_id", 0)),
                    "metadata": meta,
                }
        return records

    def _build_records(self):
        groups = {}
        for scene_dir in sorted(self.split_root.glob("scene_*")):
            for rgb_path in sorted(scene_dir.glob("*_color.png")):
                frame_id = rgb_path.name.split("_", 1)[0]
                gt_path = self.gt_root / f"results_real_train_{scene_dir.name}_{frame_id}.pkl"
                depth_path = scene_dir / f"{frame_id}_depth.png"
                mask_path = scene_dir / f"{frame_id}_mask.png"
                if self.verify_files and (not gt_path.is_file() or not depth_path.is_file() or not mask_path.is_file()):
                    continue
                gt = self._load_pickle(gt_path)
                for object_index, object_name in enumerate(gt["model_names"]):
                    object_name = str(object_name)
                    if object_name not in self.object_records:
                        continue
                    class_id = int(gt["class_ids"][object_index])
                    key = (scene_dir.name, object_name)
                    groups.setdefault(
                        key,
                        {
                            "scene_name": scene_dir.name,
                            "object_key": object_name,
                            "object_name": object_name,
                            "category": self.class_id_to_name.get(class_id, ""),
                            "frames": [],
                        },
                    )
                    groups[key]["frames"].append(
                        {
                            "image_id": int(frame_id),
                            "rgb_path": rgb_path,
                            "depth_path": depth_path,
                            "mask_path": mask_path,
                            "gt_path": gt_path,
                            "object_index": int(object_index),
                            "inst_id": int(gt["inst_ids"][object_index]),
                            "class_id": class_id,
                        }
                    )
                    if self._max_records_ready(groups):
                        return [v for v in groups.values() if len(v["frames"]) >= self.num_scene_views]
        return [v for v in groups.values() if len(v["frames"]) >= self.num_scene_views]

    def _read_depth(self, frame_rec):
        depth = np.asarray(Image.open(frame_rec["depth_path"]), dtype=np.float32) / 1000.0
        depth[~np.isfinite(depth)] = 0.0
        return depth.astype(np.float32)

    def _camera_matrices(self, frame_rec):
        extrinsic = np.concatenate(
            [np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)],
            axis=1,
        )
        return extrinsic, self.K_REAL.copy()

    def _object_model_to_world(self, frame_rec, rec):
        gt = self._load_pickle(frame_rec["gt_path"])
        idx = int(frame_rec["object_index"])
        r_align = self.r_align_by_class_id.get(int(frame_rec["class_id"]), np.eye(3, dtype=np.float32))
        r = np.asarray(gt["rotations"][idx], dtype=np.float32).reshape(3, 3) @ r_align.T
        t = np.asarray(gt["translations"][idx], dtype=np.float32).reshape(3)
        return r.astype(np.float32), t.astype(np.float32)

    def _object_size_metric(self, object_key, rec):
        del rec
        meta = self.object_records[object_key].get("metadata", {})
        bounds = meta.get("mesh", {}).get("centered_bounds")
        if bounds is None:
            return None
        bounds = np.asarray(bounds, dtype=np.float32)
        if bounds.shape != (2, 3):
            return None
        return (bounds[1] - bounds[0]).astype(np.float32)
