from pathlib import Path

import numpy as np
from PIL import Image

from data.datasets.mixed_real_pose_normalize.base import MixedPoseNormalizeBase


class HouseCat6DMultiPoseNormalizeDataset(MixedPoseNormalizeBase):
    dataset_name = "housecat6d"
    symmetry_dataset_name = "HouseCat6DCameraPose"

    def __init__(
        self,
        common_conf,
        DATA_ROOT: str = "/mnt/train-data-4-hdd/yian/freepose/housecat6d",
        OBJECT_IMAGE_ROOT: str = "/mnt/train-data-4-hdd/yian/freepose/housecat6d/housecat6d_aligned_object_refs",
        ALIGN_JSON: str = "/mnt/train-data-4-hdd/yian/freepose/baseline_0503/dataset_align.json",
        split: str = "train",
        **kwargs,
    ):
        self.data_root = Path(DATA_ROOT)
        self.object_image_root = Path(OBJECT_IMAGE_ROOT)
        align = self._load_json(Path(ALIGN_JSON))["datasets"]["housecat6d"]
        self.category_name_to_id = {str(k): int(v) for k, v in align["category_name_to_id"].items()}
        self.category_id_to_name = {v: k for k, v in self.category_name_to_id.items()}
        self.r_align_by_category = {
            str(k): np.asarray(v["R_align"], dtype=np.float32).reshape(3, 3)
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
                    "category": str(meta.get("class_name", object_dir.name.split("-", 1)[0])),
                    "metadata": meta,
                }
        return records

    def _scene_meta(self, scene_dir: Path):
        out = {}
        with (scene_dir / "meta.txt").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    inst_id, class_id, name = line.strip().split(maxsplit=2)
                    out[name] = {
                        "inst_id": int(inst_id),
                        "class_id": int(class_id),
                        "category": self.category_id_to_name.get(int(class_id), name.split("-", 1)[0]),
                    }
        return out

    def _build_records(self):
        groups = {}
        for scene_dir in sorted(self.data_root.glob("scene*")):
            if not (scene_dir / "meta.txt").is_file() or not (scene_dir / "intrinsics.txt").is_file():
                continue
            meta = self._scene_meta(scene_dir)
            for label_path in sorted((scene_dir / "labels").glob("*_label.pkl")):
                image_id = int(label_path.name.split("_", 1)[0])
                rgb_path = scene_dir / "rgb" / f"{image_id:06d}.png"
                depth_path = scene_dir / "depth" / f"{image_id:06d}.png"
                if self.verify_files and (not rgb_path.is_file() or not depth_path.is_file()):
                    continue
                label = self._load_pickle(label_path)
                for object_index, object_name in enumerate([str(x) for x in label["model_list"]]):
                    if object_name not in self.object_records:
                        continue
                    item = meta.get(object_name, {})
                    mask_path = scene_dir / "instance" / f"{image_id:06d}_{object_name}.png"
                    key = (scene_dir.name, object_name)
                    groups.setdefault(
                        key,
                        {
                            "scene_name": scene_dir.name,
                            "object_key": object_name,
                            "object_name": object_name,
                            "category": item.get("category", ""),
                            "frames": [],
                        },
                    )
                    groups[key]["frames"].append(
                        {
                            "image_id": image_id,
                            "rgb_path": rgb_path,
                            "depth_path": depth_path,
                            "mask_path": mask_path if mask_path.is_file() else None,
                            "label_path": label_path,
                            "intrinsics_path": scene_dir / "intrinsics.txt",
                            "object_index": int(object_index),
                            "inst_id": item.get("inst_id"),
                            "class_id": int(label["class_ids"][object_index]),
                            "category": item.get("category", ""),
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
        k = np.loadtxt(frame_rec["intrinsics_path"], dtype=np.float32).reshape(3, 3)
        extrinsic = np.concatenate(
            [np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)],
            axis=1,
        )
        return extrinsic, k

    def _object_model_to_world(self, frame_rec, rec):
        label = self._load_pickle(frame_rec["label_path"])
        idx = int(frame_rec["object_index"])
        category = str(frame_rec.get("category", rec.get("category", "")))
        r_align = self.r_align_by_category.get(category, np.eye(3, dtype=np.float32))
        r = np.asarray(label["rotations"][idx], dtype=np.float32).reshape(3, 3) @ r_align.T
        t = np.asarray(label["translations"][idx], dtype=np.float32).reshape(3)
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
