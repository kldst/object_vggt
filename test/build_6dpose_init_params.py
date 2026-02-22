#!/usr/bin/env python3
import argparse
import json
import os
import os.path as osp
from typing import Dict, List, Tuple

import numpy as np


def collect_object_srt(srt_root: str) -> Dict[str, List[Tuple[float, np.ndarray]]]:
    per_object: Dict[str, List[Tuple[float, np.ndarray]]] = {}
    run_names = sorted([d for d in os.listdir(srt_root) if d.startswith("run_")])

    for run_name in run_names:
        run_dir = osp.join(srt_root, run_name)
        if not osp.isdir(run_dir):
            continue

        input_names = sorted([d for d in os.listdir(run_dir) if d.startswith("input_images_")])
        for input_name in input_names:
            srt_json = osp.join(run_dir, input_name, "obj_to_vggt_srt.json")
            if not osp.isfile(srt_json):
                continue

            try:
                with open(srt_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            for obj in data.get("objects", []):
                object_name = obj.get("object_name")
                obj_srt = obj.get("obj_to_vggt_cam1")
                if object_name is None or obj_srt is None:
                    continue

                s = float(obj_srt["s"])
                t = np.array(obj_srt["t"], dtype=np.float32)
                per_object.setdefault(object_name, []).append((s, t))

    return per_object


def build_init_arrays(
    per_object: Dict[str, List[Tuple[float, np.ndarray]]],
    use_median_xy: bool = True,
    use_log_tz: bool = False,
) -> Dict[str, np.ndarray]:
    object_names = sorted(per_object.keys())
    object_to_idx = {n: i for i, n in enumerate(object_names)}
    n_obj = len(object_names)

    init_scale = np.zeros((n_obj, 1), dtype=np.float32)
    init_translate = np.zeros((n_obj, 3), dtype=np.float32)
    init_rot6d = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32), (n_obj, 1))

    for name in object_names:
        idx = object_to_idx[name]
        samples = per_object[name]
        s_arr = np.array([x[0] for x in samples], dtype=np.float32)
        t_arr = np.stack([x[1] for x in samples], axis=0).astype(np.float32)

        init_scale[idx, 0] = float(np.median(s_arr))

        if use_median_xy:
            init_translate[idx, 0] = float(np.median(t_arr[:, 0]))
            init_translate[idx, 1] = float(np.median(t_arr[:, 1]))
        else:
            init_translate[idx, 0] = float(np.mean(t_arr[:, 0]))
            init_translate[idx, 1] = float(np.mean(t_arr[:, 1]))

        tz = t_arr[:, 2]
        if use_log_tz:
            tz = np.maximum(tz, 1e-6)
            init_translate[idx, 2] = float(np.exp(np.mean(np.log(tz))))
        else:
            init_translate[idx, 2] = float(np.mean(tz))

    global_scale = np.array([float(np.mean(init_scale[:, 0]))], dtype=np.float32)
    global_translate = np.array([np.mean(init_translate[:, 0]), np.mean(init_translate[:, 1]), np.mean(init_translate[:, 2])], dtype=np.float32)
    global_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    return {
        "object_names": np.array(object_names, dtype=object),
        "init_scale_per_object": init_scale,
        "init_translate_per_object": init_translate,
        "init_rot6d_per_object": init_rot6d,
        "global_init_scale": global_scale,
        "global_init_translate": global_translate,
        "global_init_rot6d": global_rot6d,
        "object_count": np.array([n_obj], dtype=np.int32),
    }


def main():
    parser = argparse.ArgumentParser(description="Build per-object init params for 6D pose training.")
    parser.add_argument(
        "--automatic-dataset-dir",
        default="/mnt/train-data-4-hdd/yian/6dpose_obj/automatic_dataset",
        help="Path to automatic_dataset root.",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument(
        "--out",
        default="/mnt/train-data-4-hdd/yian/6dpose_obj/vggt/training/data/datasets/init_6dpose_params.npz",
        help="Output npz path.",
    )
    parser.add_argument("--mean-xy", action="store_true", default=False, help="Use mean for tx/ty instead of median.")
    parser.add_argument(
        "--log-tz",
        action="store_true",
        default=False,
        help="Use exp(mean(log(tz))) for tz instead of mean(tz).",
    )
    args = parser.parse_args()

    srt_root = osp.join(args.automatic_dataset_dir, args.split, "calculate_output_srt")
    if not osp.isdir(srt_root):
        raise FileNotFoundError(f"SRT root not found: {srt_root}")

    per_object = collect_object_srt(srt_root)
    if not per_object:
        raise RuntimeError(f"No valid SRT data found under: {srt_root}")

    arrays = build_init_arrays(
        per_object,
        use_median_xy=not args.mean_xy,
        use_log_tz=args.log_tz,
    )

    out_dir = osp.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(args.out, **arrays)

    print(f"Saved: {args.out}")
    print(f"object_count: {int(arrays['object_count'][0])}")
    print(f"global_init_scale: {arrays['global_init_scale'].tolist()}")
    print(f"global_init_translate: {arrays['global_init_translate'].tolist()}")
    print("first_5_object_names:", arrays["object_names"][:5].tolist())
    print("first_5_init_scale:", arrays["init_scale_per_object"][:5, 0].tolist())
    print("first_5_init_translate:", arrays["init_translate_per_object"][:5].tolist())


if __name__ == "__main__":
    main()
