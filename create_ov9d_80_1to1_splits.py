#!/usr/bin/env python3
"""Create OV9D single/multi train-val-test splits for 1:1 mixed training.

The generated manifests follow the existing baseline_0503 split JSON shape:

  single/{train,val,test}.json:
    scenes are oo3d9dsingle object-instance folders.

  multi/{train,val,test}.json:
    scenes are oo3d9dmulti folders with eligible_object_ids, where each eligible
    object has enough visible views and a matching oo3d9dsingle reference.

The default policy first splits object ids, then derives single and multi
manifests from the same split-specific object ids. This guarantees that
single/train and multi/train contain the same trainable object ids, and likewise
for val and test. A train_mix_1to1.json file records the recommended virtual
lengths for the existing ComposedDataset + len_train mechanism.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_DATA_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d")
DEFAULT_OUT_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/baseline_0503/splits_ov9d_80_1to1")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")


def parse_single_scene_name(scene_name: str) -> Tuple[str, str]:
    parts = scene_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected oo3d9dsingle folder name: {scene_name}")
    category = "_".join(parts[:-2])
    object_instance = "_".join(parts[:-1])
    return category, object_instance


def count_pngs(path: Path) -> int:
    return sum(1 for _ in path.glob("*.png")) if path.is_dir() else 0


def progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def largest_remainder_counts(group_sizes: Sequence[int], target_total: int) -> List[int]:
    """Allocate target_total across groups proportionally with integer counts."""
    total = sum(group_sizes)
    if total == 0:
        return [0 for _ in group_sizes]
    raw = [target_total * size / total for size in group_sizes]
    counts = [min(size, int(math.floor(value))) for size, value in zip(group_sizes, raw)]
    remaining = target_total - sum(counts)
    order = sorted(
        range(len(group_sizes)),
        key=lambda i: (raw[i] - math.floor(raw[i]), group_sizes[i]),
        reverse=True,
    )
    while remaining > 0:
        changed = False
        for i in order:
            if counts[i] < group_sizes[i]:
                counts[i] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            break
    return counts


def stratified_split_exact(
    records: List[Dict[str, Any]],
    key_name: str,
    train_ratio: float,
    val_ratio: float,
    rng: random.Random,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split records by a categorical key while matching global ratios exactly."""
    if not records:
        return {"train": [], "val": [], "test": []}

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[str(rec.get(key_name, ""))].append(rec)

    group_keys = sorted(groups)
    for key in group_keys:
        rng.shuffle(groups[key])

    total = len(records)
    train_total = int(round(total * train_ratio))
    val_total = int(round(total * val_ratio))
    if train_total + val_total > total:
        val_total = max(0, total - train_total)

    sizes = [len(groups[key]) for key in group_keys]
    train_counts = largest_remainder_counts(sizes, train_total)
    remaining_sizes = [size - count for size, count in zip(sizes, train_counts)]
    val_counts = largest_remainder_counts(remaining_sizes, val_total)

    splits = {"train": [], "val": [], "test": []}
    for key, n_train, n_val in zip(group_keys, train_counts, val_counts):
        items = groups[key]
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val :])

    for split_items in splits.values():
        rng.shuffle(split_items)
    return splits


def stratified_object_id_split(
    records: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    rng: random.Random,
) -> Dict[str, set[int]]:
    """Split object ids by category using the single-object records."""
    eligible_records = [rec for rec in records if "object_id" in rec and rec.get("has_fixed_reference_views", False)]
    grouped: Dict[str, List[int]] = defaultdict(list)
    for rec in eligible_records:
        grouped[str(rec["category"])].append(int(rec["object_id"]))

    split_records = []
    for category, object_ids in grouped.items():
        for object_id in sorted(set(object_ids)):
            split_records.append({"category": category, "object_id": object_id})

    split_items = stratified_split_exact(
        split_records,
        key_name="category",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        rng=rng,
    )
    return {split: {int(item["object_id"]) for item in items} for split, items in split_items.items()}


def has_fixed_object_views(scene_dir: Path, view_ids: Sequence[int]) -> bool:
    for image_id in view_ids:
        if not (scene_dir / "rgb" / f"{image_id:06d}.png").is_file():
            return False
        if not (scene_dir / "mask_visib" / f"{image_id:06d}_000000.png").is_file():
            return False
    return True


def build_single_records(
    data_root: Path,
    min_rgb: int,
    reference_view_ids: Sequence[int],
) -> Tuple[List[Dict[str, Any]], set[int]]:
    single_root = data_root / "oo3d9dsingle"
    if not single_root.is_dir():
        raise FileNotFoundError(f"single root not found: {single_root}")

    name_to_oid_path = data_root / "name2oid.json"
    name_to_oid = load_json(name_to_oid_path) if name_to_oid_path.is_file() else {}

    records: List[Dict[str, Any]] = []
    reference_object_ids: set[int] = set()
    scene_dirs = sorted(p for p in single_root.iterdir() if p.is_dir())
    progress(f"[single] scanning {len(scene_dirs)} scene folders")
    for idx, scene_dir in enumerate(scene_dirs, start=1):
        try:
            category, object_instance = parse_single_scene_name(scene_dir.name)
        except ValueError:
            continue
        rgb_count = count_pngs(scene_dir / "rgb")
        if rgb_count < min_rgb:
            continue
        object_id = name_to_oid.get(object_instance)
        rec = {
            "scene_name": scene_dir.name,
            "category": category,
            "object_instance": object_instance,
            "relative_path": f"oo3d9dsingle/{scene_dir.name}",
            "rgb_count": rgb_count,
        }
        if object_id is not None:
            rec["object_id"] = int(object_id)
            if has_fixed_object_views(scene_dir, reference_view_ids):
                rec["has_fixed_reference_views"] = True
                reference_object_ids.add(int(object_id))
            else:
                rec["has_fixed_reference_views"] = False
        records.append(rec)
        if idx % 1000 == 0:
            progress(f"[single] {idx}/{len(scene_dirs)} scanned, valid={len(records)}")
    return records, reference_object_ids


def invert_name_to_oid(data_root: Path) -> Dict[int, str]:
    name_to_oid = load_json(data_root / "name2oid.json")
    return {int(oid): str(name) for name, oid in name_to_oid.items()}


def object_category_from_instance(instance_name: str) -> str:
    parts = instance_name.split("_")
    if len(parts) < 2:
        return instance_name
    return "_".join(parts[:-1])


def build_multi_records(
    data_root: Path,
    min_object_views: int,
    valid_reference_object_ids: Iterable[int],
) -> List[Dict[str, Any]]:
    multi_root = data_root / "oo3d9dmulti"
    if not multi_root.is_dir():
        raise FileNotFoundError(f"multi root not found: {multi_root}")

    oid_to_name = invert_name_to_oid(data_root)
    valid_refs = {int(x) for x in valid_reference_object_ids}
    category_to_id = {
        cat: idx + 1
        for idx, cat in enumerate(
            sorted({object_category_from_instance(name) for name in oid_to_name.values()})
        )
    }
    category_id_to_category = {category_id: category for category, category_id in category_to_id.items()}

    records: List[Dict[str, Any]] = []
    scene_dirs = sorted(p for p in multi_root.iterdir() if p.is_dir())
    progress(f"[multi] scanning {len(scene_dirs)} scene folders")
    for idx, scene_dir in enumerate(scene_dirs, start=1):
        if idx % 1000 == 0:
            progress(f"[multi] {idx}/{len(scene_dirs)} scanned, valid={len(records)}")
        scene_gt_path = scene_dir / "scene_gt.json"
        if not scene_gt_path.is_file():
            continue
        scene_gt = load_json(scene_gt_path)

        object_view_counts: Counter[int] = Counter()
        for gts in scene_gt.values():
            for gt in gts:
                object_view_counts[int(gt["obj_id"])] += 1

        object_ids = sorted(object_view_counts)
        object_id_to_category_id: Dict[str, int] = {}
        object_id_to_category: Dict[int, str] = {}
        for object_id in object_ids:
            instance_name = oid_to_name.get(object_id, "")
            category = object_category_from_instance(instance_name) if instance_name else ""
            object_id_to_category[object_id] = category
            if category:
                object_id_to_category_id[str(object_id)] = category_to_id[category]

        eligible_object_ids = [
            object_id
            for object_id in object_ids
            if object_id in valid_refs and object_view_counts[object_id] >= min_object_views
        ]
        if not eligible_object_ids:
            continue

        category_ids = sorted(set(object_id_to_category_id.values()))
        categories = [category_id_to_category[category_id] for category_id in category_ids]
        primary_category = object_id_to_category.get(eligible_object_ids[0], "")

        records.append(
            {
                "scene_name": scene_dir.name,
                "relative_path": f"oo3d9dmulti/{scene_dir.name}",
                "object_ids": object_ids,
                "object_view_counts": {str(k): int(v) for k, v in sorted(object_view_counts.items())},
                "object_id_to_category_id": object_id_to_category_id,
                "category_ids": category_ids,
                "categories": categories,
                "eligible_object_ids": eligible_object_ids,
                "_stratify_category": primary_category,
            }
        )
    return records


def filter_single_records_by_object_ids(
    records: List[Dict[str, Any]],
    object_ids: set[int],
) -> List[Dict[str, Any]]:
    return [rec for rec in records if int(rec.get("object_id", -1)) in object_ids]


def filter_multi_records_by_object_ids(
    records: List[Dict[str, Any]],
    object_ids: set[int],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for rec in records:
        eligible = [int(oid) for oid in rec["eligible_object_ids"] if int(oid) in object_ids]
        if not eligible:
            continue
        next_rec = dict(rec)
        next_rec["eligible_object_ids"] = eligible
        next_rec["_stratify_category"] = rec.get("_stratify_category", "")
        filtered.append(next_rec)
    return filtered


def strip_private_keys(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: v for k, v in rec.items() if not k.startswith("_")} for rec in records]


def manifest(
    *,
    split: str,
    root_key: str,
    root_path: Path,
    records: List[Dict[str, Any]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    min_rgb: int | None = None,
    min_object_views: int | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "split": split,
        root_key: str(root_path),
        "split_policy": {
            "unit": "scene folder",
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": round(1.0 - train_ratio - val_ratio, 6),
            "seed": seed,
        },
        "count": len(records),
        "scenes": strip_private_keys(records),
    }
    if min_rgb is not None:
        payload["min_rgb"] = min_rgb
    if min_object_views is not None:
        payload["min_object_views"] = min_object_views
    return payload


def summarize_split(name: str, splits: Dict[str, List[Dict[str, Any]]], category_key: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"dataset": name}
    for split, records in splits.items():
        categories = {str(rec.get(category_key, "")) for rec in records if rec.get(category_key, "")}
        summary[split] = {
            "scenes": len(records),
            "categories": len(categories),
        }
        if name == "multi":
            summary[split]["eligible_object_records"] = sum(len(rec["eligible_object_ids"]) for rec in records)
    return summary


def write_hydra_override(out_root: Path, data_root: Path, virtual_len: int) -> None:
    """Write a small config fragment users can paste or include in Hydra configs."""
    text = f"""# Dataset override fragment for 1:1 single:multi training.
# Use these two dataset entries under data.train.dataset.dataset_configs.
# The equal len_train values make ComposedDataset sample single and multi about 1:1.
- _target_: data.datasets.ov9d_single_pose_normalize.OV9DSinglePoseNormalizeDataset
  split: train
  DATA_ROOT: {data_root}
  SPLIT_JSON: {out_root / "single" / "train.json"}
  len_train: {virtual_len}
  num_scene_views: 4
  num_object_views: 4
  min_view_gap: 5
  object_view_min_gap: 6
  object_view_max_gap: 9
  load_point_map: false
  scale_by_points: true
  negative_object_prob: 0.3
- _target_: data.datasets.ov9d_multi_pose_normalize.OV9DMultiPoseNormalizeDataset
  split: train
  DATA_ROOT: {data_root}
  SPLIT_JSON: {out_root / "multi" / "train.json"}
  len_train: {virtual_len}
  num_scene_views: 4
  num_object_views: 4
  fixed_object_view_ids: [10, 20, 30, 40]
  min_view_gap: 5
  object_view_min_gap: 6
  object_view_max_gap: 9
  load_point_map: false
  scale_by_points: true
  negative_object_prob: 0.3
"""
    (out_root / "hydra_train_dataset_configs_1to1.yaml").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--min-rgb", type=int, default=4)
    parser.add_argument("--min-object-views", type=int, default=4)
    parser.add_argument(
        "--reference-view-ids",
        type=int,
        nargs="+",
        default=[10, 20, 30, 40],
        help="Single-object reference view ids required for multi eligible objects.",
    )
    parser.add_argument(
        "--legacy-independent-scene-split",
        action="store_true",
        help="Use the older independent scene-level single/multi split instead of object-id aligned split.",
    )
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("--train-ratio and --val-ratio must leave a positive test split")

    rng = random.Random(args.seed)
    data_root = args.data_root.resolve()
    out_root = args.out_root.resolve()

    progress(f"Creating OV9D splits: data_root={data_root}, out_root={out_root}")
    single_records, valid_reference_object_ids = build_single_records(
        data_root,
        min_rgb=args.min_rgb,
        reference_view_ids=args.reference_view_ids,
    )
    progress(f"[single] valid scenes={len(single_records)}, reference object ids={len(valid_reference_object_ids)}")
    multi_records = build_multi_records(
        data_root,
        min_object_views=args.min_object_views,
        valid_reference_object_ids=valid_reference_object_ids,
    )
    progress(f"[multi] valid scenes={len(multi_records)}")

    if args.legacy_independent_scene_split:
        single_splits = stratified_split_exact(
            single_records,
            key_name="category",
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            rng=rng,
        )
        multi_splits = stratified_split_exact(
            multi_records,
            key_name="_stratify_category",
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            rng=rng,
        )
        object_id_splits = None
        split_policy_name = "independent scene-folder split"
    else:
        object_id_splits = stratified_object_id_split(
            single_records,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            rng=rng,
        )
        single_splits = {
            split: filter_single_records_by_object_ids(single_records, object_ids)
            for split, object_ids in object_id_splits.items()
        }
        multi_splits = {
            split: filter_multi_records_by_object_ids(multi_records, object_ids)
            for split, object_ids in object_id_splits.items()
        }
        split_policy_name = "object-id aligned split"

    for split, records in single_splits.items():
        write_json(
            out_root / "single" / f"{split}.json",
            manifest(
                split=split,
                root_key="single_dir",
                root_path=data_root / "oo3d9dsingle",
                records=records,
                seed=args.seed,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                min_rgb=args.min_rgb,
            ),
        )

    for split, records in multi_splits.items():
        write_json(
            out_root / "multi" / f"{split}.json",
            manifest(
                split=split,
                root_key="multi_dir",
                root_path=data_root / "oo3d9dmulti",
                records=records,
                seed=args.seed,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                min_object_views=args.min_object_views,
            ),
        )

    virtual_len = max(len(single_splits["train"]), len(multi_splits["train"]))
    mix = {
        "split": "train",
        "ratio_policy": "single:multi = 1:1 via equal virtual len_train values",
        "split_policy": split_policy_name,
        "single_train_json": str(out_root / "single" / "train.json"),
        "multi_train_json": str(out_root / "multi" / "train.json"),
        "single_train_scenes": len(single_splits["train"]),
        "multi_train_scenes": len(multi_splits["train"]),
        "single_len_train": virtual_len,
        "multi_len_train": virtual_len,
        "effective_single_to_multi_ratio": [1, 1],
        "note": "Set both dataset configs to these len_train values when using ComposedDataset.",
    }
    write_json(out_root / "train_mix_1to1.json", mix)
    write_hydra_override(out_root, data_root, virtual_len)

    summary = {
        "data_root": str(data_root),
        "out_root": str(out_root),
        "split_policy": split_policy_name,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": round(1.0 - args.train_ratio - args.val_ratio, 6),
        "single_total_valid_scenes": len(single_records),
        "multi_total_valid_scenes": len(multi_records),
        "mix_virtual_len_per_dataset": virtual_len,
        "reference_view_ids": args.reference_view_ids,
        "single": summarize_split("single", single_splits, "category"),
        "multi": summarize_split("multi", multi_splits, "_stratify_category"),
    }
    if object_id_splits is not None:
        summary["object_id_split_counts"] = {split: len(object_ids) for split, object_ids in object_id_splits.items()}
        write_json(
            out_root / "object_id_splits.json",
            {
                "split_policy": split_policy_name,
                "seed": args.seed,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "reference_view_ids": args.reference_view_ids,
                "object_ids": {split: sorted(object_ids) for split, object_ids in object_id_splits.items()},
            },
        )
    write_json(out_root / "summary.json", summary)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote splits to: {out_root}")
    print(f"1:1 train mix metadata: {out_root / 'train_mix_1to1.json'}")
    print(f"Hydra dataset config fragment: {out_root / 'hydra_train_dataset_configs_1to1.yaml'}")


if __name__ == "__main__":
    main()
