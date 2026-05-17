#!/usr/bin/env python3
"""Create OV9D splits for seen-object and same-category-unseen-object validation.

Policy:
  1. Split single-object reference-capable object ids into train/val/test.
  2. single/train contains only train object ids.
  3. multi/train uses train object ids in train scenes.
  4. multi/val_seen_object_unseen_scene uses the same train object ids, but
     only in multi scene folders held out from multi/train.
  5. multi/val_same_category_unseen_object_unseen_scene uses held-out object ids
     whose categories are present in single/train, also in multi scene folders
     held out from multi/train.

This gives two validation axes:
  - seen object, new clutter/scene context
  - unseen object instance, seen category, new clutter/scene context
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from create_ov9d_80_1to1_splits import (
    DEFAULT_DATA_ROOT,
    build_multi_records,
    build_single_records,
    filter_multi_records_by_object_ids,
    filter_single_records_by_object_ids,
    load_json,
    manifest,
    object_category_from_instance,
    stratified_object_id_split,
    write_json,
)


DEFAULT_OUT_ROOT = Path(
    "/mnt/train-data-4-hdd/yian/freepose/baseline_0503/splits_ov9d_seen_unseen_scene"
)


def write_hydra_override(out_root: Path, data_root: Path, virtual_len: int) -> None:
    text = f"""# Dataset override fragment for 1:1 single:multi training.
# Train split policy:
#   single/train object ids == multi/train target object ids.
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


def category_for_oid(oid: int, oid_to_name: Dict[int, str]) -> str:
    return object_category_from_instance(oid_to_name.get(int(oid), ""))


def multi_scene_object_map(records: Sequence[Dict[str, Any]]) -> Dict[str, set[int]]:
    scene_to_oids: Dict[str, set[int]] = defaultdict(set)
    for rec in records:
        scene_to_oids[rec["scene_name"]].update(int(oid) for oid in rec["eligible_object_ids"])
    return scene_to_oids


def split_seen_object_scenes(
    records: Sequence[Dict[str, Any]],
    train_oids: set[int],
    train_scene_ratio: float,
    val_scene_ratio: float,
    rng: random.Random,
) -> Dict[str, set[str]]:
    train_records = filter_multi_records_by_object_ids(list(records), train_oids)
    scene_to_oids = multi_scene_object_map(train_records)
    scene_names = sorted(scene_to_oids)
    rng.shuffle(scene_names)

    n_total = len(scene_names)
    n_train = int(round(n_total * train_scene_ratio))
    n_val = int(round(n_total * val_scene_ratio))
    train_scenes = set(scene_names[:n_train])
    val_seen_scenes = set(scene_names[n_train : n_train + n_val])
    test_seen_scenes = set(scene_names[n_train + n_val :])

    # Make sure every train object that appears in multi has at least one train scene.
    oid_to_scenes: Dict[int, List[str]] = defaultdict(list)
    for scene_name, oids in scene_to_oids.items():
        for oid in oids:
            oid_to_scenes[oid].append(scene_name)
    for oid in sorted(train_oids):
        if oid not in oid_to_scenes:
            continue
        if any(scene in train_scenes for scene in oid_to_scenes[oid]):
            continue
        chosen = rng.choice(sorted(oid_to_scenes[oid]))
        train_scenes.add(chosen)
        val_seen_scenes.discard(chosen)
        test_seen_scenes.discard(chosen)

    return {
        "train": train_scenes,
        "val_seen_object_unseen_scene": val_seen_scenes,
        "test_seen_object_unseen_scene": test_seen_scenes,
    }


def filter_multi_records_by_object_ids_and_scenes(
    records: Sequence[Dict[str, Any]],
    object_ids: set[int],
    scene_names: set[str] | None = None,
    exclude_scene_names: set[str] | None = None,
) -> List[Dict[str, Any]]:
    filtered = []
    for rec in records:
        scene_name = rec["scene_name"]
        if scene_names is not None and scene_name not in scene_names:
            continue
        if exclude_scene_names is not None and scene_name in exclude_scene_names:
            continue
        eligible = [int(oid) for oid in rec["eligible_object_ids"] if int(oid) in object_ids]
        if not eligible:
            continue
        next_rec = dict(rec)
        next_rec["eligible_object_ids"] = eligible
        filtered.append(next_rec)
    return filtered


def object_ids_in_multi(records: Sequence[Dict[str, Any]]) -> set[int]:
    out = set()
    for rec in records:
        out.update(int(oid) for oid in rec["eligible_object_ids"])
    return out


def scene_names(records: Sequence[Dict[str, Any]]) -> set[str]:
    return {rec["scene_name"] for rec in records}


def object_count_by_category(object_ids: Iterable[int], oid_to_name: Dict[int, str]) -> Dict[str, int]:
    counts = Counter(category_for_oid(int(oid), oid_to_name) for oid in object_ids)
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--object-train-ratio", type=float, default=0.8)
    parser.add_argument("--object-val-ratio", type=float, default=0.1)
    parser.add_argument("--seen-scene-train-ratio", type=float, default=0.8)
    parser.add_argument("--seen-scene-val-ratio", type=float, default=0.1)
    parser.add_argument("--min-rgb", type=int, default=4)
    parser.add_argument("--min-object-views", type=int, default=4)
    parser.add_argument("--reference-view-ids", type=int, nargs="+", default=[10, 20, 30, 40])
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_root = args.data_root.resolve()
    out_root = args.out_root.resolve()

    print(f"Creating seen/unseen-scene OV9D splits: data_root={data_root}, out_root={out_root}", flush=True)
    single_records, valid_reference_oids = build_single_records(
        data_root,
        min_rgb=args.min_rgb,
        reference_view_ids=args.reference_view_ids,
    )
    multi_records = build_multi_records(
        data_root,
        min_object_views=args.min_object_views,
        valid_reference_object_ids=valid_reference_oids,
    )

    object_id_splits = stratified_object_id_split(
        single_records,
        train_ratio=args.object_train_ratio,
        val_ratio=args.object_val_ratio,
        rng=rng,
    )
    train_oids = object_id_splits["train"]
    val_oids = object_id_splits["val"]
    test_oids = object_id_splits["test"]

    oid_to_name = {int(v): str(k) for k, v in load_json(data_root / "name2oid.json").items()}
    train_categories = {category_for_oid(oid, oid_to_name) for oid in train_oids}
    val_same_category_oids = {oid for oid in val_oids if category_for_oid(oid, oid_to_name) in train_categories}
    test_same_category_oids = {oid for oid in test_oids if category_for_oid(oid, oid_to_name) in train_categories}

    seen_scene_splits = split_seen_object_scenes(
        multi_records,
        train_oids=train_oids,
        train_scene_ratio=args.seen_scene_train_ratio,
        val_scene_ratio=args.seen_scene_val_ratio,
        rng=rng,
    )

    multi_train = filter_multi_records_by_object_ids_and_scenes(
        multi_records,
        train_oids,
        scene_names=seen_scene_splits["train"],
    )
    multi_val_seen = filter_multi_records_by_object_ids_and_scenes(
        multi_records,
        train_oids,
        scene_names=seen_scene_splits["val_seen_object_unseen_scene"],
    )
    multi_test_seen = filter_multi_records_by_object_ids_and_scenes(
        multi_records,
        train_oids,
        scene_names=seen_scene_splits["test_seen_object_unseen_scene"],
    )

    # For unseen-object validation, avoid scenes used by multi/train and the seen-object val/test splits.
    train_scene_names = scene_names(multi_train)
    val_seen_scene_names = scene_names(multi_val_seen)
    test_seen_scene_names = scene_names(multi_test_seen)

    multi_val_unseen = filter_multi_records_by_object_ids_and_scenes(
        multi_records,
        val_same_category_oids,
        exclude_scene_names=train_scene_names | val_seen_scene_names,
    )
    multi_test_unseen = filter_multi_records_by_object_ids_and_scenes(
        multi_records,
        test_same_category_oids,
        exclude_scene_names=train_scene_names | val_seen_scene_names | test_seen_scene_names,
    )

    single_splits = {
        "train": filter_single_records_by_object_ids(single_records, train_oids),
        "val_same_category_unseen_object": filter_single_records_by_object_ids(single_records, val_same_category_oids),
        "test_same_category_unseen_object": filter_single_records_by_object_ids(single_records, test_same_category_oids),
    }
    multi_splits = {
        "train": multi_train,
        "val_seen_object_unseen_scene": multi_val_seen,
        "val_same_category_unseen_object_unseen_scene": multi_val_unseen,
        "test_seen_object_unseen_scene": multi_test_seen,
        "test_same_category_unseen_object_unseen_scene": multi_test_unseen,
    }

    for split, records in single_splits.items():
        write_json(
            out_root / "single" / f"{split}.json",
            manifest(
                split=split,
                root_key="single_dir",
                root_path=data_root / "oo3d9dsingle",
                records=records,
                seed=args.seed,
                train_ratio=args.object_train_ratio,
                val_ratio=args.object_val_ratio,
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
                train_ratio=args.object_train_ratio,
                val_ratio=args.object_val_ratio,
                min_object_views=args.min_object_views,
            ),
        )

    virtual_len = max(len(single_splits["train"]), len(multi_train))
    write_json(
        out_root / "train_mix_1to1.json",
        {
            "split": "train",
            "ratio_policy": "single:multi = 1:1 via equal virtual len_train values",
            "single_train_json": str(out_root / "single" / "train.json"),
            "multi_train_json": str(out_root / "multi" / "train.json"),
            "single_train_scenes": len(single_splits["train"]),
            "multi_train_scenes": len(multi_train),
            "single_len_train": virtual_len,
            "multi_len_train": virtual_len,
            "effective_single_to_multi_ratio": [1, 1],
        },
    )
    write_hydra_override(out_root, data_root, virtual_len)

    write_json(
        out_root / "object_id_splits.json",
        {
            "seed": args.seed,
            "object_train_ratio": args.object_train_ratio,
            "object_val_ratio": args.object_val_ratio,
            "object_ids": {split: sorted(oids) for split, oids in object_id_splits.items()},
            "val_same_category_oids": sorted(val_same_category_oids),
            "test_same_category_oids": sorted(test_same_category_oids),
        },
    )

    summary: Dict[str, Any] = {
        "data_root": str(data_root),
        "out_root": str(out_root),
        "split_policy": "seen object unseen scene + same category unseen object unseen scene",
        "seed": args.seed,
        "reference_view_ids": args.reference_view_ids,
        "object_id_counts": {
            "train": len(train_oids),
            "val": len(val_oids),
            "test": len(test_oids),
            "val_same_category": len(val_same_category_oids),
            "test_same_category": len(test_same_category_oids),
        },
        "single_counts": {split: len(records) for split, records in single_splits.items()},
        "multi_counts": {
            split: {
                "scenes": len(records),
                "target_object_ids": len(object_ids_in_multi(records)),
                "eligible_object_records": sum(len(rec["eligible_object_ids"]) for rec in records),
            }
            for split, records in multi_splits.items()
        },
        "target_object_category_counts": {
            split: object_count_by_category(object_ids_in_multi(records), oid_to_name)
            for split, records in multi_splits.items()
        },
    }
    write_json(out_root / "summary.json", summary)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nWrote splits to: {out_root}", flush=True)


if __name__ == "__main__":
    main()
