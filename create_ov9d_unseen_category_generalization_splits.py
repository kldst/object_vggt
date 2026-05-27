#!/usr/bin/env python3
"""Create OV9D splits for same-category and unseen-category generalization.

Policy:
  1. Build eligible single-object records using fixed reference views.
  2. Randomly hold out N categories for unseen-category generalization.
  3. For all remaining categories, sample a per-category object train ratio.
  4. single/train contains the sampled train object ids from seen categories.
  5. single/test_same_category_unseen_object contains leftover object ids from
     seen categories.
  6. single/test_unseen_category_unseen_object contains all object ids from the
     held-out categories.
  7. multi/train uses train object ids only, and caps the train split to a
     fixed number of scene folders while trying to keep object coverage.
  8. multi eval splits exclude train scenes and are divided into:
     - same-category unseen object, unseen scene
     - unseen-category unseen object, unseen scene
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

from create_ov9d_80_1to1_splits import (
    DEFAULT_DATA_ROOT,
    build_multi_records,
    build_single_records,
    filter_single_records_by_object_ids,
    load_json,
    manifest,
    object_category_from_instance,
    write_json,
)


DEFAULT_OUT_ROOT = Path(
    "/mnt/train-data-4-hdd/yian/freepose/omni-object_clone/splits_ov9d_unseen_category_generalization"
)


def category_for_oid(oid: int, oid_to_name: Dict[int, str]) -> str:
    return object_category_from_instance(oid_to_name.get(int(oid), ""))


def object_count_by_category(object_ids: Iterable[int], oid_to_name: Dict[int, str]) -> Dict[str, int]:
    counts = Counter(category_for_oid(int(oid), oid_to_name) for oid in object_ids)
    return dict(sorted(counts.items()))


def object_ids_in_multi(records: Sequence[Dict[str, Any]]) -> Set[int]:
    out: Set[int] = set()
    for rec in records:
        out.update(int(oid) for oid in rec["eligible_object_ids"])
    return out


def scene_names(records: Sequence[Dict[str, Any]]) -> Set[str]:
    return {str(rec["scene_name"]) for rec in records}


def write_hydra_override(
    out_root: Path,
    data_root: Path,
    virtual_len: int,
    reference_view_ids: Sequence[int],
) -> None:
    ref_text = ", ".join(str(x) for x in reference_view_ids)
    text = f"""# Dataset override fragment for category generalization training.
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
  fixed_object_view_ids: [{ref_text}]
  min_view_gap: 5
  object_view_min_gap: 6
  object_view_max_gap: 9
  load_point_map: false
  scale_by_points: true
  negative_object_prob: 0.3
"""
    (out_root / "hydra_train_dataset_configs.yaml").write_text(text, encoding="utf-8")


def group_single_oids_by_category(records: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for rec in records:
        if "object_id" not in rec or not rec.get("has_fixed_reference_views", False):
            continue
        grouped[str(rec["category"])].append(int(rec["object_id"]))
    return {cat: sorted(set(oids)) for cat, oids in grouped.items()}


def split_seen_category_object_ids(
    grouped_oids: Dict[str, List[int]],
    seen_categories: Sequence[str],
    train_ratio: float,
    rng: random.Random,
) -> tuple[Set[int], Set[int]]:
    train_oids: Set[int] = set()
    same_category_unseen_oids: Set[int] = set()
    for category in seen_categories:
        object_ids = list(grouped_oids[category])
        rng.shuffle(object_ids)
        n_total = len(object_ids)
        if n_total == 1:
            n_train = 1
        else:
            n_train = int(math.floor(n_total * train_ratio))
            n_train = max(1, n_train)
            n_train = min(n_total - 1, n_train)
        train_subset = set(object_ids[:n_train])
        eval_subset = set(object_ids[n_train:])
        train_oids.update(train_subset)
        same_category_unseen_oids.update(eval_subset)
    return train_oids, same_category_unseen_oids


def filter_multi_records_by_object_ids_and_scenes(
    records: Sequence[Dict[str, Any]],
    object_ids: Set[int],
    include_scene_names: Set[str] | None = None,
    exclude_scene_names: Set[str] | None = None,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for rec in records:
        scene_name = str(rec["scene_name"])
        if include_scene_names is not None and scene_name not in include_scene_names:
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


def select_train_multi_scene_names(
    train_candidate_records: Sequence[Dict[str, Any]],
    train_oids: Set[int],
    target_scene_count: int,
    rng: random.Random,
) -> Set[str]:
    scene_to_oids: Dict[str, Set[int]] = {}
    oid_to_scenes: Dict[int, List[str]] = defaultdict(list)
    for rec in train_candidate_records:
        scene_name = str(rec["scene_name"])
        oids = {int(oid) for oid in rec["eligible_object_ids"] if int(oid) in train_oids}
        if not oids:
            continue
        scene_to_oids[scene_name] = oids
        for oid in oids:
            oid_to_scenes[oid].append(scene_name)

    available_scene_names = list(scene_to_oids)
    if len(available_scene_names) <= target_scene_count:
        return set(available_scene_names)

    selected: Set[str] = set()
    uncovered = {oid for oid in train_oids if oid in oid_to_scenes}
    for oid in sorted(uncovered, key=lambda x: len(oid_to_scenes[x])):
        if len(selected) >= target_scene_count:
            break
        if any(scene in selected for scene in oid_to_scenes[oid]):
            continue
        candidates = list(oid_to_scenes[oid])
        rng.shuffle(candidates)
        best_scene = max(
            candidates,
            key=lambda scene: (len(scene_to_oids[scene] & uncovered), len(scene_to_oids[scene])),
        )
        selected.add(best_scene)
        uncovered.difference_update(scene_to_oids[best_scene])

    remaining = [scene for scene in available_scene_names if scene not in selected]
    rng.shuffle(remaining)
    for scene in remaining:
        if len(selected) >= target_scene_count:
            break
        selected.add(scene)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heldout-category-count", type=int, default=20)
    parser.add_argument("--seen-category-train-object-ratio", type=float, default=0.6)
    parser.add_argument("--multi-train-scene-count", type=int, default=7000)
    parser.add_argument("--min-rgb", type=int, default=4)
    parser.add_argument("--min-object-views", type=int, default=4)
    parser.add_argument("--reference-view-ids", type=int, nargs="+", default=[1, 5, 10, 15])
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_root = args.data_root.resolve()
    out_root = args.out_root.resolve()

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

    oid_to_name = {int(v): str(k) for k, v in load_json(data_root / "name2oid.json").items()}
    grouped_oids = group_single_oids_by_category(single_records)
    all_categories = sorted(grouped_oids)
    if len(all_categories) < args.heldout_category_count:
        raise ValueError(
            f"Only {len(all_categories)} eligible categories found, "
            f"cannot hold out {args.heldout_category_count} categories."
        )

    heldout_categories = sorted(rng.sample(all_categories, args.heldout_category_count))
    heldout_category_set = set(heldout_categories)
    seen_categories = sorted(cat for cat in all_categories if cat not in heldout_category_set)

    train_oids, same_category_unseen_oids = split_seen_category_object_ids(
        grouped_oids,
        seen_categories,
        train_ratio=args.seen_category_train_object_ratio,
        rng=rng,
    )
    unseen_category_oids: Set[int] = set()
    for category in heldout_categories:
        unseen_category_oids.update(grouped_oids[category])

    single_splits = {
        "train": filter_single_records_by_object_ids(single_records, train_oids),
        "test_same_category_unseen_object": filter_single_records_by_object_ids(single_records, same_category_unseen_oids),
        "test_unseen_category_unseen_object": filter_single_records_by_object_ids(single_records, unseen_category_oids),
    }

    multi_train_candidates = filter_multi_records_by_object_ids_and_scenes(multi_records, train_oids)
    train_scene_names = select_train_multi_scene_names(
        multi_train_candidates,
        train_oids=train_oids,
        target_scene_count=args.multi_train_scene_count,
        rng=rng,
    )
    multi_train = filter_multi_records_by_object_ids_and_scenes(
        multi_records,
        train_oids,
        include_scene_names=train_scene_names,
    )

    used_train_scene_names = scene_names(multi_train)
    multi_test_same_category = filter_multi_records_by_object_ids_and_scenes(
        multi_records,
        same_category_unseen_oids,
        exclude_scene_names=used_train_scene_names,
    )
    multi_test_unseen_category = filter_multi_records_by_object_ids_and_scenes(
        multi_records,
        unseen_category_oids,
        exclude_scene_names=used_train_scene_names,
    )

    multi_splits = {
        "train": multi_train,
        "test_same_category_unseen_object_unseen_scene": multi_test_same_category,
        "test_unseen_category_unseen_object_unseen_scene": multi_test_unseen_category,
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
                train_ratio=args.seen_category_train_object_ratio,
                val_ratio=0.0,
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
                train_ratio=args.seen_category_train_object_ratio,
                val_ratio=0.0,
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
    write_hydra_override(out_root, data_root, virtual_len, args.reference_view_ids)

    write_json(
        out_root / "object_id_splits.json",
        {
            "seed": args.seed,
            "heldout_category_count": args.heldout_category_count,
            "seen_category_train_object_ratio": args.seen_category_train_object_ratio,
            "multi_train_scene_count": args.multi_train_scene_count,
            "reference_view_ids": list(args.reference_view_ids),
            "heldout_categories": heldout_categories,
            "seen_categories": seen_categories,
            "object_ids": {
                "train": sorted(train_oids),
                "test_same_category_unseen_object": sorted(same_category_unseen_oids),
                "test_unseen_category_unseen_object": sorted(unseen_category_oids),
            },
        },
    )

    summary: Dict[str, Any] = {
        "data_root": str(data_root),
        "out_root": str(out_root),
        "split_policy": (
            "hold out 20 categories for unseen-category generalization; "
            "sample 0.6 objects from remaining categories for train; "
            "cap multi train scenes to 7000"
        ),
        "seed": args.seed,
        "reference_view_ids": list(args.reference_view_ids),
        "heldout_category_count": args.heldout_category_count,
        "seen_category_train_object_ratio": args.seen_category_train_object_ratio,
        "multi_train_scene_count": args.multi_train_scene_count,
        "heldout_categories": heldout_categories,
        "category_counts": {
            "all": len(all_categories),
            "seen": len(seen_categories),
            "heldout": len(heldout_categories),
        },
        "object_id_counts": {
            "train": len(train_oids),
            "test_same_category_unseen_object": len(same_category_unseen_oids),
            "test_unseen_category_unseen_object": len(unseen_category_oids),
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
    print(f"\\nWrote splits to: {out_root}", flush=True)


if __name__ == "__main__":
    main()
