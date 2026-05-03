"""Tag eval.py's metrics.csv with whether each sample's target_object_id was
seen as a training target, then compare pose error between seen / unseen groups
within test1.

Reads `metrics.csv` produced by eval.py and the split JSONs to determine which
object_ids appear as eligible targets in train. Writes:
  - metrics_with_seen.csv: original rows + `object_seen_in_train` flag
  - seen_vs_unseen.png: boxplot+swarm of pose_l1 / rot_err / trans_l2 per group
  - summary printed to stdout (counts, per-group means, ratios)

Usage:
    python eval_object_seen.py --metrics eval_outputs/banana/metrics.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPLIT_JSON_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/baseline/splits_multi_backup_min3")
SPLITS = ("train", "test1")
METRIC_KEYS = ["pose_l1", "rot_err_deg", "trans_l1", "trans_l2"]


def load_split_target_obj_ids(split: str):
    """Return (set of all eligible target object_ids, dict obj_id -> first scene_name).

    Uses `eligible_object_ids` because those are what the dataset turns into
    training targets (the multi-pose dataset only iterates eligible_object_ids
    per scene). Falls back to `object_ids` if missing.
    """
    path = SPLIT_JSON_ROOT / f"{split}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    obj_to_scene = {}
    obj_to_category = {}
    for scene in payload.get("scenes", []):
        scene_name = scene.get("scene_name", "")
        oid_to_cid = scene.get("object_id_to_category_id", {})
        category_ids = scene.get("category_ids", [])
        categories = scene.get("categories", [])
        cid_to_cat = dict(zip(category_ids, categories))
        eligibles = scene.get("eligible_object_ids") or scene.get("object_ids") or []
        for oid in eligibles:
            oid_int = int(oid)
            obj_to_scene.setdefault(oid_int, scene_name)
            cid = oid_to_cid.get(str(oid_int), oid_to_cid.get(oid_int))
            obj_to_category.setdefault(oid_int, str(cid_to_cat.get(cid, cid)))
    return obj_to_scene, obj_to_category


def read_metrics(path: Path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def numeric(rows, key):
    vals = []
    for r in rows:
        try:
            v = float(r[key])
            if np.isfinite(v):
                vals.append(v)
        except (TypeError, ValueError, KeyError):
            continue
    return np.asarray(vals, dtype=np.float64)


def fmt_stat(arr):
    if arr.size == 0:
        return "n=0"
    return (
        f"n={arr.size:3d}  mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
        f"std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}"
    )


def print_group_summary(label, rows):
    print(f"\n[{label}]  n={len(rows)}")
    if not rows:
        return
    for k in METRIC_KEYS:
        arr = numeric(rows, k)
        print(f"  {k:14s}  {fmt_stat(arr)}")


def make_plot(out_path: Path, train, test1_seen, test1_unseen):
    keys_titles = [
        ("pose_l1", "Pose L1 (rot6d)"),
        ("rot_err_deg", "Rotation err (deg)"),
        ("trans_l2", "Translation L2"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    groups = [
        ("train\n(all)", train, "#2563eb"),
        ("test1\nSEEN obj_id", test1_seen, "#7c3aed"),
        ("test1\nUNSEEN obj_id", test1_unseen, "#dc2626"),
    ]
    for ax, (key, title) in zip(axes, keys_titles):
        data, labels, colors = [], [], []
        for label, rows, color in groups:
            arr = numeric(rows, key)
            if arr.size:
                data.append(arr)
                labels.append(f"{label}\n(n={arr.size})")
                colors.append(color)
        if not data:
            continue
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.5)
        rng = np.random.default_rng(0)
        for i, (arr, color) in enumerate(zip(data, colors), start=1):
            xs = i + rng.uniform(-0.12, 0.12, size=arr.size)
            ax.scatter(xs, arr, color=color, alpha=0.75, s=22, edgecolors="white", linewidths=0.5)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.suptitle("Pose error split by 'object_id seen as target in train'", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[PLOT] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default="eval_outputs/banana/metrics.csv",
        help="Path to metrics.csv produced by eval.py",
    )
    parser.add_argument("--output-dir", default=None,
                        help="Default = parent dir of metrics.csv")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_path}")
    out_dir = Path(args.output_dir) if args.output_dir else metrics_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"metrics = {metrics_path}")
    print(f"output  = {out_dir}")

    rows = read_metrics(metrics_path)
    print(f"loaded {len(rows)} rows  ({sorted({r['split'] for r in rows})})")

    train_obj_to_scene, train_obj_to_category = load_split_target_obj_ids("train")
    train_obj_ids = set(train_obj_to_scene.keys())
    test1_obj_to_scene, _ = load_split_target_obj_ids("test1")
    test1_obj_ids = set(test1_obj_to_scene.keys())

    print(f"\ntrain split: {len(train_obj_ids)} unique target object_ids across all categories")
    print(f"test1 split: {len(test1_obj_ids)} unique target object_ids across all categories")
    print(f"test1 ∩ train (overlap): {len(test1_obj_ids & train_obj_ids)}")
    print(f"test1 \\ train (unseen):  {len(test1_obj_ids - train_obj_ids)}")

    # Tag each row
    for r in rows:
        oid = int(r["object_id"])
        r["object_seen_in_train"] = "1" if oid in train_obj_ids else "0"

    # Save tagged CSV
    tagged_path = out_dir / "metrics_with_seen.csv"
    with open(tagged_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[CSV] {tagged_path}")

    # Group: split by (split, seen)
    train_rows = [r for r in rows if r["split"] == "train"]
    test1_rows = [r for r in rows if r["split"] == "test1"]
    test1_seen = [r for r in test1_rows if r["object_seen_in_train"] == "1"]
    test1_unseen = [r for r in test1_rows if r["object_seen_in_train"] == "0"]

    # Identify object_ids
    test1_obj_ids_in_csv = sorted({int(r["object_id"]) for r in test1_rows})
    print(f"\ntest1 banana sample object_ids: {test1_obj_ids_in_csv}")
    seen_oids = sorted({int(r["object_id"]) for r in test1_seen})
    unseen_oids = sorted({int(r["object_id"]) for r in test1_unseen})
    print(f"  ├─ seen   in train (any category): {seen_oids}")
    print(f"  └─ unseen in train (any category): {unseen_oids}")

    # Per-group summaries
    print("\n=== Per-group summary ===")
    print_group_summary("train (all)", train_rows)
    print_group_summary("test1 SEEN object_id in train", test1_seen)
    print_group_summary("test1 UNSEEN object_id in train", test1_unseen)

    # Headline ratio
    print("\n=== Seen vs unseen ratio (test1) ===")
    for k in METRIC_KEYS:
        s = numeric(test1_seen, k)
        u = numeric(test1_unseen, k)
        if s.size and u.size:
            ratio_mean = u.mean() / s.mean() if s.mean() > 0 else float("nan")
            ratio_med = np.median(u) / np.median(s) if np.median(s) > 0 else float("nan")
            print(
                f"  {k:14s}  unseen.mean / seen.mean = {ratio_mean:.2f}x   "
                f"unseen.median / seen.median = {ratio_med:.2f}x"
            )
        else:
            print(f"  {k:14s}  (one group empty)")

    make_plot(out_dir / "seen_vs_unseen.png", train_rows, test1_seen, test1_unseen)


if __name__ == "__main__":
    main()
