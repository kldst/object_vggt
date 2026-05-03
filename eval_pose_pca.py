"""PCA visualization of pose / translation predictions vs per-obj_id train mean.

Backs up the bias analysis with two visual / statistical checks:

1) Per-obj_id 2D PCA scatter of {train pred, train gt, test1 pred, test1 gt}
   plus the train_mean_pred. The hypothesis "test1 predictions collapse to
   per-obj_id training mean" predicts: test1 preds cluster around the cross
   marker, while test1 GTs spread out matching train GT spread.

2) GT distribution shift table per obj_id: ||mean_gt_test1 - mean_gt_train||
   in units of within-train std. If shift is small, test1 GTs are *not*
   conveniently coinciding with the train predicted mean -- the test1 GTs
   genuinely span the same region the model learned, and the model still
   fails to predict them.

Re-uses helpers from eval_pose_bias.py; runs inference and caches raw vectors
to a .npz so subsequent runs are instant.

Usage:
    python eval_pose_pca.py --category banana
    python eval_pose_pca.py --category banana --refresh   # ignore cache
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pose_bias import (  # noqa: E402
    REPO_ROOT, SPLITS, SPLIT_COLORS,
    build_model, collect_samples, run_sample,
)


# ----------------------------- Inference + cache -----------------------------

def gather_predictions(category: str, max_per_split: int, seed: int, cache_path: Path,
                      refresh: bool):
    if cache_path.is_file() and not refresh:
        print(f"[CACHE] loading {cache_path}")
        z = np.load(cache_path, allow_pickle=False)
        rows = []
        for i in range(int(z["n"])):
            rows.append({
                "split": str(z["splits"][i]),
                "sample_idx": int(z["sample_indices"][i]),
                "scene_name": str(z["scene_names"][i]),
                "object_id": int(z["object_ids"][i]),
                "pred_pose": z["pred_pose"][i],
                "pred_t": z["pred_t"][i],
                "gt_pose": z["gt_pose"][i],
                "gt_t": z["gt_t"][i],
            })
        return rows

    samples = collect_samples(category, max_per_split)
    counts = {s: sum(1 for x in samples if x["split"] == s) for s in SPLITS}
    print(f"sample counts: {counts}")
    if not any(counts.values()):
        return []

    model = build_model()
    rows = []
    for i, meta in enumerate(samples):
        try:
            out = run_sample(meta["split"], meta["sample_idx"], seed, model)
        except Exception as exc:
            print(f"[{i+1}/{len(samples)}] ERROR: {exc}")
            continue
        rows.append({**meta, **out})
        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            print(f"[{i+1}/{len(samples)}]")
    print(f"Collected {len(rows)} samples.")

    n = len(rows)
    np.savez_compressed(
        cache_path,
        n=np.int64(n),
        splits=np.array([r["split"] for r in rows]),
        sample_indices=np.array([r["sample_idx"] for r in rows], dtype=np.int64),
        scene_names=np.array([r["scene_name"] for r in rows]),
        object_ids=np.array([r["object_id"] for r in rows], dtype=np.int64),
        pred_pose=np.stack([r["pred_pose"] for r in rows]),
        pred_t=np.stack([r["pred_t"] for r in rows]),
        gt_pose=np.stack([r["gt_pose"] for r in rows]),
        gt_t=np.stack([r["gt_t"] for r in rows]),
    )
    print(f"[NPZ] saved {cache_path}")
    return rows


# ----------------------------- PCA plot -----------------------------

def plot_pca_grid(rows, obj_ids, kind: str, out_path: Path):
    """Per-obj_id 2D PCA of pred + gt vectors. kind in {pose, trans}."""
    pred_key, gt_key = ("pred_pose", "gt_pose") if kind == "pose" else ("pred_t", "gt_t")

    n = len(obj_ids)
    cols = 3
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.0 * cols, 4.4 * rows_n),
                             squeeze=False)
    axes = axes.flatten()

    for ax_i, oid in enumerate(obj_ids):
        ax = axes[ax_i]
        items = [r for r in rows if r["object_id"] == oid]
        if len(items) < 3:
            ax.axis("off")
            continue
        all_vecs = np.stack(
            [r[pred_key] for r in items] + [r[gt_key] for r in items], axis=0
        )
        center = all_vecs.mean(axis=0)
        centered = all_vecs - center
        # SVD-based PCA
        _, s_vals, vt = np.linalg.svd(centered, full_matrices=False)
        comp = vt[:2]  # (2, D)
        proj = centered @ comp.T  # (2N, 2)
        n_items = len(items)
        pred_proj = proj[:n_items]
        gt_proj = proj[n_items:]

        splits_arr = np.array([r["split"] for r in items])
        for split in SPLITS:
            mask = splits_arr == split
            if not mask.any():
                continue
            ax.scatter(pred_proj[mask, 0], pred_proj[mask, 1],
                       color=SPLIT_COLORS[split], s=70, alpha=0.85,
                       marker="o", edgecolors="black", linewidths=0.8,
                       label=f"{split} pred (n={int(mask.sum())})")
            ax.scatter(gt_proj[mask, 0], gt_proj[mask, 1],
                       color=SPLIT_COLORS[split], s=80, alpha=0.85,
                       marker="^", edgecolors="white", linewidths=0.6,
                       label=f"{split} gt")
            # Connect pred->gt with thin line per sample (visualizes residual)
            for i in np.where(mask)[0]:
                ax.plot([pred_proj[i, 0], gt_proj[i, 0]],
                        [pred_proj[i, 1], gt_proj[i, 1]],
                        color=SPLIT_COLORS[split], alpha=0.25, lw=0.7)

        # Mark train_mean_pred
        train_mask = splits_arr == "train"
        if train_mask.any():
            mtp = pred_proj[train_mask].mean(axis=0)
            ax.scatter(mtp[0], mtp[1], color="black", marker="X",
                       s=240, zorder=10, label="train_mean_pred",
                       edgecolors="white", linewidths=1.4)

        var_pct = (s_vals[:2] ** 2).sum() / (s_vals ** 2).sum()
        ax.text(0.02, 0.98, f"PC1+PC2 = {var_pct:.0%}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        n_tr = int((splits_arr == "train").sum())
        n_te = int((splits_arr == "test1").sum())
        ax.set_title(f"obj {oid}  (train={n_tr}, test1={n_te})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="datalim")

    for ax_i in range(n, len(axes)):
        axes[ax_i].axis("off")

    fig.suptitle(
        f"Per-obj_id PCA of {kind.upper()} predictions and GT  "
        f"(○=pred, △=gt, X=train_mean_pred, line=residual)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[PLOT] {out_path}")


# ----------------------------- GT distribution shift -----------------------------

def gt_shift_table(rows, obj_ids, out_path: Path):
    lines = []

    def w(s):
        print(s)
        lines.append(s)

    w("\n=== Distribution shift of GT between train and test1 (per obj_id) ===")
    w("If 'shift_in_std_units' is small (< 1), test1 GT lives inside train GT spread")
    w("→ rules out 'test1 GT happens to land on train_mean' lucky coincidence;")
    w("  the model fails despite test1 GT being in-distribution.")
    w("")
    header = (
        f"{'obj':>5s}  {'kind':>5s}  "
        f"{'n_tr':>4s} {'n_te':>4s}  "
        f"{'mean_shift':>10s}  "
        f"{'std_train':>10s}  {'std_test1':>10s}  "
        f"{'shift/std_train':>15s}"
    )
    w(header)
    for oid in obj_ids:
        for kind, key in [("pose", "gt_pose"), ("trans", "gt_t")]:
            tr = np.stack([r[key] for r in rows
                           if r["object_id"] == oid and r["split"] == "train"]) \
                 if any(r["object_id"] == oid and r["split"] == "train" for r in rows) else None
            te = np.stack([r[key] for r in rows
                           if r["object_id"] == oid and r["split"] == "test1"]) \
                 if any(r["object_id"] == oid and r["split"] == "test1" for r in rows) else None
            if tr is None or te is None or len(tr) < 2 or len(te) < 1:
                continue
            mean_tr = tr.mean(axis=0)
            mean_te = te.mean(axis=0)
            shift = float(np.linalg.norm(mean_te - mean_tr))
            std_tr = float(np.sqrt(np.sum(np.var(tr, axis=0))))
            std_te = float(np.sqrt(np.sum(np.var(te, axis=0))))
            ratio = shift / max(std_tr, 1e-9)
            w(f"{oid:5d}  {kind:>5s}  "
              f"{len(tr):4d} {len(te):4d}  "
              f"{shift:10.4f}  {std_tr:10.4f}  {std_te:10.4f}  {ratio:15.3f}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[SUMMARY] {out_path}")


# ----------------------------- Headline metric -----------------------------

def headline_metrics(rows):
    """Print a compact answer to: 'is test1 pred closer to train_mean_pred or to gt?'"""
    obj_ids = sorted({r["object_id"] for r in rows})
    train_mean_pose = {}
    train_mean_t = {}
    for oid in obj_ids:
        tr = [r for r in rows if r["object_id"] == oid and r["split"] == "train"]
        if tr:
            train_mean_pose[oid] = np.mean([r["pred_pose"] for r in tr], axis=0)
            train_mean_t[oid] = np.mean([r["pred_t"] for r in tr], axis=0)

    print("\n=== Headline: where does test1 pred sit? ===")
    for kind, mean_dict, pred_key, gt_key in [
        ("pose", train_mean_pose, "pred_pose", "gt_pose"),
        ("trans", train_mean_t, "pred_t", "gt_t"),
    ]:
        wins_mean = 0
        wins_gt = 0
        n = 0
        for r in rows:
            if r["split"] != "test1":
                continue
            mean_vec = mean_dict.get(r["object_id"])
            if mean_vec is None:
                continue
            d_to_mean = float(np.linalg.norm(r[pred_key] - mean_vec))
            d_to_gt = float(np.linalg.norm(r[pred_key] - r[gt_key]))
            n += 1
            if d_to_mean < d_to_gt:
                wins_mean += 1
            else:
                wins_gt += 1
        if n:
            print(f"  {kind:5s}: pred is closer to train_mean_pred than to GT in "
                  f"{wins_mean}/{n} test1 samples  ({wins_mean/n*100:.0f}%)")


# ----------------------------- Main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="banana")
    parser.add_argument("--max-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "eval_outputs"))
    parser.add_argument("--min-samples-per-obj", type=int, default=3)
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore cached .npz and re-run inference")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / args.category
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "raw_predictions.npz"

    rows = gather_predictions(args.category, args.max_per_split, args.seed,
                              cache_path, args.refresh)
    if not rows:
        print("Nothing to plot.")
        return

    # Filter obj_ids with enough samples to PCA meaningfully
    counts = {}
    for r in rows:
        counts[r["object_id"]] = counts.get(r["object_id"], 0) + 1
    obj_ids = sorted([oid for oid, c in counts.items() if c >= args.min_samples_per_obj])
    print(f"PCA on obj_ids (>= {args.min_samples_per_obj} samples): {obj_ids}")

    plot_pca_grid(rows, obj_ids, "pose", out_dir / "pose_pca.png")
    plot_pca_grid(rows, obj_ids, "trans", out_dir / "trans_pca.png")
    gt_shift_table(rows, obj_ids, out_dir / "gt_shift.txt")
    headline_metrics(rows)


if __name__ == "__main__":
    main()
