"""
Compare normalized pose distributions between two splits (e.g. train vs test1).

Usage:
    python compare_splits.py \
        --split-jsons splits_multi/train.json splits_multi/test1.json \
        --split-names train test1 \
        --multi-dir /mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d/oo3d9dmulti \
        --out-dir analysis_output
"""
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

BASELINE_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = BASELINE_DIR / "training"
for _p in [str(TRAINING_ROOT), str(BASELINE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis_data import collect_normalized_poses, rotation_to_angle_deg


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def summarise(label: str, rotations: List[np.ndarray], translations: List[np.ndarray]) -> dict:
    rot_angles = np.array([rotation_to_angle_deg(r) for r in rotations])
    trans = np.stack(translations)
    norms = np.linalg.norm(trans, axis=1)

    print(f"\n{'='*60}")
    print(f"  {label}   (n={len(rotations)})")
    print(f"{'='*60}")
    print(f"  Rotation angle (deg):")
    print(f"    mean={rot_angles.mean():.2f}  std={rot_angles.std():.2f}  "
          f"p25={np.percentile(rot_angles,25):.2f}  "
          f"p50={np.percentile(rot_angles,50):.2f}  "
          f"p75={np.percentile(rot_angles,75):.2f}")
    print(f"  Translation (X / Y / Z) mean:")
    print(f"    X mean={trans[:,0].mean():.4f}  std={trans[:,0].std():.4f}")
    print(f"    Y mean={trans[:,1].mean():.4f}  std={trans[:,1].std():.4f}")
    print(f"    Z mean={trans[:,2].mean():.4f}  std={trans[:,2].std():.4f}")
    print(f"  Translation norm:")
    print(f"    mean={norms.mean():.4f}  std={norms.std():.4f}  "
          f"p25={np.percentile(norms,25):.4f}  "
          f"p50={np.percentile(norms,50):.4f}  "
          f"p75={np.percentile(norms,75):.4f}")

    return {
        "rot_angles": rot_angles,
        "trans": trans,
        "norms": norms,
        "label": label,
        "n": len(rotations),
    }


# ---------------------------------------------------------------------------
# KL / Wasserstein helpers
# ---------------------------------------------------------------------------

def histogram_kl(a: np.ndarray, b: np.ndarray, bins: int = 80) -> float:
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    pa, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    pb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    eps = 1e-10
    pa = pa + eps
    pb = pb + eps
    pa /= pa.sum()
    pb /= pb.sum()
    return float(np.sum(pa * np.log(pa / pb)))


def print_divergence(stats_a: dict, stats_b: dict) -> None:
    la, lb = stats_a["label"], stats_b["label"]
    print(f"\n--- KL divergence ({la} || {lb}) ---")
    print(f"  Rotation angle:      {histogram_kl(stats_a['rot_angles'], stats_b['rot_angles']):.4f}")
    for i, axis in enumerate(["X", "Y", "Z"]):
        print(f"  Translation {axis}:     {histogram_kl(stats_a['trans'][:,i], stats_b['trans'][:,i]):.4f}")
    print(f"  Translation norm:    {histogram_kl(stats_a['norms'], stats_b['norms']):.4f}")


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def plot_comparison(
    stats_list: List[dict],
    out_dir: Path,
    out_name: str = "pose_comparison.png",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_splits = len(stats_list)
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    title = "  vs  ".join(f"{s['label']} (n={s['n']})" for s in stats_list)
    fig.suptitle(title, fontsize=12)

    alpha = 0.55

    def _hist(ax, key, idx, title_str, xlabel, bins=80):
        lo = min(s[key][:, idx].min() if key == "trans" else s[key].min() for s in stats_list)
        hi = max(s[key][:, idx].max() if key == "trans" else s[key].max() for s in stats_list)
        for j, s in enumerate(stats_list):
            data = s[key][:, idx] if key == "trans" else s[key]
            ax.hist(data, bins=bins, range=(lo, hi),
                    color=PALETTE[j % len(PALETTE)], alpha=alpha,
                    label=s["label"], edgecolor="none", density=True)
            ax.axvline(float(np.median(data)), color=PALETTE[j % len(PALETTE)],
                       linewidth=1.5, linestyle="--")
        ax.set_title(title_str, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.legend(fontsize=8)

    _hist(axes[0, 0], "rot_angles", None, "Rotation angle", "degrees")
    _hist(axes[0, 1], "trans", 0, "Normalized translation X", "value")
    _hist(axes[0, 2], "trans", 1, "Normalized translation Y", "value")
    _hist(axes[0, 3], "trans", 2, "Normalized translation Z", "value")
    _hist(axes[1, 0], "norms", None, "Translation norm", "value")

    # Rotation matrix elements
    for j, s in enumerate(stats_list):
        rot_flat = np.stack([r for r in s.get("_rotations", [])]).reshape(-1) if "_rotations" in s else np.array([])
        if rot_flat.size == 0:
            continue
        axes[1, 1].hist(rot_flat, bins=120, range=(-1.05, 1.05),
                        color=PALETTE[j % len(PALETTE)], alpha=alpha,
                        label=s["label"], edgecolor="none", density=True)
    axes[1, 1].set_title("Rotation matrix elements", fontsize=10)
    axes[1, 1].set_xlabel("value", fontsize=9)
    axes[1, 1].set_ylabel("density", fontsize=9)
    axes[1, 1].legend(fontsize=8)

    # XY scatter
    for j, s in enumerate(stats_list):
        axes[1, 2].scatter(s["trans"][:, 0], s["trans"][:, 1],
                           alpha=0.15, s=3, linewidths=0,
                           color=PALETTE[j % len(PALETTE)], label=s["label"])
    axes[1, 2].set_title("Translation X vs Y", fontsize=10)
    axes[1, 2].set_xlabel("X", fontsize=9)
    axes[1, 2].set_ylabel("Y", fontsize=9)
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].set_aspect("equal")

    # XZ scatter
    for j, s in enumerate(stats_list):
        axes[1, 3].scatter(s["trans"][:, 0], s["trans"][:, 2],
                           alpha=0.15, s=3, linewidths=0,
                           color=PALETTE[j % len(PALETTE)], label=s["label"])
    axes[1, 3].set_title("Translation X vs Z", fontsize=10)
    axes[1, 3].set_xlabel("X", fontsize=9)
    axes[1, 3].set_ylabel("Z", fontsize=9)
    axes[1, 3].legend(fontsize=8)
    axes[1, 3].set_aspect("equal")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    plt.savefig(out_path, dpi=150)
    print(f"\nComparison plot saved to: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-jsons", nargs="+",
                        default=[
                            str(BASELINE_DIR / "splits_multi" / "train.json"),
                            str(BASELINE_DIR / "splits_multi" / "test1.json"),
                        ])
    parser.add_argument("--split-names", nargs="+", default=None,
                        help="Labels for each split (default: derived from filename)")
    parser.add_argument("--multi-dir",
                        default="/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d/oo3d9dmulti")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--out-dir", default=str(BASELINE_DIR / "analysis_output"))
    parser.add_argument("--out-name", default="pose_comparison.png")
    args = parser.parse_args()

    names = args.split_names or [Path(p).stem for p in args.split_jsons]
    if len(names) != len(args.split_jsons):
        parser.error("--split-names must have same length as --split-jsons")

    all_stats = []
    for split_json, name in zip(args.split_jsons, names):
        print(f"\nCollecting poses for: {name}  ({split_json})")
        rotations, translations, _ = collect_normalized_poses(
            split_json=Path(split_json),
            multi_dir=Path(args.multi_dir),
            max_scenes=args.max_scenes,
            num_views=args.num_views,
        )
        if not rotations:
            print(f"  WARNING: no poses collected for {name}")
            continue
        s = summarise(name, rotations, translations)
        s["_rotations"] = rotations
        all_stats.append(s)

    if len(all_stats) >= 2:
        print_divergence(all_stats[0], all_stats[1])

    if len(all_stats) >= 1:
        plot_comparison(all_stats, Path(args.out_dir), out_name=args.out_name)


if __name__ == "__main__":
    main()
