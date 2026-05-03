"""Pose / translation bias analysis.

For each (object_id, split) in a category, run the trained model on every
matching sample, then test the hypothesis: "test1 predictions collapse toward
the per-object_id mean of train predictions" (i.e., the cross-attention has
memorized a default pose per object_id and ignores scene context).

Method
------
Per object_id O:
  train_mean_pose = mean over train samples with O of pred_pose (rot6d, 6D)
  train_mean_t    = mean over train samples with O of pred_translation (3D)

Per sample with obj_id O:
  v_gt_pose   = gt_pose   - train_mean_pose
  v_pred_pose = pred_pose - train_mean_pose
  d_pred / d_gt are L2 norms; fraction_explained = <v_pred, v_gt> / ||v_gt||^2
    (1.0 = perfect, 0.0 = predicts the mean, <0 = wrong direction)

A test1 sample with d_pred << d_gt and fraction near 0 is evidence the model
just predicted "the average pose I learned for this obj_id" instead of using
the scene to infer the actual pose.

Outputs
-------
  pose_bias.csv         per-sample metrics
  pose_bias_summary.txt per-split / per-obj_id summary
  pose_bias_scatter.png d_pred vs d_gt (pose + translation), one subplot each
  pose_bias_hist.png    fraction_explained histograms
  pose_bias_variance.png per-obj_id variance: pred vs gt across scenes

Usage:
    python eval_pose_bias.py --category banana
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------- Constants -----------------------------

REPO_ROOT = Path(__file__).resolve().parent
TRAINING_ROOT = REPO_ROOT / "training"
PROJECT_ROOT = REPO_ROOT.parent

DATA_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d")
SPLIT_JSON_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/baseline/splits_multi_backup_min3")
TRAINED_CKPT = Path(
    "/mnt/train-data-4-hdd/yian/freepose/baseline/training/logs/0431_model/checkpoint_60.pt"
)

NUM_SCENE_VIEWS = 4
NUM_OBJECT_VIEWS = 4
SPLITS = ("train", "test1")
SPLIT_COLORS = {"train": "#2563eb", "test1": "#dc2626"}

device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------- Imports (trained VGGT) -----------------------------

for p in (PROJECT_ROOT, REPO_ROOT, TRAINING_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vggt.models.vggt import VGGT as TrainedVGGT  # noqa: E402
from data.datasets.ov9d_multi_pose_normalize import OV9DMultiPoseNormalizeDataset  # noqa: E402


# ----------------------------- Helpers -----------------------------

def make_common_conf():
    return SimpleNamespace(
        debug=False, training=False, inside_random=False,
        img_size=518, patch_size=14,
        augs=SimpleNamespace(scales=[]), rescale=True, rescale_aug=False,
        landscape_check=True,
    )


_DATASET_CACHE = {}


def build_dataset(split: str):
    if split not in _DATASET_CACHE:
        _DATASET_CACHE[split] = OV9DMultiPoseNormalizeDataset(
            common_conf=make_common_conf(),
            split=split,
            DATA_ROOT=str(DATA_ROOT),
            SPLIT_JSON=str(SPLIT_JSON_ROOT / f"{split}.json"),
            verify_files=True,
            num_scene_views=NUM_SCENE_VIEWS,
            num_object_views=NUM_OBJECT_VIEWS,
            load_point_map=True,
            scale_by_points=True,
            negative_object_prob=0.0,
        )
    return _DATASET_CACHE[split]


def seed_everything(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def images_to_tensor(images):
    arrs = [np.ascontiguousarray(np.asarray(im, dtype=np.uint8)) for im in images]
    t = torch.from_numpy(np.stack(arrs, axis=0)).permute(0, 3, 1, 2).float().div(255.0)
    return t.to(device)


def matrix_to_rot6d(matrix):
    return np.asarray(matrix, dtype=np.float32)[:, :2].reshape(-1)


# ----------------------------- Model -----------------------------

def build_model() -> torch.nn.Module:
    print("Building trained VGGT...")
    model = TrainedVGGT(
        enable_camera=False, enable_depth=False, enable_point=False, enable_track=False,
        enable_object_point=False, enable_object_mask=False, enable_object_srt=True,
        use_shared_object_latent=False, enable_object_cross_attn=False,
        enable_pre_aggregator_object_cross_attn=False,
        enable_multi_layer_object_prototype_cross_attn=True,
        enable_global_pool_scene_object_pose_head=False,
        object_prototype_layer_indices=(4, 11, 17, 23),
        object_prototype_num_tokens=32,
        object_prototype_object_encoder_no_grad=False,
        object_cross_attn_heads=16,
    )
    print(f"Loading checkpoint: {TRAINED_CKPT}")
    ckpt = torch.load(TRAINED_CKPT, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    return model.eval().to(device)


# ----------------------------- Per-sample inference -----------------------------

def run_sample(split: str, idx: int, seed: int, model) -> dict:
    seed_everything(seed)
    ds = build_dataset(split)
    batch = ds.get_data(seq_index=idx, img_per_seq=NUM_SCENE_VIEWS, aspect_ratio=1.0)
    scene_t = images_to_tensor(batch["images"])
    object_t = images_to_tensor(batch["object_images"])
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            preds = model(scene_t, object_images=object_t)
    pred_pose = preds["object_pose"].float().cpu().numpy()[0]
    pred_t = preds["object_translation"].float().cpu().numpy()[0]
    gt_R = np.asarray(batch["object_rotation"], dtype=np.float32)
    gt_pose = matrix_to_rot6d(gt_R)
    gt_t = np.asarray(batch["object_translation"], dtype=np.float32)
    return {
        "pred_pose": pred_pose,
        "pred_t": pred_t,
        "gt_pose": gt_pose,
        "gt_t": gt_t,
    }


def collect_samples(category: str, max_per_split: int):
    samples = []
    for split in SPLITS:
        ds = build_dataset(split)
        count = 0
        for idx, rec in enumerate(ds.records):
            if str(rec.get("category", "")).lower() == category.lower():
                samples.append({
                    "split": split,
                    "sample_idx": idx,
                    "scene_name": rec["scene_name"],
                    "object_id": int(rec["target_object_id"]),
                })
                count += 1
                if max_per_split and count >= max_per_split:
                    break
    return samples


# ----------------------------- Bias analysis -----------------------------

def compute_bias(rows):
    """Add per-sample bias metrics referenced to per-obj_id train mean."""
    obj_ids = sorted({r["object_id"] for r in rows})

    train_mean_pose = {}
    train_mean_t = {}
    for oid in obj_ids:
        tr = [r for r in rows if r["object_id"] == oid and r["split"] == "train"]
        if not tr:
            continue
        train_mean_pose[oid] = np.mean([r["pred_pose"] for r in tr], axis=0)
        train_mean_t[oid] = np.mean([r["pred_t"] for r in tr], axis=0)

    for r in rows:
        oid = r["object_id"]
        if oid not in train_mean_pose:
            r["pose_d_pred"] = float("nan")
            r["pose_d_gt"] = float("nan")
            r["pose_bias_ratio"] = float("nan")
            r["pose_fraction_explained"] = float("nan")
            r["pose_cos"] = float("nan")
            r["trans_d_pred"] = float("nan")
            r["trans_d_gt"] = float("nan")
            r["trans_bias_ratio"] = float("nan")
            r["trans_fraction_explained"] = float("nan")
            r["trans_cos"] = float("nan")
            continue
        m_pose = train_mean_pose[oid]
        m_t = train_mean_t[oid]
        for prefix, mean_vec, pred_key, gt_key in [
            ("pose", m_pose, "pred_pose", "gt_pose"),
            ("trans", m_t, "pred_t", "gt_t"),
        ]:
            v_pred = r[pred_key] - mean_vec
            v_gt = r[gt_key] - mean_vec
            d_pred = float(np.linalg.norm(v_pred))
            d_gt = float(np.linalg.norm(v_gt))
            r[f"{prefix}_d_pred"] = d_pred
            r[f"{prefix}_d_gt"] = d_gt
            r[f"{prefix}_bias_ratio"] = d_pred / d_gt if d_gt > 1e-9 else float("nan")
            r[f"{prefix}_fraction_explained"] = (
                float(np.dot(v_pred, v_gt)) / (d_gt * d_gt) if d_gt > 1e-9 else float("nan")
            )
            r[f"{prefix}_cos"] = (
                float(np.dot(v_pred, v_gt)) / (d_pred * d_gt)
                if d_pred > 1e-9 and d_gt > 1e-9 else float("nan")
            )

    return rows, train_mean_pose, train_mean_t


# ----------------------------- Plots -----------------------------

def plot_bias_scatter(rows, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, prefix, title in [
        (axes[0], "pose", "Pose (rot6d) — d_pred vs d_gt"),
        (axes[1], "trans", "Translation — d_pred vs d_gt"),
    ]:
        for split in SPLITS:
            xs = np.array([r[f"{prefix}_d_gt"] for r in rows if r["split"] == split], dtype=np.float64)
            ys = np.array([r[f"{prefix}_d_pred"] for r in rows if r["split"] == split], dtype=np.float64)
            ok = np.isfinite(xs) & np.isfinite(ys)
            ax.scatter(xs[ok], ys[ok], color=SPLIT_COLORS[split], alpha=0.7, s=32,
                       label=f"{split} (n={ok.sum()})", edgecolors="white", linewidths=0.4)
        # Diagonal y=x
        cur_lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, cur_lim], [0, cur_lim], ls="--", color="gray", alpha=0.6, label="y = x  (pred uses scene as much as GT does)")
        ax.set_xlabel("|| gt − train_mean ||  (scene-specific GT displacement)")
        ax.set_ylabel("|| pred − train_mean ||  (scene-specific pred displacement)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle("Below the diagonal = predictions collapsed toward per-obj train mean", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[PLOT] {out_path}")


def plot_bias_hist(rows, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [
        (axes[0, 0], "pose_fraction_explained", "Pose fraction explained\n(1=perfect, 0=mean, <0=wrong dir)", (-1, 1.5)),
        (axes[0, 1], "trans_fraction_explained", "Translation fraction explained", (-1, 1.5)),
        (axes[1, 0], "pose_bias_ratio", "Pose bias ratio  d_pred / d_gt\n(<1=collapsed toward mean)", (0, 3)),
        (axes[1, 1], "trans_bias_ratio", "Translation bias ratio", (0, 3)),
    ]
    for ax, key, title, xlim in panels:
        for split in SPLITS:
            data = np.array([r[key] for r in rows if r["split"] == split], dtype=np.float64)
            data = data[np.isfinite(data)]
            if data.size == 0:
                continue
            ax.hist(data, bins=20, alpha=0.55, color=SPLIT_COLORS[split],
                    label=f"{split}  median={np.median(data):.2f}", range=xlim)
        ax.axvline(0 if "fraction" in key else 1, ls="--", color="black", alpha=0.5)
        ax.set_xlim(xlim)
        ax.set_xlabel(title)
        ax.set_ylabel("count")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle("Bias toward per-obj_id train-mean prediction", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[PLOT] {out_path}")


def plot_variance(rows, out_path: Path):
    """For each obj_id, compare std(pred) vs std(gt) across scenes, per split."""
    obj_ids = sorted({r["object_id"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, prefix, title in [
        (axes[0], "pose", "Per-obj_id std across scenes — Pose"),
        (axes[1], "trans", "Per-obj_id std across scenes — Translation"),
    ]:
        for split in SPLITS:
            xs, ys, labels = [], [], []
            for oid in obj_ids:
                items = [r for r in rows if r["object_id"] == oid and r["split"] == split]
                if len(items) < 2:
                    continue
                pred = np.stack([r[f"pred_{'pose' if prefix == 'pose' else 't'}"] for r in items])
                gt = np.stack([r[f"gt_{'pose' if prefix == 'pose' else 't'}"] for r in items])
                # Total std = sqrt(sum of per-dim variance)
                std_pred = float(np.sqrt(np.sum(np.var(pred, axis=0))))
                std_gt = float(np.sqrt(np.sum(np.var(gt, axis=0))))
                xs.append(std_gt)
                ys.append(std_pred)
                labels.append(oid)
            ax.scatter(xs, ys, color=SPLIT_COLORS[split], alpha=0.8, s=70,
                       label=f"{split} (n_obj={len(xs)})", edgecolors="white", linewidths=0.6)
            for x, y, l in zip(xs, ys, labels):
                ax.annotate(str(l), (x, y), fontsize=8, alpha=0.8,
                            xytext=(3, 3), textcoords="offset points")
        cur_lim = max(ax.get_xlim()[1], ax.get_ylim()[1], 0.001)
        ax.plot([0, cur_lim], [0, cur_lim], ls="--", color="gray", alpha=0.6, label="y = x")
        ax.set_xlabel("std(GT) across scenes (per obj_id)")
        ax.set_ylabel("std(pred) across scenes (per obj_id)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle("Below the diagonal = predictions vary less than GT across scenes  (≈ collapse)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[PLOT] {out_path}")


# ----------------------------- Summary -----------------------------

def print_summary(rows, out_path: Path):
    lines = []

    def w(s):
        print(s)
        lines.append(s)

    w("\n=== Per-split bias summary ===")
    for split in SPLITS:
        rs = [r for r in rows if r["split"] == split]
        if not rs:
            continue
        w(f"\n[{split}]  n={len(rs)}")
        for prefix, label in [("pose", "Pose"), ("trans", "Translation")]:
            ratio = np.array([r[f"{prefix}_bias_ratio"] for r in rs], dtype=np.float64)
            ratio = ratio[np.isfinite(ratio)]
            frac = np.array([r[f"{prefix}_fraction_explained"] for r in rs], dtype=np.float64)
            frac = frac[np.isfinite(frac)]
            cos = np.array([r[f"{prefix}_cos"] for r in rs], dtype=np.float64)
            cos = cos[np.isfinite(cos)]
            d_pred = np.array([r[f"{prefix}_d_pred"] for r in rs], dtype=np.float64)
            d_pred = d_pred[np.isfinite(d_pred)]
            d_gt = np.array([r[f"{prefix}_d_gt"] for r in rs], dtype=np.float64)
            d_gt = d_gt[np.isfinite(d_gt)]
            w(f"  {label}:")
            w(f"    ||pred - train_mean||  mean={d_pred.mean():.4f}  median={np.median(d_pred):.4f}")
            w(f"    ||gt   - train_mean||  mean={d_gt.mean():.4f}  median={np.median(d_gt):.4f}")
            w(f"    bias ratio (d_pred/d_gt)        median={np.median(ratio):.3f}  ({(ratio < 1).sum()}/{ratio.size} below 1)")
            w(f"    fraction explained <v_pr,v_gt>/||v_gt||^2  median={np.median(frac):+.3f}")
            w(f"    cosine(v_pred, v_gt)            median={np.median(cos):+.3f}")

    w("\n=== Per-obj_id pose variance (std across scenes) ===")
    obj_ids = sorted({r["object_id"] for r in rows})
    w(f"  {'obj_id':>6s}  {'split':>5s}  {'n':>3s}   {'std_pred_pose':>13s}  {'std_gt_pose':>11s}  "
      f"{'std_pred_t':>10s}  {'std_gt_t':>10s}")
    for oid in obj_ids:
        for split in SPLITS:
            items = [r for r in rows if r["object_id"] == oid and r["split"] == split]
            if len(items) < 2:
                continue
            pred_pose = np.stack([r["pred_pose"] for r in items])
            gt_pose = np.stack([r["gt_pose"] for r in items])
            pred_t = np.stack([r["pred_t"] for r in items])
            gt_t = np.stack([r["gt_t"] for r in items])
            sp_p = float(np.sqrt(np.sum(np.var(pred_pose, axis=0))))
            sg_p = float(np.sqrt(np.sum(np.var(gt_pose, axis=0))))
            sp_t = float(np.sqrt(np.sum(np.var(pred_t, axis=0))))
            sg_t = float(np.sqrt(np.sum(np.var(gt_t, axis=0))))
            w(f"  {oid:6d}  {split:>5s}  {len(items):3d}   "
              f"{sp_p:13.4f}  {sg_p:11.4f}  {sp_t:10.4f}  {sg_t:10.4f}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[SUMMARY] {out_path}")


# ----------------------------- CSV -----------------------------

CSV_FIELDS = [
    "split", "sample_idx", "scene_name", "object_id",
    "pose_d_pred", "pose_d_gt", "pose_bias_ratio",
    "pose_fraction_explained", "pose_cos",
    "trans_d_pred", "trans_d_gt", "trans_bias_ratio",
    "trans_fraction_explained", "trans_cos",
]


def write_csv(rows, out_path: Path):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    print(f"[CSV] {out_path}")


# ----------------------------- Main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="banana")
    parser.add_argument("--max-per-split", type=int, default=0,
                        help="0 = unlimited; defaults to all matching samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "eval_outputs"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / args.category
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(args.category, args.max_per_split)
    counts = {s: sum(1 for x in samples if x["split"] == s) for s in SPLITS}
    print(f"category = {args.category}")
    print(f"output   = {out_dir}")
    print(f"sample counts: {counts}")

    if not any(counts.values()):
        print("Nothing to evaluate.")
        return

    model = build_model()

    rows = []
    for i, meta in enumerate(samples):
        try:
            out = run_sample(meta["split"], meta["sample_idx"], args.seed, model)
        except Exception as exc:
            print(f"[{i+1}/{len(samples)}] {meta['split']} idx={meta['sample_idx']}  ERROR: {exc}")
            continue
        row = {**meta, **out}
        rows.append(row)
        print(f"[{i+1}/{len(samples)}] {meta['split']:5s}  obj={meta['object_id']:4d}  "
              f"idx={meta['sample_idx']:4d}")

    rows, train_mean_pose, train_mean_t = compute_bias(rows)

    print_summary(rows, out_dir / "pose_bias_summary.txt")
    write_csv(rows, out_dir / "pose_bias.csv")
    plot_bias_scatter(rows, out_dir / "pose_bias_scatter.png")
    plot_bias_hist(rows, out_dir / "pose_bias_hist.png")
    plot_variance(rows, out_dir / "pose_bias_variance.png")


if __name__ == "__main__":
    main()
