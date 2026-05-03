"""Diagnostic eval for the OV9D multi-object pose model.

Compares the trained model's pose error against the official pretrained VGGT's
scene reconstruction & camera estimation quality on matched samples (same object
category, e.g. banana) across train and test1 splits. The goal is to find which
upstream signal (depth quality, intrinsic estimate, camera pose estimate)
correlates most with pose error -- and whether the train/test gap is driven by
those upstream signals or by the trained cross-attention itself.

Usage:
    python eval.py --category banana
    python eval.py --category banana --max-per-split 30 --seeds 3
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TORCH_HOME", "/mnt/train-data-4-hdd/yian/freepose/vggt_model")

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------- Constants -----------------------------

REPO_ROOT = Path(__file__).resolve().parent
TRAINING_ROOT = REPO_ROOT / "training"
PROJECT_ROOT = REPO_ROOT.parent
OFFICIAL_VGGT_ROOT = PROJECT_ROOT / "vggt"

DATA_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d")
SPLIT_JSON_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose/baseline/splits_multi_backup_min3")
TRAINED_CKPT = Path(
    "/mnt/train-data-4-hdd/yian/freepose/baseline/training/logs/0431_model/checkpoint_60.pt"
)
OFFICIAL_VGGT_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"

NUM_SCENE_VIEWS = 4
NUM_OBJECT_VIEWS = 4
SPLITS = ("train", "test1")

device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------- Dual-import: trained VGGT + official VGGT ----------------- #
# Both packages are named `vggt`. We import the trained one first (with its
# helper modules), snapshot the class, then clear `vggt.*` from sys.modules and
# import the official one from a different location.

def _import_trained_modules():
    for p in (PROJECT_ROOT, REPO_ROOT, TRAINING_ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from vggt.models.vggt import VGGT as TrainedVGGT  # noqa: WPS433
    from data.datasets.ov9d_multi_pose_normalize import OV9DMultiPoseNormalizeDataset  # noqa: WPS433
    return TrainedVGGT, OV9DMultiPoseNormalizeDataset


def _import_official_modules():
    # Drop trained `vggt.*` before re-importing under a different sys.path entry.
    for mod_name in list(sys.modules):
        if mod_name == "vggt" or mod_name.startswith("vggt."):
            del sys.modules[mod_name]
    # Make sure the official root takes precedence.
    if str(REPO_ROOT) in sys.path:
        sys.path.remove(str(REPO_ROOT))
    sys.path.insert(0, str(OFFICIAL_VGGT_ROOT))
    from vggt.models.vggt import VGGT as OfficialVGGT  # noqa: WPS433
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: WPS433
    return OfficialVGGT, pose_encoding_to_extri_intri


TrainedVGGT, OV9DMultiPoseNormalizeDataset = _import_trained_modules()
OfficialVGGT, pose_encoding_to_extri_intri = _import_official_modules()


# ----------------------------- Helpers -----------------------------

def make_common_conf():
    return SimpleNamespace(
        debug=False,
        training=False,
        inside_random=False,
        img_size=518,
        patch_size=14,
        augs=SimpleNamespace(scales=[]),
        rescale=True,
        rescale_aug=False,
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


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Match training/loss.py: rotation_matrix[..., :, :2].reshape(6) (row-major)."""
    return np.asarray(matrix, dtype=np.float32)[:, :2].reshape(-1)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Inverse of matrix_to_rot6d. Re-orthonormalize via Gram-Schmidt."""
    r = np.asarray(rot6d, dtype=np.float64).reshape(3, 2)
    x_raw, y_raw = r[:, 0], r[:, 1]
    x = x_raw / max(np.linalg.norm(x_raw), 1e-12)
    y = y_raw - np.dot(x, y_raw) * x
    y = y / max(np.linalg.norm(y), 1e-12)
    z = np.cross(x, y)
    z = z / max(np.linalg.norm(z), 1e-12)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    rel = R_pred @ R_gt.T
    cos_t = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_t)))


# ----------------------------- Model loading -----------------------------

def build_trained_model() -> torch.nn.Module:
    print("Building trained VGGT (object pose head)...")
    model = TrainedVGGT(
        enable_camera=False,
        enable_depth=False,
        enable_point=False,
        enable_track=False,
        enable_object_point=False,
        enable_object_mask=False,
        enable_object_srt=True,
        use_shared_object_latent=False,
        enable_object_cross_attn=False,
        enable_pre_aggregator_object_cross_attn=False,
        enable_multi_layer_object_prototype_cross_attn=True,
        enable_global_pool_scene_object_pose_head=False,
        object_prototype_layer_indices=(4, 11, 17, 23),
        object_prototype_num_tokens=32,
        object_prototype_object_encoder_no_grad=False,
        object_cross_attn_heads=16,
    )
    print(f"Loading trained checkpoint: {TRAINED_CKPT}")
    ckpt = torch.load(TRAINED_CKPT, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Trained model missing keys: {len(missing)}")
    if unexpected:
        print(f"  Trained model unexpected keys: {len(unexpected)}")
    return model.eval().to(device)


def build_official_model() -> torch.nn.Module:
    print("Building official pretrained VGGT-1B...")
    model = OfficialVGGT()
    state = torch.hub.load_state_dict_from_url(OFFICIAL_VGGT_URL)
    model.load_state_dict(state)
    return model.eval().to(device)


# ----------------------------- Per-sample evaluation -----------------------------

def run_trained(model, scene_t, object_t, dtype):
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            preds = model(scene_t, object_images=object_t)
    return {
        "object_pose": preds["object_pose"].float().cpu().numpy()[0],
        "object_translation": preds["object_translation"].float().cpu().numpy()[0],
    }


def run_official(model, scene_t, dtype):
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            preds = model(scene_t)
    H, W = scene_t.shape[-2:]
    extr, intr = pose_encoding_to_extri_intri(preds["pose_enc"], (H, W))
    out = {
        "extrinsic": extr.float().cpu().numpy()[0],  # (S, 3, 4)
        "intrinsic": intr.float().cpu().numpy()[0],  # (S, 3, 3)
        "depth": preds["depth"].float().cpu().numpy()[0, ..., 0],  # (S, H, W)
        "depth_conf": preds["depth_conf"].float().cpu().numpy()[0],  # (S, H, W)
    }
    return out


def evaluate_sample(split: str, idx: int, seed: int, trained_model, official_model) -> dict:
    seed_everything(seed)
    dataset = build_dataset(split)
    batch = dataset.get_data(seq_index=idx, img_per_seq=NUM_SCENE_VIEWS, aspect_ratio=1.0)

    scene_t = images_to_tensor(batch["images"])
    object_t = images_to_tensor(batch["object_images"])
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    trained_out = run_trained(trained_model, scene_t, object_t, dtype)
    official_out = run_official(official_model, scene_t, dtype)

    # ---------- Pose metrics (trained model) ----------
    pred_pose = trained_out["object_pose"]
    pred_t = trained_out["object_translation"]
    gt_R = np.asarray(batch["object_rotation"], dtype=np.float32)
    gt_pose = matrix_to_rot6d(gt_R)
    gt_t = np.asarray(batch["object_translation"], dtype=np.float32)
    pose_l1 = float(np.abs(pred_pose - gt_pose).mean())
    rot_err = rotation_error_deg(rot6d_to_matrix(pred_pose), gt_R)
    trans_l2 = float(np.linalg.norm(pred_t - gt_t))
    trans_l1 = float(np.abs(pred_t - gt_t).mean())

    # ---------- Reconstruction metric (official VGGT depth vs GT depth) ----------
    pred_depth = official_out["depth"]  # (S, H, W)
    gt_depth = np.stack([d.astype(np.float32) for d in batch["depths"]], axis=0)  # (S, H, W)
    point_masks = np.stack([m.astype(bool) for m in batch["point_masks"]], axis=0)
    valid = point_masks & (gt_depth > 1e-6) & (pred_depth > 1e-6) & np.isfinite(pred_depth)
    if valid.any():
        scale = float(np.median(gt_depth[valid] / np.maximum(pred_depth[valid], 1e-6)))
        aligned = pred_depth * scale
        diff = aligned[valid] - gt_depth[valid]
        depth_rmse_aligned = float(np.sqrt(np.mean(diff ** 2)))
        depth_rel_err = float(
            np.mean(np.abs(diff) / np.maximum(np.abs(gt_depth[valid]), 1e-6))
        )
        valid_ratio = float(valid.sum() / valid.size)
    else:
        depth_rmse_aligned = float("nan")
        depth_rel_err = float("nan")
        valid_ratio = 0.0

    # ---------- Intrinsic metric ----------
    gt_intr = np.stack([np.asarray(K, dtype=np.float32) for K in batch["intrinsics"]])  # (S, 3, 3)
    pred_intr = official_out["intrinsic"]  # (S, 3, 3)
    fx_rel = float(
        np.mean(np.abs(pred_intr[:, 0, 0] - gt_intr[:, 0, 0]) / np.maximum(np.abs(gt_intr[:, 0, 0]), 1e-6))
    )
    fy_rel = float(
        np.mean(np.abs(pred_intr[:, 1, 1] - gt_intr[:, 1, 1]) / np.maximum(np.abs(gt_intr[:, 1, 1]), 1e-6))
    )
    cx_err = float(np.mean(np.abs(pred_intr[:, 0, 2] - gt_intr[:, 0, 2])))
    cy_err = float(np.mean(np.abs(pred_intr[:, 1, 2] - gt_intr[:, 1, 2])))

    # ---------- Extrinsic metric (relative camera pose vs first frame) ----------
    pred_extr = official_out["extrinsic"]  # (S, 3, 4)
    gt_extr = np.stack([np.asarray(E, dtype=np.float32) for E in batch["extrinsics"]])[:, :3, :]  # (S, 3, 4)
    rel_rot_errs = []
    rel_trans_dir_errs = []
    R0_p, R0_g = pred_extr[0, :3, :3], gt_extr[0, :3, :3]
    t0_p, t0_g = pred_extr[0, :3, 3], gt_extr[0, :3, 3]
    for s in range(1, pred_extr.shape[0]):
        Ri_p, Ri_g = pred_extr[s, :3, :3], gt_extr[s, :3, :3]
        ti_p, ti_g = pred_extr[s, :3, 3], gt_extr[s, :3, 3]
        R_rel_p = Ri_p @ R0_p.T
        R_rel_g = Ri_g @ R0_g.T
        rel_rot_errs.append(rotation_error_deg(R_rel_p, R_rel_g))
        t_rel_p = ti_p - R_rel_p @ t0_p
        t_rel_g = ti_g - R_rel_g @ t0_g
        np_p = np.linalg.norm(t_rel_p) + 1e-9
        np_g = np.linalg.norm(t_rel_g) + 1e-9
        cos_a = float(np.clip(np.dot(t_rel_p, t_rel_g) / (np_p * np_g), -1.0, 1.0))
        rel_trans_dir_errs.append(float(np.degrees(np.arccos(cos_a))))
    rel_rot_err = float(np.mean(rel_rot_errs)) if rel_rot_errs else 0.0
    rel_trans_dir_err = float(np.mean(rel_trans_dir_errs)) if rel_trans_dir_errs else 0.0

    return {
        "pose_l1": pose_l1,
        "rot_err_deg": rot_err,
        "trans_l1": trans_l1,
        "trans_l2": trans_l2,
        "depth_rmse_aligned": depth_rmse_aligned,
        "depth_rel_err": depth_rel_err,
        "depth_valid_ratio": valid_ratio,
        "fx_rel_err": fx_rel,
        "fy_rel_err": fy_rel,
        "principal_cx_err_px": cx_err,
        "principal_cy_err_px": cy_err,
        "relative_rot_err_deg": rel_rot_err,
        "relative_trans_dir_err_deg": rel_trans_dir_err,
        "normalization_scale": float(np.asarray(batch["normalization_scale"]).reshape(-1)[0]),
        "scene_views": ",".join(str(int(i)) for i in batch["ids"]),
        "object_views": ",".join(str(int(i)) for i in batch["object_cam_indices"]),
    }


# ----------------------------- Aggregation & plotting -----------------------------

def write_csv(rows, path: Path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[CSV] wrote {len(rows)} rows -> {path}")


def _scatter_with_corr(ax, xs, ys, label, color):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[valid], ys[valid]
    if xs.size == 0:
        return None
    ax.scatter(xs, ys, alpha=0.7, s=28, c=color, label=label, edgecolors="white", linewidth=0.4)
    if xs.size > 2 and np.std(xs) > 1e-9 and np.std(ys) > 1e-9:
        return float(np.corrcoef(xs, ys)[0, 1])
    return None


SPLIT_COLORS = {"train": "#2563eb", "test1": "#dc2626"}


def make_scatter_grid(rows, out_path: Path, ykey: str, ylabel: str, title_extra: str = ""):
    pairs = [
        ("depth_rmse_aligned", "Depth RMSE (scale-aligned)"),
        ("depth_rel_err", "Depth rel error"),
        ("fx_rel_err", "fx rel error"),
        ("principal_cx_err_px", "Principal-point cx err (px)"),
        ("relative_rot_err_deg", "Camera relative rot err (deg)"),
        ("relative_trans_dir_err_deg", "Camera relative trans dir err (deg)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    splits = sorted({r["split"] for r in rows})
    for ax, (xkey, xlbl) in zip(axes, pairs):
        title_corrs = []
        for split in splits:
            xs = [r[xkey] for r in rows if r["split"] == split]
            ys = [r[ykey] for r in rows if r["split"] == split]
            corr = _scatter_with_corr(ax, xs, ys, split, SPLIT_COLORS.get(split, "#888"))
            if corr is not None:
                title_corrs.append(f"{split} r={corr:+.2f}")
        # All-split correlation
        all_xs = [r[xkey] for r in rows]
        all_ys = [r[ykey] for r in rows]
        all_xs_np = np.asarray(all_xs, dtype=np.float64)
        all_ys_np = np.asarray(all_ys, dtype=np.float64)
        ok = np.isfinite(all_xs_np) & np.isfinite(all_ys_np)
        if ok.sum() > 2:
            r_all = float(np.corrcoef(all_xs_np[ok], all_ys_np[ok])[0, 1])
            title_corrs.append(f"all r={r_all:+.2f}")
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylabel)
        ax.set_title("  |  ".join(title_corrs) if title_corrs else "")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle(f"{ylabel} vs upstream signals{title_extra}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[PLOT] {out_path}")


def make_distribution_plot(rows, out_path: Path):
    keys_titles = [
        ("pose_l1", "Pose L1 (rot6d)"),
        ("rot_err_deg", "Rotation err (deg)"),
        ("trans_l2", "Translation L2"),
        ("depth_rmse_aligned", "Depth RMSE aligned"),
        ("fx_rel_err", "fx rel err"),
        ("relative_rot_err_deg", "Cam relative rot err (deg)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    splits = sorted({r["split"] for r in rows})
    for ax, (key, title) in zip(axes, keys_titles):
        for split in splits:
            data = np.asarray([r[key] for r in rows if r["split"] == split], dtype=np.float64)
            data = data[np.isfinite(data)]
            if data.size == 0:
                continue
            ax.hist(data, bins=20, alpha=0.55, label=f"{split} (n={data.size})",
                    color=SPLIT_COLORS.get(split, "#888"))
        ax.set_xlabel(title)
        ax.set_ylabel("count")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle("Per-split distributions", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[PLOT] {out_path}")


# ----------------------------- Cross-attention heatmaps -----------------------------

def _capture_cross_attn_outputs(model, scene_t, object_t, dtype):
    """Run forward pass and capture post-cross-attention scene tokens for each
    progressive object-prototype cross-attention layer.

    Returns:
        captured: dict[int, Tensor]  layer_idx -> (B, S*P, C) on CPU/float32
        layer_indices: sorted list of layer indices
        preds: dict of model predictions (object_pose, object_translation, ...)
    """
    blocks = model.object_token_cross_attn_blocks
    if blocks is None:
        raise RuntimeError(
            "Trained model has no object_token_cross_attn_blocks; "
            "heatmap generation requires enable_multi_layer_object_prototype_cross_attn=True"
        )
    layer_indices = sorted(int(k) for k in blocks.keys())

    captured = {}
    handles = []
    for L in layer_indices:
        def make_hook(l):
            def hook(_mod, _inputs, output):
                captured[l] = output.detach().float().cpu()
            return hook
        handles.append(blocks[str(L)].register_forward_hook(make_hook(L)))

    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                preds = model(scene_t, object_images=object_t)
    finally:
        for h in handles:
            h.remove()

    return captured, layer_indices, preds


def _pose_metrics_from_preds(preds, batch):
    pred_pose = preds["object_pose"].float().cpu().numpy()[0]
    pred_t = preds["object_translation"].float().cpu().numpy()[0]
    gt_R = np.asarray(batch["object_rotation"], dtype=np.float32)
    gt_pose = matrix_to_rot6d(gt_R)
    gt_t = np.asarray(batch["object_translation"], dtype=np.float32)
    return {
        "pose_l1": float(np.abs(pred_pose - gt_pose).mean()),
        "rot_err_deg": rotation_error_deg(rot6d_to_matrix(pred_pose), gt_R),
        "trans_l1": float(np.abs(pred_t - gt_t).mean()),
        "trans_l2": float(np.linalg.norm(pred_t - gt_t)),
    }


def _tokens_to_heatmaps(tokens, num_views, h_p, w_p):
    """tokens: (1, S*P, C) -> (S, H_p, W_p) per-token L2 norm."""
    t = tokens.view(1, num_views, h_p, w_p, -1)[0]
    return t.norm(dim=-1).numpy()


def generate_cross_attn_heatmaps(model, dataset, sample_meta, out_path: Path, seed: int):
    seed_everything(seed)
    batch = dataset.get_data(
        seq_index=sample_meta["sample_idx"],
        img_per_seq=NUM_SCENE_VIEWS,
        aspect_ratio=1.0,
    )
    scene_t = images_to_tensor(batch["images"])
    object_t = images_to_tensor(batch["object_images"])
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    captured, layer_indices, preds = _capture_cross_attn_outputs(model, scene_t, object_t, dtype)
    metrics = _pose_metrics_from_preds(preds, batch)

    H, W = scene_t.shape[-2:]
    patch = int(model.aggregator.patch_size)
    h_p, w_p = H // patch, W // patch
    s_scene = int(scene_t.shape[0])  # scene_t is (S, 3, H, W) before model adds batch dim

    rgb_views = [np.asarray(im, dtype=np.uint8) for im in batch["images"]]
    obj_views = [np.asarray(im, dtype=np.uint8) for im in batch["object_images"]]

    n_rows = 1 + len(layer_indices)
    n_cols = s_scene
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows), squeeze=False)

    for c in range(n_cols):
        ax = axes[0, c]
        ax.imshow(rgb_views[c])
        ax.set_title(f"scene view {c} (frame {int(batch['ids'][c])})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("input", fontsize=10)

    for r, L in enumerate(layer_indices, start=1):
        heat = _tokens_to_heatmaps(captured[L], s_scene, h_p, w_p)  # (S, h_p, w_p)
        # Per-layer global min/max so views are comparable within the same layer.
        vmin = float(np.nanmin(heat))
        vmax = float(np.nanmax(heat))
        if vmax - vmin < 1e-9:
            vmax = vmin + 1.0
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(rgb_views[c])
            heat_up = np.kron(heat[c], np.ones((patch, patch), dtype=np.float32))
            ax.imshow(heat_up, cmap="jet", alpha=0.5, vmin=vmin, vmax=vmax,
                      extent=(0, W, H, 0), interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel(f"layer {L}\n(||t||₂)", fontsize=10)

    title = (
        f"[{sample_meta['split']}] {sample_meta['scene_name']}  "
        f"obj={sample_meta['object_id']}  cat={sample_meta.get('category','')}  "
        f"sample_idx={sample_meta['sample_idx']}\n"
        f"pose_l1={metrics['pose_l1']:.4f}  rot_err={metrics['rot_err_deg']:.2f}°  "
        f"trans_l1={metrics['trans_l1']:.4f}  trans_l2={metrics['trans_l2']:.4f}"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(
        f"[HEATMAP] {out_path}  "
        f"pose_l1={metrics['pose_l1']:.4f} rot_err={metrics['rot_err_deg']:.2f} "
        f"trans_l1={metrics['trans_l1']:.4f} trans_l2={metrics['trans_l2']:.4f}"
    )

    # Also save a strip of the object reference views alongside, for context.
    obj_path = out_path.with_name(out_path.stem + "_object_views.png")
    n_obj = len(obj_views)
    fig2, axes2 = plt.subplots(1, n_obj, figsize=(3.0 * n_obj, 3.0), squeeze=False)
    for c in range(n_obj):
        axes2[0, c].imshow(obj_views[c])
        axes2[0, c].set_title(f"object view {c} (frame {int(batch['object_cam_indices'][c])})", fontsize=9)
        axes2[0, c].set_xticks([]); axes2[0, c].set_yticks([])
    fig2.suptitle(f"object reference (cat={sample_meta.get('category','')})", fontsize=11)
    fig2.tight_layout(rect=(0, 0, 1, 0.93))
    fig2.savefig(obj_path, dpi=120)
    plt.close(fig2)
    print(f"[HEATMAP] {obj_path}")

    return {
        **metrics,
        "split": sample_meta["split"],
        "sample_idx": sample_meta["sample_idx"],
        "object_id": sample_meta["object_id"],
        "scene_name": sample_meta["scene_name"],
        "category": sample_meta.get("category", ""),
        "heatmap_path": str(out_path),
    }


def find_all_samples_for_object(split: str, object_id: int, category: str = None):
    """Return ALL records (sample_idx, ...) in `split` matching the given object_id
    (and optionally category)."""
    ds = build_dataset(split)
    cat_lower = category.lower() if category else None
    out = []
    for idx, rec in enumerate(ds.records):
        if int(rec["target_object_id"]) != int(object_id):
            continue
        if cat_lower and str(rec.get("category", "")).lower() != cat_lower:
            continue
        out.append({
            "split": split,
            "sample_idx": idx,
            "scene_name": rec["scene_name"],
            "object_id": int(rec["target_object_id"]),
            "category": rec.get("category", ""),
        })
    return out


def find_sample_in_dataset(split: str, category: str, sample_idx_override=None, object_id_override=None):
    """Scan the full dataset records (not just the capped `samples` list) for a sample
    matching the given category, plus optional sample_idx/object_id constraints."""
    ds = build_dataset(split)
    cat_lower = category.lower()
    if sample_idx_override is not None:
        rec = ds.records[sample_idx_override % len(ds.records)]
        return {
            "split": split,
            "sample_idx": sample_idx_override,
            "scene_name": rec["scene_name"],
            "object_id": int(rec["target_object_id"]),
            "category": rec.get("category", ""),
        }
    fallback = None
    for idx, rec in enumerate(ds.records):
        if str(rec.get("category", "")).lower() != cat_lower:
            continue
        meta = {
            "split": split,
            "sample_idx": idx,
            "scene_name": rec["scene_name"],
            "object_id": int(rec["target_object_id"]),
            "category": rec.get("category", ""),
        }
        if fallback is None:
            fallback = meta
        if object_id_override is None or meta["object_id"] == object_id_override:
            return meta
    if object_id_override is not None and fallback is not None:
        print(f"[HEATMAP] object_id={object_id_override} not in {split} for category={category}; falling back to first match.")
    return fallback


def find_shared_object_id(category: str):
    """Return the smallest object_id that appears in BOTH train and test1 records within
    the requested category, or None. Scans full datasets, not the capped samples list."""
    cat_lower = category.lower()
    per_split = {}
    for split in SPLITS:
        ds = build_dataset(split)
        per_split[split] = {
            int(rec["target_object_id"])
            for rec in ds.records
            if str(rec.get("category", "")).lower() == cat_lower
        }
    if any(not v for v in per_split.values()):
        return None
    shared = set.intersection(*per_split.values())
    return min(shared) if shared else None


def print_summary(rows):
    keys = [
        "pose_l1", "rot_err_deg", "trans_l1", "trans_l2",
        "depth_rmse_aligned", "depth_rel_err",
        "fx_rel_err", "fy_rel_err",
        "principal_cx_err_px", "principal_cy_err_px",
        "relative_rot_err_deg", "relative_trans_dir_err_deg",
    ]
    splits = sorted({r["split"] for r in rows})
    print("\n=== Per-split summary ===")
    for split in splits:
        items = [r for r in rows if r["split"] == split]
        print(f"\n[{split}]  n={len(items)}")
        for k in keys:
            vals = np.asarray([r[k] for r in items], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                print(
                    f"  {k:32s}  mean={vals.mean():.6f}  median={np.median(vals):.6f}  "
                    f"std={vals.std():.6f}  min={vals.min():.6f}  max={vals.max():.6f}"
                )

    print("\n=== Pearson correlation: pose_l1 vs upstream signals ===")
    pose_l1 = np.asarray([r["pose_l1"] for r in rows], dtype=np.float64)
    for k in [
        "depth_rmse_aligned", "depth_rel_err",
        "fx_rel_err", "fy_rel_err",
        "principal_cx_err_px", "principal_cy_err_px",
        "relative_rot_err_deg", "relative_trans_dir_err_deg",
    ]:
        v = np.asarray([r[k] for r in rows], dtype=np.float64)
        ok = np.isfinite(pose_l1) & np.isfinite(v)
        if ok.sum() < 3 or np.std(pose_l1[ok]) < 1e-12 or np.std(v[ok]) < 1e-12:
            print(f"  pose_l1 ↔ {k:32s}  (insufficient variance)")
            continue
        r = float(np.corrcoef(pose_l1[ok], v[ok])[0, 1])
        print(f"  pose_l1 ↔ {k:32s}  r = {r:+.4f}  (n={ok.sum()})")


# ----------------------------- Main loop -----------------------------

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
                    "category": rec.get("category", ""),
                })
                count += 1
                if max_per_split and count >= max_per_split:
                    break
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="banana", help="Object category to evaluate")
    parser.add_argument("--max-per-split", type=int, default=40,
                        help="Cap samples per split. 0 = unlimited")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds per sample (averaged for stability stats)")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "eval_outputs"))
    parser.add_argument("--heatmap", action=argparse.BooleanOptionalAction, default=True,
                        help="Generate post-cross-attention token heatmaps for one fixed sample per split.")
    parser.add_argument("--heatmap-only", action="store_true",
                        help="Only generate heatmaps; skip the metric eval loop.")
    parser.add_argument("--heatmap-sample-idx", type=int, default=None,
                        help="Force a specific sample_idx to visualize (applied to whichever split contains it).")
    parser.add_argument("--heatmap-object-id", type=int, default=None,
                        help="Force a specific target_object_id for both splits. If unset, auto-pick the smallest object_id shared by train and test1 within the category.")
    parser.add_argument("--heatmap-all", action="store_true",
                        help="Generate heatmaps for ALL records in train and test1 with the chosen object_id (instead of only the first).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / args.category
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"category = {args.category}")
    print(f"output   = {out_dir}")

    samples = collect_samples(args.category, args.max_per_split)
    if not samples:
        print(f"No samples found for category '{args.category}'.")
        return
    counts = {s: sum(1 for x in samples if x["split"] == s) for s in SPLITS}
    print(f"Sample counts: {counts}")

    trained = build_trained_model()

    if args.heatmap:
        heatmap_dir = out_dir / "heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        chosen_obj_id = args.heatmap_object_id
        if chosen_obj_id is None and args.heatmap_sample_idx is None:
            chosen_obj_id = find_shared_object_id(args.category)
            if chosen_obj_id is not None:
                print(f"[HEATMAP] auto-selected object_id={chosen_obj_id} (shared by train and test1 within category={args.category})")
            else:
                print("[HEATMAP] no shared object_id found; using first sample per split")

        # Build the per-split task list of (split, picked_meta).
        heatmap_tasks = []
        for split in SPLITS:
            if args.heatmap_all:
                if chosen_obj_id is None:
                    print(f"[HEATMAP] --heatmap-all requires a fixed object_id; skipping {split}.")
                    continue
                metas = find_all_samples_for_object(split, chosen_obj_id, args.category)
                if not metas:
                    print(f"[HEATMAP] no records for object_id={chosen_obj_id} in {split}; skipping.")
                    continue
                print(f"[HEATMAP] {split}: {len(metas)} samples for object_id={chosen_obj_id}")
                heatmap_tasks.extend(metas)
            else:
                picked = find_sample_in_dataset(
                    split, args.category, args.heatmap_sample_idx, chosen_obj_id,
                )
                if picked is None:
                    print(f"[HEATMAP] No sample available for split={split}, skipping.")
                    continue
                heatmap_tasks.append(picked)

        heatmap_rows = []
        for picked in heatmap_tasks:
            ds = build_dataset(picked["split"])
            tag = f"{picked['split']}_idx{picked['sample_idx']}_obj{picked['object_id']}"
            out_path = heatmap_dir / f"crossattn_heatmap_{tag}.png"
            try:
                row = generate_cross_attn_heatmaps(
                    trained, ds, picked, out_path, seed=args.seed_base
                )
                if row is not None:
                    heatmap_rows.append(row)
            except Exception as exc:
                print(f"[HEATMAP] {picked['split']} idx={picked['sample_idx']} failed: {exc}")

        if heatmap_rows:
            csv_path = heatmap_dir / "heatmap_metrics.csv"
            write_csv(heatmap_rows, csv_path)

    if args.heatmap_only:
        print("[HEATMAP] --heatmap-only specified; skipping eval loop.")
        return

    official = build_official_model()

    rows = []
    for i, meta in enumerate(samples):
        per_seed_metrics = []
        for k in range(max(1, args.seeds)):
            seed = args.seed_base + k
            try:
                m = evaluate_sample(meta["split"], meta["sample_idx"], seed, trained, official)
            except Exception as exc:
                print(f"[{i+1}/{len(samples)}] {meta['split']} idx={meta['sample_idx']}  ERROR: {exc}")
                continue
            per_seed_metrics.append(m)
        if not per_seed_metrics:
            continue
        # Aggregate: take mean over seeds
        agg = {key: float(np.mean([m[key] for m in per_seed_metrics]))
               for key in per_seed_metrics[0]
               if isinstance(per_seed_metrics[0][key], (int, float))}
        agg["scene_views"] = per_seed_metrics[0]["scene_views"]
        agg["object_views"] = per_seed_metrics[0]["object_views"]
        agg["n_seeds"] = len(per_seed_metrics)
        if args.seeds > 1:
            agg["pose_l1_std"] = float(np.std([m["pose_l1"] for m in per_seed_metrics]))
            agg["trans_l2_std"] = float(np.std([m["trans_l2"] for m in per_seed_metrics]))
        agg.update(meta)
        rows.append(agg)
        print(
            f"[{i+1}/{len(samples)}] {meta['split']:5s} idx={meta['sample_idx']:4d} "
            f"obj={meta['object_id']:4d}  pose_l1={agg['pose_l1']:.4f}  "
            f"rot_err={agg['rot_err_deg']:.2f}  trans_l2={agg['trans_l2']:.4f}  "
            f"depth_rmse={agg['depth_rmse_aligned']:.4f}  fx_rel={agg['fx_rel_err']:.4f}  "
            f"cam_rot={agg['relative_rot_err_deg']:.2f}"
        )

    write_csv(rows, out_dir / "metrics.csv")
    make_scatter_grid(rows, out_dir / "scatter_pose_l1.png", "pose_l1", "Pose L1")
    make_scatter_grid(rows, out_dir / "scatter_trans_l2.png", "trans_l2", "Translation L2")
    make_distribution_plot(rows, out_dir / "distributions.png")
    print_summary(rows)


if __name__ == "__main__":
    main()
