import json
import os
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("GRADIO_TEMP_DIR", "/mnt/train-data-4-hdd/yian/freepose/baseline/tmp")

import cv2
import gradio as gr
import numpy as np
import torch
import trimesh
# OV9D_DEMO_DISABLE_ANCHORS=1 python /mnt/train-data-4-hdd/yian/freepose/baseline/demo_gradio_ov9d.py
REPO_ROOT = Path(__file__).resolve().parent
TRAINING_ROOT = REPO_ROOT / "training"
for import_root in (REPO_ROOT, TRAINING_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from data.datasets.ov9d_multi_pose_normalize import OV9DMultiPoseNormalizeDataset
from vggt.models.vggt import VGGT


DATA_ROOT = Path(os.environ.get("OV9D_DATA_ROOT", "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d"))
CONFIG_PATH = Path(
    os.environ.get(
        "OV9D_OBJECT_CONFIG",
        "/mnt/train-data-4-hdd/yian/freepose/baseline_0503/training/config/default_object.yaml",
    )
)
SPLIT_JSON_ROOT = Path(
    os.environ.get(
        "OV9D_SPLIT_JSON_ROOT",
        "/mnt/train-data-4-hdd/yian/freepose/baseline_0503/splits_multi_4_3000",
    )
)
CKPT_PATH = Path(
    os.environ.get(
        "OV9D_POSE_CKPT",
        "/mnt/train-data-4-hdd/yian/freepose/baseline_0503/training/logs/0508_model/checkpoint.pt",
    )
)
VIEWER_CACHE_ROOT = Path(os.environ.get("OV9D_DEMO_CACHE", "/tmp/demo_gradio_ov9d_multi"))
SPLITS = ("train", "test1")
NUM_SCENE_VIEWS = 4
NUM_OBJECT_VIEWS = 4
FIXED_OBJECT_IMAGE_IDS = [10, 20, 30, 40]
SCENE_POINT_LIMIT = int(os.environ.get("OV9D_SCENE_POINT_LIMIT", "250000"))
SERVER_NAME = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
GRADIO_SHARE = os.environ.get("GRADIO_SHARE", "0") == "1"
DISABLE_ANCHORS = os.environ.get("OV9D_DEMO_DISABLE_ANCHORS", "0") == "1"

device = "cuda" if torch.cuda.is_available() else "cpu"


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


DATASET_CACHE = {}
SPLIT_MANIFEST_CACHE = {}


def build_dataset(split: str):
    split = split if split in SPLITS else "test1"
    if split not in DATASET_CACHE:
        DATASET_CACHE[split] = OV9DMultiPoseNormalizeDataset(
            common_conf=make_common_conf(),
            split=split,
            DATA_ROOT=str(DATA_ROOT),
            SPLIT_JSON=str(SPLIT_JSON_ROOT / f"{split}.json"),
            verify_files=True,
            num_scene_views=NUM_SCENE_VIEWS,
            num_object_views=NUM_OBJECT_VIEWS,
            fixed_object_view_ids=FIXED_OBJECT_IMAGE_IDS,
            load_point_map=True,
            scale_by_points=True,
            negative_object_prob=0.0,
        )
    return DATASET_CACHE[split]


def split_manifest_by_scene(split: str):
    split = split if split in SPLITS else "test1"
    if split not in SPLIT_MANIFEST_CACHE:
        path = SPLIT_JSON_ROOT / f"{split}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        SPLIT_MANIFEST_CACHE[split] = {
            str(item["scene_name"]): item for item in payload.get("scenes", [])
        }
    return SPLIT_MANIFEST_CACHE[split]


def _format_sample_label(split: str, idx: int, rec: dict) -> str:
    return (
        f"[{split}] {idx:04d} | {rec['scene_name']} | "
        f"obj_{int(rec['target_object_id']):06d} | {rec.get('category', '')}"
    )


def sample_labels(split: str):
    dataset = build_dataset(split)
    return [_format_sample_label(split, idx, rec) for idx, rec in enumerate(dataset.records)]


def sample_labels_same_object(target_object_id: int):
    labels = []
    for split in SPLITS:
        dataset = build_dataset(split)
        for idx, rec in enumerate(dataset.records):
            if int(rec["target_object_id"]) == int(target_object_id):
                labels.append(_format_sample_label(split, idx, rec))
    return labels


def anchor_object_ids():
    """Return the ordered list of anchor_object_ids declared in the train split JSON.

    Falls back to an empty list if the field is absent (older split formats).
    """
    train_path = SPLIT_JSON_ROOT / "train.json"
    if not train_path.exists():
        return []
    payload = json.loads(train_path.read_text(encoding="utf-8"))
    return [int(o) for o in payload.get("anchor_object_ids", [])]


QUADRANT_CSV = REPO_ROOT / "eval_outputs" / "anchor_count_vs_error" / "all_anchors_sorted.csv"
SYMMETRY_CSV = REPO_ROOT / "eval_outputs" / "anchor_count_vs_error" / "all_anchors_with_symmetry.csv"
SYMMETRY_QUADRANT_PLOT = REPO_ROOT / "eval_outputs" / "anchor_count_vs_error" / "symmetry_vs_quadrant.png"
SYMMETRY_QUADRANT_TABLE = REPO_ROOT / "eval_outputs" / "anchor_count_vs_error" / "symmetry_vs_quadrant_table.txt"
QUADRANT_ORDER = ("Good", "Normal", "Worse", "Worst")


def load_quadrant_map():
    """Return ``{object_id: pose_l1_delta_class}`` from analyze_all_anchors.py output.

    Returns ``{}`` if the CSV is missing — the demo still works without
    pose-L1 class filtering.
    """
    import csv as _csv
    if not QUADRANT_CSV.exists():
        return {}
    out = {}
    with open(QUADRANT_CSV, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            try:
                out[int(row["object_id"])] = row["quadrant"]
            except (KeyError, ValueError):
                continue
    return out


def load_symmetry_map():
    """Return ``{object_id: symmetry_tag}`` from the symmetry analysis CSV."""
    import csv as _csv
    if not SYMMETRY_CSV.exists():
        return {}
    out = {}
    with open(SYMMETRY_CSV, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            try:
                out[int(row["object_id"])] = row["symmetry"]
            except (KeyError, ValueError):
                continue
    return out


def collect_anchor_object_examples():
    """Build one example per (split, anchor_object_id), tagged by pose-L1 class.

    Returns a list of three-column rows ``[[class, symmetry, label], ...]`` for the
    Gradio Dataset / Examples block. Each label encodes its split via the
    ``[split]`` prefix used by ``parse_sample_label``. Class comes from
    ``all_anchors_sorted.csv``; anchors absent from that CSV (12 anchors with
    no test1 sample) get tag ``"Untagged"``.
    """
    anchors = anchor_object_ids()
    if not anchors:
        return []
    quadrant_map = load_quadrant_map()
    symmetry_map = load_symmetry_map()
    rows = []
    for split in SPLITS:
        dataset = build_dataset(split)
        first_idx_for_obj = {}
        for idx, rec in enumerate(dataset.records):
            oid = int(rec["target_object_id"])
            if oid in first_idx_for_obj:
                continue
            first_idx_for_obj[oid] = idx
        for oid in anchors:
            idx = first_idx_for_obj.get(oid)
            if idx is None:
                continue
            rec = dataset.records[idx]
            quadrant = quadrant_map.get(int(oid), "Untagged")
            symmetry = symmetry_map.get(int(oid), "Unknown")
            rows.append([quadrant, symmetry, _format_sample_label(split, idx, rec)])
    return rows


def filter_examples_by_quadrant(all_rows, selected):
    """Return rows whose pose-L1 class tag is in ``selected``. Empty selection → all."""
    if not selected:
        return list(all_rows)
    sel = set(selected)
    return [r for r in all_rows if r[0] in sel]


def anchor_dropdown_choices(rows):
    """Format anchor rows as ``(display, value)`` tuples for ``gr.Dropdown``.

    Display includes the quadrant + symmetry tags so the user can scan; value is
    the bare sample label that ``parse_sample_label`` / ``load_example_and_generate``
    already understand.
    """
    return [(f"[{q}] [{s}] {label}", label) for q, s, label in rows]


def load_symmetry_quadrant_report():
    plot = str(SYMMETRY_QUADRANT_PLOT) if SYMMETRY_QUADRANT_PLOT.exists() else None
    if SYMMETRY_QUADRANT_TABLE.exists():
        table = SYMMETRY_QUADRANT_TABLE.read_text(encoding="utf-8")
    else:
        table = "Symmetry report not found. Run `python baseline/analyze_symmetry_vs_quadrant.py` first."
    return plot, f"```text\n{table}\n```"


def parse_sample_label(label: str):
    if not label:
        return SPLITS[0], 0
    text = str(label).strip()
    split = SPLITS[0]
    if text.startswith("["):
        end = text.find("]")
        if end > 0:
            split = text[1:end].strip()
            text = text[end + 1:].strip()
    idx = int(text.split("|", 1)[0].strip())
    return split, idx


def parse_sample_index(label: str) -> int:
    return parse_sample_label(label)[1]


def get_target_object_id(label: str) -> int:
    split, idx = parse_sample_label(label)
    dataset = build_dataset(split)
    rec = dataset.records[idx % dataset.sequence_list_len]
    return int(rec["target_object_id"])


def seed_everything(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_batch(sample_label: str, seed: int):
    seed_everything(seed)
    split, parsed_idx = parse_sample_label(sample_label)
    dataset = build_dataset(split)
    idx = parsed_idx % dataset.sequence_list_len
    batch = dataset.get_data(seq_index=idx, img_per_seq=NUM_SCENE_VIEWS, aspect_ratio=1.0)
    return split, dataset, idx, batch


def images_to_tensor(images):
    tensors = []
    for image in images:
        arr = np.asarray(image, dtype=np.uint8)
        tensors.append(torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float().div(255.0))
    return torch.stack(tensors, dim=0).to(device)


def masks_to_tensor(masks):
    arr = np.stack([np.asarray(mask, dtype=np.float32) for mask in masks], axis=0)
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def gallery_images(images, captions):
    return [(np.asarray(img, dtype=np.uint8), caption) for img, caption in zip(images, captions)]


def mask_gallery(images, masks, captions):
    out = []
    for image, mask, caption in zip(images, masks, captions):
        rgb = np.asarray(image, dtype=np.uint8).copy()
        mask_bool = np.asarray(mask).astype(bool)
        overlay = rgb.copy()
        overlay[mask_bool] = (0.45 * overlay[mask_bool] + 0.55 * np.array([255, 60, 40])).astype(np.uint8)
        out.append((overlay, caption))
    return out


def rot6d_to_matrix(rot6d):
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(3, 2)
    x_raw = rot6d[:, 0]
    y_raw = rot6d[:, 1]
    x = x_raw / max(np.linalg.norm(x_raw), 1e-12)
    y = y_raw - np.dot(x, y_raw) * x
    y = y / max(np.linalg.norm(y), 1e-12)
    z = np.cross(x, y)
    z = z / max(np.linalg.norm(z), 1e-12)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def matrix_to_rot6d(matrix):
    return np.asarray(matrix, dtype=np.float32)[:, :2].reshape(-1)


def rotation_error_degrees(pred_rot6d, gt_matrix):
    pred_matrix = rot6d_to_matrix(pred_rot6d)
    rel = pred_matrix @ np.asarray(gt_matrix, dtype=np.float32).T
    cos_theta = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def translation_error(pred_t, gt_t):
    pred_t = np.asarray(pred_t, dtype=np.float64)
    gt_t = np.asarray(gt_t, dtype=np.float64)
    diff = pred_t - gt_t
    return {
        "l2": float(np.linalg.norm(diff)),
        "abs_xyz": np.abs(diff).tolist(),
        "signed_xyz": diff.tolist(),
    }


def bbox_points_from_models_info(object_id: int, normalization_scale: float, scale_multiplier: float):
    info_path = DATA_ROOT / "models_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))[str(int(object_id))]
    min_xyz = np.array([info["min_x"], info["min_y"], info["min_z"]], dtype=np.float32)
    size_xyz = np.array([info["size_x"], info["size_y"], info["size_z"]], dtype=np.float32)
    max_xyz = min_xyz + size_xyz
    corners = np.asarray(
        [
            [min_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], max_xyz[1], min_xyz[2]],
            [min_xyz[0], max_xyz[1], min_xyz[2]],
            [min_xyz[0], min_xyz[1], max_xyz[2]],
            [max_xyz[0], min_xyz[1], max_xyz[2]],
            [max_xyz[0], max_xyz[1], max_xyz[2]],
            [min_xyz[0], max_xyz[1], max_xyz[2]],
        ],
        dtype=np.float32,
    )
    return corners / float(normalization_scale) * float(scale_multiplier)


def transform_points(points_obj, rot6d, translation):
    rot = rot6d_to_matrix(rot6d)
    trans = np.asarray(translation, dtype=np.float32)
    return np.asarray(points_obj, dtype=np.float32) @ rot.T + trans[None, :]


BBOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def project_points(points, extrinsic, intrinsic):
    points = np.asarray(points, dtype=np.float32)
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    cam = points @ extrinsic[:3, :3].T + extrinsic[:3, 3][None, :]
    z = cam[:, 2]
    valid = z > 1e-6
    uv = np.zeros((points.shape[0], 2), dtype=np.float32)
    uv[:, 0] = intrinsic[0, 0] * cam[:, 0] / np.maximum(z, 1e-6) + intrinsic[0, 2]
    uv[:, 1] = intrinsic[1, 1] * cam[:, 1] / np.maximum(z, 1e-6) + intrinsic[1, 2]
    return uv, valid


def draw_bbox_projection(image, bbox_points, extrinsic, intrinsic, color):
    canvas = np.asarray(image, dtype=np.uint8).copy()
    uv, valid = project_points(bbox_points, extrinsic, intrinsic)
    h, w = canvas.shape[:2]
    for i, j in BBOX_EDGES:
        if not (valid[i] and valid[j]):
            continue
        p0 = tuple(np.round(uv[i]).astype(int))
        p1 = tuple(np.round(uv[j]).astype(int))
        if not (-w <= p0[0] <= 2 * w and -h <= p0[1] <= 2 * h and -w <= p1[0] <= 2 * w and -h <= p1[1] <= 2 * h):
            continue
        cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)
    return canvas


def pose_axis_points(rot6d, translation, axis_length):
    trans = np.asarray(translation, dtype=np.float32)
    rot = rot6d_to_matrix(rot6d)
    axes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float32,
    )
    return axes @ rot.T + trans[None, :]


def draw_axes_projection(image, rot6d, translation, axis_length, extrinsic, intrinsic, axis_colors=None):
    canvas = np.asarray(image, dtype=np.uint8).copy()
    points = pose_axis_points(rot6d, translation, axis_length)
    uv, valid = project_points(points, extrinsic, intrinsic)
    if not valid[0]:
        return canvas

    h, w = canvas.shape[:2]
    origin = tuple(np.round(uv[0]).astype(int))
    if not (-w <= origin[0] <= 2 * w and -h <= origin[1] <= 2 * h):
        return canvas

    if axis_colors is None:
        axis_colors = [
            (255, 80, 80),   # object +X
            (80, 255, 255),  # object +Y
            (255, 220, 80),  # object +Z
        ]
    axis_labels = ["X", "Y", "Z"]
    for axis_idx in range(3):
        point_idx = axis_idx + 1
        if not valid[point_idx]:
            continue
        end = tuple(np.round(uv[point_idx]).astype(int))
        if not (-w <= end[0] <= 2 * w and -h <= end[1] <= 2 * h):
            continue
        cv2.arrowedLine(canvas, origin, end, axis_colors[axis_idx], 3, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(
            canvas,
            axis_labels[axis_idx],
            end,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            axis_colors[axis_idx],
            2,
            cv2.LINE_AA,
        )
    cv2.circle(canvas, origin, 4, (255, 255, 255), -1, cv2.LINE_AA)
    return canvas


def build_projection_gallery(batch, pred_bbox, gt_bbox, pred_pose, gt_pose, pred_translation, gt_translation):
    gallery = []
    axis_length = float(np.linalg.norm(gt_bbox.max(axis=0) - gt_bbox.min(axis=0)) * 0.45)
    for idx, image in enumerate(batch["images"]):
        canvas = np.asarray(image, dtype=np.uint8).copy()
        canvas = draw_bbox_projection(canvas, gt_bbox, batch["extrinsics"][idx], batch["intrinsics"][idx], (80, 150, 255))
        canvas = draw_bbox_projection(canvas, pred_bbox, batch["extrinsics"][idx], batch["intrinsics"][idx], (60, 255, 90))
        canvas = draw_axes_projection(canvas, gt_pose, gt_translation, axis_length, batch["extrinsics"][idx], batch["intrinsics"][idx])
        canvas = draw_axes_projection(
            canvas,
            pred_pose,
            pred_translation,
            axis_length,
            batch["extrinsics"][idx],
            batch["intrinsics"][idx],
            axis_colors=[(60, 255, 90), (60, 220, 255), (255, 230, 60)],
        )
        gallery.append((canvas, f"scene {int(batch['ids'][idx])} | green bbox=pred blue bbox=gt | axes=pose"))
    return gallery


def subsample_points(points, colors, limit):
    if limit <= 0 or points.shape[0] <= limit:
        return points, colors
    order = np.linspace(0, points.shape[0] - 1, limit).astype(np.int64)
    return points[order], colors[order]


def scene_points_and_colors(batch):
    all_points = []
    all_colors = []
    for image, points, mask in zip(batch["images"], batch["world_points"], batch["point_masks"]):
        valid = np.asarray(mask, dtype=bool) & np.isfinite(points).all(axis=-1)
        all_points.append(np.asarray(points, dtype=np.float32)[valid])
        all_colors.append(np.asarray(image, dtype=np.uint8)[valid])
    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    return subsample_points(points, colors, SCENE_POINT_LIMIT)


def bbox_mesh(bbox_points, color_rgba):
    bbox_points = np.asarray(bbox_points, dtype=np.float32)
    diag = float(np.linalg.norm(bbox_points.max(axis=0) - bbox_points.min(axis=0)))
    radius = max(diag * 0.01, 0.002)
    meshes = []
    color = np.asarray(color_rgba, dtype=np.uint8)
    for i, j in BBOX_EDGES:
        mesh = trimesh.creation.cylinder(radius=radius, segment=np.stack([bbox_points[i], bbox_points[j]], axis=0))
        mesh.visual.face_colors = np.tile(color[None, :], (len(mesh.faces), 1))
        meshes.append(mesh)
    return trimesh.util.concatenate(meshes)


def axes_mesh(rot6d, translation, length, radius, center_color):
    trans = np.asarray(translation, dtype=np.float32)
    rot = rot6d_to_matrix(rot6d)
    axes = np.asarray([[length, 0, 0], [0, length, 0], [0, 0, length]], dtype=np.float32)
    endpoints = axes @ rot.T + trans[None, :]
    meshes = []
    center = trimesh.creation.icosphere(subdivisions=2, radius=radius * 1.8)
    center.apply_translation(trans)
    center.visual.face_colors = np.tile(np.asarray(center_color, dtype=np.uint8)[None, :], (len(center.faces), 1))
    meshes.append(center)
    colors = [np.array([255, 80, 80, 255]), np.array([80, 255, 255, 255]), np.array([255, 220, 80, 255])]
    for idx in range(3):
        mesh = trimesh.creation.cylinder(radius=radius, segment=np.stack([trans, endpoints[idx]], axis=0))
        mesh.visual.face_colors = np.tile(colors[idx][None, :], (len(mesh.faces), 1))
        meshes.append(mesh)
    return trimesh.util.concatenate(meshes)


def build_glb(batch, pred_bbox, gt_bbox, pred_pose, gt_pose, pred_translation, gt_translation, overlay_mode):
    split = str(batch["seq_name"]).split("/", 1)[0]
    safe_scene = re.sub(r"[^a-zA-Z0-9_]+", "_", str(batch["scene_name"]))[:80]
    safe_obj = re.sub(r"[^a-zA-Z0-9_]+", "_", str(batch["object_name"]))[:40]
    out_dir = VIEWER_CACHE_ROOT / split / safe_scene / safe_obj
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{overlay_mode.lower().replace(' ', '_')}.glb"

    points, colors = scene_points_and_colors(batch)
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=points, colors=colors))
    if overlay_mode in {"Pred + GT BBox", "Pred BBox"}:
        scene.add_geometry(bbox_mesh(pred_bbox, [60, 255, 90, 255]))
    if overlay_mode in {"Pred + GT BBox", "GT BBox"}:
        scene.add_geometry(bbox_mesh(gt_bbox, [80, 150, 255, 255]))
    if overlay_mode in {"Pose Axes", "Pred + GT BBox"}:
        diag = float(np.linalg.norm(gt_bbox.max(axis=0) - gt_bbox.min(axis=0)))
        scene.add_geometry(axes_mesh(pred_pose, pred_translation, diag * 0.45, max(diag * 0.012, 0.002), [255, 80, 80, 255]))
        scene.add_geometry(axes_mesh(gt_pose, gt_translation, diag * 0.45, max(diag * 0.012, 0.002), [80, 150, 255, 255]))
    scene.export(out_path)
    return str(out_path), int(points.shape[0])


def build_model():
    print("Initializing OV9D multi-object pose VGGT...")
    model = VGGT(
        enable_camera=False,
        enable_depth=False,
        enable_point=False,
        enable_track=False,
        enable_object_point=False,
        enable_object_mask=True,
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
    print(f"Loading checkpoint: {CKPT_PATH}")
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    model.eval()
    return model.to(device)


model = None if os.environ.get("OV9D_DEMO_SKIP_MODEL", "0") == "1" else build_model()


def describe_selection(sample_label: str, seed: int):
    split, dataset, idx, batch = load_batch(sample_label, seed)
    scene_captions = [f"scene {int(i)}" for i in batch["ids"]]
    object_captions = [f"object {int(i)}" for i in batch["object_cam_indices"]]
    message = (
        f"Split `{split}` sample `{idx}`: scene `{batch['scene_name']}`, "
        f"target `{batch['object_name']}` category `{batch.get('category', '')}`, "
        f"reference `{batch['object_reference_scene_name']}`. Dataset records: `{dataset.sequence_list_len}`."
    )
    processed_images = batch["images"]
    return (
        gallery_images(processed_images, scene_captions),
        gallery_images(batch["object_images"], object_captions),
        message,
        "No inference yet.",
        None,
        [],
    )


def refresh_split(split: str, sample_label: str, same_object: bool, seed: int):
    if same_object:
        target_id = get_target_object_id(sample_label) if sample_label else None
        labels = sample_labels_same_object(target_id) if target_id is not None else []
    else:
        labels = sample_labels(split)
    if sample_label in labels:
        value = sample_label
    else:
        value = labels[0] if labels else None
    return (
        gr.update(choices=labels, value=value),
        *describe_selection(value, seed),
    )


def refresh_sample(sample_label: str, seed: int):
    return describe_selection(sample_label, seed)


def toggle_same_object(same_object: bool, split: str, sample_label: str, seed: int):
    if same_object:
        target_id = get_target_object_id(sample_label) if sample_label else None
        labels = sample_labels_same_object(target_id) if target_id is not None else []
    else:
        labels = sample_labels(split)
    if sample_label in labels:
        value = sample_label
    else:
        value = labels[0] if labels else None
    return (
        gr.update(choices=labels, value=value),
        *describe_selection(value, seed),
    )


def load_example(label: str, same_object: bool, seed: int):
    """Load a sample by its ``[split] idx | ...`` label.

    Wired to the hidden Examples proxy textbox so a single click both flips the
    split dropdown and repopulates the sample dropdown's choices+value before
    rendering the selection.
    """
    if not label:
        return (gr.update(), gr.update(), *describe_selection(None, seed))
    split, _ = parse_sample_label(label)
    if same_object:
        target_id = get_target_object_id(label)
        labels = sample_labels_same_object(target_id)
    else:
        labels = sample_labels(split)
    if label not in labels:
        labels = [label] + labels
    return (
        gr.update(value=split),
        gr.update(choices=labels, value=label),
        *describe_selection(label, seed),
    )


def load_example_and_generate(
    label: str,
    same_object: bool,
    seed: int,
    overlay_mode: str,
    object_scale_multiplier: float,
):
    """Load a clicked example row and auto-run inference when the model is available."""
    split_update, sample_update, scene_gallery, object_gallery, log, metrics, viewer, projection = load_example(
        label, same_object, seed
    )
    if not label:
        return split_update, sample_update, scene_gallery, object_gallery, log, metrics, viewer, projection
    if model is None or not torch.cuda.is_available():
        note = " Auto-generate is unavailable in this demo instance; sample loaded only."
        return (
            split_update,
            sample_update,
            scene_gallery,
            object_gallery,
            log + note,
            metrics,
            viewer,
            projection,
        )
    infer_log, infer_metrics, infer_viewer, infer_projection = run_inference(
        label, seed, overlay_mode, object_scale_multiplier
    )
    return (
        split_update,
        sample_update,
        scene_gallery,
        object_gallery,
        infer_log,
        infer_metrics,
        infer_viewer,
        infer_projection,
    )


def run_inference(sample_label: str, seed: int, overlay_mode: str, object_scale_multiplier: float):
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. This checkpoint is too large for CPU demo inference.")
    if model is None:
        raise ValueError("Model is not loaded because OV9D_DEMO_SKIP_MODEL=1.")
    split, _, _, batch = load_batch(sample_label, seed)
    scene_images = images_to_tensor(batch["images"])
    object_images = images_to_tensor(batch["object_images"])
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    start = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(scene_images, object_images=object_images)
    elapsed = time.time() - start

    pred_pose = predictions["object_pose"].detach().cpu().numpy()[0]
    pred_translation = predictions["object_translation"].detach().cpu().numpy()[0]
    gt_rotation = np.asarray(batch["object_rotation"], dtype=np.float32)
    gt_pose = matrix_to_rot6d(gt_rotation)
    gt_translation = np.asarray(batch["object_translation"], dtype=np.float32)
    normalization_scale = float(np.asarray(batch["normalization_scale"]).reshape(-1)[0])
    object_id = int(batch["object_id"])

    bbox_obj = bbox_points_from_models_info(object_id, normalization_scale, object_scale_multiplier)
    pred_bbox = transform_points(bbox_obj, pred_pose, pred_translation)
    gt_bbox = transform_points(bbox_obj, gt_pose, gt_translation)
    projection = build_projection_gallery(
        batch,
        pred_bbox,
        gt_bbox,
        pred_pose,
        gt_pose,
        pred_translation,
        gt_translation,
    )
    glb_path, scene_point_count = build_glb(
        batch,
        pred_bbox,
        gt_bbox,
        pred_pose,
        gt_pose,
        pred_translation,
        gt_translation,
        overlay_mode,
    )

    rot_err = rotation_error_degrees(pred_pose, gt_rotation)
    trans_err = translation_error(pred_translation, gt_translation)
    pose_l1 = float(np.abs(pred_pose - gt_pose).mean())
    trans_l1 = float(np.abs(pred_translation - gt_translation).mean())

    metrics = (
        f"**Metrics**  \n"
        f"Rotation Error: `{rot_err:.4f} deg`  \n"
        f"Translation L2: `{trans_err['l2']:.6f}`  \n"
        f"Pose L1: `{pose_l1:.6f}`  \n"
        f"Translation L1: `{trans_l1:.6f}`  \n"
        f"Normalization Scale: `{normalization_scale:.6f}`  \n"
        f"Pred Translation: `{[round(float(v), 6) for v in pred_translation.tolist()]}`  \n"
        f"GT Translation: `{[round(float(v), 6) for v in gt_translation.tolist()]}`  \n"
        f"Scene Points In GLB: `{scene_point_count}`  \n"
        f"Inference Time: `{elapsed:.2f}s`"
    )
    log = (
        f"Done. split={split}, sample={parse_sample_index(sample_label)}, "
        f"scene={batch['scene_name']}, object={batch['object_name']}, "
        f"scene_views={batch['ids'].tolist()}, object_views={batch['object_cam_indices'].tolist()}."
    )

    report = {
        "split": split,
        "sample_index": parse_sample_index(sample_label),
        "scene_name": batch["scene_name"],
        "object_id": object_id,
        "object_name": batch["object_name"],
        "category": batch.get("category", ""),
        "scene_views": batch["ids"].tolist(),
        "object_views": batch["object_cam_indices"].tolist(),
        "prediction": {
            "object_pose_rot6d": pred_pose.tolist(),
            "object_translation": pred_translation.tolist(),
        },
        "ground_truth": {
            "object_pose_rot6d": gt_pose.tolist(),
            "object_translation": gt_translation.tolist(),
        },
        "errors": {
            "rotation_deg": rot_err,
            "translation": trans_err,
            "pose_l1": pose_l1,
            "translation_l1": trans_l1,
        },
        "assets": {
            "glb": glb_path,
        },
    }
    report_path = Path(glb_path).with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return log, metrics, glb_path, projection


theme = gr.themes.Ocean()
initial_split = "test1"
initial_labels = sample_labels(initial_split)
initial_sample = initial_labels[0] if initial_labels else None
ANCHOR_OBJECT_EXAMPLES = [] if DISABLE_ANCHORS else collect_anchor_object_examples()
QUADRANT_FILTER_CHOICES = list(QUADRANT_ORDER) + (
    ["Untagged"] if any(r[0] == "Untagged" for r in ANCHOR_OBJECT_EXAMPLES) else []
)

with gr.Blocks(theme=theme) as demo:
    gr.HTML(
        """
        <h1>OV9D Multi-Object VGGT Pose Demo</h1>
        <p>
        This page samples <code>OV9DMultiPoseNormalizeDataset</code> directly from <code>ov9d</code>,
        loads the trained pose checkpoint, and visualizes predicted normalized object pose with
        2D projected boxes plus a rotatable GLB scene point cloud.
        </p>
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            split_dropdown = gr.Dropdown(choices=list(SPLITS), value=initial_split, label="Dataset Split")
            same_object_check = gr.Checkbox(
                label="Same Object Across Splits",
                value=False,
                info="When ON, the sample list shows scenes from BOTH train and test1 that share the currently selected sample's target object id.",
            )
            sample_dropdown = gr.Dropdown(choices=initial_labels, value=initial_sample, label="Sample")
            seed_input = gr.Number(label="Sampling Seed", value=42, precision=0)
            overlay_mode = gr.Radio(
                choices=["Pred + GT BBox", "Pred BBox", "GT BBox", "Pose Axes", "Scene Only"],
                value="Pred + GT BBox",
                label="3D Overlay",
            )
            scale_slider = gr.Slider(0.25, 4.0, value=1.0, step=0.05, label="Object Box Scale")
            refresh_btn = gr.Button("Refresh Sample")
            generate_btn = gr.Button("Generate", variant="primary")
            log_output = gr.Markdown("Select a sample, then click Generate.")
            metrics_output = gr.Markdown("No inference yet.")
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Inputs"):
                    scene_gallery = gr.Gallery(label="Processed Scene Views", columns=4, height="260px", object_fit="contain")
                    object_gallery = gr.Gallery(label="Object Reference Views", columns=4, height="260px", object_fit="contain")
                with gr.Tab("2D Projection"):
                    projection_gallery = gr.Gallery(label="Projected 3D Boxes", columns=2, height="620px", object_fit="contain")
                with gr.Tab("3D Viewer"):
                    model_viewer = gr.Model3D(label="Scene Point Cloud + Object Pose Overlay", height=720)
                with gr.Tab("Symmetry vs Quadrant"):
                    symmetry_plot = gr.Image(
                        value=str(SYMMETRY_QUADRANT_PLOT) if SYMMETRY_QUADRANT_PLOT.exists() else None,
                        label="Object symmetry composition by pose-L1 class",
                        type="filepath",
                        height=520,
                    )
                    symmetry_table = gr.Markdown(load_symmetry_quadrant_report()[1])

    if ANCHOR_OBJECT_EXAMPLES:
        with gr.Accordion(
            f"Anchor object examples ({len(ANCHOR_OBJECT_EXAMPLES)} entries — "
            f"one scene per anchor_object_id, train + test1). Pick from dropdown to load.",
            open=True,
        ):
            quadrant_filter = gr.CheckboxGroup(
                choices=QUADRANT_FILTER_CHOICES,
                value=QUADRANT_FILTER_CHOICES,
                label=(
                    "Pose L1 delta filter — delta = test1_pose_l1 - train_pose_l1. "
                    "Good ≤ 0.10, Normal ≤ 0.15, Worse ≤ 0.20, Worst > 0.20. "
                    "Untick a tag to narrow the dropdown."
                ),
            )
            anchor_dropdown = gr.Dropdown(
                choices=anchor_dropdown_choices(ANCHOR_OBJECT_EXAMPLES),
                value=None,
                label="Anchor sample (type to search)",
                filterable=True,
            )

    demo.load(
        fn=lambda: describe_selection(initial_sample, 42),
        inputs=[],
        outputs=[scene_gallery, object_gallery, log_output, metrics_output, model_viewer, projection_gallery],
    )
    split_dropdown.change(
        fn=refresh_split,
        inputs=[split_dropdown, sample_dropdown, same_object_check, seed_input],
        outputs=[sample_dropdown, scene_gallery, object_gallery, log_output, metrics_output, model_viewer, projection_gallery],
    )
    same_object_check.change(
        fn=toggle_same_object,
        inputs=[same_object_check, split_dropdown, sample_dropdown, seed_input],
        outputs=[sample_dropdown, scene_gallery, object_gallery, log_output, metrics_output, model_viewer, projection_gallery],
    )
    sample_dropdown.change(
        fn=refresh_sample,
        inputs=[sample_dropdown, seed_input],
        outputs=[scene_gallery, object_gallery, log_output, metrics_output, model_viewer, projection_gallery],
    )
    seed_input.change(
        fn=refresh_sample,
        inputs=[sample_dropdown, seed_input],
        outputs=[scene_gallery, object_gallery, log_output, metrics_output, model_viewer, projection_gallery],
    )
    refresh_btn.click(
        fn=refresh_sample,
        inputs=[sample_dropdown, seed_input],
        outputs=[scene_gallery, object_gallery, log_output, metrics_output, model_viewer, projection_gallery],
    )
    generate_btn.click(
        fn=run_inference,
        inputs=[sample_dropdown, seed_input, overlay_mode, scale_slider],
        outputs=[log_output, metrics_output, model_viewer, projection_gallery],
    )
    if ANCHOR_OBJECT_EXAMPLES:
        anchor_dropdown.change(
            fn=load_example_and_generate,
            inputs=[anchor_dropdown, same_object_check, seed_input, overlay_mode, scale_slider],
            outputs=[split_dropdown, sample_dropdown, scene_gallery, object_gallery, log_output, metrics_output, model_viewer, projection_gallery],
        )
        quadrant_filter.change(
            fn=lambda sel: gr.update(
                choices=anchor_dropdown_choices(filter_examples_by_quadrant(ANCHOR_OBJECT_EXAMPLES, sel)),
                value=None,
            ),
            inputs=[quadrant_filter],
            outputs=[anchor_dropdown],
        )


if __name__ == "__main__":
    demo.queue(max_size=8).launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        share=GRADIO_SHARE,
        show_error=True,
    )
