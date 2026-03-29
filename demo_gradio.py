import glob
import json
import math
import os
import random
import re
import struct
import sys
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

sys.path.append("vggt/")

from vggt.models.vggt import VGGT

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

OBJ_ROOT = Path("/mnt/train-data-4-hdd/yian/6dpose_obj/obj")
DATA_ROOT = Path("/mnt/train-data-4-hdd/yian/6dpose_obj/0315_fixedCam_test")
OUT_IMAGE_ROOT = DATA_ROOT / "out_image"
OUT_POSE_ROOT = DATA_ROOT / "out_pose"
# OBJECT_IMAGE_ROOT = DATA_ROOT / "object_space_rgb"
OBJECT_IMAGE_ROOT = Path("/mnt/train-data-4-hdd/yian/6dpose_obj/0315_fixedCam_1k/object_space_rgb")
OUT_CAM_PARAM_ROOT = DATA_ROOT / "out_cam_param"
FIXED_VIEWS = (1, 3, 8, 12, 15, 18)
NUM_OBJECT_VIEWS = 4
PROJECTION_POINT_LIMIT = 12000
MODEL_CKPT_PATH = Path(
    "/mnt/train-data-4-hdd/yian/6dpose_obj/vggt_objectspc/training/logs/test_0321_object_dataset_AGGREGATOR_ALL_white_random_object_image/ckpts/checkpoint_19.pt"
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def list_runs():
    return sorted(p.name for p in OUT_IMAGE_ROOT.iterdir() if p.is_dir() and p.name.startswith("run_"))


def image_path_for_view(run_name: str, view_idx: int) -> str:
    filename = "Main_Camera.jpg" if view_idx == 0 else f"Main_Camera_({view_idx}).jpg"
    return str(OUT_IMAGE_ROOT / run_name / filename)


def object_image_glob(object_name: str):
    return sorted(glob.glob(str(OBJECT_IMAGE_ROOT / object_name / "*_rgb.png")))


def extract_view_idx(path: str):
    name = Path(path).name
    if name == "Main_Camera_rgb.png":
        return 0
    match = re.match(r"Main_Camera_\((\d+)\)_rgb\.png$", name)
    return int(match.group(1)) if match else None


def get_available_object_view_indices(object_name: str):
    view_indices = []
    for path in object_image_glob(object_name):
        view_idx = extract_view_idx(path)
        if view_idx is not None:
            view_indices.append(view_idx)
    return sorted(set(view_indices))


def object_image_path(object_name: str, view_idx: int):
    filename = "Main_Camera_rgb.png" if int(view_idx) == 0 else f"Main_Camera_({int(view_idx)})_rgb.png"
    path = OBJECT_IMAGE_ROOT / object_name / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing object image: {path}")
    return str(path)


def object_ply_path(object_name: str) -> Path:
    dataset_name, obj_id = object_name.split("_obj_")
    path = OBJ_ROOT / dataset_name / f"obj_{obj_id}.ply"
    if not path.exists():
        raise FileNotFoundError(f"Missing object point cloud: {path}")
    return path


def _dataset_scale(name_or_path: str) -> float:
    s = str(name_or_path).lower()
    if "ycbv" in s:
        return 0.002
    if "handal" in s:
        return 0.0015
    if "hope" in s:
        return 0.003
    if "rupac" in s:
        return 0.002
    return 1.0


def camera_param_path(run_name: str, view_idx: int) -> Path:
    filename = "camera_Main_Camera.npz" if int(view_idx) == 0 else f"camera_Main_Camera_({int(view_idx)}).npz"
    path = OUT_CAM_PARAM_ROOT / run_name / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing camera param file: {path}")
    return path


def load_camera_params(run_name: str, view_idx: int):
    data = np.load(camera_param_path(run_name, view_idx))
    K = np.asarray(data["intrinsics.K_flat9"], dtype=np.float32).reshape(3, 3)
    extrinsic = np.asarray(data["extrinsics.opencv.worldToCamera16"], dtype=np.float32).reshape(4, 4)[:3, :4]
    width = int(np.asarray(data["image.width"]).reshape(-1)[0])
    height = int(np.asarray(data["image.height"]).reshape(-1)[0])
    return K, extrinsic, width, height


def load_ply_xyz(path: Path):
    with path.open("rb") as f:
        if f.readline().decode("ascii", errors="ignore").strip() != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        fmt = None
        vertex_count = None
        properties = []
        in_vertex = False
        while True:
            line = f.readline().decode("ascii", errors="ignore")
            if not line:
                raise ValueError(f"Unexpected EOF in header: {path}")
            line = line.strip()
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element "):
                in_vertex = False
            elif line.startswith("property ") and in_vertex:
                parts = line.split()
                properties.append((parts[1], parts[2]))
            elif line == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY missing vertex count: {path}")
        xyz = []
        if fmt == "ascii":
            for _ in range(vertex_count):
                parts = f.readline().decode("ascii", errors="ignore").strip().split()
                xyz.append([float(parts[0]), float(parts[1]), float(parts[2])])
            return np.asarray(xyz, dtype=np.float32)
        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format {fmt}: {path}")
        type_map = {
            "char": "b", "uchar": "B", "int8": "b", "uint8": "B",
            "short": "h", "ushort": "H", "int16": "h", "uint16": "H",
            "int": "i", "uint": "I", "int32": "i", "uint32": "I",
            "float": "f", "float32": "f", "double": "d", "float64": "d",
        }
        fmt_str = "<" + "".join(type_map[t] for t, _ in properties)
        row_size = struct.calcsize(fmt_str)
        for _ in range(vertex_count):
            row = struct.unpack(fmt_str, f.read(row_size))
            xyz.append([row[0], row[1], row[2]])
        return np.asarray(xyz, dtype=np.float32)


def project_points(points_world: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray):
    points_cam = points_world @ extrinsic[:, :3].T + extrinsic[:, 3][None, :]
    z = points_cam[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    points_cam = points_cam[valid]
    z = z[valid]
    uvw = points_cam @ intrinsic.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return uv.astype(np.float32), z.astype(np.float32)


def draw_projected_points(image_path: str, uv: np.ndarray, depth: np.ndarray, width: int, height: int):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    if uv.shape[0] == 0:
        return image
    inside = (
        (uv[:, 0] >= 0) & (uv[:, 0] < width) &
        (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    uv = uv[inside]
    depth = depth[inside]
    if uv.shape[0] == 0:
        return image
    if uv.shape[0] > PROJECTION_POINT_LIMIT:
        order = np.linspace(0, uv.shape[0] - 1, PROJECTION_POINT_LIMIT).astype(np.int32)
        uv = uv[order]
        depth = depth[order]
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    denom = max(depth_max - depth_min, 1e-6)
    norm = (depth - depth_min) / denom
    colors = cv2.applyColorMap((255.0 * (1.0 - norm)).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = image.copy()
    for (u, v), color in zip(np.round(uv).astype(np.int32), colors.reshape(-1, 3)):
        cv2.circle(overlay, (int(u), int(v)), 1, tuple(int(c) for c in color.tolist()), -1)
    blended = cv2.addWeighted(image, 0.55, overlay, 0.45, 0.0)
    return blended


def build_projection_gallery(run_name: str, object_name: str, pred_translation: np.ndarray, pred_rot6d: np.ndarray):
    object_points = load_ply_xyz(object_ply_path(object_name))
    object_points = object_points * float(_dataset_scale(object_name))
    rot = rot6d_to_matrix(pred_rot6d).astype(np.float32)
    trans = np.asarray(pred_translation, dtype=np.float32)
    world_points = object_points @ rot.T + trans[None, :]
    gallery = []
    for view_idx in FIXED_VIEWS:
        image_path = image_path_for_view(run_name, view_idx)
        K, extrinsic, width, height = load_camera_params(run_name, view_idx)
        uv, depth = project_points(world_points, extrinsic, K)
        overlay = draw_projected_points(image_path, uv, depth, width, height)
        gallery.append((overlay, f"View {view_idx}"))
    return gallery


def compute_bbox_corners(points_obj: np.ndarray):
    pmin = points_obj.min(axis=0)
    pmax = points_obj.max(axis=0)
    return np.asarray([
        [pmin[0], pmin[1], pmin[2]],
        [pmax[0], pmin[1], pmin[2]],
        [pmax[0], pmax[1], pmin[2]],
        [pmin[0], pmax[1], pmin[2]],
        [pmin[0], pmin[1], pmax[2]],
        [pmax[0], pmin[1], pmax[2]],
        [pmax[0], pmax[1], pmax[2]],
        [pmin[0], pmax[1], pmax[2]],
    ], dtype=np.float32)


def project_points_with_mask(points_world: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray):
    points_cam = points_world @ extrinsic[:, :3].T + extrinsic[:, 3][None, :]
    z = points_cam[:, 2]
    valid = z > 1e-6
    uv = np.full((points_world.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        uvw = points_cam[valid] @ intrinsic.T
        uv[valid] = (uvw[:, :2] / uvw[:, 2:3]).astype(np.float32)
    return uv, z.astype(np.float32), valid, points_cam.astype(np.float32)


def draw_projected_bbox(
    image_path: str,
    uv: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
    center_uv: np.ndarray,
    center_valid: bool,
    axis_uv: np.ndarray,
    axis_valid: np.ndarray,
):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    overlay = image.copy()
    inside = valid.copy()
    inside &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    for i, j in edges:
        if inside[i] and inside[j]:
            p1 = tuple(np.round(uv[i]).astype(np.int32))
            p2 = tuple(np.round(uv[j]).astype(np.int32))
            cv2.line(overlay, p1, p2, (0, 255, 0), 2)

    if center_valid:
        c = tuple(np.round(center_uv).astype(np.int32))
        axis_colors = [(255, 64, 64), (0, 255, 255), (255, 215, 0)]
        for idx, color in enumerate(axis_colors):
            if axis_valid[idx]:
                end = tuple(np.round(axis_uv[idx]).astype(np.int32))
                cv2.arrowedLine(overlay, c, end, color, 3, tipLength=0.18)

    return overlay


def build_bbox_projection_gallery(run_name: str, object_name: str, pred_translation: np.ndarray, pred_rot6d: np.ndarray):
    object_points = load_ply_xyz(object_ply_path(object_name))
    object_points = object_points * float(_dataset_scale(object_name))
    bbox_obj = compute_bbox_corners(object_points)
    rot = rot6d_to_matrix(pred_rot6d).astype(np.float32)
    trans = np.asarray(pred_translation, dtype=np.float32)
    bbox_world = bbox_obj @ rot.T + trans[None, :]

    center_obj = np.zeros((1, 3), dtype=np.float32)
    axis_len = max(np.linalg.norm(bbox_obj.max(axis=0) - bbox_obj.min(axis=0)) * 0.25, 1e-3)
    axis_obj = np.asarray([
        [axis_len, 0.0, 0.0],
        [0.0, axis_len, 0.0],
        [0.0, 0.0, axis_len],
    ], dtype=np.float32)
    center_world = center_obj @ rot.T + trans[None, :]
    axis_world = axis_obj @ rot.T + trans[None, :]

    gallery = []
    for view_idx in FIXED_VIEWS:
        image_path = image_path_for_view(run_name, view_idx)
        K, extrinsic, width, height = load_camera_params(run_name, view_idx)
        uv, _, valid, _ = project_points_with_mask(bbox_world, extrinsic, K)
        center_uv, _, center_valid, _ = project_points_with_mask(center_world, extrinsic, K)
        axis_uv, _, axis_valid, _ = project_points_with_mask(axis_world, extrinsic, K)
        center_ok = bool(center_valid[0])
        axis_ok = np.asarray(axis_valid, dtype=bool)
        overlay = draw_projected_bbox(
            image_path,
            uv,
            valid,
            width,
            height,
            center_uv[0],
            center_ok,
            axis_uv,
            axis_ok,
        )
        gallery.append((overlay, f"View {view_idx}"))
    return gallery


def decode_name(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_pose_npz(run_name: str):
    pose_path = OUT_POSE_ROOT / f"{run_name}.npz"
    data = np.load(pose_path, allow_pickle=True)
    names = [decode_name(x) for x in data["names"]]
    positions = np.asarray(data["positions"], dtype=np.float32)
    rot_quat_wxyz = np.asarray(data["rot_quat_wxyz"], dtype=np.float32)
    return names, positions, rot_quat_wxyz


def get_objects_for_run(run_name: str):
    names, _, _ = load_pose_npz(run_name)
    return names


def get_scene_images(run_name: str):
    image_paths = []
    for view_idx in FIXED_VIEWS:
        path = image_path_for_view(run_name, view_idx)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing fixed-view image: {path}")
        image_paths.append(path)
    return image_paths


def sample_object_view_indices(object_name: str, seed: int):
    candidates = get_available_object_view_indices(object_name)
    if len(candidates) < NUM_OBJECT_VIEWS:
        raise ValueError(f"Object {object_name} only has {len(candidates)} images, need {NUM_OBJECT_VIEWS}.")
    rng = random.Random(int(seed))
    return sorted(rng.sample(candidates, NUM_OBJECT_VIEWS))


def resolve_object_images(object_name: str, selected_view_indices):
    if selected_view_indices is None:
        raise ValueError("selected_view_indices is required")
    cleaned = [int(v) for v in selected_view_indices]
    if len(cleaned) != NUM_OBJECT_VIEWS:
        raise ValueError(f"Please provide exactly {NUM_OBJECT_VIEWS} object view indices.")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("Object view indices must be unique.")
    return [object_image_path(object_name, v) for v in cleaned]


def quat_wxyz_to_matrix(quat):
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_rot6d(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    # Match training/loss.py: rotation_matrix[..., :, :2].reshape(..., 6)
    return matrix[:, :2].reshape(-1)


def quat_wxyz_to_rot6d(quat):
    return matrix_to_rot6d(quat_wxyz_to_matrix(quat))


def rot6d_to_matrix(rot6d):
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(3, 2)
    x_raw = rot6d[:, 0]
    y_raw = rot6d[:, 1]
    x_norm = np.linalg.norm(x_raw)
    if x_norm < 1e-12:
        x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x = x_raw / x_norm
    y = y_raw - np.dot(x, y_raw) * x
    y_norm = np.linalg.norm(y)
    if y_norm < 1e-12:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y = fallback - np.dot(x, fallback) * x
        y = y / max(np.linalg.norm(y), 1e-12)
    else:
        y = y / y_norm
    z = np.cross(x, y)
    z = z / max(np.linalg.norm(z), 1e-12)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def rotation_error_degrees(pred_rot6d, gt_quat_wxyz):
    r_pred = rot6d_to_matrix(pred_rot6d)
    r_gt = quat_wxyz_to_matrix(gt_quat_wxyz)
    rel = r_pred @ r_gt.T
    trace = np.trace(rel)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
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


def vector_loss(pred, gt, loss_type="l1"):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if loss_type == "l1":
        return float(np.abs(pred - gt).mean())
    if loss_type == "l2":
        return float(np.square(pred - gt).mean())
    raise ValueError(f"Unknown loss_type: {loss_type}")


def build_model():
    print("Initializing dataset-driven pose-only VGGT model...")
    model = VGGT(
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
        enable_track=False,
        enable_object_point=False,
        enable_object_srt=True,
        use_shared_object_latent=False,
        enable_object_cross_attn=True,
        object_cross_attn_heads=16,
    )
    print(f"Loading checkpoint from local path: {MODEL_CKPT_PATH}")
    checkpoint = torch.load(MODEL_CKPT_PATH, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"Unexpected keys when loading checkpoint: {unexpected}")
    model.eval()
    return model.to(device)


model = build_model()
RUN_CHOICES = list_runs()


def _load_images_to_device(image_paths):
    images = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).to(torch.float32).div(255.0)
        images.append(tensor)
    return torch.stack(images, dim=0).to(device)


def parse_view_index_text(view_index_text: str, strict: bool = True):
    raw = (view_index_text or "").strip()
    if not raw:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != NUM_OBJECT_VIEWS:
        if strict:
            raise ValueError(f"Please enter exactly {NUM_OBJECT_VIEWS} comma-separated object view indices.")
        return None
    try:
        parsed = [int(part) for part in parts]
    except ValueError:
        if strict:
            raise ValueError(
                f"Object view indices must be integers, got: {view_index_text}"
            )
        return None
    return parsed


def format_view_index_text(view_indices):
    return ", ".join(str(int(v)) for v in view_indices)


def describe_selection(run_name: str, object_name: str, seed: int, view_index_text: str):
    if not run_name:
        return gr.update(choices=[], value=None), [], [], "", "Please select a run.", [], None, [], None
    object_choices = get_objects_for_run(run_name)
    chosen_object = object_name if object_name in object_choices else object_choices[0]
    available_view_indices = get_available_object_view_indices(chosen_object)
    parsed_view_indices = parse_view_index_text(view_index_text, strict=False)
    selected_view_indices = parsed_view_indices if parsed_view_indices is not None else sample_object_view_indices(chosen_object, seed)
    if any(v not in available_view_indices for v in selected_view_indices):
        selected_view_indices = sample_object_view_indices(chosen_object, seed)
    object_paths = resolve_object_images(chosen_object, selected_view_indices)
    scene_paths = get_scene_images(run_name)
    selected_text = format_view_index_text(selected_view_indices)
    message = (
        f"Run {run_name}: using fixed scene views {FIXED_VIEWS}. "
        f"Object {chosen_object}: selected object view indices [{selected_text}]. Predicted point-cloud projection will be shown after Generate."
    )
    return gr.update(choices=object_choices, value=chosen_object), scene_paths, object_paths, selected_text, message, [], None, [], None


def run_inference(run_name: str, object_name: str, seed: int, view_index_text: str):
    if not run_name:
        raise ValueError("Please select a run.")
    if not object_name:
        raise ValueError("Please select an object.")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    scene_paths = get_scene_images(run_name)
    selected_view_indices = parse_view_index_text(view_index_text)
    if selected_view_indices is None:
        selected_view_indices = sample_object_view_indices(object_name, seed)
    object_paths = resolve_object_images(object_name, selected_view_indices)

    scene_images = _load_images_to_device(scene_paths)
    object_images = _load_images_to_device(object_paths)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    start_time = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(scene_images, object_images=object_images)
    elapsed = time.time() - start_time

    pred_pose = predictions["object_pose"].detach().cpu().numpy()[0]
    pred_translation = predictions["object_translation"].detach().cpu().numpy()[0]

    names, positions, rot_quat_wxyz = load_pose_npz(run_name)
    obj_idx = names.index(object_name)
    gt_translation = positions[obj_idx]
    gt_quat = rot_quat_wxyz[obj_idx]
    gt_rot6d = quat_wxyz_to_rot6d(gt_quat)

    rot_err_deg = rotation_error_degrees(pred_pose, gt_quat)
    trans_err = translation_error(pred_translation, gt_translation)
    loss_object_pose = vector_loss(pred_pose, gt_rot6d, loss_type="l1")
    loss_object_translation = vector_loss(pred_translation, gt_translation, loss_type="l1")

    projection_gallery = build_projection_gallery(run_name, object_name, pred_translation, pred_pose)
    bbox_gallery = build_bbox_projection_gallery(run_name, object_name, pred_translation, pred_pose)

    result = {
        "run": run_name,
        "object_name": object_name,
        "seed": int(seed),
        "scene_views": list(FIXED_VIEWS),
        "object_view_indices": [int(v) for v in selected_view_indices],
        "scene_image_paths": scene_paths,
        "object_image_paths": object_paths,
        "prediction": {
            "object_pose_rot6d": pred_pose.tolist(),
            "object_translation": pred_translation.tolist(),
        },
        "ground_truth": {
            "translation": gt_translation.tolist(),
            "rotation_rot6d": gt_rot6d.tolist(),
        },
        "errors": {
            "rotation_deg": rot_err_deg,
            "translation_abs_xyz": trans_err["abs_xyz"],
            "translation_signed_xyz": trans_err["signed_xyz"],
        },
        "losses": {
            "loss_object_pose": loss_object_pose,
            "loss_object_translation": loss_object_translation,
        },
        "timing": {
            "inference_seconds": elapsed,
        },
    }

    output_dir = Path("pose_eval_outputs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{object_name}_seed{int(seed)}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    log_msg = (
        f"Done. run={run_name}, object={object_name}, seed={seed}, views={format_view_index_text(selected_view_indices)}, "
        f"rot_err={rot_err_deg:.4f} deg, loss_object_translation={loss_object_translation:.6f}, "
        f"loss_object_pose={loss_object_pose:.6f}, time={elapsed:.2f}s"
    )
    projection_viewer = projection_gallery[0][0] if projection_gallery else None
    bbox_viewer = bbox_gallery[0][0] if bbox_gallery else None
    return log_msg, scene_paths, object_paths, format_view_index_text(selected_view_indices), projection_gallery, projection_viewer, bbox_gallery, bbox_viewer


def refresh_from_run(run_name: str, seed: int, view_index_text: str):
    return describe_selection(run_name, None, seed, view_index_text)


def refresh_from_object(run_name: str, object_name: str, seed: int, view_index_text: str):
    return describe_selection(run_name, object_name, seed, view_index_text)


def refresh_from_seed(run_name: str, object_name: str, seed: int, view_index_text: str):
    return describe_selection(run_name, object_name, seed, view_index_text)


def clear_outputs():
    return "Select a run and object, then click Generate.", [], None, [], None


def select_gallery_image(gallery, evt: gr.SelectData):
    if gallery is None:
        return None
    if isinstance(gallery, list) and 0 <= evt.index < len(gallery):
        item = gallery[evt.index]
        if isinstance(item, tuple):
            return item[0]
        return item
    return None


theme = gr.themes.Ocean()

with gr.Blocks(theme=theme) as demo:
    gr.HTML(
        """
        <h1>VGGT Object Pose Evaluation</h1>
        <p>Select a dataset run and object. The demo uses fixed scene views <code>(1, 3, 8, 12, 15, 18)</code>,
        samples 4 object reference images from <code>object_space_rgb</code>, runs pose inference, compares against GT from <code>out_pose</code>,
        and also projects the predicted object point cloud back to the fixed scene images using <code>out_cam_param</code>.</p>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            run_dropdown = gr.Dropdown(choices=RUN_CHOICES, label="Run", value=RUN_CHOICES[0] if RUN_CHOICES else None)
            object_dropdown = gr.Dropdown(choices=[], label="Object", value=None)
            seed_input = gr.Number(label="Random Seed For 4 Object Views", value=42, precision=0)
            object_view_indices_input = gr.Textbox(label="Object View Indices", value="", placeholder="e.g. 1, 3, 8, 12")
            refresh_btn = gr.Button("Resample Object Views")
            generate_btn = gr.Button("Generate", variant="primary")
            clear_btn = gr.Button("Clear Output")
            log_output = gr.Markdown("Select a run and object, then click Generate.")
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Input Views"):
                    scene_gallery = gr.Gallery(label="Scene Fixed Views", columns=3, height="520px", object_fit="contain")
                    object_gallery = gr.Gallery(label="Sampled Object Views", columns=4, height="260px", object_fit="contain")
                with gr.Tab("Projection"):
                    projection_gallery = gr.Gallery(label="Pred Point Cloud Projection", columns=3, height="320px", object_fit="contain")
                    projection_viewer = gr.Image(label="Projection Viewer", height=900, interactive=False)
                with gr.Tab("Bounding Box"):
                    bbox_gallery = gr.Gallery(label="Pred Bounding Box Projection", columns=3, height="320px", object_fit="contain")
                    bbox_viewer = gr.Image(label="Bounding Box Viewer", height=900, interactive=False)

    demo.load(
        fn=lambda: describe_selection(RUN_CHOICES[0], None, 42, "") if RUN_CHOICES else (gr.update(choices=[], value=None), [], [], "", "No runs found.", [], None, [], None),
        inputs=[],
        outputs=[object_dropdown, scene_gallery, object_gallery, object_view_indices_input, log_output, projection_gallery, projection_viewer, bbox_gallery, bbox_viewer],
    )

    run_dropdown.change(
        fn=refresh_from_run,
        inputs=[run_dropdown, seed_input, object_view_indices_input],
        outputs=[object_dropdown, scene_gallery, object_gallery, object_view_indices_input, log_output, projection_gallery, projection_viewer, bbox_gallery, bbox_viewer],
    )
    object_dropdown.change(
        fn=refresh_from_object,
        inputs=[run_dropdown, object_dropdown, seed_input, object_view_indices_input],
        outputs=[object_dropdown, scene_gallery, object_gallery, object_view_indices_input, log_output, projection_gallery, projection_viewer, bbox_gallery, bbox_viewer],
    )
    seed_input.change(
        fn=refresh_from_seed,
        inputs=[run_dropdown, object_dropdown, seed_input, object_view_indices_input],
        outputs=[object_dropdown, scene_gallery, object_gallery, object_view_indices_input, log_output, projection_gallery, projection_viewer, bbox_gallery, bbox_viewer],
    )
    refresh_btn.click(
        fn=lambda run_name, object_name, seed: describe_selection(run_name, object_name, seed, ""),
        inputs=[run_dropdown, object_dropdown, seed_input],
        outputs=[object_dropdown, scene_gallery, object_gallery, object_view_indices_input, log_output, projection_gallery, projection_viewer, bbox_gallery, bbox_viewer],
    )
    generate_btn.click(
        fn=run_inference,
        inputs=[run_dropdown, object_dropdown, seed_input, object_view_indices_input],
        outputs=[log_output, scene_gallery, object_gallery, object_view_indices_input, projection_gallery, projection_viewer, bbox_gallery, bbox_viewer],
    )
    clear_btn.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[log_output, projection_gallery, projection_viewer, bbox_gallery, bbox_viewer],
    )
    projection_gallery.select(
        fn=select_gallery_image,
        inputs=[projection_gallery],
        outputs=[projection_viewer],
    )
    bbox_gallery.select(
        fn=select_gallery_image,
        inputs=[bbox_gallery],
        outputs=[bbox_viewer],
    )

    demo.queue(max_size=20).launch(show_error=True, share=True)
