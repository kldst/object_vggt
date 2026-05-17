import json
import os
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("GRADIO_TEMP_DIR", "/mnt/train-data-4-hdd/yian/freepose/baseline/tmp")

import gradio as gr
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parent
TRAINING_ROOT = REPO_ROOT / "training"
for import_root in (REPO_ROOT, TRAINING_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from data.datasets.ov9d_multi_pose_normalize import OV9DMultiPoseNormalizeDataset


DATA_ROOT = Path(os.environ.get("OV9D_DATA_ROOT", "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d"))
SPLIT_JSON_ROOT = Path(
    os.environ.get(
        "OV9D_SPLIT_JSON_ROOT",
        "/mnt/train-data-4-hdd/yian/freepose/baseline_0503/splits_multi_4_3000",
    )
)
VIEWER_CACHE_ROOT = Path(os.environ.get("OV9D_GT_BBOX_DEMO_CACHE", "/tmp/demo_gradio_ov9d_gt_size_bbox"))
SPLITS = ("train", "test1")
SCENE_POINT_LIMIT = int(os.environ.get("OV9D_SCENE_POINT_LIMIT", "250000"))
SERVER_NAME = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7861"))
GRADIO_SHARE = os.environ.get("GRADIO_SHARE", "0") == "1"


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
MODELS_INFO_CACHE = None


def build_dataset(split: str):
    split = split if split in SPLITS else "test1"
    if split not in DATASET_CACHE:
        DATASET_CACHE[split] = OV9DMultiPoseNormalizeDataset(
            common_conf=make_common_conf(),
            split=split,
            DATA_ROOT=str(DATA_ROOT),
            SPLIT_JSON=str(SPLIT_JSON_ROOT / f"{split}.json"),
            verify_files=True,
            num_scene_views=1,
            num_object_views=1,
            load_point_map=True,
            scale_by_points=True,
            negative_object_prob=0.0,
        )
    return DATASET_CACHE[split]


def load_models_info():
    global MODELS_INFO_CACHE
    if MODELS_INFO_CACHE is None:
        MODELS_INFO_CACHE = json.loads((DATA_ROOT / "models_info.json").read_text(encoding="utf-8"))
    return MODELS_INFO_CACHE


def format_sample_label(split: str, idx: int, rec: dict) -> str:
    return (
        f"[{split}] {idx:04d} | {rec['scene_name']} | "
        f"obj_{int(rec['target_object_id']):06d} | {rec.get('category', '')}"
    )


def sample_labels(split: str):
    dataset = build_dataset(split)
    return [format_sample_label(split, idx, rec) for idx, rec in enumerate(dataset.records)]


def parse_sample_label(label: str):
    match = re.match(r"\[(?P<split>[^\]]+)\]\s+(?P<idx>\d+)", str(label or ""))
    if match:
        return match.group("split"), int(match.group("idx"))
    return "test1", 0


def load_batch(sample_label: str, seed: int):
    split, idx = parse_sample_label(sample_label)
    dataset = build_dataset(split)
    rng_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    try:
        batch = dataset.get_data(seq_index=idx % dataset.sequence_list_len, img_per_seq=1, aspect_ratio=1.0)
    finally:
        random.setstate(rng_state)
        np.random.set_state(np_state)
    return split, idx, batch


BBOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def centered_bbox_points_from_gt_size(object_id: int, normalization_scale: float, scale_multiplier: float):
    info = load_models_info()[str(int(object_id))]
    size = np.array([info["size_x"], info["size_y"], info["size_z"]], dtype=np.float32)
    size = size / float(normalization_scale) * float(scale_multiplier)
    half = size * 0.5
    return np.asarray(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float32,
    )


def transform_points(points_obj, rotation, translation):
    rot = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    trans = np.asarray(translation, dtype=np.float32).reshape(3)
    return np.asarray(points_obj, dtype=np.float32) @ rot.T + trans[None, :]


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


def draw_bbox_projection(image, bbox_points, extrinsic, intrinsic, color=(60, 255, 90)):
    canvas = np.asarray(image, dtype=np.uint8).copy()
    uv, valid = project_points(bbox_points, extrinsic, intrinsic)
    h, w = canvas.shape[:2]
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    for i, j in BBOX_EDGES:
        if not (valid[i] and valid[j]):
            continue
        p0 = tuple(np.round(uv[i]).astype(int))
        p1 = tuple(np.round(uv[j]).astype(int))
        if not (-w <= p0[0] <= 2 * w and -h <= p0[1] <= 2 * h and -w <= p1[0] <= 2 * w and -h <= p1[1] <= 2 * h):
            continue
        draw.line([p0, p1], fill=tuple(color), width=3)
    return np.asarray(pil, dtype=np.uint8)


def draw_arrow(draw, origin, end, color, width=3):
    draw.line([origin, end], fill=tuple(color), width=width)
    vec = np.asarray(end, dtype=np.float32) - np.asarray(origin, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return
    direction = vec / norm
    perp = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    head_len = min(18.0, max(8.0, norm * 0.18))
    head_w = head_len * 0.45
    tip = np.asarray(end, dtype=np.float32)
    left = tip - direction * head_len + perp * head_w
    right = tip - direction * head_len - perp * head_w
    draw.polygon([tuple(tip), tuple(left), tuple(right)], fill=tuple(color))


def draw_pose_axes(image, rotation, translation, axis_length, extrinsic, intrinsic):
    axes_obj = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float32,
    )
    points = transform_points(axes_obj, rotation, translation)
    uv, valid = project_points(points, extrinsic, intrinsic)
    canvas = np.asarray(image, dtype=np.uint8).copy()
    if not valid[0]:
        return canvas
    h, w = canvas.shape[:2]
    origin = tuple(np.round(uv[0]).astype(int))
    if not (-w <= origin[0] <= 2 * w and -h <= origin[1] <= 2 * h):
        return canvas
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()
    colors = [(255, 80, 80), (80, 255, 255), (255, 220, 80)]
    labels = ["X", "Y", "Z"]
    for axis_idx in range(3):
        end_idx = axis_idx + 1
        if not valid[end_idx]:
            continue
        end = tuple(np.round(uv[end_idx]).astype(int))
        if not (-w <= end[0] <= 2 * w and -h <= end[1] <= 2 * h):
            continue
        draw_arrow(draw, origin, end, colors[axis_idx], width=4)
        draw.text(end, labels[axis_idx], fill=tuple(colors[axis_idx]), font=font)
    r = 4
    draw.ellipse((origin[0] - r, origin[1] - r, origin[0] + r, origin[1] + r), fill=(255, 255, 255))
    return np.asarray(pil, dtype=np.uint8)


def subsample_points(points, colors, limit):
    if limit <= 0 or points.shape[0] <= limit:
        return points, colors
    order = np.linspace(0, points.shape[0] - 1, limit).astype(np.int64)
    return points[order], colors[order]


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


def axes_mesh(rotation, translation, length, radius):
    trans = np.asarray(translation, dtype=np.float32).reshape(3)
    rot = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    endpoints = np.asarray([[length, 0, 0], [0, length, 0], [0, 0, length]], dtype=np.float32) @ rot.T + trans[None, :]
    meshes = []
    center = trimesh.creation.icosphere(subdivisions=2, radius=radius * 1.8)
    center.apply_translation(trans)
    center.visual.face_colors = np.tile(np.array([[255, 255, 255, 255]], dtype=np.uint8), (len(center.faces), 1))
    meshes.append(center)
    colors = [np.array([255, 80, 80, 255]), np.array([80, 255, 255, 255]), np.array([255, 220, 80, 255])]
    for idx in range(3):
        mesh = trimesh.creation.cylinder(radius=radius, segment=np.stack([trans, endpoints[idx]], axis=0))
        mesh.visual.face_colors = np.tile(colors[idx][None, :], (len(mesh.faces), 1))
        meshes.append(mesh)
    return trimesh.util.concatenate(meshes)


def build_glb(batch, bbox_points):
    split = str(batch["seq_name"]).split("/", 1)[0]
    safe_scene = re.sub(r"[^a-zA-Z0-9_]+", "_", str(batch["scene_name"]))[:80]
    safe_obj = re.sub(r"[^a-zA-Z0-9_]+", "_", str(batch["object_name"]))[:40]
    out_dir = VIEWER_CACHE_ROOT / split / safe_scene / safe_obj
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"view_{int(batch['ids'][0]):06d}_gt_size_centered.glb"

    image = np.asarray(batch["images"][0], dtype=np.uint8)
    points = np.asarray(batch["world_points"][0], dtype=np.float32)
    mask = np.asarray(batch["point_masks"][0], dtype=bool) & np.isfinite(points).all(axis=-1)
    points = points[mask]
    colors = image[mask]
    points, colors = subsample_points(points, colors, SCENE_POINT_LIMIT)

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=points, colors=colors))
    scene.add_geometry(bbox_mesh(bbox_points, [60, 255, 90, 255]))
    diag = float(np.linalg.norm(bbox_points.max(axis=0) - bbox_points.min(axis=0)))
    scene.add_geometry(axes_mesh(batch["object_rotation"], batch["object_translation"], diag * 0.35, max(diag * 0.012, 0.002)))
    scene.export(out_path)
    return str(out_path), int(points.shape[0])


def render_sample(sample_label: str, seed: int, box_scale: float):
    split, idx, batch = load_batch(sample_label, seed)
    object_id = int(batch["object_id"])
    normalization_scale = float(np.asarray(batch["normalization_scale"]).reshape(-1)[0])
    bbox_obj = centered_bbox_points_from_gt_size(object_id, normalization_scale, box_scale)
    bbox_world = transform_points(bbox_obj, batch["object_rotation"], batch["object_translation"])
    glb_path, point_count = build_glb(batch, bbox_world)

    axis_length = float(np.linalg.norm(bbox_world.max(axis=0) - bbox_world.min(axis=0)) * 0.35)
    projection = draw_bbox_projection(batch["images"][0], bbox_world, batch["extrinsics"][0], batch["intrinsics"][0])
    projection = draw_pose_axes(
        projection,
        batch["object_rotation"],
        batch["object_translation"],
        axis_length,
        batch["extrinsics"][0],
        batch["intrinsics"][0],
    )

    log = (
        f"Loaded [{split}] sample `{idx}`  \n"
        f"Scene: `{batch['scene_name']}`  \n"
        f"Frame: `{int(batch['ids'][0])}`  \n"
        f"Object: `{batch['object_name']}` / `{batch.get('category', '')}`  \n"
        f"GT translation: `{[round(float(v), 6) for v in np.asarray(batch['object_translation']).tolist()]}`  \n"
        f"Normalization scale: `{normalization_scale:.6f}`  \n"
        f"Point count in GLB: `{point_count}`"
    )
    return projection, glb_path, log


def refresh_split(split: str):
    labels = sample_labels(split)
    value = labels[0] if labels else None
    return gr.update(choices=labels, value=value)


initial_split = "test1"
initial_labels = sample_labels(initial_split)
initial_sample = initial_labels[0] if initial_labels else None

theme = gr.themes.Ocean()
with gr.Blocks(theme=theme) as demo:
    gr.HTML(
        """
        <h1>OV9D GT Size Centered BBox Demo</h1>
        <p>
        Uses GT rotation + GT translation, then draws a box centered at translation with size from models_info.
        One sampled scene view is shown in 2D and as a normalized GT point cloud.
        </p>
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            split_dropdown = gr.Dropdown(choices=list(SPLITS), value=initial_split, label="Dataset Split")
            sample_dropdown = gr.Dropdown(choices=initial_labels, value=initial_sample, label="Sample")
            seed_input = gr.Number(label="Sampling Seed", value=42, precision=0)
            scale_slider = gr.Slider(0.25, 4.0, value=1.0, step=0.05, label="GT Size Scale")
            render_btn = gr.Button("Render", variant="primary")
            log_output = gr.Markdown("Pick a sample, then click Render.")
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("2D Camera View"):
                    projection_image = gr.Image(label="GT size bbox centered at GT translation", type="numpy", height=620)
                with gr.Tab("3D Point Cloud"):
                    model_viewer = gr.Model3D(label="GT Scene Point Cloud + Centered GT Size BBox", height=720)

    demo.load(
        fn=render_sample,
        inputs=[sample_dropdown, seed_input, scale_slider],
        outputs=[projection_image, model_viewer, log_output],
    )
    split_dropdown.change(fn=refresh_split, inputs=[split_dropdown], outputs=[sample_dropdown])
    render_btn.click(
        fn=render_sample,
        inputs=[sample_dropdown, seed_input, scale_slider],
        outputs=[projection_image, model_viewer, log_output],
    )
    sample_dropdown.change(
        fn=render_sample,
        inputs=[sample_dropdown, seed_input, scale_slider],
        outputs=[projection_image, model_viewer, log_output],
    )
    seed_input.change(
        fn=render_sample,
        inputs=[sample_dropdown, seed_input, scale_slider],
        outputs=[projection_image, model_viewer, log_output],
    )
    scale_slider.change(
        fn=render_sample,
        inputs=[sample_dropdown, seed_input, scale_slider],
        outputs=[projection_image, model_viewer, log_output],
    )


if __name__ == "__main__":
    demo.launch(server_name=SERVER_NAME, server_port=SERVER_PORT, share=GRADIO_SHARE)
