import os
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("GRADIO_TEMP_DIR", "/mnt/train-data-4-hdd/yian/freepose/baseline_0503/tmp")

import gradio as gr
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parent
TRAINING_ROOT = REPO_ROOT / "training"
for import_root in (REPO_ROOT, TRAINING_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from data.datasets.mixed_real_pose_normalize import (
    HouseCat6DMultiPoseNormalizeDataset,
    Real275MultiPoseNormalizeDataset,
    YCBVMultiPoseNormalizeDataset,
)
from data.datasets.ov9d_single_pose_normalize import OV9DSinglePoseNormalizeDataset


CONFIG_PATH = REPO_ROOT / "training/config/default_object_oo9d_real275_ycbv_hc.yaml"
FREEPOSE_ROOT = Path("/mnt/train-data-4-hdd/yian/freepose")
ALIGN_JSON = REPO_ROOT / "dataset_align.json"
OV9D_ROOT = FREEPOSE_ROOT / "ov9d/ov9d"
OV9D_OBJECT_IMAGE_ROOT = FREEPOSE_ROOT / "ov9d/ov9d_around_image"
OV9D_SPLIT_JSON = REPO_ROOT / "splits_ov9d_unseen_category_generalization/single/train.json"
REAL275_ROOT = FREEPOSE_ROOT / "real275"
YCBV_ROOT = FREEPOSE_ROOT / "datasets_real/ycbv"
HOUSECAT6D_ROOT = FREEPOSE_ROOT / "housecat6d"

SAMPLE_COUNT = int(os.environ.get("MIXED_GT_BBOX_SAMPLE_COUNT", "20"))
SAMPLE_SEED = int(os.environ.get("MIXED_GT_BBOX_SAMPLE_SEED", "30"))
MAX_RECORDS_ENV = os.environ.get("MIXED_GT_BBOX_MAX_RECORDS", "1000").strip()
MAX_RECORDS = int(MAX_RECORDS_ENV) if MAX_RECORDS_ENV and int(MAX_RECORDS_ENV) > 0 else None
SCENE_POINT_LIMIT = int(os.environ.get("MIXED_GT_BBOX_SCENE_POINT_LIMIT", "280000"))
VIEWER_CACHE_ROOT = Path(os.environ.get("MIXED_GT_BBOX_DEMO_CACHE", "/tmp/demo_gradio_mixed_gt_size_bbox"))
SERVER_NAME = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7863"))
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


def dataset_common_kwargs():
    kwargs = dict(
        verify_files=True,
        num_scene_views=4,
        num_object_views=4,
        fixed_object_view_ids=[1, 5, 10, 15],
        strict_fixed_object_view_ids=True,
        min_view_gap=5,
        load_point_map=True,
        scale_by_points=True,
        negative_object_prob=0.0,
    )
    if MAX_RECORDS is not None:
        kwargs["max_records"] = MAX_RECORDS
    return kwargs


DATASETS = None
SAMPLES = []


def build_datasets():
    global DATASETS
    if DATASETS is not None:
        return DATASETS

    common_conf = make_common_conf()
    common = dataset_common_kwargs()
    DATASETS = [
        (
            "oo9d_single",
            OV9DSinglePoseNormalizeDataset(
                common_conf=common_conf,
                split="train",
                DATA_ROOT=str(OV9D_ROOT),
                OBJECT_IMAGE_ROOT=str(OV9D_OBJECT_IMAGE_ROOT),
                SPLIT_JSON=str(OV9D_SPLIT_JSON),
                **common,
            ),
        ),
        (
            "real275",
            Real275MultiPoseNormalizeDataset(
                common_conf=common_conf,
                split="train",
                DATA_ROOT=str(REAL275_ROOT),
                SPLIT_ROOT=str(REAL275_ROOT / "real_train"),
                GT_ROOT=str(REAL275_ROOT / "gts/real_train_umeyama"),
                OBJECT_IMAGE_ROOT=str(REAL275_ROOT / "real275_aligned_object_refs"),
                ALIGN_JSON=str(ALIGN_JSON),
                **common,
            ),
        ),
        (
            "ycbv",
            YCBVMultiPoseNormalizeDataset(
                common_conf=common_conf,
                split="train_real",
                DATA_ROOT=str(YCBV_ROOT),
                SPLIT_ROOT=str(YCBV_ROOT / "train_real"),
                OBJECT_IMAGE_ROOT=str(YCBV_ROOT / "ycbv_aligned_object_refs"),
                ALIGN_JSON=str(ALIGN_JSON),
                **common,
            ),
        ),
        (
            "housecat6d",
            HouseCat6DMultiPoseNormalizeDataset(
                common_conf=common_conf,
                split="train",
                DATA_ROOT=str(HOUSECAT6D_ROOT),
                OBJECT_IMAGE_ROOT=str(HOUSECAT6D_ROOT / "housecat6d_aligned_object_refs"),
                ALIGN_JSON=str(ALIGN_JSON),
                **common,
            ),
        ),
    ]
    return DATASETS


def make_label(dataset_name: str, sample_idx: int, seq_index: int, rec: dict) -> str:
    scene = rec.get("scene_name", "")
    obj = rec.get("object_name") or rec.get("object_instance") or f"obj_{int(rec.get('object_id', 0)):06d}"
    category = rec.get("category", "")
    return f"{sample_idx:02d} | {dataset_name} | idx={seq_index} | {scene} | {obj} | {category}"


def refresh_samples(seed: int = SAMPLE_SEED):
    global SAMPLES
    datasets = build_datasets()
    candidates = []
    for dataset_name, dataset in datasets:
        for seq_index in range(dataset.sequence_list_len):
            candidates.append((dataset_name, dataset, seq_index))

    rng = random.Random(int(seed))
    selected = rng.sample(candidates, min(SAMPLE_COUNT, len(candidates)))
    SAMPLES = []
    for sample_idx, (dataset_name, dataset, seq_index) in enumerate(selected, start=1):
        rec = dataset.records[seq_index % dataset.sequence_list_len]
        SAMPLES.append(
            {
                "label": make_label(dataset_name, sample_idx, seq_index, rec),
                "dataset_name": dataset_name,
                "dataset": dataset,
                "seq_index": seq_index,
            }
        )
    return [sample["label"] for sample in SAMPLES]


def get_sample(label: str):
    if not SAMPLES:
        refresh_samples(SAMPLE_SEED)
    for sample in SAMPLES:
        if sample["label"] == label:
            return sample
    return SAMPLES[0]


def load_batch(sample_label: str, view_seed: int):
    sample = get_sample(sample_label)
    rng_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(int(view_seed))
    np.random.seed(int(view_seed) % (2**32 - 1))
    try:
        batch = sample["dataset"].get_data(
            seq_index=sample["seq_index"],
            img_per_seq=4,
            aspect_ratio=1.0,
        )
    finally:
        random.setstate(rng_state)
        np.random.set_state(np_state)
    return sample, batch


BBOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def centered_bbox_points(object_size, scale_multiplier: float):
    size = np.asarray(object_size, dtype=np.float32).reshape(3) * float(scale_multiplier)
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


def draw_bbox_and_pose(image, bbox_points, rotation, translation, axis_length, extrinsic, intrinsic):
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
        if not (-w <= p0[0] <= 2 * w and -h <= p0[1] <= 2 * h):
            continue
        if not (-w <= p1[0] <= 2 * w and -h <= p1[1] <= 2 * h):
            continue
        draw.line([p0, p1], fill=(60, 255, 90), width=3)

    axes_obj = np.asarray(
        [[0.0, 0.0, 0.0], [axis_length, 0.0, 0.0], [0.0, axis_length, 0.0], [0.0, 0.0, axis_length]],
        dtype=np.float32,
    )
    axes = transform_points(axes_obj, rotation, translation)
    axis_uv, axis_valid = project_points(axes, extrinsic, intrinsic)
    if axis_valid[0]:
        origin = tuple(np.round(axis_uv[0]).astype(int))
        font = ImageFont.load_default()
        colors = [(255, 80, 80), (80, 255, 255), (255, 220, 80)]
        labels = ["X", "Y", "Z"]
        if -w <= origin[0] <= 2 * w and -h <= origin[1] <= 2 * h:
            for axis_idx in range(3):
                end_idx = axis_idx + 1
                if not axis_valid[end_idx]:
                    continue
                end = tuple(np.round(axis_uv[end_idx]).astype(int))
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


def build_glb(sample, batch, bbox_points):
    safe_sample = re.sub(r"[^a-zA-Z0-9_]+", "_", sample["label"])[:120]
    out_dir = VIEWER_CACHE_ROOT / safe_sample
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "normalized_point_cloud_gt_pose_bbox.glb"

    all_points = []
    all_colors = []
    for image, points, point_mask in zip(batch["images"], batch["world_points"], batch["point_masks"]):
        image = np.asarray(image, dtype=np.uint8)
        points = np.asarray(points, dtype=np.float32)
        mask = np.asarray(point_mask, dtype=bool) & np.isfinite(points).all(axis=-1)
        if not np.any(mask):
            continue
        all_points.append(points[mask])
        all_colors.append(image[mask])
    points = np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 3), dtype=np.float32)
    colors = np.concatenate(all_colors, axis=0) if all_colors else np.zeros((0, 3), dtype=np.uint8)
    points, colors = subsample_points(points, colors, SCENE_POINT_LIMIT)

    scene = trimesh.Scene()
    if points.shape[0] > 0:
        scene.add_geometry(trimesh.PointCloud(vertices=points, colors=colors))
    scene.add_geometry(bbox_mesh(bbox_points, [60, 255, 90, 255]))
    diag = float(np.linalg.norm(bbox_points.max(axis=0) - bbox_points.min(axis=0)))
    scene.add_geometry(
        axes_mesh(
            batch["object_rotation"],
            batch["object_translation"],
            diag * 0.35,
            max(diag * 0.012, 0.002),
        )
    )
    scene.export(out_path)
    return str(out_path), int(points.shape[0])


def image_grid_rows(images, labels):
    rows = []
    for image, label in zip(images, labels):
        rows.append((np.asarray(image, dtype=np.uint8), str(label)))
    return rows


def sample_paths(sample, batch):
    paths = {
        "scene_rgb_paths": list(batch.get("scene_rgb_paths", [])),
        "scene_depth_paths": list(batch.get("scene_depth_paths", [])),
        "object_rgb_paths": list(batch.get("object_rgb_paths", [])),
    }
    if all(paths.values()):
        return paths

    dataset = sample["dataset"]
    if isinstance(dataset, OV9DSinglePoseNormalizeDataset):
        rec = dataset.records[sample["seq_index"] % dataset.sequence_list_len]
        scene_dir = Path(rec["scene_dir"])
        scene_ids = np.asarray(batch["ids"]).reshape(-1).tolist()
        object_ids = np.asarray(batch["object_cam_indices"]).reshape(-1).tolist()
        object_scene_dir = dataset.object_image_root / str(batch["object_reference_scene_name"])
        paths["scene_rgb_paths"] = [str(scene_dir / "rgb" / f"{int(i):06d}.png") for i in scene_ids]
        paths["scene_depth_paths"] = [str(scene_dir / "depth" / f"{int(i):06d}.png") for i in scene_ids]
        paths["object_rgb_paths"] = [str(object_scene_dir / "rgb" / f"{int(i):06d}.png") for i in object_ids]

    return paths


def render_sample(sample_label: str, view_seed: int, box_scale: float):
    sample, batch = load_batch(sample_label, view_seed)
    if "object_size" not in batch:
        raise RuntimeError(f"Dataset sample has no object_size: {sample['label']}")

    bbox_obj = centered_bbox_points(batch["object_size"], box_scale)
    bbox_world = transform_points(bbox_obj, batch["object_rotation"], batch["object_translation"])
    axis_length = float(np.linalg.norm(bbox_world.max(axis=0) - bbox_world.min(axis=0)) * 0.35)
    glb_path, point_count = build_glb(sample, batch, bbox_world)

    scene_images = []
    for view_idx, image in enumerate(batch["images"]):
        scene_images.append(
            draw_bbox_and_pose(
                image,
                bbox_world,
                batch["object_rotation"],
                batch["object_translation"],
                axis_length,
                batch["extrinsics"][view_idx],
                batch["intrinsics"][view_idx],
            )
        )
    scene_labels = [f"scene view {i}: frame {int(frame_id)}" for i, frame_id in enumerate(batch["ids"])]
    object_labels = [
        f"object view {i}: frame {int(frame_id)}"
        for i, frame_id in enumerate(np.asarray(batch["object_cam_indices"]).reshape(-1))
    ]

    rotation = np.asarray(batch["object_rotation"], dtype=np.float32).reshape(3, 3)
    translation = np.asarray(batch["object_translation"], dtype=np.float32).reshape(3)
    object_size = np.asarray(batch["object_size"], dtype=np.float32).reshape(3)
    scale = float(np.asarray(batch["normalization_scale"]).reshape(-1)[0])
    paths = []
    path_values = sample_paths(sample, batch)
    for title, key in (("scene_rgb_paths", "scene_rgb_paths"), ("scene_depth_paths", "scene_depth_paths"), ("object_rgb_paths", "object_rgb_paths")):
        values = path_values.get(key, [])
        paths.append(f"**{title}**")
        paths.extend(f"`{path}`" for path in values)

    info = (
        f"Config: `{CONFIG_PATH}`  \n"
        f"Dataset: `{sample['dataset_name']}`  \n"
        f"Seq index: `{sample['seq_index']}`  \n"
        f"Seq name: `{batch['seq_name']}`  \n"
        f"Scene: `{batch.get('scene_name', '')}`  \n"
        f"Object: `{batch.get('object_name', '')}` / `{batch.get('category', '')}`  \n"
        f"Object reference: `{batch.get('object_reference_name', '')}` / `{batch.get('object_reference_category', '')}`  \n"
        f"has_object: `{float(np.asarray(batch['has_object']).reshape(-1)[0]):.1f}`  \n"
        f"symmetry_object_id: `{batch.get('symmetry_object_id', '')}`  \n"
        f"GT normalized translation: `{[round(float(v), 6) for v in translation.tolist()]}`  \n"
        f"GT normalized rotation:\n```text\n{np.array2string(rotation, precision=5, suppress_small=True)}\n```  \n"
        f"GT normalized bbox size: `{[round(float(v), 6) for v in object_size.tolist()]}`  \n"
        f"normalization_scale: `{scale:.6f}`  \n"
        f"GLB point count: `{point_count}`  \n\n"
        + "  \n".join(paths)
    )
    return (
        image_grid_rows(scene_images, scene_labels),
        image_grid_rows(batch["object_images"], object_labels),
        glb_path,
        info,
    )


def regenerate(seed: int):
    labels = refresh_samples(int(seed))
    value = labels[0] if labels else None
    return gr.update(choices=labels, value=value)


initial_labels = refresh_samples(SAMPLE_SEED)
initial_sample = initial_labels[0] if initial_labels else None

theme = gr.themes.Ocean()
with gr.Blocks() as demo:
    gr.HTML(
        """
        <h1>Mixed Real Dataset GT Normalize Pose + BBox Demo</h1>
        <p>
        Randomly samples 20 records from the same four-dataset setup used by
        default_object_oo9d_real275_ycbv_hc.yaml. Scene RGB has projected GT pose/bbox;
        3D view shows the GT normalized point cloud with bbox and axes.
        </p>
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            sample_dropdown = gr.Dropdown(choices=initial_labels, value=initial_sample, label="Random Sample")
            seed_input = gr.Number(label="Scene/Object View Sampling Seed", value=SAMPLE_SEED, precision=0)
            sample_seed_input = gr.Number(label="Resample 20 Records Seed", value=SAMPLE_SEED, precision=0)
            scale_slider = gr.Slider(0.25, 4.0, value=1.0, step=0.05, label="BBox Scale")
            with gr.Row():
                resample_btn = gr.Button("Resample 20")
                render_btn = gr.Button("Render", variant="primary")
            info_output = gr.Markdown("Pick a sample, then click Render.")
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Scene RGB"):
                    scene_gallery = gr.Gallery(
                        label="4 sampled scene views with GT normalized pose and bbox projection",
                        columns=2,
                        height=720,
                    )
                with gr.Tab("Object RGB"):
                    object_gallery = gr.Gallery(label="4 fixed object reference views", columns=4, height=360)
                with gr.Tab("Normalized Point Cloud"):
                    model_viewer = gr.Model3D(label="Normalized point cloud + GT bbox + GT axes", height=720)

    demo.load(
        fn=render_sample,
        inputs=[sample_dropdown, seed_input, scale_slider],
        outputs=[scene_gallery, object_gallery, model_viewer, info_output],
    )
    resample_btn.click(fn=regenerate, inputs=[sample_seed_input], outputs=[sample_dropdown]).then(
        fn=render_sample,
        inputs=[sample_dropdown, seed_input, scale_slider],
        outputs=[scene_gallery, object_gallery, model_viewer, info_output],
    )
    render_btn.click(
        fn=render_sample,
        inputs=[sample_dropdown, seed_input, scale_slider],
        outputs=[scene_gallery, object_gallery, model_viewer, info_output],
    )
    sample_dropdown.change(
        fn=render_sample,
        inputs=[sample_dropdown, seed_input, scale_slider],
        outputs=[scene_gallery, object_gallery, model_viewer, info_output],
    )
    seed_input.change(
        fn=render_sample,
        inputs=[sample_dropdown, seed_input, scale_slider],
        outputs=[scene_gallery, object_gallery, model_viewer, info_output],
    )
    scale_slider.change(
        fn=render_sample,
        inputs=[sample_dropdown, seed_input, scale_slider],
        outputs=[scene_gallery, object_gallery, model_viewer, info_output],
    )


if __name__ == "__main__":
    demo.launch(server_name=SERVER_NAME, server_port=SERVER_PORT, share=GRADIO_SHARE, theme=theme)
