import glob
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

sys.path.append("vggt/")

from vggt.models.vggt import VGGT

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

DATA_ROOT = Path("/mnt/train-data-4-hdd/yian/6dpose_obj/0327_fixedCam_1k")
OUT_IMAGE_ROOT = DATA_ROOT / "out_image"
OUT_POSE_ROOT = DATA_ROOT / "out_pose"
OBJECT_IMAGE_ROOT = DATA_ROOT / "object_space_rgb"
FIXED_VIEWS = (1, 3, 8, 12, 15, 18)
NUM_OBJECT_VIEWS = 4
MODEL_CKPT_PATH = Path(
    "/mnt/train-data-4-hdd/yian/6dpose_obj/vggt_objectspc/training/logs/"
    "test_0327_object_dataset_mask_test_1k/ckpts/checkpoint_10.pt"
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


def decode_name(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_pose_npz(run_name: str):
    pose_path = OUT_POSE_ROOT / f"{run_name}.npz"
    data = np.load(pose_path, allow_pickle=True)
    names = [decode_name(x) for x in data["names"]]
    return names


def get_objects_for_run(run_name: str):
    return load_pose_npz(run_name)


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
    cleaned = [int(v) for v in selected_view_indices]
    if len(cleaned) != NUM_OBJECT_VIEWS:
        raise ValueError(f"Please provide exactly {NUM_OBJECT_VIEWS} object view indices.")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("Object view indices must be unique.")
    return [object_image_path(object_name, v) for v in cleaned]


def _load_images_to_numpy(image_paths):
    images = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return images


def _images_to_gallery(images, prefix: str):
    return [(image, f"{prefix} {idx + 1}") for idx, image in enumerate(images)]


def _load_images_to_device(image_paths):
    tensors = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).to(torch.float32).div(255.0)
        tensors.append(tensor)
    return torch.stack(tensors, dim=0).to(device)


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
        return [int(part) for part in parts]
    except ValueError as exc:
        if strict:
            raise ValueError(f"Object view indices must be integers, got: {view_index_text}") from exc
        return None


def format_view_index_text(view_indices):
    return ", ".join(str(int(v)) for v in view_indices)


def build_model():
    print("Initializing VGGT heatmap model...")
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


def _infer_patch_grid(num_patches: int):
    side = int(round(math.sqrt(num_patches)))
    if side * side != num_patches:
        raise ValueError(f"Patch count {num_patches} is not a square grid.")
    return side, side


def _compute_heatmap_values(tokens: torch.Tensor):
    return torch.linalg.norm(tokens, dim=-1)


def _compute_relative_delta_values(scene_tokens: torch.Tensor, delta_tokens: torch.Tensor):
    scene_norm = torch.linalg.norm(scene_tokens, dim=-1)
    delta_norm = torch.linalg.norm(delta_tokens, dim=-1)
    return delta_norm / scene_norm.clamp_min(1e-6)


def _compute_cosine_change_values(scene_tokens: torch.Tensor, fused_tokens: torch.Tensor):
    cosine = torch.nn.functional.cosine_similarity(scene_tokens, fused_tokens, dim=-1, eps=1e-6)
    return 1.0 - cosine


def _tokens_to_heatmaps(token_values: np.ndarray, images, prefix: str):
    gallery = []
    for idx, image in enumerate(images):
        heat = token_values[idx]
        h_patch, w_patch = _infer_patch_grid(int(heat.size))
        heat = heat.reshape(h_patch, w_patch)
        heat = heat.astype(np.float32)
        heat = heat - float(heat.min())
        heat = heat / max(float(heat.max()), 1e-6)
        heat_resized = cv2.resize(heat, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        heat_uint8 = np.clip(255.0 * heat_resized, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_TURBO)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(image, 0.45, color, 0.55, 0.0)
        gallery.append((overlay, f"{prefix} {idx + 1}"))
    return gallery


def _summarize_tensor(name: str, tensor: torch.Tensor):
    arr = tensor.detach().float().cpu()
    return (
        f"{name}: mean={arr.mean().item():.4f}, std={arr.std().item():.4f}, "
        f"min={arr.min().item():.4f}, max={arr.max().item():.4f}"
    )


def _collect_token_visualizations(scene_images: torch.Tensor, object_images: torch.Tensor):
    if scene_images.dim() == 4:
        scene_images = scene_images.unsqueeze(0)
    if object_images.dim() == 4:
        object_images = object_images.unsqueeze(0)

    aggregated_tokens_list, patch_start_idx = model.aggregator(scene_images)
    object_aggregated_tokens_list, object_patch_start_idx = model.aggregator(object_images)

    scene_tokens = aggregated_tokens_list[-1]
    scene_patch_tokens = scene_tokens[:, :, patch_start_idx:, :]
    object_tokens = object_aggregated_tokens_list[-1]
    object_patch_tokens = object_tokens[:, :, object_patch_start_idx:, :]

    bsz, s_obj, p_obj, c_obj = object_patch_tokens.shape
    object_context = object_patch_tokens.reshape(bsz, s_obj * p_obj, c_obj)

    bsz, s_scene, p_scene, c_scene = scene_patch_tokens.shape
    scene_query = scene_patch_tokens.reshape(bsz, s_scene * p_scene, c_scene)
    fused_scene_patch_tokens = model.object_token_cross_attn(scene_query, object_context)
    fused_scene_patch_tokens = fused_scene_patch_tokens.view(bsz, s_scene, p_scene, c_scene)

    delta_tokens = fused_scene_patch_tokens - scene_patch_tokens

    return {
        "scene_patch_tokens": scene_patch_tokens[0],
        "object_patch_tokens": object_patch_tokens[0],
        "fused_scene_patch_tokens": fused_scene_patch_tokens[0],
        "delta_scene_patch_tokens": delta_tokens[0],
    }


def describe_selection(run_name: str, object_name: str, seed: int, view_index_text: str):
    empty = [], [], "", "Please select a run.", [], [], [], [], [], [], []
    if not run_name:
        return (gr.update(choices=[], value=None),) + empty

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
        f"Run {run_name}: fixed scene views {FIXED_VIEWS}. "
        f"Object {chosen_object}: selected object view indices [{selected_text}]. "
        f"Generate will visualize scene tokens, object tokens, fused scene tokens, token delta, "
        f"relative delta, cosine change, and optional wrong-object ablation."
    )
    return (
        gr.update(choices=object_choices, value=chosen_object),
        _images_to_gallery(_load_images_to_numpy(scene_paths), "Scene"),
        _images_to_gallery(_load_images_to_numpy(object_paths), "Object"),
        selected_text,
        message,
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )


def _choose_wrong_object(run_name: str, object_name: str):
    object_choices = get_objects_for_run(run_name)
    for candidate in object_choices:
        if candidate != object_name:
            return candidate
    return None


def run_inference(run_name: str, object_name: str, seed: int, view_index_text: str, enable_wrong_object_ablation: bool):
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

    scene_image_arrays = _load_images_to_numpy(scene_paths)
    object_image_arrays = _load_images_to_numpy(object_paths)
    scene_images = _load_images_to_device(scene_paths)
    object_images = _load_images_to_device(object_paths)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    start_time = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            token_dict = _collect_token_visualizations(scene_images, object_images)
    elapsed = time.time() - start_time

    scene_norm = _compute_heatmap_values(token_dict["scene_patch_tokens"]).cpu().numpy()
    object_norm = _compute_heatmap_values(token_dict["object_patch_tokens"]).cpu().numpy()
    fused_norm = _compute_heatmap_values(token_dict["fused_scene_patch_tokens"]).cpu().numpy()
    delta_norm = _compute_heatmap_values(token_dict["delta_scene_patch_tokens"]).cpu().numpy()
    relative_delta = _compute_relative_delta_values(
        token_dict["scene_patch_tokens"], token_dict["delta_scene_patch_tokens"]
    ).cpu().numpy()
    cosine_change = _compute_cosine_change_values(
        token_dict["scene_patch_tokens"], token_dict["fused_scene_patch_tokens"]
    ).cpu().numpy()

    scene_gallery = _tokens_to_heatmaps(scene_norm, scene_image_arrays, "Scene")
    object_gallery = _tokens_to_heatmaps(object_norm, object_image_arrays, "Object")
    fused_gallery = _tokens_to_heatmaps(fused_norm, scene_image_arrays, "Fused")
    delta_gallery = _tokens_to_heatmaps(delta_norm, scene_image_arrays, "Delta")
    relative_delta_gallery = _tokens_to_heatmaps(relative_delta, scene_image_arrays, "Relative Delta")
    cosine_change_gallery = _tokens_to_heatmaps(cosine_change, scene_image_arrays, "Cosine Change")

    wrong_object_gallery = []
    wrong_object_message = "wrong_object_ablation: disabled"
    if enable_wrong_object_ablation:
        wrong_object_name = _choose_wrong_object(run_name, object_name)
        if wrong_object_name is None:
            wrong_object_message = "wrong_object_ablation: no alternative object available in this run"
        else:
            wrong_object_view_indices = sample_object_view_indices(wrong_object_name, seed)
            wrong_object_paths = resolve_object_images(wrong_object_name, wrong_object_view_indices)
            wrong_object_images = _load_images_to_device(wrong_object_paths)
            wrong_object_arrays = _load_images_to_numpy(wrong_object_paths)
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    wrong_token_dict = _collect_token_visualizations(scene_images, wrong_object_images)
            wrong_relative_delta = _compute_relative_delta_values(
                wrong_token_dict["scene_patch_tokens"], wrong_token_dict["delta_scene_patch_tokens"]
            ).cpu().numpy()
            wrong_object_gallery = _tokens_to_heatmaps(
                wrong_relative_delta,
                scene_image_arrays,
                f"Wrong Object Relative Delta ({wrong_object_name})",
            )
            wrong_object_message = (
                f"wrong_object_ablation: using {wrong_object_name} with views "
                f"[{format_view_index_text(wrong_object_view_indices)}]"
            )

    log_msg = "\n".join(
        [
            f"run={run_name}, object={object_name}, object_views=[{format_view_index_text(selected_view_indices)}], time={elapsed:.2f}s",
            _summarize_tensor("scene_patch_tokens", token_dict["scene_patch_tokens"]),
            _summarize_tensor("object_patch_tokens", token_dict["object_patch_tokens"]),
            _summarize_tensor("fused_scene_patch_tokens", token_dict["fused_scene_patch_tokens"]),
            _summarize_tensor("delta_scene_patch_tokens", token_dict["delta_scene_patch_tokens"]),
            (
                "relative_delta: "
                f"mean={float(relative_delta.mean()):.4f}, std={float(relative_delta.std()):.4f}, "
                f"min={float(relative_delta.min()):.4f}, max={float(relative_delta.max()):.4f}"
            ),
            (
                "cosine_change: "
                f"mean={float(cosine_change.mean()):.4f}, std={float(cosine_change.std()):.4f}, "
                f"min={float(cosine_change.min()):.4f}, max={float(cosine_change.max()):.4f}"
            ),
            wrong_object_message,
        ]
    )

    return (
        log_msg,
        _images_to_gallery(scene_image_arrays, "Scene"),
        _images_to_gallery(object_image_arrays, "Object"),
        format_view_index_text(selected_view_indices),
        scene_gallery,
        object_gallery,
        fused_gallery,
        delta_gallery,
        relative_delta_gallery,
        cosine_change_gallery,
        wrong_object_gallery,
    )


def refresh_from_run(run_name: str, seed: int, view_index_text: str):
    return describe_selection(run_name, None, seed, view_index_text)


def refresh_from_object(run_name: str, object_name: str, seed: int, view_index_text: str):
    return describe_selection(run_name, object_name, seed, view_index_text)


def refresh_from_seed(run_name: str, object_name: str, seed: int, view_index_text: str):
    return describe_selection(run_name, object_name, seed, view_index_text)


def clear_outputs():
    return "Select a run and object, then click Generate.", [], [], [], [], [], [], []


theme = gr.themes.Ocean()

with gr.Blocks(theme=theme) as demo:
    gr.HTML(
        """
        <h1>VGGT Token Heatmap Viewer</h1>
        <p>Use the same dataset inputs as <code>demo_gradio.py</code>. This page loads the requested checkpoint,
        extracts the final-layer <code>object tokens</code>, <code>scene tokens</code>, and <code>cross-attention fused scene tokens</code>,
        then visualizes token L2 norm heatmaps overlaid on the input images. The delta tab shows
        <code>||fused - scene||</code> per patch.</p>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            run_dropdown = gr.Dropdown(choices=RUN_CHOICES, label="Run", value=RUN_CHOICES[0] if RUN_CHOICES else None)
            object_dropdown = gr.Dropdown(choices=[], label="Object", value=None)
            seed_input = gr.Number(label="Random Seed For 4 Object Views", value=42, precision=0)
            object_view_indices_input = gr.Textbox(label="Object View Indices", value="", placeholder="e.g. 1, 3, 8, 12")
            wrong_object_ablation_checkbox = gr.Checkbox(label="Enable Wrong Object Ablation", value=False)
            refresh_btn = gr.Button("Resample Object Views")
            generate_btn = gr.Button("Generate Heatmaps", variant="primary")
            clear_btn = gr.Button("Clear Output")
            log_output = gr.Markdown("Select a run and object, then click Generate.")
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Input Views"):
                    scene_input_gallery = gr.Gallery(label="Scene Fixed Views", columns=3, height="520px", object_fit="contain")
                    object_input_gallery = gr.Gallery(label="Sampled Object Views", columns=4, height="260px", object_fit="contain")
                with gr.Tab("Scene Tokens"):
                    scene_token_gallery = gr.Gallery(label="Scene Token Norm Heatmap", columns=3, height="520px", object_fit="contain")
                with gr.Tab("Object Tokens"):
                    object_token_gallery = gr.Gallery(label="Object Token Norm Heatmap", columns=4, height="320px", object_fit="contain")
                with gr.Tab("Fused Scene Tokens"):
                    fused_scene_gallery = gr.Gallery(label="Cross-Attention Fused Scene Heatmap", columns=3, height="520px", object_fit="contain")
                with gr.Tab("Delta"):
                    delta_scene_gallery = gr.Gallery(label="Token Delta Heatmap", columns=3, height="520px", object_fit="contain")
                with gr.Tab("Relative Delta"):
                    relative_delta_gallery = gr.Gallery(label="Relative Delta Heatmap", columns=3, height="520px", object_fit="contain")
                with gr.Tab("Cosine Change"):
                    cosine_change_gallery = gr.Gallery(label="Cosine Change Heatmap", columns=3, height="520px", object_fit="contain")
                with gr.Tab("Wrong Object Ablation"):
                    wrong_object_gallery = gr.Gallery(label="Wrong Object Relative Delta Heatmap", columns=3, height="520px", object_fit="contain")

    demo.load(
        fn=lambda: describe_selection(RUN_CHOICES[0], None, 42, "") if RUN_CHOICES else (gr.update(choices=[], value=None), [], [], "", "No runs found.", [], [], [], [], [], [], []),
        inputs=[],
        outputs=[
            object_dropdown,
            scene_input_gallery,
            object_input_gallery,
            object_view_indices_input,
            log_output,
            scene_token_gallery,
            object_token_gallery,
            fused_scene_gallery,
            delta_scene_gallery,
            relative_delta_gallery,
            cosine_change_gallery,
            wrong_object_gallery,
        ],
    )

    run_dropdown.change(
        fn=refresh_from_run,
        inputs=[run_dropdown, seed_input, object_view_indices_input],
        outputs=[
            object_dropdown,
            scene_input_gallery,
            object_input_gallery,
            object_view_indices_input,
            log_output,
            scene_token_gallery,
            object_token_gallery,
            fused_scene_gallery,
            delta_scene_gallery,
            relative_delta_gallery,
            cosine_change_gallery,
            wrong_object_gallery,
        ],
    )
    object_dropdown.change(
        fn=refresh_from_object,
        inputs=[run_dropdown, object_dropdown, seed_input, object_view_indices_input],
        outputs=[
            object_dropdown,
            scene_input_gallery,
            object_input_gallery,
            object_view_indices_input,
            log_output,
            scene_token_gallery,
            object_token_gallery,
            fused_scene_gallery,
            delta_scene_gallery,
            relative_delta_gallery,
            cosine_change_gallery,
            wrong_object_gallery,
        ],
    )
    seed_input.change(
        fn=refresh_from_seed,
        inputs=[run_dropdown, object_dropdown, seed_input, object_view_indices_input],
        outputs=[
            object_dropdown,
            scene_input_gallery,
            object_input_gallery,
            object_view_indices_input,
            log_output,
            scene_token_gallery,
            object_token_gallery,
            fused_scene_gallery,
            delta_scene_gallery,
            relative_delta_gallery,
            cosine_change_gallery,
            wrong_object_gallery,
        ],
    )
    refresh_btn.click(
        fn=lambda run_name, object_name, seed: describe_selection(run_name, object_name, seed, ""),
        inputs=[run_dropdown, object_dropdown, seed_input],
        outputs=[
            object_dropdown,
            scene_input_gallery,
            object_input_gallery,
            object_view_indices_input,
            log_output,
            scene_token_gallery,
            object_token_gallery,
            fused_scene_gallery,
            delta_scene_gallery,
            relative_delta_gallery,
            cosine_change_gallery,
            wrong_object_gallery,
        ],
    )
    generate_btn.click(
        fn=run_inference,
        inputs=[run_dropdown, object_dropdown, seed_input, object_view_indices_input, wrong_object_ablation_checkbox],
        outputs=[
            log_output,
            scene_input_gallery,
            object_input_gallery,
            object_view_indices_input,
            scene_token_gallery,
            object_token_gallery,
            fused_scene_gallery,
            delta_scene_gallery,
            relative_delta_gallery,
            cosine_change_gallery,
            wrong_object_gallery,
        ],
    )
    clear_btn.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[
            log_output,
            scene_token_gallery,
            object_token_gallery,
            fused_scene_gallery,
            delta_scene_gallery,
            relative_delta_gallery,
            cosine_change_gallery,
            wrong_object_gallery,
        ],
    )

    demo.queue(max_size=20).launch(show_error=True)
