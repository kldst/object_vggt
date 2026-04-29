"""
Analyze the distribution of normalized object poses (rotation + translation)
for all (scene, object) pairs in the train split after applying the same
normalization as OV9DMultiPoseNormalizeDataset.

Only camera JSON and depth maps are loaded — RGB images are skipped for speed.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

BASELINE_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = BASELINE_DIR / "training"
for _p in [str(TRAINING_ROOT), str(BASELINE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.dataset_util import depth_to_world_coords_points
from data.datasets.ov9d_pose_normalize import OV9DPoseNormalizeDataset

MULTI_DIR_DEFAULT = "/mnt/train-data-4-hdd/yian/freepose/ov9d/ov9d/oo3d9dmulti"
SPLIT_JSON_DEFAULT = str(BASELINE_DIR / "splits_multi" / "train.json")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_depth(path: Path, depth_scale: float) -> np.ndarray:
    depth_raw = np.asarray(Image.open(path), dtype=np.float32)
    depth = depth_raw * float(depth_scale)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0
    return depth.astype(np.float32)


# ---------------------------------------------------------------------------
# Core collection logic
# ---------------------------------------------------------------------------

def collect_normalized_poses(
    split_json: Path,
    multi_dir: Path,
    max_scenes: Optional[int],
    num_views: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Iterate over every (scene, eligible_object) pair in train.json.
    For each pair:
      1. Load `num_views` depth maps + camera parameters.
      2. Compute avg_scale via normalize_extrinsics_and_world_points.
      3. Compute the normalized object rotation (3×3) and translation (3,)
         using normalize_object_pose_to_first_camera.

    Returns:
        rotations   – list of (3, 3) float32 arrays
        translations – list of (3,) float32 arrays
        object_ids  – list of int object ids (parallel to the above)
    """
    payload = load_json(split_json)
    scenes: List[Dict[str, Any]] = payload.get("scenes", [])
    if max_scenes is not None:
        scenes = scenes[:max_scenes]

    rotations: List[np.ndarray] = []
    translations: List[np.ndarray] = []
    object_ids: List[int] = []

    n_scenes = len(scenes)
    n_skipped_scene = 0
    n_skipped_obj = 0

    for scene_idx, item in enumerate(scenes):
        scene_name: str = item["scene_name"]
        scene_dir = multi_dir / scene_name

        scene_gt_path = scene_dir / "scene_gt.json"
        scene_camera_path = scene_dir / "scene_camera.json"
        if not scene_gt_path.is_file() or not scene_camera_path.is_file():
            n_skipped_scene += 1
            continue

        scene_gt = load_json(scene_gt_path)
        scene_camera = load_json(scene_camera_path)

        eligible_object_ids = [int(x) for x in item.get("eligible_object_ids", [])]
        if not eligible_object_ids:
            n_skipped_scene += 1
            continue

        # ------------------------------------------------------------------
        # Select the first `num_views` image ids for scale estimation
        # ------------------------------------------------------------------
        all_image_ids = sorted(int(k) for k in scene_gt.keys())
        selected_ids = all_image_ids[:num_views]

        extrinsics_list: List[np.ndarray] = []
        world_points_list: List[np.ndarray] = []
        point_masks_list: List[np.ndarray] = []

        for image_id in selected_ids:
            cam = scene_camera[str(image_id)]
            extrinsic = OV9DPoseNormalizeDataset.bop_camera_to_extrinsic(cam)
            intrinsic = OV9DPoseNormalizeDataset.bop_camera_to_intrinsic(cam)
            depth_path = scene_dir / "depth" / f"{image_id:06d}.png"
            if not depth_path.is_file():
                continue
            depth = read_depth(depth_path, float(cam.get("depth_scale", 1.0)))
            world_pts, _, point_mask = depth_to_world_coords_points(depth, extrinsic, intrinsic)
            extrinsics_list.append(extrinsic)
            world_points_list.append(world_pts)
            point_masks_list.append(point_mask)

        if not extrinsics_list:
            n_skipped_scene += 1
            continue

        extrinsics_np = np.stack(extrinsics_list).astype(np.float32)
        _, avg_scale, _ = OV9DPoseNormalizeDataset.normalize_extrinsics_and_world_points(
            extrinsics=extrinsics_np,
            world_points=world_points_list,
            point_masks=point_masks_list,
            scale_by_points=True,
        )
        first_extrinsic = extrinsics_np[0]

        # ------------------------------------------------------------------
        # For each eligible object find the first frame where it is visible
        # and compute the normalized pose
        # ------------------------------------------------------------------
        for object_id in eligible_object_ids:
            found = False
            for image_id in all_image_ids:
                gts = scene_gt.get(str(image_id), [])
                obj_gt = next(
                    (gt for gt in gts if int(gt.get("obj_id", -1)) == object_id),
                    None,
                )
                if obj_gt is None:
                    continue
                cam = scene_camera[str(image_id)]
                r_m2w, t_m2w = OV9DPoseNormalizeDataset.model_to_world_from_bop(cam, obj_gt)
                rot, trans = OV9DPoseNormalizeDataset.normalize_object_pose_to_first_camera(
                    first_extrinsic=first_extrinsic,
                    object_rotation_m2w=r_m2w,
                    object_translation_m2w=t_m2w,
                    avg_scale=avg_scale,
                    scale_by_points=True,
                )
                rotations.append(rot)
                translations.append(trans)
                object_ids.append(object_id)
                found = True
                break
            if not found:
                n_skipped_obj += 1

        if (scene_idx + 1) % 200 == 0 or scene_idx + 1 == n_scenes:
            print(
                f"  [{scene_idx + 1}/{n_scenes}] collected {len(rotations)} pose samples"
                f"  (skipped scenes={n_skipped_scene}, skipped objects={n_skipped_obj})"
            )

    return rotations, translations, object_ids


# ---------------------------------------------------------------------------
# Statistics & plotting
# ---------------------------------------------------------------------------

def rotation_to_angle_deg(rot: np.ndarray) -> float:
    """Axis-angle magnitude (degrees) for a rotation matrix."""
    trace = float(np.trace(rot))
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def print_stats(rotations: List[np.ndarray], translations: List[np.ndarray]) -> None:
    rot_angles = np.array([rotation_to_angle_deg(r) for r in rotations])
    trans = np.stack(translations)
    trans_norms = np.linalg.norm(trans, axis=1)

    print(f"\n{'='*55}")
    print(f"  Normalized Pose Distribution  (n={len(rotations)})")
    print(f"{'='*55}")

    print("\nRotation angle (degrees):")
    print(f"  mean={rot_angles.mean():.2f}  std={rot_angles.std():.2f}")
    print(f"  min={rot_angles.min():.2f}  p25={np.percentile(rot_angles, 25):.2f}"
          f"  p50={np.percentile(rot_angles, 50):.2f}"
          f"  p75={np.percentile(rot_angles, 75):.2f}  max={rot_angles.max():.2f}")

    print("\nNormalized translation (X, Y, Z):")
    for i, label in enumerate(["X", "Y", "Z"]):
        v = trans[:, i]
        print(f"  {label}: mean={v.mean():.4f}  std={v.std():.4f}"
              f"  min={v.min():.4f}  max={v.max():.4f}")

    print("\nNormalized translation norm:")
    print(f"  mean={trans_norms.mean():.4f}  std={trans_norms.std():.4f}")
    print(f"  min={trans_norms.min():.4f}  p25={np.percentile(trans_norms, 25):.4f}"
          f"  p50={np.percentile(trans_norms, 50):.4f}"
          f"  p75={np.percentile(trans_norms, 75):.4f}  max={trans_norms.max():.4f}")
    print()


def plot_distributions(
    rotations: List[np.ndarray],
    translations: List[np.ndarray],
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rot_angles = np.array([rotation_to_angle_deg(r) for r in rotations])
    trans = np.stack(translations)
    trans_norms = np.linalg.norm(trans, axis=1)
    rot_flat = np.stack(rotations).reshape(-1)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(f"Normalized Pose Distribution  (n={len(rotations)})", fontsize=13)

    def _hist(ax, data, title, xlabel, bins=80, color="#4C72B0"):
        ax.hist(data, bins=bins, color=color, edgecolor="none", alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("count", fontsize=9)
        ax.axvline(float(np.median(data)), color="red", linewidth=1, linestyle="--", label=f"median={np.median(data):.3f}")
        ax.legend(fontsize=8)

    _hist(axes[0, 0], rot_angles, "Rotation angle", "degrees")
    _hist(axes[0, 1], trans[:, 0], "Normalized translation X", "value", color="#55A868")
    _hist(axes[0, 2], trans[:, 1], "Normalized translation Y", "value", color="#C44E52")
    _hist(axes[0, 3], trans[:, 2], "Normalized translation Z", "value", color="#8172B2")

    _hist(axes[1, 0], trans_norms, "Normalized translation norm", "value", color="#CCB974")
    _hist(axes[1, 1], rot_flat, "Rotation matrix elements", "value", bins=120, color="#64B5CD")

    # XY scatter
    axes[1, 2].scatter(trans[:, 0], trans[:, 1], alpha=0.2, s=4, linewidths=0, color="#4C72B0")
    axes[1, 2].set_title("Translation X vs Y", fontsize=10)
    axes[1, 2].set_xlabel("X", fontsize=9)
    axes[1, 2].set_ylabel("Y", fontsize=9)
    axes[1, 2].set_aspect("equal")

    # XZ scatter
    axes[1, 3].scatter(trans[:, 0], trans[:, 2], alpha=0.2, s=4, linewidths=0, color="#55A868")
    axes[1, 3].set_title("Translation X vs Z", fontsize=10)
    axes[1, 3].set_xlabel("X", fontsize=9)
    axes[1, 3].set_ylabel("Z", fontsize=9)
    axes[1, 3].set_aspect("equal")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pose_distribution.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze normalized pose distributions from the train split."
    )
    parser.add_argument("--split-json", default=SPLIT_JSON_DEFAULT,
                        help="Path to splits_multi/train.json")
    parser.add_argument("--multi-dir", default=MULTI_DIR_DEFAULT,
                        help="Root directory for oo3d9dmulti scenes")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Limit number of scenes to process (default: all)")
    parser.add_argument("--num-views", type=int, default=4,
                        help="Number of views used for avg_scale estimation per scene")
    parser.add_argument("--out-dir", default=str(BASELINE_DIR / "analysis_output"),
                        help="Directory where pose_distribution.png will be saved")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip saving the plot (print stats only)")
    args = parser.parse_args()

    print(f"Split JSON : {args.split_json}")
    print(f"Multi dir  : {args.multi_dir}")
    print(f"Max scenes : {args.max_scenes or 'all'}")
    print(f"Num views  : {args.num_views}")
    print()

    rotations, translations, _ = collect_normalized_poses(
        split_json=Path(args.split_json),
        multi_dir=Path(args.multi_dir),
        max_scenes=args.max_scenes,
        num_views=args.num_views,
    )

    if not rotations:
        print("No pose samples collected — check paths and data.")
        return

    print_stats(rotations, translations)

    if not args.no_plot:
        plot_distributions(rotations, translations, Path(args.out_dir))


if __name__ == "__main__":
    main()
