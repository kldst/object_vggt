# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.obj_dpt_head import OBJ_DPTHead
from vggt.heads.object_mask_head import ObjectMaskHead
from vggt.heads.track_head import TrackHead
from vggt.heads.object_pose_camera_head import ObjectPoseCameraHead
from vggt.heads.object_pose_head import ObjectPoseHead

class ObjectTokenCrossAttentionBlock(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True,
        )
        hidden_dim = int(query_dim * mlp_ratio)
        self.mlp_norm = nn.LayerNorm(query_dim)
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, query_dim),
        )

    def forward(self, query_tokens: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        context_tokens = self.context_norm(context_tokens)
        attn_out, _ = self.cross_attn(
            self.query_norm(query_tokens),
            context_tokens,
            context_tokens,
            need_weights=False,
        )
        query_tokens = query_tokens + attn_out
        query_tokens = query_tokens + self.mlp(self.mlp_norm(query_tokens))
        return query_tokens


class ObjectPrototypePool(nn.Module):
    def __init__(self, dim: int, num_prototypes: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.prototype_queries = nn.Parameter(torch.randn(1, num_prototypes, dim))
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.normal_(self.prototype_queries, std=1e-6)

    def forward(self, object_patch_tokens: torch.Tensor) -> torch.Tensor:
        B, S_obj, P_obj, C = object_patch_tokens.shape
        context_tokens = self.context_norm(object_patch_tokens.reshape(B, S_obj * P_obj, C))
        prototype_queries = self.prototype_queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(
            self.query_norm(prototype_queries),
            context_tokens,
            context_tokens,
            need_weights=False,
        )
        prototype_queries = prototype_queries + attn_out
        prototype_queries = prototype_queries + self.mlp(self.mlp_norm(prototype_queries))
        return prototype_queries


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True,
                 enable_object_point=True, enable_object_mask=False, enable_object_srt=True, use_shared_object_latent=False,
                 enable_object_cross_attn=True, object_cross_attn_heads=16,
                 enable_pre_aggregator_object_cross_attn=False,
                 enable_multi_layer_object_prototype_cross_attn=False,
                 object_prototype_layer_indices=(4, 11, 17, 23),
                 object_prototype_num_tokens=4,
                 object_prototype_object_encoder_no_grad=False,
                 enable_global_pool_scene_object_pose_head=False,
                 enable_camera_style_object_pose_head=False,
                 object_pose_cfg=None):
        super().__init__()

        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        self.use_shared_object_latent = bool(use_shared_object_latent)
        self.shared_object_latent = None
        if self.use_shared_object_latent:
            # Shared latent vector used by both object heads (head-side latent, not an aggregator token).
            self.shared_object_latent = nn.Parameter(torch.zeros(1, 1, 2 * embed_dim))
            nn.init.normal_(self.shared_object_latent, std=1e-6)

        self.enable_object_cross_attn = bool(enable_object_cross_attn)
        self.enable_pre_aggregator_object_cross_attn = bool(enable_pre_aggregator_object_cross_attn)
        self.enable_multi_layer_object_prototype_cross_attn = bool(enable_multi_layer_object_prototype_cross_attn)
        self.enable_global_pool_scene_object_pose_head = bool(enable_global_pool_scene_object_pose_head)
        self.enable_camera_style_object_pose_head = bool(enable_camera_style_object_pose_head)
        self.object_pose_cfg = object_pose_cfg
        self.object_prototype_layer_indices = tuple(int(idx) for idx in object_prototype_layer_indices)
        self.object_prototype_num_tokens = int(object_prototype_num_tokens)
        self.object_prototype_object_encoder_no_grad = bool(object_prototype_object_encoder_no_grad)
        self.pre_object_token_cross_attn = (
            ObjectTokenCrossAttentionBlock(
                query_dim=embed_dim,
                context_dim=embed_dim,
                num_heads=object_cross_attn_heads,
                mlp_ratio=4.0,
            )
            if self.enable_pre_aggregator_object_cross_attn
            else None
        )
        self.object_token_cross_attn = (
            ObjectTokenCrossAttentionBlock(
                query_dim=2 * embed_dim,
                context_dim=2 * embed_dim,
                num_heads=object_cross_attn_heads,
                mlp_ratio=4.0,
            )
            if self.enable_object_cross_attn and not self.enable_pre_aggregator_object_cross_attn
            else None
        )
        self.object_token_cross_attn_blocks = None
        self.object_prototype_poolers = None
        if self.enable_multi_layer_object_prototype_cross_attn and not self.enable_pre_aggregator_object_cross_attn:
            progressive_layer_indices = self._resolve_object_prototype_layer_indices(self.aggregator.depth)
            self.object_token_cross_attn_blocks = nn.ModuleDict(
                {
                    str(layer_idx): ObjectTokenCrossAttentionBlock(
                        query_dim=embed_dim,
                        context_dim=embed_dim,
                        num_heads=object_cross_attn_heads,
                        mlp_ratio=4.0,
                    )
                    for layer_idx in progressive_layer_indices
                }
            )
            self.object_prototype_poolers = nn.ModuleDict(
                {
                    str(layer_idx): ObjectPrototypePool(
                        dim=embed_dim,
                        num_prototypes=self.object_prototype_num_tokens,
                        num_heads=object_cross_attn_heads,
                        mlp_ratio=4.0,
                    )
                    for layer_idx in progressive_layer_indices
                }
            )

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
        self.object_pt_head = (
            OBJ_DPTHead(
                dim_in=2 * embed_dim,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
                use_object_latent=self.use_shared_object_latent,
            )
            if enable_object_point
            else None
        )
        self.object_mask_head = (
            ObjectMaskHead(
                dim_in=2 * embed_dim,
                patch_size=patch_size,
                use_object_latent=self.use_shared_object_latent,
            )
            if enable_object_mask
            else None
        )
        self.object_srt_head = (
            (
                ObjectPoseCameraHead(
                    dim_in=2 * embed_dim,
                    object_pose_cfg=self.object_pose_cfg,
                    use_global_scene_object_concat=self.enable_global_pool_scene_object_pose_head,
                )
                if self.enable_camera_style_object_pose_head
                else ObjectPoseHead(
                    dim_in=2 * embed_dim,
                    object_pose_cfg=self.object_pose_cfg,
                    use_global_scene_object_concat=self.enable_global_pool_scene_object_pose_head,
                )
            )
            if enable_object_srt
            else None
        )

    def _ensure_batched_images(self, images: torch.Tensor):
        if images is None:
            return None
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        return images

    def _encode_object_tokens(self, object_images: torch.Tensor):
        object_aggregated_tokens_list, object_patch_start_idx = self.aggregator(object_images)
        object_tokens = object_aggregated_tokens_list[-1][:, :, object_patch_start_idx:, :]
        return object_tokens, object_aggregated_tokens_list, object_patch_start_idx

    def _encode_object_prototypes(self, object_images: torch.Tensor):
        selected_layers = self._resolve_object_prototype_layer_indices(self.aggregator.depth)
        requested_layers = tuple(dict.fromkeys((*selected_layers, self.aggregator.depth - 1)))
        if self.object_prototype_object_encoder_no_grad:
            with torch.no_grad():
                _, object_patch_start_idx, object_layer_tokens = self.aggregator(
                    object_images,
                    return_layer_tokens=True,
                    layer_token_indices=requested_layers,
                    collect_output_list=False,
                )
        else:
            _, object_patch_start_idx, object_layer_tokens = self.aggregator(
                object_images,
                return_layer_tokens=True,
                layer_token_indices=requested_layers,
                collect_output_list=False,
            )

        prototypes_by_idx = {
            layer_idx: self._build_object_prototypes(object_layer_tokens[layer_idx], object_patch_start_idx, layer_idx)
            for layer_idx in selected_layers
        }
        final_object_tokens = object_layer_tokens[self.aggregator.depth - 1][:, :, object_patch_start_idx:, :]
        del object_layer_tokens
        return prototypes_by_idx, final_object_tokens

    def _embed_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        B, S, _, _, _ = images.shape
        patch_tokens = self.aggregator.embed_images(images)
        return patch_tokens.view(B, S, patch_tokens.shape[1], patch_tokens.shape[2])

    def _apply_pre_aggregator_object_cross_attention(self, scene_images: torch.Tensor, object_images: torch.Tensor):
        if self.pre_object_token_cross_attn is None or object_images is None:
            return None, None

        scene_patch_tokens = self._embed_patch_tokens(scene_images)
        object_patch_tokens = self._embed_patch_tokens(object_images)

        B_scene, S_scene, P_scene, C_scene = scene_patch_tokens.shape
        B_obj, S_obj, P_obj, C_obj = object_patch_tokens.shape
        if B_scene != B_obj:
            raise ValueError(f"Scene/object batch size mismatch: {B_scene} vs {B_obj}")

        scene_query = scene_patch_tokens.reshape(B_scene, S_scene * P_scene, C_scene)
        object_context = object_patch_tokens.reshape(B_obj, S_obj * P_obj, C_obj)
        fused_scene_patch_tokens = self.pre_object_token_cross_attn(scene_query, object_context)
        fused_scene_patch_tokens = fused_scene_patch_tokens.view(B_scene, S_scene, P_scene, C_scene)
        return fused_scene_patch_tokens, object_patch_tokens

    def _apply_object_cross_attention(self, aggregated_tokens_list, object_patch_tokens, patch_start_idx):
        if self.object_token_cross_attn is None or object_patch_tokens is None:
            return aggregated_tokens_list

        B, S_obj, P_obj, C_obj = object_patch_tokens.shape
        object_context = object_patch_tokens.reshape(B, S_obj * P_obj, C_obj)
        fused_tokens_list = list(aggregated_tokens_list)
        tokens = fused_tokens_list[-1]                        # 只取最後 aggregator output 做 cross attentation
        special_tokens = tokens[:, :, :patch_start_idx, :]
        scene_patch_tokens = tokens[:, :, patch_start_idx:, :]
        if scene_patch_tokens.numel() == 0:
            return fused_tokens_list

        B_scene, S_scene, P_scene, C_scene = scene_patch_tokens.shape
        scene_query = scene_patch_tokens.reshape(B_scene, S_scene * P_scene, C_scene)
        fused_scene_patch_tokens = self.object_token_cross_attn(scene_query, object_context)
        fused_scene_patch_tokens = fused_scene_patch_tokens.view(B_scene, S_scene, P_scene, C_scene)
        fused_tokens_list[-1] = torch.cat([special_tokens, fused_scene_patch_tokens], dim=2)
        return fused_tokens_list

    def _resolve_object_prototype_layer_indices(self, num_layers: int):
        resolved_indices = []
        for layer_idx in self.object_prototype_layer_indices:
            resolved_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx
            if resolved_idx < 0 or resolved_idx >= num_layers:
                raise ValueError(
                    f"object_prototype_layer_indices contains invalid layer index {layer_idx} for {num_layers} layers"
                )
            resolved_indices.append(resolved_idx)
        return tuple(dict.fromkeys(resolved_indices))

    def _build_object_prototypes(self, object_layer_tokens: torch.Tensor, object_patch_start_idx: int, layer_idx: int):
        if self.object_prototype_poolers is None:
            raise RuntimeError("object_prototype_poolers is not initialized")
        object_patch_tokens = object_layer_tokens[:, :, object_patch_start_idx:, :]
        if object_patch_tokens.numel() == 0:
            raise ValueError("Object patch tokens are empty; cannot build object prototypes")
        return self.object_prototype_poolers[str(layer_idx)](object_patch_tokens)

    def _apply_progressive_object_prototype_cross_attention(
        self,
        layer_idx: int,
        scene_layer_tokens: torch.Tensor,
        scene_patch_start_idx: int,
        object_prototypes_by_idx,
    ):
        if self.object_token_cross_attn_blocks is None or layer_idx not in object_prototypes_by_idx:
            return scene_layer_tokens

        scene_special_tokens = scene_layer_tokens[:, :, :scene_patch_start_idx, :]
        scene_patch_tokens = scene_layer_tokens[:, :, scene_patch_start_idx:, :]
        if scene_patch_tokens.numel() == 0:
            return scene_layer_tokens

        object_prototypes = object_prototypes_by_idx[layer_idx]
        if scene_layer_tokens.shape[0] != object_prototypes.shape[0]:
            raise ValueError(
                f"Scene/object batch size mismatch at layer {layer_idx}: "
                f"{scene_layer_tokens.shape[0]} vs {object_prototypes.shape[0]}"
            )
        if scene_layer_tokens.shape[-1] != object_prototypes.shape[-1]:
            raise ValueError(
                f"Scene/object channel mismatch at layer {layer_idx}: "
                f"{scene_layer_tokens.shape[-1]} vs {object_prototypes.shape[-1]}"
            )

        B_scene, S_scene, P_scene, C_scene = scene_patch_tokens.shape
        scene_query = scene_patch_tokens.reshape(B_scene, S_scene * P_scene, C_scene)
        fused_scene_query = self.object_token_cross_attn_blocks[str(layer_idx)](scene_query, object_prototypes)
        fused_scene_patch_tokens = fused_scene_query.view(B_scene, S_scene, P_scene, C_scene)
        return torch.cat([scene_special_tokens, fused_scene_patch_tokens], dim=2)

    def forward(self, scene_images: torch.Tensor, object_images: torch.Tensor = None, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            scene_images (torch.Tensor): Scene images with shape [S, 3, H, W] or [B, S, 3, H, W].
            object_images (torch.Tensor, optional): Object crop images with shape [S_obj, 3, H, W]
                or [B, S_obj, 3, H, W]. They are encoded by the full aggregator, and the
                object tokens can then condition scene patch tokens either with final-layer
                cross-attention or with multi-layer prototype cross-attention, depending on flags.
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - object_points (torch.Tensor): Object-space 3D point map with shape [B, S, H, W, 3]
                - object_points_conf (torch.Tensor): Confidence scores for object point map with shape [B, S, H, W]
                - object_pose (torch.Tensor): Object rotation in 6D representation with shape [B, 6]
                - object_scale (torch.Tensor): Object scale with shape [B, 1]
                - object_translation (torch.Tensor): Object translation with shape [B, 3]
                - images (torch.Tensor): Original scene images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """
        scene_images = self._ensure_batched_images(scene_images)
        object_images = self._ensure_batched_images(object_images)

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        object_patch_tokens = None
        if self.enable_global_pool_scene_object_pose_head:
            aggregated_tokens_list, patch_start_idx = self.aggregator(scene_images)
            if object_images is None:
                raise ValueError("object_images must be provided when enable_global_pool_scene_object_pose_head=True")
            object_patch_tokens, _, _ = self._encode_object_tokens(object_images)
        elif self.enable_pre_aggregator_object_cross_attn and object_images is not None:
            fused_scene_patch_tokens, object_patch_tokens = self._apply_pre_aggregator_object_cross_attention(
                scene_images,
                object_images,
            )
            B, S, _, H, W = scene_images.shape
            fused_scene_patch_tokens = fused_scene_patch_tokens.reshape(
                B * S,
                fused_scene_patch_tokens.shape[2],
                fused_scene_patch_tokens.shape[3],
            )
            aggregated_tokens_list, patch_start_idx = self.aggregator.forward_from_patch_tokens(
                fused_scene_patch_tokens,
                batch_size=B,
                seq_len=S,
                height=H,
                width=W,
            )
        elif self.enable_multi_layer_object_prototype_cross_attn and object_images is not None:
            object_prototypes_by_idx, object_patch_tokens = self._encode_object_prototypes(object_images)

            def progressive_object_fusion(layer_idx, scene_layer_tokens, scene_patch_start_idx):
                return self._apply_progressive_object_prototype_cross_attention(
                    layer_idx,
                    scene_layer_tokens,
                    scene_patch_start_idx,
                    object_prototypes_by_idx,
                )

            aggregated_tokens_list, patch_start_idx = self.aggregator(
                scene_images,
                layer_postprocessor=progressive_object_fusion,
            )
        else:
            aggregated_tokens_list, patch_start_idx = self.aggregator(scene_images)

        if (
            not self.enable_global_pool_scene_object_pose_head
            and
            object_images is not None
            and not self.enable_pre_aggregator_object_cross_attn
            and not self.enable_multi_layer_object_prototype_cross_attn
        ):
            object_patch_tokens, object_aggregated_tokens_list, object_patch_start_idx = self._encode_object_tokens(
                object_images
            )
            aggregated_tokens_list = self._apply_object_cross_attention(
                aggregated_tokens_list,
                object_patch_tokens,
                patch_start_idx,
            )

        object_latent = None
        if self.shared_object_latent is not None:
            B, S = scene_images.shape[:2]
            object_latent = self.shared_object_latent.expand(B, S, -1)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=scene_images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=scene_images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.object_pt_head is not None:
                object_pts3d, object_pts3d_conf = self.object_pt_head(
                    aggregated_tokens_list,
                    images=scene_images,
                    patch_start_idx=patch_start_idx,
                    object_latent=object_latent,
                )
                predictions["object_points"] = object_pts3d
                predictions["object_points_conf"] = object_pts3d_conf

            if self.object_mask_head is not None:
                predictions.update(
                    self.object_mask_head(
                        aggregated_tokens_list,
                        images=scene_images,
                        patch_start_idx=patch_start_idx,
                        object_latent=object_latent,
                    )
                )

            if self.object_srt_head is not None:
                predictions.update(
                    self.object_srt_head(
                        aggregated_tokens_list,
                        patch_start_idx=patch_start_idx,
                        object_latent=object_latent,
                        object_tokens=object_patch_tokens,
                    )
                )

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=scene_images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = scene_images
            if object_patch_tokens is not None:
                predictions["object_patch_tokens"] = object_patch_tokens

        return predictions
