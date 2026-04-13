from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from vggt.heads.head_act import activate_pose
from vggt.layers import Mlp
from vggt.layers.block import Block


@dataclass(frozen=True)
class ObjectPoseCameraHeadConfig:
    ief_iters: int = 4
    pose_encoding_type: str = "absT_quaR"
    object_hidden_dim: int = -1
    object_trunk_depth: int = 4
    object_num_heads: int = 16
    object_mlp_ratio: int = 4
    object_init_values: float = 0.01
    trans_act: str = "linear"
    quat_act: str = "linear"


def _coerce_cfg(cfg: Optional[Union[ObjectPoseCameraHeadConfig, Dict]]) -> ObjectPoseCameraHeadConfig:
    if cfg is None:
        return ObjectPoseCameraHeadConfig()
    if isinstance(cfg, ObjectPoseCameraHeadConfig):
        return cfg
    if isinstance(cfg, Mapping):
        return ObjectPoseCameraHeadConfig(**cfg)
    raise TypeError(f"Unsupported object camera pose config type: {type(cfg)!r}")


def _build_object_context_tokens(
    tokens: torch.Tensor,
    patch_start_idx: int,
    context_pool: str,
    object_latent: Optional[torch.Tensor],
    object_tokens: Optional[torch.Tensor],
    use_global_scene_object_concat: bool,
) -> torch.Tensor:
    patch_tokens = tokens[:, :, patch_start_idx:, :]

    if use_global_scene_object_concat:
        if object_tokens is None:
            raise ValueError("object_tokens must be provided when use_global_scene_object_concat=True")
        scene_global = patch_tokens.mean(dim=(1, 2))
        object_global = object_tokens.mean(dim=(1, 2))
        return torch.cat([scene_global, object_global], dim=-1).unsqueeze(1)

    if context_pool == "mean":
        context_tokens = patch_tokens.mean(dim=2)
    elif context_pool == "flatten":
        batch_size, num_views, num_patches, channels = patch_tokens.shape
        context_tokens = patch_tokens.reshape(batch_size, num_views * num_patches, channels)
    else:
        raise ValueError(f"Unknown context_pool: {context_pool}")

    if object_latent is not None:
        if object_latent.dim() != 3:
            raise ValueError(f"object_latent should be (B,S,C), got {tuple(object_latent.shape)}")
        context_tokens = torch.cat([object_latent, context_tokens], dim=1)

    return context_tokens


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class ObjectPoseCameraDecoderHead(nn.Module):
    """Camera-style iterative object pose decoder with a learnable 9D pose token."""

    def __init__(self, *, context_dim: int, cfg: Optional[Union[ObjectPoseCameraHeadConfig, Dict]] = None):
        super().__init__()
        self.cfg = _coerce_cfg(cfg)
        if self.cfg.pose_encoding_type != "absT_quaR":
            raise ValueError(f"Unsupported object pose encoding type: {self.cfg.pose_encoding_type}")

        self.target_dim = 7  # 3D translation + 4D quaternion rotation
        hidden_dim = self.cfg.object_hidden_dim if self.cfg.object_hidden_dim > 0 else context_dim
        self.context_proj = nn.Identity() if hidden_dim == context_dim else nn.Linear(context_dim, hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.trunk_norm = nn.LayerNorm(hidden_dim)
        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=hidden_dim,
                    num_heads=self.cfg.object_num_heads,
                    mlp_ratio=self.cfg.object_mlp_ratio,
                    init_values=self.cfg.object_init_values,
                )
                for _ in range(self.cfg.object_trunk_depth)
            ]
        )

        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, hidden_dim)
        self.pose_ln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 3 * hidden_dim, bias=True))
        self.adaln_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(
            in_features=hidden_dim,
            hidden_features=max(hidden_dim // 2, 1),
            out_features=self.target_dim,
            drop=0,
        )

    def forward(self, context_tokens: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        batch_size = context_tokens.shape[0]
        pooled_context = self.context_proj(context_tokens).mean(dim=1, keepdim=True)
        pooled_context = self.context_norm(pooled_context)

        pred_pose_enc = None
        pred_quat_list = []
        pred_translation_list = []
        for _ in range(int(self.cfg.ief_iters)):
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(batch_size, 1, -1))
            else:
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            shift_msa, scale_msa, gate_msa = self.pose_ln_modulation(module_input).chunk(3, dim=-1)
            pose_tokens = gate_msa * _modulate(self.adaln_norm(pooled_context), shift_msa, scale_msa)
            pose_tokens = pose_tokens + pooled_context
            pose_tokens = self.trunk(pose_tokens)

            pred_pose_delta = self.pose_branch(self.trunk_norm(pose_tokens))
            pred_pose_enc = pred_pose_delta if pred_pose_enc is None else pred_pose_enc + pred_pose_delta

            activated_pose = activate_pose(
                pred_pose_enc,
                trans_act=self.cfg.trans_act,
                quat_act=self.cfg.quat_act,
                fl_act="linear",
            ).squeeze(1)
            pred_translation_list.append(activated_pose[:, :3])
            pred_quat_list.append(activated_pose[:, 3:])
        return pred_quat_list, pred_translation_list


class ObjectPoseCameraHead(nn.Module):
    """Object head using camera-style iterative refinement with a learnable token."""

    def __init__(
        self,
        *,
        dim_in: int,
        object_pose_cfg: Optional[Union[ObjectPoseCameraHeadConfig, Dict]] = None,
        context_pool: str = "flatten",
        use_global_scene_object_concat: bool = False,
    ):
        super().__init__()
        self.context_pool = context_pool
        self.use_global_scene_object_concat = bool(use_global_scene_object_concat)
        decoder_context_dim = 2 * dim_in if self.use_global_scene_object_concat else dim_in
        self.decoder = ObjectPoseCameraDecoderHead(context_dim=decoder_context_dim, cfg=object_pose_cfg)

    def forward(
        self,
        aggregated_tokens_list,
        patch_start_idx: int,
        object_latent: Optional[torch.Tensor] = None,
        object_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        tokens = aggregated_tokens_list[-1]
        context_tokens = _build_object_context_tokens(
            tokens=tokens,
            patch_start_idx=patch_start_idx,
            context_pool=self.context_pool,
            object_latent=object_latent,
            object_tokens=object_tokens,
            use_global_scene_object_concat=self.use_global_scene_object_concat,
        )
        object_pose_list, object_translation_list = self.decoder(context_tokens)
        return {
            "object_pose": object_pose_list[-1],
            "object_translation": object_translation_list[-1],
            "object_pose_list": object_pose_list,
            "object_translation_list": object_translation_list,
        }


__all__ = [
    "ObjectPoseCameraHead",
    "ObjectPoseCameraDecoderHead",
    "ObjectPoseCameraHeadConfig",
]
