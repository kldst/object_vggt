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
class ObjectPoseQueryHeadConfig:
    ief_iters: int = 4
    pose_encoding_type: str = "absT_quaR"
    object_hidden_dim: int = -1
    object_trunk_depth: int = 4
    object_num_heads: int = 16
    object_mlp_ratio: int = 4
    object_init_values: float = 0.01
    trans_act: str = "linear"
    quat_act: str = "linear"


def _coerce_cfg(cfg: Optional[Union[ObjectPoseQueryHeadConfig, Dict]]) -> ObjectPoseQueryHeadConfig:
    if cfg is None:
        return ObjectPoseQueryHeadConfig()
    if isinstance(cfg, ObjectPoseQueryHeadConfig):
        return cfg
    if isinstance(cfg, Mapping):
        return ObjectPoseQueryHeadConfig(**cfg)
    raise TypeError(f"Unsupported object pose query config type: {type(cfg)!r}")


def _build_scene_context_tokens(
    tokens: torch.Tensor,
    patch_start_idx: int,
    context_pool: str,
) -> torch.Tensor:
    patch_tokens = tokens[:, :, patch_start_idx:, :]

    if context_pool == "mean":
        context_tokens = patch_tokens.mean(dim=2)
    elif context_pool == "flatten":
        batch_size, num_views, num_patches, channels = patch_tokens.shape
        context_tokens = patch_tokens.reshape(batch_size, num_views * num_patches, channels)
    else:
        raise ValueError(f"Unknown context_pool: {context_pool}")

    return context_tokens


class ObjectPoseQueryDecoderHead(nn.Module):
    """Object pose decoder that uses a learnable pose query to read fused scene tokens."""

    def __init__(self, *, context_dim: int, cfg: Optional[Union[ObjectPoseQueryHeadConfig, Dict]] = None):
        super().__init__()
        self.cfg = _coerce_cfg(cfg)
        if self.cfg.pose_encoding_type != "absT_quaR":
            raise ValueError(f"Unsupported object pose encoding type: {self.cfg.pose_encoding_type}")

        self.target_dim = 7
        hidden_dim = self.cfg.object_hidden_dim if self.cfg.object_hidden_dim > 0 else context_dim
        self.context_proj = nn.Identity() if hidden_dim == context_dim else nn.Linear(context_dim, hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
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
        self.pose_branch = Mlp(
            in_features=hidden_dim,
            hidden_features=max(hidden_dim // 2, 1),
            out_features=self.target_dim,
            drop=0,
        )

    def forward(self, context_tokens: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        batch_size = context_tokens.shape[0]
        context_tokens = self.context_norm(self.context_proj(context_tokens))

        pred_pose_enc = None
        pred_quat_list = []
        pred_translation_list = []
        for _ in range(int(self.cfg.ief_iters)):
            if pred_pose_enc is None:
                object_query = self.embed_pose(self.empty_pose_tokens.expand(batch_size, 1, -1))
            else:
                pred_pose_enc = pred_pose_enc.detach()
                object_query = self.embed_pose(pred_pose_enc)

            query_context_tokens = torch.cat([self.query_norm(object_query), context_tokens], dim=1)
            query_context_tokens = self.trunk(query_context_tokens)
            query_out = query_context_tokens[:, :1, :]

            pred_pose_delta = self.pose_branch(self.trunk_norm(query_out))
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


class ObjectPoseQueryHead(nn.Module):
    """Query-readout object pose head for fused multi-view scene tokens."""

    def __init__(
        self,
        *,
        dim_in: int,
        object_pose_cfg: Optional[Union[ObjectPoseQueryHeadConfig, Dict]] = None,
        context_pool: str = "flatten",
    ):
        super().__init__()
        self.context_pool = context_pool
        self.decoder = ObjectPoseQueryDecoderHead(context_dim=dim_in, cfg=object_pose_cfg)

    def forward(
        self,
        aggregated_tokens_list,
        patch_start_idx: int,
    ) -> Dict[str, torch.Tensor]:
        tokens = aggregated_tokens_list[-1]
        context_tokens = _build_scene_context_tokens(
            tokens=tokens,
            patch_start_idx=patch_start_idx,
            context_pool=self.context_pool,
        )
        object_pose_list, object_translation_list = self.decoder(context_tokens)
        return {
            "object_pose": object_pose_list[-1],
            "object_translation": object_translation_list[-1],
            "object_pose_list": object_pose_list,
            "object_translation_list": object_translation_list,
        }


__all__ = [
    "ObjectPoseQueryHead",
    "ObjectPoseQueryDecoderHead",
    "ObjectPoseQueryHeadConfig",
]
