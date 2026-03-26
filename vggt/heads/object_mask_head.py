from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectMaskHead(nn.Module):
    """A lightweight scene-view mask decoder for object-conditioned grounding."""

    def __init__(
        self,
        *,
        dim_in: int,
        patch_size: int = 14,
        hidden_dim: int = 256,
        use_object_latent: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.norm = nn.LayerNorm(dim_in)
        self.use_object_latent = bool(use_object_latent)
        self.object_latent_proj = nn.Linear(dim_in, dim_in) if self.use_object_latent else None
        self.decoder = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        object_latent: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        tokens = aggregated_tokens_list[-1][:, :, patch_start_idx:, :]
        bsz, seq_len, num_patches, dim = tokens.shape
        img_h, img_w = images.shape[-2:]
        patch_h, patch_w = img_h // self.patch_size, img_w // self.patch_size
        if patch_h * patch_w != num_patches:
            raise ValueError(
                f"Patch grid mismatch: patch_h={patch_h}, patch_w={patch_w}, num_patches={num_patches}"
            )

        if object_latent is not None and self.object_latent_proj is not None:
            if object_latent.shape[:2] != (bsz, seq_len):
                raise ValueError(
                    f"object_latent should be (B,S,C), got {tuple(object_latent.shape)} for {(bsz, seq_len)}"
                )
            tokens = tokens + self.object_latent_proj(object_latent).unsqueeze(2)

        x = tokens.reshape(bsz * seq_len, num_patches, dim)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(bsz * seq_len, dim, patch_h, patch_w)
        logits = self.decoder(x)
        logits = F.interpolate(logits, size=(img_h, img_w), mode="bilinear", align_corners=False)
        logits = logits.reshape(bsz, seq_len, img_h, img_w)

        return {
            "object_mask_logits": logits,
            "object_mask": torch.sigmoid(logits),
        }


__all__ = ["ObjectMaskHead"]
