"""Patch embedding + position stem shared by the Vim backbone (one stem per sensor)."""
from __future__ import annotations

import torch
from torch import nn


class PatchStem(nn.Module):
    """Modality patch embedding and positions, gathered before the trunk."""
    def __init__(self, patch_embed, pos):
        super().__init__()
        self.patch_embed = patch_embed
        self.pos = pos
        self.num_patches = patch_embed.num_patches

    def forward(self, x, keep_idx=None):
        tokens = self.patch_embed(x) + self.pos
        if keep_idx is not None:
            tokens = torch.gather(tokens, 1, keep_idx[..., None].expand(-1, -1, tokens.size(-1)))
        return tokens
