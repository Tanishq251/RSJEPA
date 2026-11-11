from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def sincos_pos_embed(dim: int, grid: int) -> torch.Tensor:
    """2D sine-cosine position embedding, (grid*grid, dim)."""
    def _1d(d, pos):
        omega = torch.arange(d // 2, dtype=torch.float64) / (d / 2.0)
        omega = 1.0 / 10000 ** omega
        out = pos.reshape(-1)[:, None] * omega[None, :]
        return torch.cat([out.sin(), out.cos()], dim=1)

    g = torch.arange(grid, dtype=torch.float64)
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    emb = torch.cat([_1d(dim // 2, gx), _1d(dim // 2, gy)], dim=1)
    return emb.float()


class Mlp(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    """Standard pre-norm transformer block."""
    def __init__(self, dim, heads, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.dp = drop_path

    def _drop(self, x):
        if self.dp == 0.0 or not self.training:
            return x
        keep = 1 - self.dp
        m = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep) / keep
        return x * m

    def forward(self, x):
        h = self.norm1(x)
        x = x + self._drop(self.attn(h, h, h, need_weights=False)[0])
        return x + self._drop(self.mlp(self.norm2(x)))


class Stem(nn.Module):
    """Patch embedding + modality-specific positional embeddings."""
    def __init__(self, in_chans, dim, img_size, patch_size):
        super().__init__()
        grid = img_size // patch_size
        self.num_patches = grid ** 2
        self.proj = nn.Conv2d(in_chans, dim, patch_size, patch_size)
        self.pos = nn.Parameter(sincos_pos_embed(dim, grid)[None].clone())

    def forward(self, x, keep_idx=None):
        t = self.proj(x).flatten(2).transpose(1, 2) + self.pos
        if keep_idx is not None:
            t = torch.gather(t, 1, keep_idx[..., None].expand(-1, -1, t.size(-1)))
        return t


def _pred_block(L, x, ctx):
    x = x + L["sa"](L["n1"](x), L["n1"](x), L["n1"](x), need_weights=False)[0]
    kv = L["nkv"](ctx)
    x = x + L["ca"](L["nq"](x), kv, kv, need_weights=False)[0]
    return x + L["mlp"](L["n3"](x))


class QueryPredictor(nn.Module):
    """Learnable mask queries + target-position embeddings, then self-attention,
    cross-attention to visible context, and MLP."""
    def __init__(self, dim, depth, heads, num_patches, grad_ckpt=False):
        super().__init__()
        self.grad_ckpt = grad_ckpt
        self.mask_query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_query, std=0.02)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict(dict(
                n1=nn.LayerNorm(dim), sa=nn.MultiheadAttention(dim, heads, batch_first=True),
                nq=nn.LayerNorm(dim), nkv=nn.LayerNorm(dim),
                ca=nn.MultiheadAttention(dim, heads, batch_first=True),
                n3=nn.LayerNorm(dim), mlp=Mlp(dim, dim * 4))))
        self.norm = nn.LayerNorm(dim)

    def forward(self, ctx, tgt_idx, tgt_pos):
        B = ctx.size(0)
        p = torch.gather(tgt_pos.expand(B, -1, -1), 1, tgt_idx[..., None].expand(-1, -1, ctx.size(-1)))
        x = self.mask_query.expand(B, tgt_idx.size(1), -1) + p
        for L in self.layers:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                x = checkpoint(_pred_block, L, x, ctx, use_reentrant=False)
            else:
                x = _pred_block(L, x, ctx)
        return self.norm(x)


class RetrievalHead(nn.Module):
    """Raw projection r for SIGReg, l2-normalised embedding e for retrieval."""
    def __init__(self, dim, out, in_norm=False):
        super().__init__()
        layers = [nn.LayerNorm(dim)] if in_norm else []
        self.net = nn.Sequential(*layers, nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, out))

    def forward(self, x):
        r = self.net(x)
        return r, F.normalize(r, dim=-1)
