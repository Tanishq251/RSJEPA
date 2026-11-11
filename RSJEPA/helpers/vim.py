"""Official Vision Mamba (Vim-Tiny) token backbone for the final RSJEPA model."""
import copy
from pathlib import Path
import sys
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .stem import PatchStem


def load_official_vim():
    path = Path(__file__).resolve().parent / 'vim_official'   # official Vim code (hustvl/Vim)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from models_mamba import VisionMamba, PatchEmbed
    return VisionMamba, PatchEmbed


def trajectory_ranks(grid):
    """(4, grid*grid) scan position of each patch under the four scan trajectories:
    row-major, column-major, row-major mirrored, column-major mirrored. Bidirectional
    Vim also scans each one reversed, so these are the four distinct dynamics."""
    p = torch.arange(grid * grid)
    r, c = p // grid, p % grid
    return torch.stack((r * grid + c, c * grid + r,
                        r * grid + (grid - 1 - c), c * grid + (grid - 1 - r)))


def _gaps(pos):
    """Scan-position gap to the previous kept token; the first token gets 1."""
    return torch.cat((torch.ones_like(pos[:, :1]), pos.diff(dim=1)), dim=1)


def _gap_branch(xz, pos, gap, L, conv, x_proj, dt_proj, A_log, D, d_state, dt_rank):
    """One scan direction of a Vim mixer on an irregularly sampled sequence.

    The width-4 causal conv runs on the full-length sequence with zeros at the
    skipped positions. The scan uses step gap*delta, and input u/gap, so that
    h_i = exp(gap*delta*A) h_{i-1} + delta*B*u_i: the state decays over the skipped
    positions as if they had zero input, and each kept input is injected once.
    """
    from causal_conv1d import causal_conv1d_fn
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    x, z = xz.chunk(2, dim=1)
    b, d, n = x.shape
    at = pos[:, None, :].expand(-1, d, -1)
    full = x.new_zeros(b, d, L).scatter(2, at, x)
    full = causal_conv1d_fn(full, conv.weight.squeeze(1), conv.bias, activation="silu")
    x = full.gather(2, at)
    x_dbl = x_proj(x.transpose(1, 2).reshape(b * n, d))
    dt, B, C = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=-1)
    dt = (dt_proj.weight @ dt.t()).reshape(d, b, n).permute(1, 0, 2)
    g = gap[:, None, :].to(torch.float32)
    delta = (torch.nn.functional.softplus(dt.float() + dt_proj.bias.float()[:, None]) * g).to(x.dtype)
    B = B.reshape(b, n, d_state).transpose(1, 2).contiguous()
    C = C.reshape(b, n, d_state).transpose(1, 2).contiguous()
    u = (x.float() / g).to(x.dtype)
    y = selective_scan_fn(u, delta, -torch.exp(A_log.float()), B, C, None, None,
                          delta_bias=None, delta_softplus=False)
    return (y + x * D.to(x.dtype)[:, None]) * torch.nn.functional.silu(z)


def gap_mixer(m, h, pos, L):
    """Vim bimamba-v2 mixer with gap-aware discretisation (unfused path)."""
    b, n, _ = h.shape
    xz = (h @ m.in_proj.weight.t()).transpose(1, 2)
    if m.in_proj.bias is not None:
        xz = xz + m.in_proj.bias.to(xz.dtype)[:, None]
    kw = dict(L=L, d_state=m.d_state, dt_rank=m.dt_rank)
    y_f = _gap_branch(xz, pos, _gaps(pos), conv=m.conv1d, x_proj=m.x_proj, dt_proj=m.dt_proj,
                      A_log=m.A_log, D=m.D, **kw)
    pos_b = (L - 1) - pos.flip(1)
    y_b = _gap_branch(xz.flip(-1), pos_b, _gaps(pos_b), conv=m.conv1d_b, x_proj=m.x_proj_b,
                      dt_proj=m.dt_proj_b, A_log=m.A_b_log, D=m.D_b, **kw)
    y = y_f + y_b.flip(-1)
    if m.if_divide_out:
        y = y / 2
    out = torch.nn.functional.linear(y.transpose(1, 2), m.out_proj.weight, m.out_proj.bias)
    return out * m.gamma if m.init_layer_scale is not None else out


def _gap_block(block, x, residual, pos, L):
    """models_mamba.Block.forward (fused_add_norm=False) with the gap-aware mixer."""
    residual = x if residual is None else residual + block.drop_path(x)
    x = block.norm(residual.to(dtype=block.norm.weight.dtype))
    if block.residual_in_fp32:
        residual = residual.to(torch.float32)
    return gap_mixer(block.mixer, x, pos, L), residual


class VimTrunk(nn.Module):
    """Bidirectional Mamba blocks with spatially ordered masked sequences.

    idx_*: patch indices of the given tokens. traj_*: (B,) trajectory ids (see
    trajectory_ranks); None = row-major. With gap_delta, a masked sequence is scanned
    as an irregular sample of the full-image trajectory instead of a contiguous one.
    """
    def __init__(self, vim, grad_ckpt, gap_delta=False):
        super().__init__()
        self.blocks, self.norm = vim.layers, vim.norm_f
        self.drop_path = vim.drop_path
        self.grad_ckpt = grad_ckpt
        self.gap_delta = gap_delta
        self.num_patches = vim.patch_embed.num_patches
        grid = int(round(self.num_patches ** 0.5))
        self.register_buffer("traj_rank", trajectory_ranks(grid), persistent=False)

    def scan_positions(self, tokens, idx, traj):
        if idx is None and traj is None:
            return None
        if idx is None:
            idx = torch.arange(tokens.size(1), device=tokens.device).expand(tokens.size(0), -1)
        if traj is None:
            return idx
        return torch.gather(self.traj_rank[traj], 1, idx)

    @staticmethod
    def order(tokens, positions):
        if positions is None:
            return tokens, None, None
        order = positions.argsort(dim=1)
        sorted_tokens = torch.gather(tokens, 1, order[..., None].expand_as(tokens))
        return sorted_tokens, order.argsort(dim=1), torch.gather(positions, 1, order)

    def forward(self, ta, tb=None, cross=False, idx_a=None, idx_b=None, traj_a=None, traj_b=None):
        if cross:
            raise ValueError('Vision Mamba requires the final --no-inter recipe')
        ta, undo_a, pos_a = self.order(ta, self.scan_positions(ta, idx_a, traj_a))
        undo_b = pos_b = None
        if tb is not None:
            tb, undo_b, pos_b = self.order(tb, self.scan_positions(tb, idx_b, traj_b))
        x = ta if tb is None else torch.cat((ta, tb), dim=0)
        pos = None
        if self.gap_delta and (pos_a is not None or pos_b is not None):
            ar = lambda t: torch.arange(t.size(1), device=t.device).expand(t.size(0), -1)
            pos = pos_a if pos_a is not None else ar(ta)
            if tb is not None:
                pos = torch.cat((pos, pos_b if pos_b is not None else ar(tb)), dim=0)
        residual = None
        for block in self.blocks:
            ckpt = self.grad_ckpt and self.training and torch.is_grad_enabled()
            if pos is not None:
                args = (block, x, residual, pos, self.num_patches)
                x, residual = checkpoint(_gap_block, *args, use_reentrant=False) if ckpt else _gap_block(*args)
            elif ckpt:
                x, residual = checkpoint(block, x, residual, use_reentrant=False)
            else:
                x, residual = block(x, residual)
        residual = x if residual is None else residual + self.drop_path(x)
        x = self.norm(residual.to(dtype=self.norm.weight.dtype))
        sa, sb = (x, None) if tb is None else x.split(ta.size(0), dim=0)
        if undo_a is not None:
            sa = torch.gather(sa, 1, undo_a[..., None].expand_as(sa))
        if tb is not None and undo_b is not None:
            sb = torch.gather(sb, 1, undo_b[..., None].expand_as(sb))
        return sa, sb, None, None


def build_vim(cfg):
    VisionMamba, PatchEmbed = load_official_vim()
    vim = VisionMamba(
        img_size=cfg.img_size, patch_size=cfg.patch_size, stride=cfg.patch_size,
        channels=cfg.s1_chans, embed_dim=cfg.dim, depth=cfg.depth, num_classes=0,
        rms_norm=True, residual_in_fp32=True, fused_add_norm=False,
        drop_path_rate=0, final_pool_type='mean', if_abs_pos_embed=True,
        if_rope=False, bimamba_type='v2', if_cls_token=False, if_divide_out=True,
    )
    stem_a = PatchStem(vim.patch_embed, vim.pos_embed)
    patch_b = PatchEmbed(img_size=cfg.img_size, patch_size=cfg.patch_size,
                         stride=cfg.patch_size, in_chans=cfg.s2_chans, embed_dim=cfg.dim)
    stem_b = PatchStem(patch_b, copy.deepcopy(vim.pos_embed))
    return stem_a, stem_b, VimTrunk(vim, cfg.grad_ckpt, getattr(cfg, 'gap_delta', False))
