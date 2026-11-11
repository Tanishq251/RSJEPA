"""RS-JEPA: single-view base, two-view (dual), and BiDA (coupled) architectures.

RSJEPABiDA + I1 (adversarial view router) is the final, best-performing
configuration (71.97% mean F1@5 on BEN-14K) -- this is the model referred to
as "RSJEPA" in results and docs. RSJEPA (base) and RSJEPADual exist to
reproduce the ablation series showing why the BiDA coupling matters
(docs/results/ABLATIONS.md section 1).
"""
from __future__ import annotations
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .helpers.layers import Stem, QueryPredictor, RetrievalHead, Mlp, sincos_pos_embed


# ============================================================================
# Base: single view (RSJEPA) + shared-trunk two-view variant (RSJEPAShared)
# ============================================================================

@dataclass
class RSJEPAConfig:
    img_size: int = 224
    patch_size: int = 16
    s1_chans: int = 2
    s2_chans: int = 12
    dim: int = 512
    depth: int = 12
    backbone: str = "vit"           # train.py always passes vim_tiny (the final model)
    inter_cross: bool = True    # false disables cross-sensor trunk prediction routes
    heads: int = 8
    pred_depth: int = 6
    pred_heads: int = 8
    retr_dim: int = 256
    mask_ratio: float = 0.5
    grad_ckpt: bool = True
    fold_cross_predictor: bool = True
    temperature: float = 0.07
    lam_aa: float = 1.0
    lam_bb: float = 1.0
    lam_ab: float = 1.0
    lam_ba: float = 1.0
    lam_cross: float = 1.0
    lam_uni: float = 1.0
    lam_sigreg: float = 1.0
    pred_loss: str = "mse"          # base-model masked-feature loss: mse | smooth_l1 | cosine
    uni_mode: str = "decoupled"
    ema_target: bool = False
    ema: tuple = (0.996, 1.0)
    gap_delta: bool = False         # Vim only: scan masked sequences as irregular samples (step x gap)
    # N3 sensor-physics-consistent JEPA: each sensor's *context* is corrupted by its own
    # imaging physics; the masked targets are encoded from the clean image.
    sensor_noise: bool = False
    sensor_kinds: tuple = ()        # per sensor: "sar_db" | "sar_amplitude" | "optical"
    sensor_mean: tuple = ()         # per sensor, per channel z-score stats (to work in physical units)
    sensor_std: tuple = ()
    speckle_looks: float = 8.0      # SAR: equivalent number of looks of the added Gamma speckle
    haze_max: float = 0.3           # optical: transmission t ~ U(1 - haze_max, 1) toward the airlight
    gain_jitter: float = 0.1        # optical: per-band calibration gain ~ U(1 -/+ gain_jitter)
    read_noise: float = 0.01        # optical: Gaussian read noise, as a fraction of each band's std
    # N2 cross-sensor neighbourhood consensus (twin-free; no sensor is assumed to be the teacher)
    lam_rel: float = 0.0
    rel_tau: float = 0.1
    rel_mode: str = "consensus"     # "consensus" (ours) | "cyclip" (CyCLIP-style cyclic consistency, prior work)


class QuadAttention(nn.Module):
    """One shared qkv projection -> four attention routes."""
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def _split(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def _attend(self, q, k, v):
        x = F.scaled_dot_product_attention(q, k, v)
        B, H, N, D = x.shape
        return self.proj(x.transpose(1, 2).reshape(B, N, H * D))

    def forward(self, xa, xb, cross: bool):
        qa, ka, va = self._split(xa)
        if xb is None:
            return self._attend(qa, ka, va), None, None, None
        qb, kb, vb = self._split(xb)
        oa = self._attend(qa, ka, va)
        ob = self._attend(qb, kb, vb)
        if not cross:
            return oa, ob, None, None
        q = torch.cat((qa, qb), 0)
        k = torch.cat((kb, ka), 0)
        v = torch.cat((vb, va), 0)
        o = self._attend(q, k, v)
        oab, oba = o.split(xa.size(0), dim=0)
        return oa, ob, oab, oba


class QuadBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = QuadAttention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, sa, sb, sab, sba, cross: bool):
        if sb is None:
            o, _, _, _ = self.attn(self.n1(sa), None, cross=False)
            sa = sa + o
            return sa + self.mlp(self.n2(sa)), None, None, None
        oa, ob, oab, oba = self.attn(self.n1(sa), self.n1(sb), cross=cross)
        sa = sa + oa; sa = sa + self.mlp(self.n2(sa))
        sb = sb + ob; sb = sb + self.mlp(self.n2(sb))
        if cross:
            sab = sab + oab; sab = sab + self.mlp(self.n2(sab))
            sba = sba + oba; sba = sba + self.mlp(self.n2(sba))
        return sa, sb, sab, sba


class QuadTrunk(nn.Module):
    def __init__(self, dim, depth, heads, grad_ckpt=False):
        super().__init__()
        self.blocks = nn.ModuleList([QuadBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.grad_ckpt = grad_ckpt

    def forward(self, ta, tb=None, cross=True):
        sab, sba = (tb, ta) if (tb is not None and cross) else (None, None)
        sa, sb = ta, tb
        for blk in self.blocks:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                sa, sb, sab, sba = checkpoint(blk, sa, sb, sab, sba, cross,
                                              use_reentrant=False)
            else:
                sa, sb, sab, sba = blk(sa, sb, sab, sba, cross)
        n = self.norm
        return n(sa), (n(sb) if sb is not None else None), \
               (n(sab) if sab is not None else None), (n(sba) if sba is not None else None)


class RSJEPA(nn.Module):
    def __init__(self, cfg: RSJEPAConfig = None):
        super().__init__()
        cfg = cfg or RSJEPAConfig()
        self.cfg = cfg
        if cfg.backbone == "vim_tiny":
            if cfg.depth not in (6, 12, 24) or cfg.dim != 192:
                raise ValueError("vim_tiny requires depth in (6, 12, 24) and dim=192")
            if getattr(cfg, "inter_cross", True):
                raise ValueError("Vision Mamba requires final BiDA with --no-inter")
            from .helpers.vim import build_vim
            self.stem_a, self.stem_b, self.trunk = build_vim(cfg)
        elif cfg.backbone == "vit":   # plain transformer trunk used by the base/dual model classes and CPU tests
            self.stem_a = Stem(cfg.s1_chans, cfg.dim, cfg.img_size, cfg.patch_size)
            self.stem_b = Stem(cfg.s2_chans, cfg.dim, cfg.img_size, cfg.patch_size)
            self.trunk = QuadTrunk(cfg.dim, cfg.depth, cfg.heads, cfg.grad_ckpt)
        else:
            raise ValueError(f"Unknown backbone: {cfg.backbone}")
        self.num_patches = self.stem_a.num_patches

        if cfg.ema_target:
            import copy
            self.stem_a_t = copy.deepcopy(self.stem_a)
            self.stem_b_t = copy.deepcopy(self.stem_b)
            self.trunk_t = copy.deepcopy(self.trunk)
            self.trunk_t.grad_ckpt = False
            for p in (list(self.stem_a_t.parameters()) + list(self.stem_b_t.parameters())
                      + list(self.trunk_t.parameters())):
                p.requires_grad = False
        else:
            self.stem_a_t = self.stem_b_t = self.trunk_t = None

        pk = dict(dim=cfg.dim, depth=cfg.pred_depth, heads=cfg.pred_heads,
                  num_patches=self.num_patches, grad_ckpt=cfg.grad_ckpt)
        self.pred_aa = QueryPredictor(**pk)
        self.pred_bb = QueryPredictor(**pk)
        self.pred_cross = None if cfg.fold_cross_predictor else QueryPredictor(**pk)

        self.head_uni = RetrievalHead(cfg.dim, cfg.retr_dim)
        self.head_cross = RetrievalHead(cfg.dim, cfg.retr_dim)

    def sensor_noise(self, x, sensor):
        """Training-only physical corruption of one sensor's image (normalised in, normalised out).

        SAR: multiplicative, unit-mean Gamma(L, 1/L) intensity speckle -- additive and de-biased
        in dB, or multiplying the amplitude by sqrt(speckle). Optical: per-band calibration gain,
        haze toward a per-image airlight (the image's own bright level, so no units are assumed),
        and read noise. Further sensor kinds plug in here.
        """
        c = self.cfg
        if not (self.training and c.sensor_noise):
            return x
        kind = c.sensor_kinds[sensor]
        mean = x.new_tensor(c.sensor_mean[sensor], dtype=torch.float32).view(1, -1, 1, 1)
        std = x.new_tensor(c.sensor_std[sensor], dtype=torch.float32).view(1, -1, 1, 1)
        xf = x.float()
        if kind.startswith("sar"):
            L = torch.full_like(xf, c.speckle_looks)
            g = torch.distributions.Gamma(L, L).sample()
            if kind == "sar_db":
                bias = 10.0 / math.log(10.0) * (torch.digamma(L[:1, :1, :1, :1]) - math.log(c.speckle_looks))
                out = xf + (10.0 * torch.log10(g) - bias) / std
            else:
                a = (xf * std + mean).clamp_min(0) * g.sqrt()
                out = (a - mean) / std
        elif kind == "optical":
            B, C = x.shape[:2]
            phys = xf * std + mean
            gain = 1 + (torch.rand(B, C, 1, 1, device=x.device) * 2 - 1) * c.gain_jitter
            t = 1 - torch.rand(B, 1, 1, 1, device=x.device) * c.haze_max
            air = torch.quantile(phys.flatten(2), 0.99, dim=2)[..., None, None]
            phys = t * gain * phys + (1 - t) * air
            out = (phys - mean) / std + torch.randn_like(xf) * c.read_noise
        else:
            raise ValueError(f"unknown sensor kind {kind!r}")
        return out.to(x.dtype)

    def sample_masks(self, B, device):
        n = self.num_patches
        k = int(round(n * (1 - self.cfg.mask_ratio)))
        perm = torch.argsort(torch.rand(B, n, device=device), dim=1)
        return perm[:, :k].contiguous(), perm[:, k:].contiguous()

    def forward(self, xa, xb):
        B = xa.size(0)
        vis_a, msk_a = self.sample_masks(B, xa.device)
        vis_b, msk_b = self.sample_masks(B, xa.device)

        index_kw = dict(idx_a=vis_a, idx_b=vis_b) if self.cfg.backbone == "vim_tiny" else {}
        ca, cb, cab, cba = self.trunk(self.stem_a(xa, vis_a),
                                      self.stem_b(xb, vis_b), cross=self.cfg.inter_cross, **index_kw)
        target_trunk = self.trunk_t if self.trunk_t is not None else self.trunk
        target_stem_a = self.stem_a_t if self.stem_a_t is not None else self.stem_a
        target_stem_b = self.stem_b_t if self.stem_b_t is not None else self.stem_b
        with torch.no_grad():
            index_kw = dict(idx_a=msk_a, idx_b=msk_b) if self.cfg.backbone == "vim_tiny" else {}
            ta, tb, _, _ = target_trunk(target_stem_a(xa, msk_a),
                                        target_stem_b(xb, msk_b), cross=False, **index_kw)

        pa, pb = self.stem_a.pos, self.stem_b.pos
        p_aa = self.pred_aa(ca, msk_a, pa)
        p_bb = self.pred_bb(cb, msk_b, pb)
        xp = self.pred_cross
        if self.cfg.inter_cross:
            p_ab = (self.pred_bb if xp is None else xp)(cab, msk_b, pb)
            p_ba = (self.pred_aa if xp is None else xp)(cba, msk_a, pa)
        else:
            p_ab, p_ba = p_bb, p_aa

        r_ua, e_ua = self.head_uni(ca.mean(1)); r_ub, e_ub = self.head_uni(cb.mean(1))
        r_ca, e_ca = self.head_cross(ca.mean(1)); r_cb, e_cb = self.head_cross(cb.mean(1))
        return dict(p_aa=p_aa, p_bb=p_bb, p_ab=p_ab, p_ba=p_ba, t_a=ta, t_b=tb,
                    r_uni_a=r_ua, r_uni_b=r_ub, e_uni_a=e_ua, e_uni_b=e_ub,
                    r_cross_a=r_ca, r_cross_b=r_cb, e_cross_a=e_ca, e_cross_b=e_cb)

    @torch.no_grad()
    def update_ema(self, m):
        if self.trunk_t is None:
            return
        pairs = zip(
            list(self.stem_a_t.parameters()) + list(self.stem_b_t.parameters()) + list(self.trunk_t.parameters()),
            list(self.stem_a.parameters()) + list(self.stem_b.parameters()) + list(self.trunk.parameters()))
        for pt, ps in pairs:
            pt.mul_(m).add_(ps.detach(), alpha=1 - m)

    @torch.no_grad()
    def embed_single(self, x, modality: int, head):
        stem = self.stem_a if modality == 0 else self.stem_b
        s, _, _, _ = self.trunk(stem(x), None, cross=False)
        return head(s.mean(1))[1]

    @torch.no_grad()
    def embed_uni(self, x, modality: int):
        return self.embed_single(x, modality, self.head_uni)

    @torch.no_grad()
    def embed(self, x1=None, x2=None, mode="cross"):
        head = self.head_uni if mode == "uni" else self.head_cross
        return (None if x1 is None else self.embed_single(x1, 0, head),
                None if x2 is None else self.embed_single(x2, 1, head))


@dataclass
class RSJEPASharedConfig(RSJEPAConfig):
    uni_mode: str = "paper"
    lam_view: float = 1.0


class RSJEPAShared(RSJEPA):
    def __init__(self, cfg: RSJEPASharedConfig = None):
        super().__init__(cfg or RSJEPASharedConfig())

    def sample_views(self, B, device):
        n = self.num_patches
        k = int(round(n * (1 - self.cfg.mask_ratio)))
        k -= k % 2
        h = k // 2
        perm = torch.argsort(torch.rand(B, n, device=device), dim=1)
        return perm[:, :h].contiguous(), perm[:, h:k].contiguous(), perm[:, k:].contiguous()

    def forward(self, xa, xb):
        B = xa.size(0)
        aA, aB, msk_a = self.sample_views(B, xa.device)
        bA, bB, msk_b = self.sample_views(B, xa.device)
        ta_in = torch.cat((self.stem_a(xa, aA), self.stem_a(xa, aB)), 0)
        tb_in = torch.cat((self.stem_b(xb, bA), self.stem_b(xb, bB)), 0)
        ca2, cb2, cab2, cba2 = self.trunk(ta_in, tb_in, cross=True)
        halves = lambda z: (z[:B], z[B:])
        join = lambda z: torch.cat(halves(z), 1)
        ca, cb, cab, cba = join(ca2), join(cb2), join(cab2), join(cba2)

        target_trunk = self.trunk_t if self.trunk_t is not None else self.trunk
        target_stem_a = self.stem_a_t if self.stem_a_t is not None else self.stem_a
        target_stem_b = self.stem_b_t if self.stem_b_t is not None else self.stem_b
        with torch.no_grad():
            ta, tb, _, _ = target_trunk(target_stem_a(xa, msk_a),
                                        target_stem_b(xb, msk_b), cross=False)

        pa, pb = self.stem_a.pos, self.stem_b.pos
        xp = self.pred_cross
        p_aa = self.pred_aa(ca, msk_a, pa)
        p_bb = self.pred_bb(cb, msk_b, pb)
        p_ab = (self.pred_bb if xp is None else xp)(cab, msk_b, pb)
        p_ba = (self.pred_aa if xp is None else xp)(cba, msk_a, pa)

        r_ua, e_ua = self.head_uni(ca.mean(1)); r_ub, e_ub = self.head_uni(cb.mean(1))
        r_ca, e_ca = self.head_cross(ca.mean(1)); r_cb, e_cb = self.head_cross(cb.mean(1))

        (caA, caB), (cbA, cbB) = halves(ca2), halves(cb2)
        r_aA, e_aA = self.head_uni(caA.mean(1)); r_aB, e_aB = self.head_uni(caB.mean(1))
        r_bA, e_bA = self.head_uni(cbA.mean(1)); r_bB, e_bB = self.head_uni(cbB.mean(1))

        return dict(p_aa=p_aa, p_bb=p_bb, p_ab=p_ab, p_ba=p_ba, t_a=ta, t_b=tb,
                    r_uni_a=r_ua, r_uni_b=r_ub, e_uni_a=e_ua, e_uni_b=e_ub,
                    r_cross_a=r_ca, r_cross_b=r_cb, e_cross_a=e_ca, e_cross_b=e_cb,
                    e_view_aA=e_aA, e_view_aB=e_aB, e_view_bA=e_bA, e_view_bB=e_bB,
                    r_view=(r_aA, r_aB, r_bA, r_bB))


# ============================================================================
# Dual: two independent encoders per modality, no interaction between views
# (ablation-1 evidence that splitting alone does nothing -- see ABLATIONS.md §1)
# ============================================================================

@dataclass
class RSJEPADualConfig(RSJEPASharedConfig):
    readout: str = "concat"      # "concat" (2 x retr_dim) or "avg" (retr_dim)


class RSJEPADual(RSJEPAShared):
    def __init__(self, cfg: RSJEPADualConfig = None):
        cfg = cfg or RSJEPADualConfig()
        super().__init__(cfg)
        self.trunk_a2 = QuadTrunk(cfg.dim, cfg.depth, cfg.heads, cfg.grad_ckpt)
        self.trunk_b2 = QuadTrunk(cfg.dim, cfg.depth, cfg.heads, cfg.grad_ckpt)

    def _enc2(self, tokens, modality: int):
        trunk = self.trunk_a2 if modality == 0 else self.trunk_b2
        s, _, _, _ = trunk(tokens, None, cross=False)
        return s

    def _heads(self, pooled):
        r_u, e_u = self.head_uni(pooled)
        r_c, e_c = self.head_cross(pooled)
        return r_u, e_u, r_c, e_c

    def forward(self, xa, xb):
        B = xa.size(0)
        aA, aB, msk_a = self.sample_views(B, xa.device)
        bA, bB, msk_b = self.sample_views(B, xa.device)

        ca_A, cb_A, cab, cba = self.trunk(self.stem_a(xa, aA),
                                          self.stem_b(xb, bA), cross=True)
        ca_B = self._enc2(self.stem_a(xa, aB), 0)
        cb_B = self._enc2(self.stem_b(xb, bB), 1)

        with torch.no_grad():
            ta, tb, _, _ = self.trunk(self.stem_a(xa, msk_a),
                                      self.stem_b(xb, msk_b), cross=False)

        pa, pb = self.stem_a.pos, self.stem_b.pos
        xp = self.pred_cross
        p_aa = self.pred_aa(ca_A, msk_a, pa)
        p_aa2 = self.pred_aa(ca_B, msk_a, pa)
        p_bb = self.pred_bb(cb_A, msk_b, pb)
        p_bb2 = self.pred_bb(cb_B, msk_b, pb)
        p_ab = (self.pred_bb if xp is None else xp)(cab, msk_b, pb)
        p_ba = (self.pred_aa if xp is None else xp)(cba, msk_a, pa)

        r_uaA, e_uaA, r_caA, _ = self._heads(ca_A.mean(1))
        r_uaB, e_uaB, r_caB, _ = self._heads(ca_B.mean(1))
        r_ubA, e_ubA, r_cbA, _ = self._heads(cb_A.mean(1))
        r_ubB, e_ubB, r_cbB, _ = self._heads(cb_B.mean(1))

        avg = lambda p, q: F.normalize(0.5 * (p + q), dim=-1)
        r_ua, r_ub = 0.5 * (r_uaA + r_uaB), 0.5 * (r_ubA + r_ubB)
        r_ca, r_cb = 0.5 * (r_caA + r_caB), 0.5 * (r_cbA + r_cbB)
        e_ua, e_ub = avg(r_uaA, r_uaB), avg(r_ubA, r_ubB)
        e_ca, e_cb = avg(r_caA, r_caB), avg(r_cbA, r_cbB)

        return dict(p_aa=p_aa, p_aa2=p_aa2, p_bb=p_bb, p_bb2=p_bb2,
                    p_ab=p_ab, p_ba=p_ba, t_a=ta, t_b=tb,
                    r_uni_a=r_ua, r_uni_b=r_ub, e_uni_a=e_ua, e_uni_b=e_ub,
                    r_cross_a=r_ca, r_cross_b=r_cb, e_cross_a=e_ca, e_cross_b=e_cb,
                    e_view_aA=e_uaA, e_view_aB=e_uaB, e_view_bA=e_ubA, e_view_bB=e_ubB,
                    r_view=(r_uaA, r_uaB, r_ubA, r_ubB))

    @torch.no_grad()
    def embed_single(self, x, modality: int, head, readout: str = None):
        ro = readout or self.cfg.readout
        t = (self.stem_a if modality == 0 else self.stem_b)(x)
        sA, _, _, _ = self.trunk(t, None, cross=False)
        sB = self._enc2(t, modality)
        rA, rB = head(sA.mean(1))[0], head(sB.mean(1))[0]
        r = torch.cat((rA, rB), dim=-1) if ro == "concat" else 0.5 * (rA + rB)
        return F.normalize(r, dim=-1)

    @torch.no_grad()
    def embed(self, x1=None, x2=None, mode="cross", readout: str = None):
        head = self.head_uni if mode == "uni" else self.head_cross
        return (None if x1 is None else self.embed_single(x1, 0, head, readout),
                None if x2 is None else self.embed_single(x2, 1, head, readout))


# ============================================================================
# BiDA: crossing at TWO levels -- inter-modal (S1<->S2, full context, QuadTrunk)
# and intra-modal (view A<->B, two disjoint halves per modality, BiDABlock).
# This + I1 (view_router="adversarial") is the final "RSJEPA" model (71.97%
# mean F1@5 on BEN-14K). Also carries the ablation-5 cross-modal-alternative
# code paths (comp_mask, inter_synergy, inter_views, inter_pred, shared_mask,
# synergy) -- see ABLATIONS.md §11 for what each one tested and why it didn't
# beat the final model.
# ============================================================================

@dataclass
class RSJEPABiDAConfig(RSJEPASharedConfig):
    view_depth: int = 4          # depth of each per-modality view stack
    view_interaction: bool = True  # CVI ablation: keep view processing without coupling
    gated_cvi: bool = False      # learn a per-token gate on each cross-view update
    cvi_pairing: str = "aligned" # crossed-stream residual init: "aligned" (s_AB <- t_A) | "legacy" (s_AB <- t_B, pre-2026-10-03 runs)
    lam_view_pred: float = 1.0   # weight of the view prediction term
    inter_cross: bool = True     # False = ABLATION: intra-modal BiDA only, no S1<->S2 routes
    # ---- I1: adversarial view router (kept -- the final model's own ablation) ----
    view_router: str = "random"         # "random" | "adversarial"
    lam_router: float = 1.0             # gradient-reversal scale for the adversarial router
    # ---- ablation 5: cross-modal interaction alternatives (ABLATIONS.md §11) ----
    synergy: bool = False               # I3: single-view paths + synergy hinge
    synergy_ref: str = "trunk"          # reference the joint path must beat: "trunk" | "singles"
    lam_synergy: float = 1.0            # weight of the synergy hinge
    synergy_margin: float = 0.02        # margin m in max(0, L_joint - L_ref + m)
    inter_synergy: bool = False         # predict each modality's targets from own / other / both contexts
    lam_inter_other: float = 0.5        # weight on "predict m's targets from the OTHER modality"
    lam_inter_joint: float = 1.0        # weight on "predict m's targets from BOTH contexts"
    lam_inter_syn: float = 1.0          # weight on max(0, L_joint - sg[min(L_own, L_other)] + margin)
    comp_mask: bool = False             # complementary cross-sensor masks: S1 visible only where S2 is hidden
    inter_views: bool = False           # sensor-to-sensor BiDA: a second BiDA stack coupling the two SENSORS
    inter_pred: bool = False            # plain cross-modal prediction: each sensor's visible tokens predict the other's hidden ones
    shared_mask: bool = False           # both sensors hide exactly the same patches
    # ---- Trajectory-JEPA (docs/final/RSJEPA_XFACTOR_TRAJECTORY_JEPA.md): replaces I1 + CVI ----
    traj_views: bool = False            # views = scan trajectories of the trunk itself (Vim only)
    traj_select: str = "hardest"        # second trajectory: "hardest" (max current error) | "random"
    # Faithful BiDA (IEEE TCSVT, Block_triple_branches) coupling of two views AFTER the trunk:
    # "none" | "traj" (the trunk's features of the same visible patches along paths g1 and g2;
    # rows correspond, as BiDA's semantic tokens do) | "adv" (I1 adversarial split A / B)
    bida_mode: str = "none"


class _GradReverse(torch.autograd.Function):
    """Identity forward, -scale * grad backward (I1 adversarial router)."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.scale * g, None


class BiDABlock(nn.Module):
    """BiDA's block, Fig. 2: two branches with their OWN qkv, plus a coupled
    cross-attention whose two outputs get their own Add&Norm + FFN.

        Qst = [Qa; Qb]   Kst = [Kb; Ka]   Vst = [Vb; Va]

    Deviation from the figure, to keep the parameter budget sane: the two
    crossed directions share one FFN instead of one each, and the two pure
    branches share theirs (they are the same modality here, not two domains).
    """

    def __init__(self, dim, heads, mlp_ratio=4.0, interaction=True, gated=False):
        super().__init__()
        self.h = heads
        self.interaction = interaction
        self.gated = gated
        self.n1_a, self.n1_b = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv_a = nn.Linear(dim, dim * 3, bias=False)    # encoder for view A
        self.qkv_b = nn.Linear(dim, dim * 3, bias=False)    # encoder for view B
        self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))           # pure branches
        self.n2c = nn.LayerNorm(dim) if interaction else None
        self.mlp_c = Mlp(dim, int(dim * mlp_ratio)) if interaction else None         # coupled branches
        if interaction and gated:
            self.cross_gate = nn.Linear(2 * dim, 1)
            nn.init.zeros_(self.cross_gate.weight)
            nn.init.constant_(self.cross_gate.bias, 2.0)

    def _split(self, qkv, x):
        B, N, C = x.shape
        t = qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        return t[0], t[1], t[2]

    def _attend(self, q, k, v):
        x = F.scaled_dot_product_attention(q, k, v)
        B, H, N, D = x.shape
        return self.proj(x.transpose(1, 2).reshape(B, N, H * D))

    def forward(self, sa, sb, sab, sba):
        ha, hb = self.n1_a(sa), self.n1_b(sb)
        qa, ka, va = self._split(self.qkv_a, ha)
        qb, kb, vb = self._split(self.qkv_b, hb)
        oa, ob = self._attend(qa, ka, va), self._attend(qb, kb, vb)
        sa = sa + oa; sa = sa + self.mlp(self.n2(sa))
        sb = sb + ob; sb = sb + self.mlp(self.n2(sb))
        if not self.interaction:
            return sa, sb, sa, sb
        # coupled cross-attention, both directions in one call
        o = self._attend(torch.cat((qa, qb), 0), torch.cat((kb, ka), 0),
                         torch.cat((vb, va), 0))
        oab, oba = o.split(qa.size(0), dim=0)
        if self.gated:
            oab = torch.sigmoid(self.cross_gate(torch.cat((sab, oab), dim=-1))) * oab
            oba = torch.sigmoid(self.cross_gate(torch.cat((sba, oba), dim=-1))) * oba
        sab = sab + oab; sab = sab + self.mlp_c(self.n2c(sab))
        sba = sba + oba; sba = sba + self.mlp_c(self.n2c(sba))
        return sa, sb, sab, sba


class FaithfulBiDABlock(nn.Module):
    """BiDA's Block_triple_branches (Zhang et al., IEEE TCSVT; github.com/YuxiangZhang-BIT/IEEE_TCSVT_BiDA)
    with dropout 0: one LayerNorm, one qkv, one proj and one MLP shared by all four streams;
    coupled attention Q=[q;q2], K=[k2;k], V=[v2;v] in a single batch-stacked call; residuals
    xab = fusion + attn(q, k2, v2) and xba = x + attn(q2, k, v), each followed by the MLP."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.h = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def _qkv(self, x):
        B, N, C = x.shape
        t = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        return t[0], t[1], t[2]

    def _out(self, o):
        B, H, N, D = o.shape
        return self.proj(o.transpose(1, 2).reshape(B, N, H * D))

    def forward(self, x, x2, fusion):
        B = x.size(0)
        q, k, v = self._qkv(self.norm1(x))
        q2, k2, v2 = self._qkv(self.norm1(x2))
        a = self._out(F.scaled_dot_product_attention(q, k, v))
        a2 = self._out(F.scaled_dot_product_attention(q2, k2, v2))
        a3, a4 = self._out(F.scaled_dot_product_attention(
            torch.cat((q, q2)), torch.cat((k2, k)), torch.cat((v2, v)))).split(B)
        ff = lambda r: r + self.mlp(self.norm2(r))
        return ff(x + a), ff(x2 + a2), ff(fusion + a3), ff(x + a4)


class FaithfulBiDAStack(nn.Module):
    """BiDAnet's block loop: the fusion stream starts as the second view and is carried across
    blocks; the reverse crossed stream is recomputed from the first view in every block."""

    def __init__(self, dim, depth, heads, grad_ckpt=False):
        super().__init__()
        self.blocks = nn.ModuleList([FaithfulBiDABlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.grad_ckpt = grad_ckpt

    def forward(self, x, x2):
        fusion, rev = x2, x
        for blk in self.blocks:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                x, x2, fusion, rev = checkpoint(blk, x, x2, fusion, use_reentrant=False)
            else:
                x, x2, fusion, rev = blk(x, x2, fusion)
        return self.norm(fusion), self.norm(rev)


class ViewStack(nn.Module):
    """One modality's view encoders: A, B, and the two coupled streams."""

    def __init__(self, dim, depth, heads, grad_ckpt=False, interaction=True, gated=False,
                 pairing="aligned"):
        super().__init__()
        if pairing not in ("aligned", "legacy"):
            raise ValueError(f"cvi_pairing must be 'aligned' or 'legacy', got {pairing!r}")
        self.blocks = nn.ModuleList([BiDABlock(dim, heads, interaction=interaction, gated=gated) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.grad_ckpt = grad_ckpt
        self.pairing = pairing

    def forward(self, ta, tb, ia=None, ib=None):
        """ia/ib: patch indices of the view tokens (unused here; kept for call-site
        symmetry with the router path, which needs them)."""
        # "ab" = A's queries reading B's content. Its rows are A's query rows, so its
        # residual stream starts from A's tokens. "legacy" started it from B's tokens,
        # which added A-query i's result onto the unrelated B-token i.
        if self.pairing == "aligned":
            sa, sb, sab, sba = ta, tb, ta, tb
        else:
            sa, sb, sab, sba = ta, tb, tb, ta
        for blk in self.blocks:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                sa, sb, sab, sba = checkpoint(blk, sa, sb, sab, sba, use_reentrant=False)
            else:
                sa, sb, sab, sba = blk(sa, sb, sab, sba)
        n = self.norm
        return n(sa), n(sb), n(sab), n(sba)


class RSJEPABiDA(RSJEPAShared):
    def __init__(self, cfg: RSJEPABiDAConfig = None):
        cfg = cfg or RSJEPABiDAConfig()
        super().__init__(cfg)
        if cfg.traj_views:
            # A non-SSM trunk (ViT) is allowed as the negative control: attention ignores token
            # order, so every trajectory gives the same features and the objective degenerates.
            if cfg.traj_select not in ("hardest", "random"):
                raise ValueError(f"traj_select must be 'hardest' or 'random', got {cfg.traj_select!r}")
            # one learned code per target trajectory, added to the predictor's query positions
            self.traj_emb = nn.Parameter(torch.zeros(4, cfg.dim))
            nn.init.trunc_normal_(self.traj_emb, std=0.02)
            if cfg.bida_mode not in ("none", "traj", "adv"):
                raise ValueError(f"bida_mode must be none/traj/adv, got {cfg.bida_mode!r}")
            if cfg.bida_mode != "none":
                # one faithful BiDA stack per sensor, on the TRUNK's features of the two views
                self.bida_x = nn.ModuleList([FaithfulBiDAStack(cfg.dim, cfg.view_depth, cfg.heads, cfg.grad_ckpt)
                                             for _ in range(2)])
            if cfg.bida_mode == "adv":
                self.router_a, self.router_b = nn.Linear(cfg.dim, 1), nn.Linear(cfg.dim, 1)
                for r in (self.router_a, self.router_b):
                    nn.init.zeros_(r.weight); nn.init.zeros_(r.bias)
            return      # no stem-level view stacks: the trunk produces the views
        vk = dict(dim=cfg.dim, depth=cfg.view_depth, heads=cfg.heads, grad_ckpt=cfg.grad_ckpt,
                  interaction=cfg.view_interaction, gated=cfg.gated_cvi, pairing=cfg.cvi_pairing)
        self.views_a = ViewStack(**vk)      # S1's two view encoders + coupling
        self.views_b = ViewStack(**vk)      # S2's two view encoders + coupling
        if cfg.inter_views:
            # sensor-to-sensor BiDA: the two SENSORS as a BiDA pair. Its crossed streams
            # ("S1 after asking S2") must predict that sensor's hidden patches, so the
            # coupling has to earn its keep -- unlike the trunk's inter-modal routes,
            # which had no objective of their own.
            self.views_x = ViewStack(**vk)
        if cfg.inter_synergy or cfg.inter_pred:
            # ONE shared predictor for the other-modality and joint predictions, plus a
            # learnable modality tag added to each context token so it knows which sensor a
            # token came from. Training-only: retrieval never runs it.
            self.pred_inter = QueryPredictor(dim=cfg.dim, depth=cfg.pred_depth, heads=cfg.pred_heads,
                                             num_patches=self.num_patches, grad_ckpt=cfg.grad_ckpt)
            self.mod_emb = nn.Parameter(torch.zeros(2, cfg.dim))
            nn.init.trunc_normal_(self.mod_emb, std=0.02)
        if cfg.view_router == "adversarial":
            # I1: one scoring layer per modality; zero init = no preference at start
            self.router_a, self.router_b = nn.Linear(cfg.dim, 1), nn.Linear(cfg.dim, 1)
            for r in (self.router_a, self.router_b):
                nn.init.zeros_(r.weight); nn.init.zeros_(r.bias)

    def route_views(self, x, stem, router, views=None):
        """I1: split the visible set into two HARD, disjoint, equal-size views.

        Tokens are scored, the top half goes to A and the rest to B. The forward
        pass uses the hard split; the router learns through a straight-through
        weight (value exactly 1, gradient from sigmoid(score)). The gradient is
        reversed, so the router seeks the split that makes view prediction
        HARDEST -- mask 0.70 showed a harder task scores better. Returns token
        tensors and their patch indices.
        """
        B = x.size(0)
        A, Bv, msk = views if views is not None else self.sample_views(B, x.device)
        vis = torch.cat((A, Bv), 1)
        t = stem(x, vis)
        score = _GradReverse.apply(router(t).squeeze(-1), self.cfg.lam_router)
        w = torch.sigmoid(score)
        order = torch.argsort(score.detach(), dim=1, descending=True)
        h = vis.size(1) // 2
        ia, ib = order[:, :h], order[:, h:]
        g = lambda z, i: torch.gather(z, 1, i if z.dim() == 2 else i[..., None].expand(-1, -1, z.size(-1)))
        wa, wb = g(w, ia), 1.0 - g(w, ib)
        tA = g(t, ia) * (1.0 + wa - wa.detach())[..., None]
        tB = g(t, ib) * (1.0 + wb - wb.detach())[..., None]
        return tA, tB, g(vis, ia), g(vis, ib), msk

    def sample_comp_views(self, B, device):
        """Complementary cross-sensor masks. One permutation per sample: the first k patches
        are S1's visible set and the next k are S2's visible set, so the two sensors never
        see the same patch and each one's visible set lies inside the other's hidden set.
        Each modality still hides (n - k) patches, exactly as with independent masks."""
        n = self.num_patches
        k = int(round(n * (1 - self.cfg.mask_ratio)))
        k -= k % 2                      # views must split evenly (see sample_views)
        h = k // 2
        perm = torch.argsort(torch.rand(B, n, device=device), dim=1)
        vis_a, vis_b, rest = perm[:, :k], perm[:, k:2 * k], perm[:, 2 * k:]
        msk_a = torch.cat((vis_b, rest), 1)          # everything S1 cannot see
        msk_b = torch.cat((vis_a, rest), 1)          # everything S2 cannot see
        return ((vis_a[:, :h].contiguous(), vis_a[:, h:].contiguous(), msk_a.contiguous()),
                (vis_b[:, :h].contiguous(), vis_b[:, h:].contiguous(), msk_b.contiguous()))

    def _traj_trunk(self, ta, tb=None, idx_a=None, idx_b=None, traj_a=None, traj_b=None):
        if self.cfg.backbone == "vim_tiny":
            return self.trunk(ta, tb, idx_a=idx_a, idx_b=idx_b, traj_a=traj_a, traj_b=traj_b)
        return self.trunk(ta, tb, cross=False)      # order-free trunk: trajectories are a no-op

    def _traj_predict(self, ctx, x, stem, vis, msk, pred, g1):
        """One sensor of Trajectory-JEPA: the context is scanned along trajectory g1; the
        hidden patches' targets are computed under all four trajectories (stop-gradient);
        the predictor, told the target trajectory, predicts the g1 targets and those of a
        second trajectory g2 != g1. "hardest" picks the g2 whose targets disagree most with
        g1's -- the traversal the trunk is currently least invariant to."""
        B, G = x.size(0), self.traj_emb.size(0)
        b = torch.arange(B, device=x.device)
        with torch.no_grad():
            tr = torch.arange(G, device=x.device).repeat_interleave(B)
            t_all, _, _, _ = self._traj_trunk(stem(x, msk).repeat(G, 1, 1), idx_a=msk.repeat(G, 1), traj_a=tr)
            t_all = t_all.view(G, B, *t_all.shape[1:])
            if self.cfg.traj_select == "hardest":
                gap = (t_all.float() - t_all[g1, b].float()).pow(2).mean((2, 3))
                gap[g1, b] = -1.0
                g2 = gap.argmax(0)
            else:
                g2 = (g1 + torch.randint(1, G, (B,), device=x.device)) % G
        g = torch.cat((g1, g2))
        p = pred(ctx.repeat(2, 1, 1), msk.repeat(2, 1), stem.pos + self.traj_emb[g][:, None, :])
        return p[:B], t_all[g1, b], p[B:], t_all[g2, b], g2

    def _traj_targets(self, x, stem, msk, g1):
        """Stop-gradient targets of the hidden patches under every trajectory, and the
        second trajectory g2 (hardest = largest target disagreement with g1, or random)."""
        B, G = x.size(0), self.traj_emb.size(0)
        b = torch.arange(B, device=x.device)
        with torch.no_grad():
            tr = torch.arange(G, device=x.device).repeat_interleave(B)
            t_all, _, _, _ = self._traj_trunk(stem(x, msk).repeat(G, 1, 1), idx_a=msk.repeat(G, 1), traj_a=tr)
            t_all = t_all.view(G, B, *t_all.shape[1:])
            if self.cfg.traj_select == "hardest":
                gap = (t_all.float() - t_all[g1, b].float()).pow(2).mean((2, 3))
                gap[g1, b] = -1.0
                g2 = gap.argmax(0)
            else:
                g2 = (g1 + torch.randint(1, G, (B,), device=x.device)) % G
        return t_all, g2

    def _coupled_predict(self, ctx, x, stem, vis, msk, pred, g1, sensor):
        """BiDA-coupled views after the trunk; the crossed streams predict the g2 targets."""
        B = x.size(0)
        b = torch.arange(B, device=x.device)
        t_all, g2 = self._traj_targets(x, stem, msk, g1)
        xn = self.sensor_noise(x, sensor)
        if self.cfg.bida_mode == "traj":
            # the same visible patches along g1 (= ctx) and along g2: row i is one patch in both
            ctx2, _, _, _ = self._traj_trunk(stem(xn, vis), idx_a=vis, traj_a=g2)
            v1, v2 = ctx, ctx2
        else:
            # I1: adversarial hard split of the visible patches; each half through the trunk
            # BiDA's coupled attention needs equal-size views: with an odd visible count
            # (e.g. 43 at 70% masking) the last visible patch is left out of this branch only
            h = vis.size(1) // 2
            router = self.router_a if sensor == 0 else self.router_b
            tA, tB, ia, ib, _ = self.route_views(xn, stem, router, (vis[:, :h], vis[:, h:2 * h], msk))
            v1, _, _, _ = self._traj_trunk(tA, idx_a=ia, traj_a=g1)
            v2, _, _, _ = self._traj_trunk(tB, idx_a=ib, traj_a=g1)
        fused, rev = self.bida_x[sensor](v1, v2)
        p_main = pred(ctx, msk, stem.pos + self.traj_emb[g1][:, None, :])
        p_view = pred(torch.cat((fused, rev), 1), msk, stem.pos + self.traj_emb[g2][:, None, :])
        return p_main, t_all[g1, b], p_view, t_all[g2, b], g2

    def _forward_traj(self, xa, xb):
        B, dev = xa.size(0), xa.device
        vis_a, msk_a = self.sample_masks(B, dev)
        vis_b, msk_b = self.sample_masks(B, dev)
        g1_a = torch.randint(0, 4, (B,), device=dev)
        g1_b = torch.randint(0, 4, (B,), device=dev)
        ca, cb, _, _ = self._traj_trunk(self.stem_a(self.sensor_noise(xa, 0), vis_a),
                                        self.stem_b(self.sensor_noise(xb, 1), vis_b),
                                  idx_a=vis_a, idx_b=vis_b, traj_a=g1_a, traj_b=g1_b)
        if self.cfg.bida_mode == "none":
            p_aa, ta, p_va, tva, g2_a = self._traj_predict(ca, xa, self.stem_a, vis_a, msk_a, self.pred_aa, g1_a)
            p_bb, tb, p_vb, tvb, g2_b = self._traj_predict(cb, xb, self.stem_b, vis_b, msk_b, self.pred_bb, g1_b)
        else:
            p_aa, ta, p_va, tva, g2_a = self._coupled_predict(ca, xa, self.stem_a, vis_a, msk_a, self.pred_aa, g1_a, 0)
            p_bb, tb, p_vb, tvb, g2_b = self._coupled_predict(cb, xb, self.stem_b, vis_b, msk_b, self.pred_bb, g1_b, 1)
        r_ua, e_ua = self.head_uni(ca.mean(1)); r_ub, e_ub = self.head_uni(cb.mean(1))
        r_ca, e_ca = self.head_cross(ca.mean(1)); r_cb, e_cb = self.head_cross(cb.mean(1))
        return dict(p_aa=p_aa, p_bb=p_bb, p_ab=p_bb, p_ba=p_aa, t_a=ta, t_b=tb,
                    p_view_a=p_va, p_view_b=p_vb, t_view_a=tva, t_view_b=tvb,
                    g2_a=g2_a, g2_b=g2_b, msk_a=msk_a, msk_b=msk_b,
                    r_uni_a=r_ua, r_uni_b=r_ub, e_uni_a=e_ua, e_uni_b=e_ub,
                    r_cross_a=r_ca, r_cross_b=r_cb, e_cross_a=e_ca, e_cross_b=e_cb)

    def forward(self, xa, xb):
        if self.cfg.traj_views:
            return self._forward_traj(xa, xb)
        # N3: the context sees each sensor's physical noise; the targets are encoded clean
        (xa_clean, xb_clean), xa, xb = (xa, xb), self.sensor_noise(xa, 0), self.sensor_noise(xb, 1)
        B = xa.size(0)
        # complementary cross-sensor masks are drawn jointly for the two modalities
        cv = self.sample_comp_views(B, xa.device) if self.cfg.comp_mask else (None, None)
        if self.cfg.shared_mask:
            # ONE mask per batch, shared by both sensors AND by every sample. A per-sample
            # mask would identify each pair: the pooled embedding carries a signature of which
            # positions were visible, so the contrastive loss matched partners by mask pattern
            # instead of content (l_cross 0.039 vs 2.31, retrieval collapsed to 44.38).
            v = tuple(t.expand(B, -1) for t in self.sample_views(1, xa.device))
            cv = (v, v)
        if self.cfg.view_router == "adversarial":
            tA_a, tB_a, aA, aB, msk_a = self.route_views(xa, self.stem_a, self.router_a, cv[0])
            tA_b, tB_b, bA, bB, msk_b = self.route_views(xb, self.stem_b, self.router_b, cv[1])
        else:
            aA, aB, msk_a = cv[0] if self.cfg.comp_mask else self.sample_views(B, xa.device)
            bA, bB, msk_b = cv[1] if self.cfg.comp_mask else self.sample_views(B, xa.device)
            tA_a = tB_a = tA_b = tB_b = None
        vis_a = torch.cat((aA, aB), 1)      # the full visible context, before the split
        vis_b = torch.cat((bA, bB), 1)

        # ---- level 1: inter-modal crossing on the FULL context ----
        ic = self.cfg.inter_cross
        index_kw = dict(idx_a=vis_a, idx_b=vis_b) if self.cfg.backbone == "vim_tiny" else {}
        ca, cb, cab, cba = self.trunk(self.stem_a(xa, vis_a),
                                      self.stem_b(xb, vis_b), cross=ic, **index_kw)
        target_trunk = self.trunk_t if self.trunk_t is not None else self.trunk
        target_stem_a = self.stem_a_t if self.stem_a_t is not None else self.stem_a
        target_stem_b = self.stem_b_t if self.stem_b_t is not None else self.stem_b
        with torch.no_grad():
            # CR-JEPA eq. 5: the trunk sees the masked tokens alone
            index_kw = dict(idx_a=msk_a, idx_b=msk_b) if self.cfg.backbone == "vim_tiny" else {}
            ta, tb, _, _ = target_trunk(target_stem_a(xa_clean, msk_a),
                                        target_stem_b(xb_clean, msk_b), cross=False, **index_kw)

        # ---- level 2: intra-modal crossing between the two 25% views ----
        if tA_a is None:
            tA_a, tB_a = self.stem_a(xa, aA), self.stem_a(xa, aB)
            tA_b, tB_b = self.stem_b(xb, bA), self.stem_b(xb, bB)
        vaA, vaB, vaAB, vaBA = self.views_a(tA_a, tB_a, aA, aB)
        vbA, vbB, vbAB, vbBA = self.views_b(tA_b, tB_b, bA, bB)

        if self.cfg.inter_views:
            # S1's visible tokens and S2's visible tokens are the two views (equal counts)
            xA, xB, xAB, xBA = self.views_x(self.stem_a(xa, vis_a), self.stem_b(xb, vis_b))

        pa, pb = self.stem_a.pos, self.stem_b.pos
        xp = self.pred_cross
        p_aa = self.pred_aa(ca, msk_a, pa)
        p_bb = self.pred_bb(cb, msk_b, pb)
        if ic:
            # cross routes: S1 tokens predicting S2 targets (and back); the grids are
            # co-registered, so the context sits at its own modality's patch indices
            p_ab = (self.pred_bb if xp is None else xp)(cab, msk_b, pb)
            p_ba = (self.pred_aa if xp is None else xp)(cba, msk_a, pa)
        else:   # ablation: no inter-modal routes exist; lam_ab/lam_ba are 0 (see losses)
            p_ab, p_ba = p_bb, p_aa
        # the CROSSED view streams predict the same targets: this is what gives
        # the intra-modal coupling a gradient of its own. vaAB is view A's tokens
        # (having asked B), so it sits at aA; vaBA sits at aB.
        p_view_a = self.pred_aa(torch.cat((vaAB, vaBA), 1), msk_a, pa)
        p_view_b = self.pred_bb(torch.cat((vbAB, vbBA), 1), msk_b, pb)

        extra = {}
        if self.cfg.inter_views:
            # "S1 after asking S2" predicts S1's own hidden patches, and vice versa
            extra["p_xview_a"] = self.pred_aa(xAB, msk_a, pa)
            extra["p_xview_b"] = self.pred_bb(xBA, msk_b, pb)
        if self.cfg.synergy:
            # I3: each view alone predicts the same targets (trained); concat-without-BiDA
            # is a measurement only, isolating what the interaction adds.
            extra["p_single_a"] = [self.pred_aa(vaA, msk_a, pa), self.pred_aa(vaB, msk_a, pa)]
            extra["p_single_b"] = [self.pred_bb(vbA, msk_b, pb), self.pred_bb(vbB, msk_b, pb)]
        if self.cfg.inter_pred and not self.cfg.inter_synergy:
            # plain cross-modal prediction: each sensor's visible context predicts the OTHER
            # sensor's hidden patches. The predictor is given the positions of those patches
            # (target position embeddings) but never their contents.
            extra["p_inter_other_a"] = self.pred_inter(cb + self.mod_emb[1], msk_a, pa)
            extra["p_inter_other_b"] = self.pred_inter(ca + self.mod_emb[0], msk_b, pb)
        if self.cfg.inter_synergy:
            # same hidden patches predicted from the own, the other, and both contexts.
            # The contexts are the TRUNK's features, so these terms train the retrieval network.
            ca_t, cb_t = ca + self.mod_emb[0], cb + self.mod_emb[1]
            both = torch.cat((ca_t, cb_t), 1)
            extra["p_inter_other_a"] = self.pred_inter(cb_t, msk_a, pa)     # S1 targets from S2
            extra["p_inter_other_b"] = self.pred_inter(ca_t, msk_b, pb)     # S2 targets from S1
            extra["p_inter_joint_a"] = self.pred_inter(both, msk_a, pa)
            extra["p_inter_joint_b"] = self.pred_inter(both, msk_b, pb)

        r_ua, e_ua = self.head_uni(ca.mean(1)); r_ub, e_ub = self.head_uni(cb.mean(1))
        r_ca, e_ca = self.head_cross(ca.mean(1)); r_cb, e_cb = self.head_cross(cb.mean(1))

        return dict(**extra, p_aa=p_aa, p_bb=p_bb, p_ab=p_ab, p_ba=p_ba,
                    p_view_a=p_view_a, p_view_b=p_view_b, t_a=ta, t_b=tb,
                    r_uni_a=r_ua, r_uni_b=r_ub, e_uni_a=e_ua, e_uni_b=e_ub,
                    msk_a=msk_a, msk_b=msk_b,
                    r_cross_a=r_ca, r_cross_b=r_cb, e_cross_a=e_ca, e_cross_b=e_cb)

    # inference: inherited from RSJEPA -- full image, shared trunk, one head.
    # The view stacks are training-only, so retrieval cost is unchanged.


__all__ = [
    "RSJEPA", "RSJEPAConfig",
    "RSJEPAShared", "RSJEPASharedConfig",
    "QuadAttention", "QuadBlock", "QuadTrunk",
    "RSJEPADual", "RSJEPADualConfig",
    "RSJEPABiDA", "RSJEPABiDAConfig", "BiDABlock",
]
