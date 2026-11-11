from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

UNI_MODES = ("paper", "decoupled", "nce_only")
PRED_LOSSES = ("mse", "smooth_l1", "cosine")


def prediction_distance(pred, target, kind):
    pred, target = pred.float(), target.float()
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "smooth_l1":
        return F.smooth_l1_loss(pred, target, beta=1.0)
    if kind == "cosine":
        return (1.0 - F.cosine_similarity(pred, target, dim=-1, eps=1e-8)).mean()
    raise ValueError(f"pred_loss must be one of {PRED_LOSSES}, got {kind!r}")


def info_nce(qa, qb, temperature=0.07):
    """Symmetric batchwise InfoNCE between two sets of normalised embeddings."""
    qa, qb = qa.float(), qb.float()
    logits = qa @ qb.t() / temperature
    tgt = torch.arange(len(qa), device=qa.device)
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))


def sigreg(r, n_dirs=64, n_freq=17, t_max=3.0, scale_by_batch=False):
    """Sketched Isotropic Gaussian Regularization."""
    r = r.float()
    B, D = r.shape
    u = F.normalize(torch.randn(D, n_dirs, device=r.device, dtype=r.dtype), dim=0)
    s = r @ u
    t = torch.linspace(0, t_max, n_freq, device=r.device, dtype=r.dtype)
    ts = s[..., None] * t
    emp_re, emp_im = ts.cos().mean(0), ts.sin().mean(0)
    gauss = torch.exp(-0.5 * t ** 2)
    w = torch.full_like(t, t_max / (n_freq - 1))
    w[0] *= 0.5
    w[-1] *= 0.5
    w = w * gauss
    diff = (emp_re - gauss) ** 2 + emp_im ** 2
    stat = (diff * w).sum(-1).mean()
    return stat * B if scale_by_batch else stat


def _neighbours(q, g, tau, log=False):
    """Row-wise distribution of each query over the OTHER batch scenes (own pair masked)."""
    eye = torch.eye(len(q), dtype=torch.bool, device=q.device)
    z = (q.float() @ g.float().t() / tau).masked_fill(eye, float("-inf"))
    return z.log_softmax(-1).masked_fill(eye, 0.0) if log else z.softmax(-1)


def neighbourhood_consensus(e_uni, e_cross, tau):
    """N2: cross-sensor neighbourhood consensus, sensor-agnostic.

    e_uni / e_cross: lists of per-sensor (B, D) embeddings of the SAME scenes. The target is the
    mean of every sensor's own neighbour distribution (stop-gradient); every same-sensor and
    every cross-sensor neighbour distribution is pulled toward it. The scene's own pair is
    excluded everywhere, so this aligns semantic neighbourhoods, not instance twins.
    """
    with torch.no_grad():
        target = torch.stack([_neighbours(e, e, tau) for e in e_uni]).mean(0)
    logt = target.clamp_min(1e-12).log()
    kl = lambda logp: (target * (logt - logp)).sum(-1).mean()
    terms = [kl(_neighbours(e, e, tau, log=True)) for e in e_uni]
    terms += [kl(_neighbours(q, g, tau, log=True))
              for i, q in enumerate(e_cross) for j, g in enumerate(e_cross) if i != j]
    return sum(terms) / len(terms)


def cyclip_consistency(e_uni, e_cross):
    """CyCLIP-style baseline (Goel et al., NeurIPS 2022) for two sensors a, b:
    in-modal  0.25 * B * mean((S_aa - S_bb)^2) on the same-sensor (uni) head, and
    cross-modal 0.25 * B * mean((S_ab - S_ba)^2) on the cross head, with cosine similarities
    (the reference implementation's logit-scale normalisation and lambda = 0.25)."""
    (ua, ub), (xa, xb) = [t.float() for t in e_uni], [t.float() for t in e_cross]
    B = len(ua)
    inmodal = ((ua @ ua.t()) - (ub @ ub.t())).square().mean() * B
    crossmodal = ((xa @ xb.t()) - (xb @ xa.t())).square().mean() * B
    return 0.25 * inmodal + 0.25 * crossmodal


def synergy_hinge(l_joint, l_ref, margin):
    """max(0, L_joint - L_ref + m) with reference under stop-gradient."""
    return F.relu(l_joint - l_ref.detach() + margin)


class RSJEPALoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.uni_mode not in UNI_MODES:
            raise ValueError(f"uni_mode must be one of {UNI_MODES}, got {cfg.uni_mode!r}")
        if cfg.pred_loss not in PRED_LOSSES:
            raise ValueError(f"pred_loss must be one of {PRED_LOSSES}, got {cfg.pred_loss!r}")
        self.c = cfg

    def _uni(self, o):
        c = self.c
        if c.uni_mode == "decoupled":
            return torch.zeros((), device=o["e_uni_a"].device)
        nce = info_nce(o["e_uni_a"], o["e_uni_b"], c.temperature)
        if c.uni_mode == "nce_only":
            return nce
        cos = 1.0 - (o["e_uni_a"].float() * o["e_uni_b"].float()).sum(-1).mean()
        return nce + cos

    def forward(self, o: dict):
        c = self.c
        mse = lambda p, t: prediction_distance(p, t, c.pred_loss)
        l_pred = (c.lam_aa * mse(o["p_aa"], o["t_a"]) +
                  c.lam_bb * mse(o["p_bb"], o["t_b"]) +
                  c.lam_ab * mse(o["p_ab"], o["t_b"]) +
                  c.lam_ba * mse(o["p_ba"], o["t_a"]))
        l_cross = info_nce(o["e_cross_a"], o["e_cross_b"], c.temperature)
        l_uni = self._uni(o)
        l_retr = c.lam_cross * l_cross + c.lam_uni * l_uni
        total = l_pred + l_retr
        if c.lam_sigreg > 0:    # earlier models only; RSJEPA v2 has no SIGReg (not computed at all)
            l_sig = 0.5 * (sigreg(o["r_uni_a"]) + sigreg(o["r_uni_b"]) +
                           sigreg(o["r_cross_a"]) + sigreg(o["r_cross_b"]))
            total = total + c.lam_sigreg * l_sig
        lam_rel = getattr(c, "lam_rel", 0.0)
        if lam_rel > 0:
            # N2: every sensor's neighbourhoods follow the cross-sensor consensus
            if getattr(c, "rel_mode", "consensus") == "cyclip":
                l_rel = cyclip_consistency([o["e_uni_a"], o["e_uni_b"]], [o["e_cross_a"], o["e_cross_b"]])
            else:
                l_rel = neighbourhood_consensus([o["e_uni_a"], o["e_uni_b"]],
                                                [o["e_cross_a"], o["e_cross_b"]], c.rel_tau)
            total = total + lam_rel * l_rel
        log = dict(pred=l_pred.detach(),
                   l_cross=l_cross.detach(),
                   l_uni=l_uni.detach(),
                   cross=l_cross.detach(),
                   uni=l_uni.detach(),
                   **({"sigreg": l_sig.detach(), "sig": l_sig.detach()} if c.lam_sigreg > 0 else {}),
                   total=total.detach(),
                   **({"l_rel": l_rel.detach()} if lam_rel > 0 else {}),
                   pred_aa=mse(o["p_aa"], o["t_a"]).detach(),
                   pred_bb=mse(o["p_bb"], o["t_b"]).detach(),
                   pred_ab=mse(o["p_ab"], o["t_b"]).detach(),
                   pred_ba=mse(o["p_ba"], o["t_a"]).detach())
        return total, log


class RSJEPASharedLoss(RSJEPALoss):
    def forward(self, o: dict):
        total, log = super().forward(o)
        cos = lambda p, q: (p.float() * q.float()).sum(-1).mean()
        inv = 1.0 - 0.5 * (cos(o["e_view_aA"], o["e_view_aB"]) +
                           cos(o["e_view_bA"], o["e_view_bB"]))
        sg = sum(sigreg(r) for r in o["r_view"]) / len(o["r_view"])
        l_view = inv + sg
        total = total + self.c.lam_view * l_view
        log.update(l_view=l_view.detach(), view_inv=inv.detach(), view_sig=sg.detach(), total=total.detach())
        return total, log


class RSJEPADualLoss(RSJEPALoss):
    def forward(self, o: dict):
        total, log = super().forward(o)
        mse = lambda p, t: F.mse_loss(p.float(), t.float())
        total = total + 0.5 * self.c.lam_aa * (mse(o["p_aa2"], o["t_a"]) - mse(o["p_aa"], o["t_a"]))
        total = total + 0.5 * self.c.lam_bb * (mse(o["p_bb2"], o["t_b"]) - mse(o["p_bb"], o["t_b"]))

        cos = lambda p, q: (p.float() * q.float()).sum(-1).mean()
        inv = 1.0 - 0.5 * (cos(o["e_view_aA"], o["e_view_aB"]) +
                           cos(o["e_view_bA"], o["e_view_bB"]))
        sg = sum(sigreg(r) for r in o["r_view"]) / len(o["r_view"])
        l_view = inv + sg
        total = total + self.c.lam_view * l_view
        log.update(pred_aa2=mse(o["p_aa2"], o["t_a"]).detach(),
                   pred_bb2=mse(o["p_bb2"], o["t_b"]).detach(), l_view=l_view.detach(),
                   view_inv=inv.detach(), view_sig=sg.detach(), total=total.detach())
        return total, log


class RSJEPABiDALoss(RSJEPALoss):
    def forward(self, o: dict):
        c = self.c
        total, log = super().forward(o)
        mse = lambda p, t: F.mse_loss(p.float(), t.float())
        mv = lambda p, t: (sum(mse(q, t) for q in p) / len(p)) if isinstance(p, (list, tuple)) else mse(p, t)
        # Trajectory-JEPA predicts a second trajectory's targets; BiDA predicts the same ones
        pred_va = mv(o["p_view_a"], o.get("t_view_a", o["t_a"]))
        pred_vb = mv(o["p_view_b"], o.get("t_view_b", o["t_b"]))
        l_vp = 0.5 * (pred_va + pred_vb)
        total = total + c.lam_view_pred * l_vp
        log.update(pred_view_a=pred_va.detach(), pred_view_b=pred_vb.detach(), l_view_pred=l_vp.detach())

        if c.inter_views:
            l_xv = 0.5 * (mse(o["p_xview_a"], o["t_a"]) + mse(o["p_xview_b"], o["t_b"]))
            total = total + l_xv
            log.update(l_xview=l_xv.detach())

        if c.synergy:
            m = c.synergy_margin
            if c.synergy_ref == "trunk":
                ha = synergy_hinge(pred_va, log["pred_aa"], m)
                hb = synergy_hinge(pred_vb, log["pred_bb"], m)
            elif c.synergy_ref == "singles":
                sa = 0.5 * (mse(o["p_single_a"][0], o["t_a"]) + mse(o["p_single_a"][1], o["t_a"]))
                sb = 0.5 * (mse(o["p_single_b"][0], o["t_b"]) + mse(o["p_single_b"][1], o["t_b"]))
                ha, hb = synergy_hinge(pred_va, sa, m), synergy_hinge(pred_vb, sb, m)
                log.update(single_a=sa.detach(), single_b=sb.detach())
            else:
                raise ValueError(f"unknown synergy_ref: {c.synergy_ref}")
            l_syn = 0.5 * (ha + hb)
            total = total + c.lam_synergy * l_syn
            log.update(hinge_a=ha.detach(), hinge_b=hb.detach(), l_synergy=l_syn.detach())

        if c.inter_synergy:
            ref_a = torch.minimum(log["pred_aa"], mse(o["p_inter_other_a"], o["t_a"]))
            ref_b = torch.minimum(log["pred_bb"], mse(o["p_inter_other_b"], o["t_b"]))
            ha = synergy_hinge(mse(o["p_inter_joint_a"], o["t_a"]), ref_a, c.synergy_margin)
            hb = synergy_hinge(mse(o["p_inter_joint_b"], o["t_b"]), ref_b, c.synergy_margin)
            l_isyn = 0.5 * (ha + hb)
            total = total + c.lam_inter_syn * l_isyn
            log.update(inter_hinge_a=ha.detach(), inter_hinge_b=hb.detach(), l_inter_synergy=l_isyn.detach())
        elif c.inter_pred:
            # plain cross-modal prediction (no synergy hinge): each sensor's targets
            # predicted straight from the OTHER modality's context, weighted by lam_inter_other.
            l_ip = 0.5 * (mse(o["p_inter_other_a"], o["t_a"]) + mse(o["p_inter_other_b"], o["t_b"]))
            total = total + c.lam_inter_other * l_ip
            log.update(l_inter_pred=l_ip.detach())

        log["total"] = total.detach()
        return total, log
