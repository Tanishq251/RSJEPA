#!/usr/bin/env python
"""Pretrain RS-JEPA on BEN-14K.

Optimisation follows CR-JEPA 4.3 exactly -- AdamW, lr 1e-4 -> 1e-3 -> 1e-6,
weight decay 0.04, grad clip 1.0, cosine with 15 warmup epochs, bf16, batch 512
-- so any difference against runs/crjepa_ep100 is attributable to the model and
the uni_mode ablation, not the schedule.

Batch 512 is a TRUE batch, not accumulation. L_cross (and L_uni in the 'paper'
and 'nce_only' modes) are InfoNCE, so the batch IS the negative set; accumulating
k micro-batches averages k small-batch InfoNCE terms, each still seeing only
micro-batch-1 negatives. Activation checkpointing is what makes 512 fit.

--uni-mode selects the ablation (see RSJEPA/losses.py):
    paper      CR-JEPA eq. (18) verbatim -- the control
    decoupled  no cross-modal term on the same-modal head -- the hypothesis
    nce_only   NCE without the cosine term
"""
from __future__ import annotations
import argparse, json, math, os, time
from pathlib import Path
import torch

from RSJEPA import (
    RSJEPA, RSJEPAConfig, RSJEPALoss, UNI_MODES,
    RSJEPAShared, RSJEPASharedConfig, RSJEPASharedLoss,
    RSJEPADual, RSJEPADualConfig, RSJEPADualLoss,
    RSJEPABiDA, RSJEPABiDAConfig, RSJEPABiDALoss
)
from RSJEPA.data import loaders, ROOT


def args():
    p = argparse.ArgumentParser("RS-JEPA pretraining")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512, help="micro-batch")
    p.add_argument("--accum", type=int, default=0, help="0 = auto, to reach --target-batch")
    p.add_argument("--target-batch", type=int, default=512)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--train-split", default="train",
                   help='pretraining split(s), e.g. "train+validation" (test is never trained on)')
    p.add_argument("--data-root", default=str(ROOT))
    p.add_argument("--s1-chans", type=int, default=2)
    p.add_argument("--s2-chans", type=int, default=12)
    p.add_argument("--patch-size", type=int, default=16, help="must divide --img-size (e.g. 120 with 10)")
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--uni-mode", choices=UNI_MODES, default="decoupled")
    p.add_argument("--keep-cross-pred", action="store_true",
                   help="build the separate cross-modal predictor (116M instead of 91M)")
    p.add_argument("--ema-target", action="store_true",
                   help="momentum target trunk instead of stop-gradient on the same trunk "
                        "(base RS-JEPA, --views, --bida; not --dual; not with --bida --moe-experts)")
    p.add_argument("--lr-start", type=float, default=1e-4)
    p.add_argument("--lr-peak", type=float, default=1e-3)
    p.add_argument("--lr-final", type=float, default=1e-6)
    p.add_argument("--warmup-epochs", type=int, default=15)
    p.add_argument("--wd", type=float, default=0.04)
    p.add_argument("--lam-sigreg", type=float, default=1.0)
    p.add_argument("--bida", action="store_true",
                   help="RSJEPA (final model): intra-modal A<->B coupling + inter-modal crossing")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--pred-heads", type=int, default=8)
    p.add_argument("--backbone", choices=["vim_tiny"], default="vim_tiny")
    p.add_argument("--depth", type=int, default=12, help="depth of the trunk transformer blocks")
    p.add_argument("--view-depth", type=int, default=4, help="depth of each view stack (--bida)")
    p.add_argument("--lam-view-pred", type=float, default=1.0, help="weight of the intra-modal prediction term (--bida)")
    p.add_argument("--pred-loss", choices=("mse", "smooth_l1", "cosine"), default="mse",
                   help="masked-feature prediction distance for no-BiDA loss comparison")
    p.add_argument("--no-cvi", action="store_true", help="disable cross-view coupling, keeping independent view processing (--bida)")
    p.add_argument("--gated-cvi", action="store_true", help="learn per-token gates for cross-view updates (--bida)")
    p.add_argument("--gap-delta", action="store_true",
                   help="Vim: scan masked sequences as irregular samples of the full-image trajectory")
    p.add_argument("--traj-views", action="store_true",
                   help="Trajectory-JEPA (--bida): scan-trajectory views in the trunk replace I1 + CVI")
    p.add_argument("--traj-select", choices=("hardest", "random"), default="hardest",
                   help="second target trajectory for --traj-views")
    p.add_argument("--bida-mode", choices=("none", "traj", "adv"), default="none",
                   help="--traj-views: faithful BiDA coupling after the trunk between the two scan-path "
                        "views (traj) or the I1 adversarial A/B views (adv)")
    p.add_argument("--sensor-noise", action="store_true",
                   help="N3: corrupt each sensor's context with its imaging physics (SAR speckle; optical "
                        "haze/gain/read noise); masked targets stay clean")
    p.add_argument("--speckle-looks", type=float, default=8.0, help="N3 SAR speckle looks")
    p.add_argument("--lam-rel", type=float, default=0.0,
                   help="N2: weight of cross-sensor neighbourhood consensus (twin-free, sensor-agnostic)")
    p.add_argument("--rel-tau", type=float, default=0.1)
    p.add_argument("--rel-mode", choices=("consensus", "cyclip"), default="consensus",
                   help="relational term: our consensus, or the CyCLIP-style prior-work baseline")
    p.add_argument("--cvi-pairing", choices=("aligned", "legacy"), default="aligned",
                   help="crossed-stream residual init (--bida); legacy reproduces runs before 2026-10-03")
    p.add_argument("--no-inter", action="store_true", help="ABLATION: intra-modal BiDA only, no S1<->S2 routes")
    p.add_argument("--seed", type=int, default=None, help="random seed; unset = unseeded, as in earlier runs")
    p.add_argument("--view-router", default="random", choices=["random", "adversarial"],
                   help="I1: how the two views are formed (--bida)")
    p.add_argument("--synergy", action="store_true", help="I3: single-view paths + synergy hinge (--bida)")
    p.add_argument("--synergy-ref", default="trunk", choices=["trunk", "singles"], help="I3 reference path")
    p.add_argument("--synergy-margin", type=float, default=0.02, help="I3 hinge margin")
    p.add_argument("--inter-synergy", action="store_true", help="ablation 5: own/other/joint inter-modal prediction + synergy hinge")
    p.add_argument("--comp-mask", action="store_true", help="ablation 5: complementary cross-sensor masks: S1 visible only where S2 is hidden")
    p.add_argument("--inter-views", action="store_true", help="ablation 5: sensor-to-sensor BiDA, extra stack coupling the two sensors (training-only)")
    p.add_argument("--inter-pred", action="store_true", help="ablation 5: plain cross-modal prediction: each sensor's visible tokens predict the other's hidden patches")
    p.add_argument("--shared-mask", action="store_true", help="ablation 5: both sensors hide exactly the same patches")
    p.add_argument("--dual", action="store_true",
                   help="RS-JEPA-dual: S1 gets a second encoder for its view B (rsjepa_dual/SPEC.md)")
    p.add_argument("--views", action="store_true",
                   help="RS-JEPA-shared: two disjoint views per modality (rsjepa_shared/SPEC.md)")
    p.add_argument("--lam-view", type=float, default=1.0)
    p.add_argument("--no-grad-ckpt", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=str, default="runs/RSJEPA")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--preset", choices=("rsjepa_v2",), default=None,
                   help="rsjepa_v2: the frozen final architecture (docs/final/RSJEPA_V2_FINAL_ARCHITECTURE.md); "
                        "sets the architecture/loss flags, while mask ratio, data and seed stay as given")
    a = p.parse_args()
    if a.preset == "rsjepa_v2":
        # Vim-Tiny-12 + scan-path views + BiDA on I1 adversarial views after the trunk + consensus, no SIGReg
        for k, v in dict(backbone="vim_tiny", depth=12, dim=192, heads=8, pred_heads=8, view_depth=4,
                         bida=True, no_inter=True, traj_views=True, traj_select="hardest", bida_mode="adv",
                         view_router="random", lam_rel=1.0, rel_tau=0.1, rel_mode="consensus", lam_sigreg=0.0,
                         uni_mode="paper", lam_view_pred=1.0, pred_loss="mse").items():
            setattr(a, k, v)
    return a


def cosine_lr(it, total, warm, lo, peak, final):
    if it < warm:
        return lo + (peak - lo) * it / max(warm, 1)
    t = (it - warm) / max(total - warm, 1)
    return final + 0.5 * (peak - final) * (1 + math.cos(math.pi * t))


def main():
    a = args()
    if a.seed is not None:
        import random, numpy as np
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.img_size % a.patch_size:
        raise SystemExit(f"--patch-size {a.patch_size} does not divide --img-size {a.img_size}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    if a.bida:
        Cfg, Model, Loss = RSJEPABiDAConfig, RSJEPABiDA, RSJEPABiDALoss
    elif a.dual:
        Cfg, Model, Loss = RSJEPADualConfig, RSJEPADual, RSJEPADualLoss
    elif a.views:
        Cfg, Model, Loss = RSJEPASharedConfig, RSJEPAShared, RSJEPASharedLoss
    else:
        Cfg, Model, Loss = RSJEPAConfig, RSJEPA, RSJEPALoss
    if a.traj_views and not a.bida:
        raise SystemExit("--traj-views requires --bida")
    if a.bida_mode != "none" and not a.traj_views:
        raise SystemExit("--bida-mode couples trunk views of the --traj-views model")
    if a.traj_views and a.view_router != "random":
        raise SystemExit("--traj-views replaces the I1 router; leave --view-router at random")
    speckle_kw = {}
    if a.sensor_noise:
        # S1 is SAR (BEN: z-scored dB, has band names; CBRSIR: z-scored amplitude), S2 optical
        st = json.load(open(Path(a.data_root) / "norm_stats.json"))
        speckle_kw = dict(sensor_noise=True, speckle_looks=a.speckle_looks,
                          sensor_kinds=("sar_db" if "s1_bands" in st else "sar_amplitude", "optical"),
                          sensor_mean=(tuple(st["s1_mean"]), tuple(st["s2_mean"])),
                          sensor_std=(tuple(st["s1_std"]), tuple(st["s2_std"])))
        if len(st["s1_mean"]) != a.s1_chans or len(st["s2_mean"]) != a.s2_chans:
            raise SystemExit("norm_stats.json channel counts do not match --s1-chans/--s2-chans")
        print(f"N3 sensor noise: kinds={speckle_kw['sensor_kinds']}, SAR looks={a.speckle_looks}")
    if a.no_cvi and not a.bida:
        raise SystemExit("--no-cvi requires --bida")
    if a.gated_cvi and (not a.bida or a.no_cvi):
        raise SystemExit("--gated-cvi requires --bida with CVI enabled")
    if a.pred_loss != "mse" and (a.bida or a.dual or a.views):
        raise SystemExit("non-MSE --pred-loss is currently a no-BiDA base-model experiment")
    kw = dict(lam_view=a.lam_view) if (a.views or a.dual) else dict(ema_target=a.ema_target)
    if a.bida:
        kw = dict(lam_view=a.lam_view, view_depth=a.view_depth,
                  lam_view_pred=a.lam_view_pred, inter_cross=not a.no_inter,
                  view_interaction=not a.no_cvi, gated_cvi=a.gated_cvi, cvi_pairing=a.cvi_pairing,
                  traj_views=a.traj_views, traj_select=a.traj_select, bida_mode=a.bida_mode,
                  view_router=a.view_router, synergy=a.synergy,
                  synergy_ref=a.synergy_ref, synergy_margin=a.synergy_margin,
                  inter_synergy=a.inter_synergy, comp_mask=a.comp_mask, inter_views=a.inter_views,
                  inter_pred=a.inter_pred, shared_mask=a.shared_mask, ema_target=a.ema_target)
    if a.no_inter:
        kw.update(inter_cross=False, lam_ab=0.0, lam_ba=0.0)
    cfg = Cfg(img_size=a.img_size, patch_size=a.patch_size, depth=a.depth, backbone=a.backbone,
              s1_chans=a.s1_chans, s2_chans=a.s2_chans, mask_ratio=a.mask_ratio,
              dim=a.dim, heads=a.heads, pred_heads=a.pred_heads,
              lam_sigreg=a.lam_sigreg, uni_mode=a.uni_mode,
              pred_loss=a.pred_loss, gap_delta=a.gap_delta,
              lam_rel=a.lam_rel, rel_tau=a.rel_tau, rel_mode=a.rel_mode, **speckle_kw,
              fold_cross_predictor=not a.keep_cross_pred,
              grad_ckpt=not a.no_grad_ckpt, **kw)
    accum = a.accum or max(1, a.target_batch // a.batch_size)
    model = Model(cfg).to(dev)
    crit = Loss(cfg).to(dev)
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    name = ('RSJEPA' if a.bida else
            'RS-JEPA-dual (2 encoders per modality)') if (a.bida or a.dual) else ('RS-JEPA-shared (views)' if a.views else 'RS-JEPA')
    print(f"{name} | trainable {tr/1e6:.2f} M | uni_mode={cfg.uni_mode} | "
          + (f"lam_view={cfg.lam_view} | " if (a.views or a.dual or a.bida) else "") +
          f"cross predictor {'kept' if a.keep_cross_pred else 'folded'}")
    if cfg.uni_mode != "paper":
        print(f"  NOTE: uni_mode='{cfg.uni_mode}' is a DELIBERATE deviation from "
              f"CR-JEPA eq. (18) -- record it in PAPER_VS_CODE.md before reporting.")
    print()

    train_dl, _ = loaders(a.batch_size, a.img_size, a.workers, augment=True,
                           root=a.data_root, train_split=a.train_split,
                           eval_split="eval" if __import__("pathlib").Path(a.data_root, "split_70_30.csv").exists() else "test")
    ipe = len(train_dl); total_it = ipe * a.epochs; warm_it = ipe * a.warmup_epochs
    print(f"{len(train_dl.dataset):,} pairs | {ipe} it/epoch | batch {a.batch_size}"
          f" x accum {accum} = effective {a.batch_size*accum} | "
          f"InfoNCE negatives {a.batch_size-1}\n")

    decay, no_decay = [], []
    for n, p_ in model.named_parameters():
        if p_.requires_grad:
            (no_decay if p_.ndim <= 1 or "pos" in n or "query" in n else decay).append(p_)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr_start, betas=(0.9, 0.95))

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda" and not use_bf16))
    if not use_bf16 and dev == "cuda":
        print(f"Hardware precision: using Float16 + GradScaler (optimal for {torch.cuda.get_device_name(0)})")

    start_ep, it = 0, 0
    if a.resume and Path(a.resume).exists():
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_ep, it = ck["epoch"] + 1, ck["it"]
        print(f"resumed @ epoch {start_ep}")

    for ep in range(start_ep, a.epochs):

        if torch.cuda.is_available():

            torch.cuda.reset_peak_memory_stats()
        model.train(); t0 = time.time(); run = {}; nb = 0
        opt.zero_grad(set_to_none=True)
        for step, (s1, s2, _, _) in enumerate(train_dl):
            lr = cosine_lr(it, total_it, warm_it, a.lr_start, a.lr_peak, a.lr_final)
            for g in opt.param_groups:
                g["lr"] = lr
            m = cfg.ema[0] + (cfg.ema[1] - cfg.ema[0]) * it / max(total_it, 1) if a.ema_target else None
            s1, s2 = s1.to(dev, non_blocking=True), s2.to(dev, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=dev == "cuda"):
                loss, log = crit(model(s1, s2))
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {ep}, step {step}; stopping before updating weights")
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if a.ema_target:
                    if not hasattr(model, "update_ema"):
                        raise SystemExit("--ema-target is only wired up for base RS-JEPA, not --bida/--dual/--views")
                    model.update_ema(m)
            it += 1; nb += 1
            for k, v in log.items():
                run[k] = run.get(k, 0.0) + float(v)
            if step % max(1, ipe // 4) == 0:
                p_val = log.get('pred', log.get('pred_aa', 0.0))
                c_val = log.get('l_cross', log.get('cross', 0.0))
                u_val = log.get('l_uni', log.get('uni', 0.0))
                s_val = log.get('sigreg', log.get('sig'))
                tot_val = log.get('total', 0.0)
                print(f"ep{ep:3d} {step:5d}/{ipe} total {float(tot_val):.4f} "
                      f"pred {float(p_val):.4f} cross {float(c_val):.4f} "
                      f"uni {float(u_val):+.4f} "
                      + (f"sig {float(s_val):.4f} " if s_val is not None else "")
                      + (f"view {float(log['l_view']):.4f} " if 'l_view' in log else "")
                      + (f"vpred {float(log['l_view_pred']):.4f} " if 'l_view_pred' in log else "")
                      + f"lr {lr:.2e}", flush=True)
            if a.smoke and step >= min(2 * accum, ipe - 1):
                print(f"\nsmoke passed | peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
                return
        rec = {"epoch": ep, "secs": round(time.time() - t0, 1),
               "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0,
               "params_m": round(sum(q.numel() for q in model.parameters()) / 1e6, 2),
               **{k: round(v / max(nb, 1), 5) for k, v in run.items()}}
        print("== epoch " + " ".join(f"{k} {v}" for k, v in rec.items()))
        # atomic save: a crash mid-write must never leave a truncated last.pt
        # (a killed session once left a 0-byte checkpoint that could not resume)
        tmp = out / "last.pt.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep, "it": it, "cfg": cfg.__dict__}, tmp)
        os.replace(tmp, out / "last.pt")
        # log only AFTER the weights are safely on disk, so a logged epoch always
        # has a checkpoint behind it
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
