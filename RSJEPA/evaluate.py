#!/usr/bin/env python
"""Evaluate a CR-JEPA or RS-JEPA checkpoint with CR-JEPA's inference protocol
(paper 3.8): same-modal retrieval uses the UNIFIED head, cross-modal uses the
CROSS head.

The model class is taken from the checkpoint -- RS-JEPA configs carry uni_mode,
CR-JEPA's do not -- so one script serves both and the two are guaranteed to be
scored by identical code."""
from __future__ import annotations
import argparse
import dataclasses
import numpy as np, torch

from RSJEPA import (
    RSJEPA, RSJEPAConfig, RSJEPAShared, RSJEPASharedConfig,
    RSJEPADual, RSJEPADualConfig,
    RSJEPABiDA, RSJEPABiDAConfig
)
from RSJEPA.data import BEN14K, dataset_for_root, ROOT
from RSJEPA.helpers.retrieval import retrieve, f1_at_k, baselines

PAPER_X = {"S1->S1": 72.98, "S2->S2": 82.65, "S1->S2": 61.23, "S2->S1": 63.73}
PAPER_CR = {"S1->S1": 75.11, "S2->S2": 82.87, "S1->S2": 75.82, "S2->S1": 75.40}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt"); p.add_argument("--split", default="test")
    p.add_argument("--batch", type=int, default=256)  # 4.3
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--metric", default="cosine", choices=["cosine", "chi2"],
                   help="retrieval metric: cosine (papers' protocol) or chi-square")
    p.add_argument("--readout", default="", choices=["", "concat", "avg"],
                   help="RS-JEPA-dual only: concat (2x dim) or avg (dim, comparable to other runs)")
    p.add_argument("--data-root", default=str(ROOT))
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    # CR-JEPA configs also carry uni_mode now (the --uni-mode ablation switch
    # added 2026-09-24), so uni_mode alone no longer distinguishes RS-JEPA from
    # CR-JEPA -- sigreg_scale_b/head_ln are CR-JEPA-only fields RSJEPAConfig
    # never has, so exclude those first.
    is_rs = "uni_mode" in ck["cfg"] and "sigreg_scale_b" not in ck["cfg"]
    if "view_depth" in ck["cfg"]:
        Model, Cfg, tag = RSJEPABiDA, RSJEPABiDAConfig, "RSJEPA"
    elif "readout" in ck["cfg"]:
        Model, Cfg, tag = RSJEPADual, RSJEPADualConfig, "RS-JEPA-dual"
    elif "lam_view" in ck["cfg"]:
        Model, Cfg, tag = RSJEPAShared, RSJEPASharedConfig, "RS-JEPA-shared"
    elif is_rs:
        Model, Cfg, tag = RSJEPA, RSJEPAConfig, "RS-JEPA"
    else:
        raise SystemExit("Not an RS-JEPA checkpoint")
    if hasattr(Cfg, "from_ckpt"):
        cfg = Cfg.from_ckpt(ck["cfg"])
    else:
        # older checkpoints (esp. RSJEPABiDA) carry ablation-only fields that were
        # later removed once they were confirmed not to beat the final model
        # (docs/results/ABLATIONS.md) -- drop anything the current dataclass no
        # longer has; every dropped field was already at the value the removed
        # code hardwired anyway, so this reconstructs the identical model.
        known = {f.name for f in dataclasses.fields(Cfg)}
        cfg = Cfg(**{k: v for k, v in ck["cfg"].items() if k in known})
    m = Model(cfg).to(dev); m.load_state_dict(ck["model"], strict=False); m.eval()
    n_par = sum(q.numel() for q in m.parameters())
    # load test images at the size the model was trained on (224 default, 128 for the img128 run)
    ds = dataset_for_root(a.data_root)(a.data_root, img_size=cfg.img_size,
                                       split=a.split, augment=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True)

    E = {k: ([], []) for k in ("uni", "cross")}; Y = []
    for s1, s2, y, _ in dl:
        s1, s2 = s1.to(dev), s2.to(dev)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=dev == "cuda"):
            for mode in ("uni", "cross"):
                kw = {"readout": a.readout} if (Model is RSJEPADual and a.readout) else {}
                e1, e2 = m.embed(s1, s2, mode=mode, **kw)
                E[mode][0].append(e1.float().cpu()); E[mode][1].append(e2.float().cpu())
        Y.append(y)
    Y = torch.cat(Y).numpy().astype(bool)
    emb = {k: (torch.cat(v[0]).numpy(), torch.cat(v[1]).numpy()) for k, v in E.items()}

    res = {}
    for name, mode, qi, gi, same in (("S1->S1", "uni", 0, 0, True), ("S2->S2", "uni", 1, 1, True),
                                     ("S1->S2", "cross", 0, 1, False), ("S2->S1", "cross", 1, 0, False)):
        q, g = emb[mode][qi], emb[mode][gi]
        rank = retrieve(q, g, k=5, exclude_self=same, metric=a.metric)
        res[name] = float(f1_at_k(Y, Y, rank, 5, "per_item")) * 100

    ro = (a.readout or getattr(cfg, "readout", "")) if Model is RSJEPADual else ""
    print(f"\n{tag}{' readout=' + ro if ro else ''} {a.ckpt} (epoch {ck.get('epoch','?')}) | {n_par/1e6:.2f} M params"
          + (f" | uni_mode={cfg.uni_mode}" if is_rs else "")
          + f" | {len(Y):,} pairs, '{a.split}' split")
    print(f"head routing per paper 3.8: same-modal -> unified, cross-modal -> cross | metric={a.metric}\n")
    print(f"{'direction':10s} {'ours':>8s} {'CR-JEPA':>9s} {'X-JEPA':>8s}")
    for k in PAPER_CR:
        print(f"{k:10s} {res[k]:7.2f}% {PAPER_CR[k]:8.2f}% {PAPER_X[k]:7.2f}%")
    # 3.6 says SIGReg exists to "mitigate representation collapse"; effective rank
    # of the retrieval space is the direct check that it did.
    print()
    for name, mode in (("unified", "uni"), ("cross-modal", "cross")):
        for mi, mn in ((0, "S1"), (1, "S2")):
            e = emb[mode][mi]
            sv = np.linalg.svd(np.cov(e.T), compute_uv=False)
            sv = sv / sv.sum()
            er = float(np.exp(-(sv * np.log(sv + 1e-12)).sum()))
            print(f"{name:12s} head / {mn}: effective rank {er:6.1f} / {e.shape[1]}")

    b = baselines(Y, k=5, variant="per_item")
    print(f"\nmean {np.mean(list(res.values())):.2f}%   floor {b['random']:.2f}%  ceiling {b['oracle']:.2f}%")
    print(f"same-cross gap: S1 {res['S1->S1']-res['S1->S2']:+.2f}  S2 {res['S2->S2']-res['S2->S1']:+.2f}")


if __name__ == "__main__":
    main()
