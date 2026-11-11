"""
BEN-14K cross-modal retrieval evaluation (F1@K), as used by X-JEPA.

Evaluated as RETRIEVAL, not 19-class classification: a retrieved item counts as
relevant to the degree its multi-label set overlaps the query's. Classes with
zero support (Agro-forestry areas, Coastal wetlands, Marine waters in any
Serbia-derived subset) never enter an intersection and are therefore inert --
they cannot affect the score.

X-JEPA is vague ("F1 ... computed over the top-5 retrieved neighbors"), but
CR-JEPA (arXiv:2606.00706), which reuses the identical benchmark, pins it down:

    "The final F1@K is obtained by averaging the item-level F1 scores over the
     top-K retrieved images and then over all queries."
    "Trivial self-matches are excluded when the query and gallery contain the
     same indexed sample."
    P(q,r) = |Yq n Yr| / |Yr|
    R(q,r) = |Yq n Yr| / |Yq|
    F1(q,r) = 2PR / (P + R + eps)

=> "per_item" is the OFFICIAL protocol and is the default. The other two exist
only to diagnose a mismatched reproduction.

    per_item  (i)   OFFICIAL. F1 of each retrieved item vs query, meaned over K.
    pooled    (ii)  one P/R over all K retrieved at once, then F1.
    union     (iii) union of the K label sets treated as a predicted label set.

BEN-14K IS the dataset in data/benv1_14k: CR-JEPA describes it as "14,832 paired
Sentinel-1 SAR and Sentinel-2 multispectral image pairs acquired over Serbia
during summer" -- an exact match for our 14,832 Serbia pairs. So the 19-class
taxonomy with only 16 populated is a property of the benchmark itself, not a
defect in our copy. The baselines below are therefore computed on the real
benchmark data and are directly comparable to published numbers.

Reference points on BEN-14K (14,832 patches, 2.74 labels/patch), variant (i):
    random retrieval ........ 35.86%   <- below this, the code is wrong
    X-JEPA  S1->S2 .......... 61.23%   CR-JEPA 75.82%
    X-JEPA  S2->S2 .......... 82.65%   CR-JEPA 82.87%
    oracle ceiling .......... 99.47%
Interpret a score by where it falls in [35.86, 99.47], not against 0.
"""
from __future__ import annotations
import numpy as np

VARIANTS = ("per_item", "pooled", "union")


def _f1(inter, n_pred, n_true):
    p = inter / np.maximum(n_pred, 1)
    r = inter / np.maximum(n_true, 1)
    return np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)


def f1_at_k(query_labels, gallery_labels, ranking, k=5, variant="per_item"):
    """
    query_labels   (Q,C) bool/0-1 multi-hot
    gallery_labels (N,C) bool/0-1 multi-hot
    ranking        (Q,k) gallery indices, already ordered best-first
    returns        scalar mean F1@k
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    q = np.asarray(query_labels, dtype=bool)
    g = np.asarray(gallery_labels, dtype=bool)
    r = g[ranking[:, :k]]                       # (Q,k,C)
    qe = q[:, None, :]                          # (Q,1,C)
    inter = (qe & r).sum(-1)                    # (Q,k)
    n_q, n_r = q.sum(-1), r.sum(-1)

    if variant == "per_item":
        return _f1(inter, n_r, n_q[:, None]).mean(1).mean()
    if variant == "pooled":
        return _f1(inter.sum(1), n_r.sum(1), n_q * r.shape[1]).mean()
    u = r.any(1)                                # (Q,C)
    return _f1((q & u).sum(-1), u.sum(-1), n_q).mean()


def _hist(x, eps=1e-8):
    """Map real embeddings to histogram-like vectors so chi-square is defined.

    chi2 needs non-negative inputs; our embeddings are l2-normalised and have
    negative components, where (a-b)^2/(a+b) is undefined. Shift each vector by
    its own minimum, then l1-normalise -- the standard treatment, and it leaves
    the relative geometry of a single vector intact.
    """
    x = np.asarray(x, np.float32)
    x = x - x.min(axis=1, keepdims=True)
    return x / np.maximum(x.sum(axis=1, keepdims=True), eps)


def _chi2_block(Q, G, eps=1e-8):
    """0.5 * sum_i (q_i - g_i)^2 / (q_i + g_i), computed for a block of queries."""
    d = Q[:, None, :] - G[None, :, :]
    s = Q[:, None, :] + G[None, :, :] + eps
    return 0.5 * np.sum(d * d / s, axis=-1)


def retrieve(query_emb, gallery_emb, k=5, exclude_self=False, block=1024, metric="cosine"):
    """Ranking by cosine similarity (default) or chi-square distance.

    metric="chi2" maps both sides through _hist() first, since chi-square is a
    histogram measure. exclude_self=True for same-modality retrieval, which
    assumes query i corresponds to gallery row i.
    """
    if metric == "chi2":
        Q, G, block = _hist(query_emb), _hist(gallery_emb), min(block, 128)
        out = np.empty((len(Q), k), dtype=np.int64)
        for s0 in range(0, len(Q), block):
            e = min(s0 + block, len(Q))
            d = _chi2_block(Q[s0:e], G)
            if exclude_self:
                d[np.arange(e - s0), np.arange(s0, e)] = np.inf
            idx = np.argpartition(d, k, axis=1)[:, :k]
            ord_ = np.argsort(np.take_along_axis(d, idx, 1), axis=1)
            out[s0:e] = np.take_along_axis(idx, ord_, 1)
        return out
    if metric != "cosine":
        raise ValueError(f"metric must be 'cosine' or 'chi2', got {metric!r}")
    Q = _l2(np.asarray(query_emb, np.float32))
    G = _l2(np.asarray(gallery_emb, np.float32))
    out = np.empty((len(Q), k), dtype=np.int64)
    for s in range(0, len(Q), block):
        e = min(s + block, len(Q))
        sim = Q[s:e] @ G.T
        if exclude_self:
            sim[np.arange(e - s), np.arange(s, e)] = -np.inf
        idx = np.argpartition(-sim, k, axis=1)[:, :k]
        ord_ = np.argsort(-np.take_along_axis(sim, idx, 1), axis=1)
        out[s:e] = np.take_along_axis(idx, ord_, 1)
    return out


def _l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def evaluate_all(s1_emb, s2_emb, labels, k=5, variant="per_item"):
    """The four directions of Table 1. Row i of every array is the same patch,
    so s1_emb[i], s2_emb[i] and labels[i] describe one S1/S2 pair."""
    res = {}
    for name, qe, ge, same in (
        ("S1->S1", s1_emb, s1_emb, True),
        ("S2->S2", s2_emb, s2_emb, True),
        ("S1->S2", s1_emb, s2_emb, False),
        ("S2->S1", s2_emb, s1_emb, False),
    ):
        rank = retrieve(qe, ge, k=k, exclude_self=same)
        res[name] = float(f1_at_k(labels, labels, rank, k, variant)) * 100
    return res


def baselines(labels, k=5, variant="per_item", n_queries=2000, seed=0):
    """Random floor and oracle ceiling for the given label set."""
    rng = np.random.default_rng(seed)
    Y = np.asarray(labels, dtype=bool)
    n = len(Y)
    qi = rng.choice(n, min(n_queries, n), replace=False)
    rand = f1_at_k(Y[qi], Y, rng.choice(n, (len(qi), k)), k, variant)

    Yf = Y.astype(np.float32)
    inter = Yf[qi] @ Yf.T
    F = _f1(inter, Yf.sum(1)[None, :], Yf.sum(1)[qi][:, None]).astype(np.float32)
    F[np.arange(len(qi)), qi] = -1
    top = np.argpartition(-F, k, axis=1)[:, :k]
    oracle = f1_at_k(Y[qi], Y, top, k, variant)
    return {"random": float(rand) * 100, "oracle": float(oracle) * 100}

