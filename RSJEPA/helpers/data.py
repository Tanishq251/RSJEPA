"""BEN-14K paired Sentinel-1 / Sentinel-2 loading.

Channel order was verified against the per-band GeoTIFFs rather than assumed
(the earlier Kaggle release stored bands alphabetically, which silently swaps
polarisations and three S2 bands). Verified order for this release:

    S1 all.npy : [VV, VH]                       ch0 mean -11.20 dB, ch1 -17.30 dB
    S2 all.npy : [B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12]

Statistics live in data/benv1_14k/norm_stats.json, computed over all 14,832 pairs.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = _REPO_ROOT / "data" / "benv1_14k"
CBRSIR_ROOT = _REPO_ROOT / "data" / "cbrsir_vs"


class BEN14K(Dataset):
    """Returns (s1, s2, labels19, index). Images are z-scored per band and
    resized to `img_size` (patch-16 needs 224; 120 is native)."""

    def __init__(self, root: str | Path = ROOT, img_size: int = 256,
                 split: str | None = None, augment: bool = False):
        self.root = Path(root)
        self.img_size = img_size
        self.augment = augment
        # Official BEN-14K: the 13,683 BigEarthNet-v2 Serbia/summer pairs with the
        # ~52/24/24 split, and v2 labels. `benv1_14k` on disk is v1 and carries
        # 1,149 extra patches that are not part of the benchmark at all.
        df = pd.read_parquet(self.root / "ben14k_official_splits.parquet")
        if split is not None:
            # "validation+test" = union of splits (never overlapping), e.g. a larger
            # evaluation gallery made only of images no model trained on
            df = df[df.split.isin(split.split("+"))]
        self.df = df.reset_index(drop=True)
        self.labels = np.array(self.df.onehot_19.tolist(), dtype=np.float32)

        st = json.load(open(self.root / "norm_stats.json"))
        f = lambda k: torch.tensor(st[k], dtype=torch.float32)[:, None, None]
        self.s1_mean, self.s1_std = f("s1_mean"), f("s1_std")
        self.s2_mean, self.s2_std = f("s2_mean"), f("s2_std")
        self.s1_bands, self.s2_bands = st["s1_bands"], st["s2_bands"]

        # Fast RAM cache check
        cache_candidates = [
            self.root / f"cache_{split}_{img_size}.pt",
            self.root / "cache_trainval_120.pt" if (split in ("train+validation", "trainval") and img_size == 120) else None,
        ]
        self.cached = False
        for c in cache_candidates:
            if c and c.exists():
                print(f"[BEN14K] Loading preloaded RAM cache: {c.name} (blazing fast zero-disk training)...", flush=True)
                ck = torch.load(c, weights_only=True)
                self.s1_cache = ck["s1"].share_memory_()
                self.s2_cache = ck["s2"].share_memory_()
                self.cached = True
                break

    def __len__(self):
        return len(self.df)

    def _prep(self, arr, mean, std):
        x = torch.from_numpy(np.ascontiguousarray(arr)).float()
        x = (x - mean) / std
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x[None], size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)[0]
        return x

    def __getitem__(self, i):
        if self.cached:
            s1 = self.s1_cache[i]
            s2 = self.s2_cache[i]
        else:
            r = self.df.iloc[i]
            s1 = self._prep(np.load(self.root / "s1" / r.s1_rel_path), self.s1_mean, self.s1_std)
            s2 = self._prep(np.load(self.root / "s2" / r.s2_rel_path), self.s2_mean, self.s2_std)
        if self.augment:
            if torch.rand(()) < 0.5:
                s1, s2 = torch.flip(s1, [-1]), torch.flip(s2, [-1])
            k = int(torch.randint(0, 4, ()))
            if k:
                s1, s2 = torch.rot90(s1, k, [-2, -1]), torch.rot90(s2, k, [-2, -1])
        lbl = self.labels[i]
        return s1, s2, torch.from_numpy(lbl) if isinstance(lbl, np.ndarray) else lbl, i


def loaders(batch_size=32, img_size=256, workers=8, augment=True, root=ROOT,
            train_split="train", eval_split="test"):
    """Official protocol: pretrain on the TRAIN split only; retrieval is measured
    inside the TEST split, which serves as both the query set and the gallery."""
    # never pretrain on the test split: retrieval is measured there
    assert "test" not in train_split.split("+"), f"train_split {train_split!r} includes the test split"
    DatasetClass = dataset_for_root(root)
    train = DatasetClass(root, img_size, split=train_split, augment=augment)
    evald = DatasetClass(root, img_size, split=eval_split, augment=False)
    mk = lambda ds, sh: torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=sh, num_workers=workers,
        pin_memory=True, drop_last=sh, persistent_workers=workers > 0)
    return mk(train, True), mk(evald, False)


class CBRSIRVS(Dataset):
    """CBRSIR-VS paired loader.

    The models expect both modalities on one patch grid.  CBRSIR-VS stores
    SAR at 64x64 and RGB at 256x256, so SAR is resized to ``img_size`` here.
    Return order remains S1/S2 = SAR/optical, matching BEN's convention.

    Per-channel z-scoring (2026-09-22): earlier versions only divided by 255,
    leaving SAR at mean 0.52/std 0.11 and RGB channels at std ~0.15-0.19 in
    [0,1] -- much lower contrast than BEN14K's per-band z-scored input (mean
    0, std ~1). This is not something the paper specifies either way (no
    normalization/preprocessing procedure is stated anywhere in the PDF for
    any of its three datasets), so it was our own choice, and the low-contrast
    version is the one that trained under a shared trunk that collapsed to
    near-rank-1 on both modalities (PAPER_VS_CODE.md). Stats are computed by
    `data.py`'s own norm_stats.json if present (see the __main__ block at the
    end of this file to regenerate), else raw [0,1] scaling as before.
    """

    CLASSES = sorted([
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
        "Industrial", "Pasture", "PermanentCrop", "Residential",
        "River", "SeaLake",
    ])

    def __init__(self, root=CBRSIR_ROOT, img_size=256, split=None, augment=False,
                 use_cache=True):
        self.root = Path(root)
        self.img_size = img_size
        self.augment = augment
        stats_file = self.root / "norm_stats.json"
        if stats_file.exists():
            st = json.load(open(stats_file))
            f = lambda k: torch.tensor(st[k], dtype=torch.float32)[:, None, None]
            self.s1_mean, self.s1_std = f("s1_mean"), f("s1_std")
            self.s2_mean, self.s2_std = f("s2_mean"), f("s2_std")
        else:
            self.s1_mean = self.s2_mean = torch.tensor(0.0)
            self.s1_std = self.s2_std = torch.tensor(1.0)
        split_file = self.root / "split_70_30.csv"
        df = pd.read_csv(split_file if split_file.exists() else self.root / "index.csv")
        if "split" in df and split is not None:
            wanted = {"test": "eval", "validation": "eval"}.get(split, split)
            df = df[df.split.isin(wanted.split("+"))]
        # The supplied split file has blank class cells for its train rows.
        df["class"] = df["class"].fillna(df["optical"].str.split("/").str[-2])
        self.df = df.reset_index(drop=True)
        class_to_i = {name: i for i, name in enumerate(self.CLASSES)}
        self.labels = np.zeros((len(self.df), len(self.CLASSES)), dtype=np.float32)
        for i, name in enumerate(self.df["class"]):
            self.labels[i, class_to_i[name]] = 1.0

        self.cached = False
        cache_split = {"eval": "test", "validation": "test"}.get(split, split)
        cache_path = self.root / f"cache_{cache_split}_{img_size}.pt"
        if use_cache and split in ("train", "eval", "test", "validation") and cache_path.exists():
            ck = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=False)  # each job loads its own private copy into RAM
            expected = ((len(self.df), 1, img_size, img_size),
                        (len(self.df), 3, img_size, img_size), self.labels.shape)
            if any(tuple(ck[k].shape) != shape for k, shape in
                   zip(("s1", "s2", "labels"), expected)):
                raise ValueError(f"Cache shape mismatch: {cache_path}")
            if not torch.equal(ck["labels"], torch.from_numpy(self.labels)):
                raise ValueError(f"Cache labels do not match split order: {cache_path}")
            if "sar_paths" in ck and ck["sar_paths"] != self.df.sar.tolist():
                raise ValueError(f"Cache SAR order mismatch: {cache_path}")
            if "optical_paths" in ck and ck["optical_paths"] != self.df.optical.tolist():
                raise ValueError(f"Cache optical order mismatch: {cache_path}")
            self.s1_cache, self.s2_cache = ck["s1"], ck["s2"]
            self.cached = True
            print(f"[CBRSIRVS] Using {cache_path.name}: {len(self.df):,} cached pairs", flush=True)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _read(path, mode):
        arr = np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[None]
        else:
            arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr)

    def __getitem__(self, i):
        if self.cached:
            s1, s2 = self.s1_cache[i], self.s2_cache[i]
        else:
            r = self.df.iloc[i]
            # S1/S2 = SAR/optical, as in the BEN loader.
            s1 = (self._read(self.root / r.sar, "L") - self.s1_mean) / self.s1_std
            s2 = (self._read(self.root / r.optical, "RGB") - self.s2_mean) / self.s2_std
            s1 = F.interpolate(s1[None], size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False)[0]
            s2 = F.interpolate(s2[None], size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False)[0]
        if self.augment:
            if torch.rand(()) < 0.5:
                s1, s2 = torch.flip(s1, [-1]), torch.flip(s2, [-1])
            k = int(torch.randint(0, 4, ()))
            if k:
                s1, s2 = torch.rot90(s1, k, [-2, -1]), torch.rot90(s2, k, [-2, -1])
        return s1, s2, torch.from_numpy(self.labels[i]), i


class DSRSID(Dataset):
    """DSRSID paired loader (panchromatic / 4-band multispectral, 8 single labels), from the native-120 caches
    built by cache_dsrsid.py (full repo): S1 = PAN (1 channel), S2 = MUL (4 channels), per-channel z-scored, stratified
    70/30 split (split_70_30.csv; "eval" is the 30%). Same interface and augmentation as CBRSIRVS."""

    CLASSES = sorted(["aquafarm", "cloud", "forest", "high building", "low building", "farm land", "river", "water"])

    def __init__(self, root, img_size=120, split=None, augment=False, use_cache=True):
        self.root = Path(root); self.img_size = img_size; self.augment = augment
        df = pd.read_csv(self.root / "split_70_30.csv")
        wanted = {"test": "eval", "validation": "eval"}.get(split, split)
        if split is not None:
            df = df[df.split.isin(wanted.split("+"))]
        self.df = df.reset_index(drop=True)
        cls = {c: i for i, c in enumerate(self.CLASSES)}
        self.labels = np.zeros((len(self.df), len(self.CLASSES)), dtype=np.float32)
        self.labels[np.arange(len(self.df)), [cls[c] for c in self.df["class"]]] = 1.0
        name = {"train": "cache_train_120.pt", "eval": "cache_test_120.pt"}.get(wanted)
        if name is None or not (self.root / name).exists():
            raise ValueError("DSRSID is only available through its 120 px caches (build them with cache_dsrsid.py from the full repo); "
                             f"split={split!r}, img_size={img_size}")
        # other input sizes (e.g. 224 for patch-16 baselines) are bilinearly resized from the native-120 cache per item,
        # as the BEN-14K loader does from its native 120 px patches
        ck = torch.load(self.root / name, map_location="cpu", weights_only=True, mmap=False)  # each job loads its own private copy into RAM
        if not torch.equal(ck["labels"], torch.from_numpy(self.labels)) or ck["ids"] != self.df["id"].tolist():
            raise ValueError(f"Cache order does not match split_70_30.csv: {name}")
        self.s1_cache, self.s2_cache = ck["s1"], ck["s2"]
        self.cached = True
        print(f"[DSRSID] Using {name}: {len(self.df):,} cached pairs", flush=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        s1, s2 = self.s1_cache[i], self.s2_cache[i]
        if self.img_size != s1.shape[-1]:
            s1 = F.interpolate(s1[None].float(), size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)[0]
            s2 = F.interpolate(s2[None].float(), size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)[0]
        if self.augment:
            if torch.rand(()) < 0.5:
                s1, s2 = torch.flip(s1, [-1]), torch.flip(s2, [-1])
            k = int(torch.randint(0, 4, ()))
            if k:
                s1, s2 = torch.rot90(s1, k, [-2, -1]), torch.rot90(s2, k, [-2, -1])
        return s1, s2, torch.from_numpy(self.labels[i]), i


def dataset_for_root(root):
    if (Path(root) / "split_70_30.csv").exists():
        cols = pd.read_csv(Path(root) / "split_70_30.csv", nrows=0).columns
        return DSRSID if "index" in cols and "optical" not in cols else CBRSIRVS
    return BEN14K


CBRSIR = CBRSIRVS

__all__ = ["BEN14K", "CBRSIR", "CBRSIRVS", "DSRSID", "loaders", "ROOT", "CBRSIR_ROOT", "dataset_for_root"]
