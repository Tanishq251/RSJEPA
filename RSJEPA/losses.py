"""RS-JEPA losses: base, dual, and BiDA."""
from __future__ import annotations

from .helpers.losses import (
    RSJEPALoss,
    RSJEPASharedLoss,
    RSJEPADualLoss,
    RSJEPABiDALoss,
    UNI_MODES,
    info_nce,
    sigreg,
    synergy_hinge,
)

__all__ = [
    "RSJEPALoss",
    "RSJEPASharedLoss",
    "RSJEPADualLoss",
    "RSJEPABiDALoss",
    "UNI_MODES",
    "info_nce",
    "sigreg",
    "synergy_hinge",
]
