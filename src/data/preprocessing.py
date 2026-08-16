"""SAR / EO preprocessing utilities (Section 5).

Documents explicitly:
  - band selection (VV/VH for SAR; a documented RGB+NIR subset for EO, not all 13 bands)
  - normalization (SAR: dB clipping + z-score; EO: reflectance scaling + z-score)
  - resizing to a fixed patch size
  - missing-value handling (NaN -> band mean)

Real SEN12MS SAR/EO tiles are GeoTIFFs read via rasterio on Kaggle. This module also
exposes pure-numpy functions that operate on already-loaded arrays, so the same
normalization logic is unit-testable on CPU without rasterio/GDAL installed.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

SAR_DB_CLIP_RANGE: Tuple[float, float] = (-30.0, 5.0)   # typical Sentinel-1 VV/VH dB range
EO_REFLECTANCE_SCALE: float = 10000.0                    # Sentinel-2 L2A digital-number -> reflectance


def load_geotiff(path: str) -> np.ndarray:
    """Load a GeoTIFF as a (C, H, W) float32 array. Requires rasterio (Kaggle-side only)."""
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    return arr


def fill_missing(arr: np.ndarray) -> np.ndarray:
    """Replace NaN / Inf per-band with that band's finite mean (Section 5: 'missing values')."""
    arr = arr.copy()
    for c in range(arr.shape[0]):
        band = arr[c]
        mask = ~np.isfinite(band)
        if mask.any():
            fill_value = np.nanmean(band[np.isfinite(band)]) if np.isfinite(band).any() else 0.0
            band[mask] = fill_value
            arr[c] = band
    return arr


def preprocess_sar(arr: np.ndarray, bands: Sequence[str], all_bands: Sequence[str] = ("VV", "VH")) -> np.ndarray:
    """Select SAR bands, clip to a realistic dB range, and min-max scale to [0, 1].

    arr: (C, H, W) raw dB-scale SAR array with channel order == all_bands.
    """
    arr = fill_missing(arr)
    idx = [all_bands.index(b) for b in bands]
    arr = arr[idx]
    lo, hi = SAR_DB_CLIP_RANGE
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return arr.astype(np.float32)


def preprocess_eo(arr: np.ndarray, bands: Sequence[str], all_bands: Sequence[str]) -> np.ndarray:
    """Select EO bands and scale digital numbers to reflectance in [0, 1].

    arr: (C, H, W) raw Sentinel-2 digital-number array with channel order == all_bands.
    """
    arr = fill_missing(arr)
    idx = [all_bands.index(b) for b in bands]
    arr = arr[idx]
    arr = np.clip(arr / EO_REFLECTANCE_SCALE, 0.0, 1.0)
    return arr.astype(np.float32)


def normalize(arr: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Per-channel z-score normalization. arr: (C, H, W)."""
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)
    return (arr - mean_arr) / std_arr


def resize_chw(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize a (C, H, W) array to (C, size, size) using skimage (CPU-safe, no torch needed)."""
    from skimage.transform import resize as sk_resize

    c = arr.shape[0]
    out = np.zeros((c, size, size), dtype=np.float32)
    for i in range(c):
        out[i] = sk_resize(arr[i], (size, size), preserve_range=True, anti_aliasing=True)
    return out
