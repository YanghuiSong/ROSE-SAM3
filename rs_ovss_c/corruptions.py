"""The six deterministic image-degradation operators used in the paper."""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
from PIL import Image

CORRUPTIONS = ("gaussian", "motion_blur", "downsample", "fog", "shadow", "stripe")
ALIASES = {
    "none": "none", "clean": "none",
    "gaussian_noise": "gaussian", "motion": "motion_blur",
    "resolution": "downsample", "lowres": "downsample",
    "haze": "fog", "illumination": "shadow",
    "banding": "stripe", "sensor_stripe": "stripe",
}

def normalize_corruption(name: str | None) -> str:
    key = str(name or "none").strip().lower()
    return ALIASES.get(key, key)


def iter_corruptions(value: str | Iterable[str] | None = None) -> list[str]:
    if value is None:
        value = " ".join(CORRUPTIONS)
    if isinstance(value, str):
        items = value.replace(",", " ").split()
    else:
        items = list(value)
    return [normalize_corruption(x) for x in items if normalize_corruption(x) != "none"]


def _rng(seed: int, key: str | None = None) -> np.random.Generator:
    if key:
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        seed = (int(seed) + int(digest[:8], 16)) % (2**32)
    return np.random.default_rng(seed)


def _to_uint8(arr) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def corrupt_array(arr, corruption_type: str | None, severity: int = 3,
                  seed: int = 3407, key: str | None = None) -> np.ndarray:
    ctype = normalize_corruption(corruption_type)
    if ctype == "none" or int(severity) <= 0:
        return _to_uint8(arr)
    if ctype not in CORRUPTIONS:
        raise ValueError(f"Unsupported corruption '{corruption_type}'. "
                         f"Supported: {', '.join(CORRUPTIONS)}")

    severity = min(5, max(1, int(severity)))
    arr = _to_uint8(arr)
    x = arr.astype(np.float32)
    rng = _rng(seed, key)

    if ctype == "gaussian":
        sigma = [8, 12, 18, 25, 35][severity - 1]
        return _to_uint8(x + rng.normal(0, sigma, x.shape))

    if ctype == "motion_blur":
        size = [3, 5, 7, 9, 11][severity - 1]
        pad = size // 2
        padded = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode="edge")
        acc = np.zeros_like(x)
        for offset in range(size):
            acc += padded[:, offset:offset + x.shape[1], :]
        return _to_uint8(acc / float(size))

    if ctype == "downsample":
        scale = [0.75, 0.60, 0.50, 0.40, 0.32][severity - 1]
        img = Image.fromarray(arr)
        w, h = img.size
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
        return np.array(small.resize((w, h), Image.Resampling.BILINEAR))

    if ctype == "fog":
        alpha = [0.10, 0.16, 0.24, 0.32, 0.42][severity - 1]
        veil = np.full_like(x, 235.0)
        return _to_uint8((1.0 - alpha) * x + alpha * veil)

    if ctype == "shadow":
        yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
        cx = arr.shape[1] * [0.30, 0.35, 0.40, 0.45, 0.50][severity - 1]
        cy = arr.shape[0] * [0.35, 0.40, 0.45, 0.50, 0.55][severity - 1]
        rx = arr.shape[1] * 0.42
        ry = arr.shape[0] * 0.32
        mask = (((xx - cx) / (rx + 1e-6)) ** 2 + ((yy - cy) / (ry + 1e-6)) ** 2) < 1
        factor = [0.78, 0.68, 0.58, 0.48, 0.38][severity - 1]
        out = x.copy()
        out[mask] *= factor
        return _to_uint8(out)

    if ctype == "stripe":
        strength = [0.08, 0.12, 0.18, 0.24, 0.32][severity - 1]
        period = max(4, int(arr.shape[1] / [18, 16, 14, 12, 10][severity - 1]))
        cols = np.arange(arr.shape[1], dtype=np.float32)
        gain = 1.0 - strength * (0.5 + 0.5 * np.sin(2.0 * np.pi * cols / period))
        return _to_uint8(x * gain[None, :, None])

    return arr


def corrupt_pil(image: Image.Image, corruption_type: str | None, severity: int = 3,
                seed: int = 3407, key: str | None = None) -> Image.Image:
    arr = np.array(image.convert("RGB"))
    return Image.fromarray(corrupt_array(arr, corruption_type, severity, seed, key))


