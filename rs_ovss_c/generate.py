#!/usr/bin/env python3
"""Materialize deterministic RS-OVSS-C images without running a model."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from rs_ovss_c.corruptions import CORRUPTIONS, corrupt_pil, iter_corruptions


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def image_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--corruptions", default=" ".join(CORRUPTIONS))
    parser.add_argument("--severity", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = image_paths(args.source)
    if not paths:
        raise FileNotFoundError(f"No supported images found under {args.source}")
    root = args.source if args.source.is_dir() else args.source.parent

    written = 0
    for corruption in iter_corruptions(args.corruptions):
        for source_path in paths:
            relative = source_path.relative_to(root)
            target = args.output / f"{corruption}_s{args.severity}" / relative
            target = target.with_suffix(".png")
            if target.exists() and not args.overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            image = Image.open(source_path).convert("RGB")
            output = corrupt_pil(
                image,
                corruption,
                severity=args.severity,
                seed=args.seed,
                key=relative.as_posix(),
            )
            output.save(target)
            written += 1
    print(f"[OK] wrote {written} images from {len(paths)} sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
