#!/usr/bin/env python3
"""Read-only integrity check; missing PromptBanks require a separate DAG run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASETS = (
    "openearthmap", "udd5", "vdd", "uavid", "vaihingen", "potsdam",
    "loveda", "isaid",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bank(path: Path, expected_classes: int) -> int:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = [item.strip() for item in line.split(",") if item.strip()]
        if not row:
            raise ValueError(f"Empty PromptBank row at {path}:{line_number}")
        rows.append(row)
    if len(rows) != expected_classes:
        raise ValueError(
            f"PromptBank class-row count mismatch: expected {expected_classes}, found {len(rows)}"
        )
    return sum(map(len, rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    prompt_root = repo / "calibrated_prompts"
    final_path = prompt_root / "final_promptbank" / f"cls_{args.dataset}.txt"
    recovery_root = prompt_root / "recovery_inputs"
    anchor_path = recovery_root / "ov3_anchors" / f"cls_{args.dataset}.txt"
    manifest_path = recovery_root / "frozen_bank_manifest.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["datasets"][args.dataset]
    expected_sha = record["sha256"].lower()
    if record["file"] != final_path.name:
        raise ValueError(f"Manifest filename mismatch for {args.dataset}")
    expected_classes = sum(
        1 for line in anchor_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )

    if final_path.is_file():
        actual_sha = sha256(final_path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Existing frozen PromptBank has unexpected SHA-256: {final_path}\n"
                f"expected={expected_sha}\nactual={actual_sha}. Refusing to overwrite it."
            )
        query_count = validate_bank(final_path, expected_classes)
        print(json.dumps({
            "dataset": args.dataset,
            "status": "verified_existing",
            "path": str(final_path),
            "class_count": expected_classes,
            "query_count": query_count,
            "sha256": actual_sha,
            "source": "released frozen PromptBank checksum",
        }, ensure_ascii=False))
        return

    raise FileNotFoundError(
        f"Frozen PromptBank is missing: {final_path}. Run scripts/rebuild_promptbank.py "
        f"--dataset {args.dataset} --calibration-image-dir <unlabeled-images>. "
        "The recorded DAG protocol selects and verifies the rebuilt bank before installation."
    )


if __name__ == "__main__":
    main()
