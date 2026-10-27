#!/usr/bin/env python3
"""Rebuild a missing frozen PromptBank from SAM 3 response-guided DAG search."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.promptbank_compiler import SUPPORTED_DATASETS, compile_promptbank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--calibration-image-dir", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = compile_promptbank(**vars(args))
    if result["status"] == "verified_existing":
        print(json.dumps(result))


if __name__ == "__main__":
    main()
