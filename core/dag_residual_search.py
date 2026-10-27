#!/usr/bin/env python3
"""Select bounded residual prompts from cached, unlabeled SAM 3 responses."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


RESPONSE_MAP_SIZE = 256


def read_bank(path: Path) -> list[list[str]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = [value.strip() for value in line.split(",") if value.strip()]
        if not row:
            raise ValueError(f"Empty PromptBank row at {path}:{line_number}")
        rows.append(row)
    return rows


def strict_top1(maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if maps.shape[0] < 2:
        raise ValueError("Anchor protection requires at least two semantic classes")
    order = np.argsort(maps, axis=0)
    top = order[-1]
    second = np.take_along_axis(maps, order[-2:-1], axis=0)[0]
    return top, maps.max(axis=0) > second


def select_consensus(
    candidates: list[dict],
    stats: dict[str, list[dict]],
    class_count: int,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Apply per-fold winner selection and the fixed four-of-five rule."""
    node_by_id = {node["id"]: node for node in candidates}
    fold_decisions: list[dict[str, str]] = []

    def fold_stat(node_id: str, fold: int) -> dict:
        rows = stats.get(node_id, [])
        if fold >= len(rows):
            return {"unique_images": 0, "unique_pixels": 0, "anchor_violations": 0}
        return rows[fold]

    for fold in range(5):
        selected: dict[str, str] = {}
        for class_index in range(class_count):
            eligible = []
            for node in candidates:
                if node["class_index"] != class_index:
                    continue
                stat = fold_stat(node["id"], fold)
                if stat["unique_images"] > 0 and stat["anchor_violations"] == 0:
                    eligible.append(node)
            eligible.sort(
                key=lambda node: (
                    -fold_stat(node["id"], fold)["unique_images"],
                    -fold_stat(node["id"], fold)["unique_pixels"],
                    node["depth"],
                    node["phrase"],
                )
            )
            if eligible:
                selected[str(class_index)] = eligible[0]["id"]
        fold_decisions.append(selected)

    consensus: dict[str, str] = {}
    for class_index in range(class_count):
        counts = Counter(
            decision[str(class_index)]
            for decision in fold_decisions
            if str(class_index) in decision
        )
        if not counts:
            continue
        node_id, count = sorted(
            counts.items(), key=lambda item: (-item[1], node_by_id[item[0]]["phrase"])
        )[0]
        if count >= 4:
            consensus[str(class_index)] = node_id
    return fold_decisions, consensus


def _response_maps(record: dict) -> dict:
    if "clean" in record:
        return record["clean"]
    if "responses" in record:
        return record["responses"]
    raise ValueError("Response cache record must contain clean or responses maps")


def _to_array(value: torch.Tensor) -> np.ndarray:
    value = value.detach().float().cpu()
    if value.ndim != 2:
        raise ValueError(f"Expected a two-dimensional response map, got {tuple(value.shape)}")
    if value.shape != (RESPONSE_MAP_SIZE, RESPONSE_MAP_SIZE):
        value = F.interpolate(
            value[None, None], size=(RESPONSE_MAP_SIZE, RESPONSE_MAP_SIZE), mode="area"
        )[0, 0]
    return value.numpy().astype(np.float32, copy=False)


def _class_maps(responses: dict, bank: list[list[str]], branch: str) -> np.ndarray:
    class_values = []
    for aliases in bank:
        maps = []
        for phrase in aliases:
            if phrase not in responses:
                raise RuntimeError(f"Response cache is missing PromptBank phrase: {phrase}")
            value = responses[phrase][branch]
            maps.append(_to_array(value))
        class_values.append(np.maximum.reduce(maps))
    return np.stack(class_values, axis=0)


def run_search(
    cache_dir: Path,
    dag_path: Path,
    base_bank_path: Path,
    partitions_path: Path,
    output_dir: Path,
) -> tuple[list[list[str]], dict]:
    base_bank = read_bank(base_bank_path)
    dag = json.loads(dag_path.read_text(encoding="utf-8"))
    partitions_payload = json.loads(partitions_path.read_text(encoding="utf-8"))
    partitions = partitions_payload.get("partitions", partitions_payload)
    if len(base_bank) != len({node["class_index"] for node in dag["nodes"] if node["immutable"]}):
        raise ValueError("Base PromptBank and candidate DAG class counts differ")
    if len(partitions) != 5:
        raise ValueError("The residual DAG requires exactly five fixed folds")

    fold_for_index: dict[int, int] = {}
    for fold_index, indices in enumerate(partitions.values()):
        for index in indices:
            index = int(index)
            if index in fold_for_index:
                raise ValueError(f"Calibration image index {index} appears in multiple folds")
            fold_for_index[index] = fold_index
    image_count = len(fold_for_index)
    expected_indices = set(range(image_count))
    if set(fold_for_index) != expected_indices:
        raise ValueError("Folds must cover each calibration image index exactly once")
    cache_files = []
    for index in range(image_count):
        four_digit = cache_dir / f"{index:04d}.pt"
        five_digit = cache_dir / f"{index:05d}.pt"
        path = four_digit if four_digit.is_file() else five_digit
        if not path.is_file():
            raise RuntimeError(f"Response cache is incomplete; missing index {index}")
        cache_files.append(path)

    first_record = torch.load(cache_files[0], map_location="cpu", weights_only=False)
    prompt_names = set(_response_maps(first_record))
    base_pairs = {(class_index, phrase) for class_index, row in enumerate(base_bank) for phrase in row}
    phrase_owners: dict[str, set[int]] = {}
    for class_index, row in enumerate(base_bank):
        for phrase in row:
            phrase_owners.setdefault(phrase.strip().casefold(), set()).add(class_index)

    candidates = []
    rejected_cross_class_aliases = []
    for node in dag["nodes"]:
        if node["immutable"]:
            continue
        phrase = node["phrase"]
        class_index = int(node["class_index"])
        owners = phrase_owners.get(phrase.strip().casefold(), set()) - {class_index}
        if owners:
            rejected_cross_class_aliases.append({
                "node_id": node["id"], "phrase": phrase, "owner_classes": sorted(owners)
            })
            continue
        if (class_index, phrase) in base_pairs:
            continue
        if phrase not in prompt_names:
            raise RuntimeError(f"Response cache is missing DAG candidate phrase: {phrase}")
        candidates.append(node)

    stats = {
        node["id"]: [
            {"unique_images": 0, "unique_pixels": 0, "unique_fraction_sum": 0.0,
             "unique_mass": 0.0, "anchor_violations": 0, "violating_images": 0,
             "non_target_core_pixels": 0}
            for _ in range(5)
        ]
        for node in candidates
    }
    core_pixel_counts = np.zeros(len(base_bank), dtype=np.int64)
    core_image_counts = np.zeros(len(base_bank), dtype=np.int64)

    for image_index, cache_path in enumerate(cache_files):
        record = torch.load(cache_path, map_location="cpu", weights_only=False)
        responses = _response_maps(record)
        semantic = _class_maps(responses, base_bank, "semantic")
        instance = _class_maps(responses, base_bank, "instance")
        fused = _class_maps(responses, base_bank, "fused")
        semantic_top, semantic_strict = strict_top1(semantic)
        instance_top, instance_strict = strict_top1(instance)
        fused_top, fused_strict = strict_top1(fused)
        agreement = (
            semantic_strict & instance_strict & fused_strict
            & (semantic_top == instance_top) & (semantic_top == fused_top)
        )
        core = np.stack(
            [agreement & (fused_top == class_index) for class_index in range(len(base_bank))],
            axis=0,
        )
        for class_index in range(len(base_bank)):
            pixels = int(core[class_index].sum())
            core_pixel_counts[class_index] += pixels
            core_image_counts[class_index] += int(pixels > 0)

        if not candidates:
            continue

        candidate_maps = np.stack(
            [_to_array(responses[node["phrase"]]["fused"]) for node in candidates],
            axis=0,
        )
        candidate_class_indices = np.asarray([int(node["class_index"]) for node in candidates])
        baseline_for_candidates = fused[candidate_class_indices]
        delta = np.maximum(0.0, candidate_maps - baseline_for_candidates)
        unique = delta > 1e-6
        unique_pixels = unique.sum(axis=(1, 2), dtype=np.int64)
        unique_mass = delta.sum(axis=(1, 2), dtype=np.float64)
        unique_fraction = unique.mean(axis=(1, 2), dtype=np.float64)
        fused_top_score = fused.max(axis=0)
        non_target_core = np.stack(
            [core[np.arange(len(base_bank)) != class_index].any(axis=0)
             for class_index in candidate_class_indices],
            axis=0,
        )
        invade = candidate_maps > fused_top_score[None, ...]
        violations = (non_target_core & invade).sum(axis=(1, 2), dtype=np.int64)
        non_target_pixels = non_target_core.sum(axis=(1, 2), dtype=np.int64)
        fold_index = fold_for_index[image_index]
        for candidate_index, node in enumerate(candidates):
            stat = stats[node["id"]][fold_index]
            pixels = int(unique_pixels[candidate_index])
            violation_count = int(violations[candidate_index])
            stat["unique_images"] += int(pixels > 0)
            stat["unique_pixels"] += pixels
            stat["unique_fraction_sum"] += float(unique_fraction[candidate_index])
            stat["unique_mass"] += float(unique_mass[candidate_index])
            stat["anchor_violations"] += violation_count
            stat["violating_images"] += int(violation_count > 0)
            stat["non_target_core_pixels"] += int(non_target_pixels[candidate_index])

        if (image_index + 1) % 20 == 0 or image_index + 1 == image_count:
            print(f"scored {image_index + 1}/{image_count} cached calibration images")

    fold_decisions, consensus = select_consensus(candidates, stats, len(base_bank))
    node_by_id = {node["id"]: node for node in candidates}
    residuals = {
        int(class_index): node_by_id[node_id]["phrase"]
        for class_index, node_id in consensus.items()
    }
    final_bank = []
    for class_index, aliases in enumerate(base_bank):
        final_row = list(aliases)
        residual = residuals.get(class_index)
        if residual and residual not in final_row:
            final_row.append(residual)
        final_bank.append(final_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "complementarity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_id", "class_index", "phrase", "fold", "unique_images", "unique_pixels", "unique_fraction_sum", "unique_mass"])
        for node in candidates:
            for fold_index in range(5):
                stat = stats[node["id"]][fold_index]
                writer.writerow([node["id"], node["class_index"], node["phrase"], fold_index + 1,
                                 stat["unique_images"], stat["unique_pixels"], stat["unique_fraction_sum"], stat["unique_mass"]])
    with (output_dir / "anchor_protection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_id", "class_index", "phrase", "fold", "anchor_violations", "violating_images", "non_target_core_pixels", "zero_violation"])
        for node in candidates:
            for fold_index in range(5):
                stat = stats[node["id"]][fold_index]
                writer.writerow([node["id"], node["class_index"], node["phrase"], fold_index + 1,
                                 stat["anchor_violations"], stat["violating_images"],
                                 stat["non_target_core_pixels"], stat["anchor_violations"] == 0])
    (output_dir / "fold_selection.json").write_text(
        json.dumps({"fold_decisions": fold_decisions, "consensus": consensus}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    certificate = {
        "schema": "rose_sam3_anchor_protection_dag_v1",
        "dataset": dag.get("dataset"),
        "candidate_count": len(candidates),
        "eligible_rule": "same-class unique support and zero non-target Anchor-Core violations",
        "fold_consensus_rule": "same fold-local winner in >=4/5 folds",
        "residual_budget": "at most one new residual per class",
        "fold_decisions": fold_decisions,
        "final_residual_nodes": consensus,
        "base_query_count": sum(map(len, base_bank)),
        "final_query_count": sum(map(len, final_bank)),
        "calibration_image_count": image_count,
        "native_anchor_core": {
            str(index): {"images": int(core_image_counts[index]), "pixels": int(core_pixel_counts[index])}
            for index in range(len(base_bank))
        },
        "cross_class_alias_rejections": rejected_cross_class_aliases,
        "labels_used": False,
        "validation_used": False,
        "test_images_used": False,
    }
    (output_dir / "selection_certificate.json").write_text(
        json.dumps(certificate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return final_bank, certificate
