#!/usr/bin/env python3
"""Rebuild a missing frozen PromptBank through the two-stage response DAG."""
from __future__ import annotations

import hashlib
import json
import os
import random
from types import SimpleNamespace
from pathlib import Path


SUPPORTED_DATASETS = ("openearthmap", "udd5", "vaihingen", "vdd", "potsdam", "uavid", "isaid", "loveda")
RECOMPILED_DATASETS = ("uavid", "isaid")
SEARCH_PROTOCOL = {
    "openearthmap": {"query_budget": 16, "base_consensus": 4},
    "udd5": {"query_budget": 32, "base_consensus": 4},
    "vaihingen": {"query_budget": 16, "base_consensus": 4},
    "vdd": {"query_budget": 32, "base_consensus": 2},
    "potsdam": {"query_budget": 32, "base_consensus": 4},
    "uavid": {"query_budget": 32, "base_consensus": 2},
    "isaid": {"query_budget": 32, "base_consensus": 4},
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
CALIBRATION_LIMITS = {dataset: 512 for dataset in SUPPORTED_DATASETS}
CALIBRATION_SEED = 42
DEFAULT_NATIVE_REFERENCES = {
    "uavid": "native_references/cls_uavid_proc.txt",
    "isaid": "native_references/cls_isaid_proc.txt",
}
FORBIDDEN_PATH_PARTS = {
    "ann", "anns", "ann_dir", "annotation", "annotations", "label", "labels",
    "mask", "masks", "gt", "ground_truth", "groundtruth", "val", "valid",
    "validation", "test", "testing",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[list[str]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = [value.strip() for value in line.split(",") if value.strip()]
        if not row:
            raise ValueError(f"Empty prompt row at {path}:{line_number}")
        rows.append(row)
    return rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_name(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").replace("-", " ").split())


def list_calibration_images(image_dir: Path, dataset: str) -> list[Path]:
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)
    if {part.casefold() for part in image_dir.parts} & FORBIDDEN_PATH_PARTS:
        raise ValueError(f"Calibration directory must be unlabeled training imagery, got: {image_dir}")
    images = sorted(
        (
            path.resolve()
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.as_posix(),
    )
    if len(images) < 5:
        raise ValueError(f"Need at least five unlabeled training images; found {len(images)}")
    limit = CALIBRATION_LIMITS[dataset]
    if limit is not None and len(images) > limit:
        images = sorted(random.Random(CALIBRATION_SEED).sample(images, limit), key=lambda path: path.as_posix())
    return images


def validate_residual_calibration(repo: Path, dataset: str, train_dir: Path, images: list[Path], partitions: dict):
    """Reject a different calibration population before expensive model calls."""
    source = repo / "calibrated_prompts/recovery_inputs/residual_calibration" / f"{dataset}.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    names = [path.relative_to(train_dir).as_posix() for path in images]
    actual = {names[index]: fold for fold, indices in enumerate(partitions.values()) for index in indices}
    recorded_names = manifest["relative_images"]
    expected = {recorded_names[index]: fold for fold, indices in enumerate(manifest["partitions"].values()) for index in indices}
    if actual != expected:
        raise ValueError(
            f"Residual calibration cohort/folds differ from the release manifest for {dataset}: "
            f"missing={len(set(expected)-set(actual))}, extra={len(set(actual)-set(expected))}. "
            "Use the canonical dataset split and the recorded image filenames."
        )
    return source


def stage1_calibration(repo: Path, dataset: str, train_dir: Path):
    """Resolve the recorded base-calibration images and five partitions."""
    source = repo / "calibrated_prompts/recovery_inputs/base_calibration" / f"{dataset}.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    split = manifest["calibration_split"]
    directory = (train_dir if split == "train" else train_dir.parent / "val").resolve()
    forbidden = FORBIDDEN_PATH_PARTS - ({"val"} if split == "val" else set())
    if {part.casefold() for part in directory.parts} & forbidden:
        raise ValueError(f"Invalid Stage-I image directory: {directory}")
    images = []
    for relative in manifest["relative_images"]:
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory):
            raise ValueError(f"Stage-I image escapes its calibration directory: {relative}")
        if not path.is_file():
            raise FileNotFoundError(
                f"Stage-I image missing: {path}; expected the recorded {dataset} {split} images."
            )
        images.append(path)
    partitions = manifest["partitions"]
    if dataset == "loveda" and manifest.get("selection_uses_folds") is False:
        return images, {}, split, source
    assigned = [index for indices in partitions.values() for index in indices]
    if len(partitions) != 5 or sorted(assigned) != list(range(len(images))):
        raise ValueError("Stage-I manifest must assign every image exactly once to five folds")
    return images, partitions, split, source


def write_stage2_queries(candidate_dag_path: Path, base_bank_path: Path, output_path: Path) -> Path:
    """Collect only the residual-DAG nodes and the selected immutable base aliases."""
    dag = json.loads(candidate_dag_path.read_text(encoding="utf-8"))
    phrases = [str(node["phrase"]).strip() for node in dag["nodes"] if str(node.get("phrase", "")).strip()]
    for row in read_rows(base_bank_path):
        phrases.extend(row)
    unique = list(dict.fromkeys(phrases))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(unique) + "\n", encoding="utf-8")
    return output_path


def build_recovery_inputs(
    repo: Path, dataset: str, work: Path,
) -> dict[str, Path]:
    """Bind native phrases to fixed dataset row IDs and prepare all response queries."""
    recovery = repo / "calibrated_prompts" / "recovery_inputs"
    anchors_path = recovery / "ov3_anchors" / f"cls_{dataset}.txt"
    base_candidates_path = recovery / "base_bank_candidates" / f"{dataset}.json"
    candidate_dag_path = recovery / "ap_candidate_dags" / dataset / "candidate_dag.json"
    evidence_path = recovery / "evidence_candidates" / f"{dataset}.json"
    required = (anchors_path, candidate_dag_path,
                evidence_path if dataset == "loveda" else base_candidates_path)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required DAG recovery input is missing: {path}")

    anchors = read_rows(anchors_path)
    default_native = recovery / DEFAULT_NATIVE_REFERENCES[dataset] if dataset in DEFAULT_NATIVE_REFERENCES else anchors_path
    native_source = default_native.resolve()
    native_rows = read_rows(native_source)
    if len(native_rows) != len(anchors):
        raise ValueError(
            f"Native PromptBank row count differs from dataset class count: "
            f"{len(native_rows)} != {len(anchors)}"
        )
    base_candidates = ([] if dataset == "loveda" else
                       json.loads(base_candidates_path.read_text(encoding="utf-8"))["banks"])
    candidate_dag = json.loads(candidate_dag_path.read_text(encoding="utf-8"))
    if candidate_dag.get("dataset") != dataset:
        raise ValueError(f"Candidate DAG dataset mismatch: expected {dataset}")
    class_count = len(anchors)
    if (not base_candidates and dataset != "loveda") or any(len(bank) != class_count for bank in base_candidates):
        raise ValueError("Each complete-bank candidate must have one row for every OV3 class")
    for bank in base_candidates:
        if any(not row or any(not str(phrase).strip() for phrase in row) for row in bank):
            raise ValueError("Complete-bank candidates must contain non-empty phrase rows")

    dag_class_names = {}
    phrases_by_class: dict[int, list[str]] = {index: [] for index in range(class_count)}
    for node in candidate_dag["nodes"]:
        class_index = int(node["class_index"])
        if not 0 <= class_index < class_count:
            raise ValueError(f"Candidate DAG class index out of range: {node}")
        dag_class_names.setdefault(class_index, node["target_class"])
        phrases_by_class[class_index].append(node["phrase"])
    for class_index, anchor_row in enumerate(anchors):
        if normalize_name(dag_class_names.get(class_index, "")) != normalize_name(anchor_row[0]):
            raise ValueError(
                f"Candidate DAG class order differs from OV3 anchors at row {class_index}: "
                f"{dag_class_names.get(class_index)!r} != {anchor_row[0]!r}"
            )

    classes = [row[0] for row in anchors]
    native_bank = {name: row for name, row in zip(classes, native_rows)}
    evidence_candidates = json.loads(evidence_path.read_text()) if dataset == "loveda" else {}
    stage1_candidates: dict[str, list[str]] = {}
    response_rows: list[list[str]] = []
    for class_index, (class_name, anchor_row) in enumerate(zip(classes, anchors)):
        # Class IDs stay fixed even if the supplied reference uses different wording.
        route = phrases_by_class[class_index]
        base_phrases = [phrase for bank in base_candidates for phrase in bank[class_index]]
        stage1_candidates[class_name] = list(dict.fromkeys(
            anchor_row + native_rows[class_index] + route + base_phrases
            + evidence_candidates.get(class_name, [])))
        response_rows.append(stage1_candidates[class_name])

    work.mkdir(parents=True, exist_ok=True)
    query_bank = work / "dag_response_queries.txt"
    query_bank.write_text("\n".join(",".join(row) for row in response_rows) + "\n", encoding="utf-8")
    classes_path = work / "classes.json"
    candidates_path = work / "stage1_candidates.json"
    seeds_path = work / "base_bank_candidates.json"
    write_json(classes_path, {"classes": classes, "native_bank": native_bank})
    write_json(candidates_path, stage1_candidates)
    write_json(seeds_path, {"banks": base_candidates})
    return {
        "query_bank": query_bank,
        "classes": classes_path,
        "candidates": candidates_path,
        "base_bank_candidates": seeds_path,
        "candidate_dag": candidate_dag_path,
        "evidence_candidates": evidence_path,
        "native_source": native_source,
    }


def finish_compilation(selected_path, final_path, work, certificate, expected_sha):
    selected_sha = sha256(selected_path)
    certificate.update(selected_sha256=selected_sha, matches_frozen_promptbank=selected_sha == expected_sha,
                       installed=False, status="matched" if selected_sha == expected_sha else "checksum_mismatch")
    write_json(work / "rebuild_certificate.json", certificate)
    if selected_sha != expected_sha:
        raise RuntimeError("DAG selection did not match the released checksum; no file was installed")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_name(final_path.name + ".rebuild.tmp")
    temp_path.write_bytes(selected_path.read_bytes())
    os.replace(temp_path, final_path)
    certificate.update(installed=True, installed_sha256=sha256(final_path))
    write_json(work / "rebuild_certificate.json", certificate)
    print(json.dumps(certificate, indent=2, ensure_ascii=False))
    return certificate


def compile_promptbank(dataset, calibration_image_dir, *, work_dir=None, device="cuda", repo=None):
    """Verify an existing bank or rebuild it from recorded SAM 3 response searches."""
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    repo = Path(repo or Path(__file__).resolve().parents[1]).resolve()
    config_name = "cfg_iSAID.py" if dataset == "isaid" else f"cfg_{dataset}.py"
    args = SimpleNamespace(dataset=dataset, config=repo / "configs" / config_name,
                           calibration_image_dir=Path(calibration_image_dir),
                           work_dir=Path(work_dir) if work_dir else None, device=device)
    final_path = repo / "calibrated_prompts" / "final_promptbank" / f"cls_{args.dataset}.txt"
    manifest_path = repo / "calibrated_prompts" / "recovery_inputs" / "frozen_bank_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sha = manifest["datasets"][args.dataset]["sha256"].lower()
    if final_path.is_file():
        actual_sha = sha256(final_path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Frozen PromptBank checksum mismatch; refusing to overwrite {final_path}: "
                f"expected={expected_sha}, actual={actual_sha}"
            )
        return {"dataset": args.dataset, "status": "verified_existing", "sha256": actual_sha}

    images = list_calibration_images(args.calibration_image_dir.resolve(), args.dataset)
    base_images, base_partitions, base_split, base_manifest = stage1_calibration(
        repo, args.dataset, args.calibration_image_dir.resolve(),
    )
    work = (args.work_dir or repo / "work_dirs" / "promptbank_rebuild" / args.dataset).resolve()
    work.mkdir(parents=True, exist_ok=True)
    inputs = build_recovery_inputs(repo, args.dataset, work)
    pool_path = work / "pool.json"
    write_json(pool_path, {"images": [str(path) for path in base_images]})
    base_partitions_path = work / "base_partitions.json"
    write_json(base_partitions_path, {"partitions": base_partitions})
    partitions = {f"fold_{fold + 1}": list(range(fold, len(images), 5)) for fold in range(5)}
    if any(not indices for indices in partitions.values()):
        raise ValueError("Every calibration fold must contain at least one image")
    partitions_path = work / "partitions.json"
    write_json(partitions_path, {"partitions": partitions})
    residual_manifest = validate_residual_calibration(
        repo, args.dataset, args.calibration_image_dir.resolve(), images, partitions,
    )
    cache_dir = work / "response_cache"
    certificate = {
        "native_promptbank_source": str(inputs["native_source"]),
        "native_promptbank_sha256": sha256(inputs["native_source"]),
        "native_kind": ("dataset_proc_reference" if args.dataset in DEFAULT_NATIVE_REFERENCES else
                        "recorded_ov3_reference"),
        "schema": "rose_sam3_two_stage_promptbank_recovery_v1",
        "dataset": args.dataset,
        "stage1_calibration_image_count": len(base_images),
        "stage1_fold_count": len(base_partitions),
        "stage1_calibration_split": base_split,
        "stage1_calibration_manifest_sha256": sha256(base_manifest),
        "stage2_calibration_image_count": len(images),
        "stage2_calibration_split": "train",
        "stage2_calibration_manifest_sha256": sha256(residual_manifest),
        "calibration_image_limit": CALIBRATION_LIMITS[args.dataset],
        "calibration_seed": CALIBRATION_SEED,
        "fold_count": 5,
        "required_final_sha256": expected_sha,
        "labels_used": False,
        "validation_used": base_split == "val",
        "validation_images_used": base_split == "val",
        "validation_labels_used": False,
        "test_images_used": False,
        "installed": False,
        "status": "running",
        "protocol_kind": "train_only_recompilation" if args.dataset in RECOMPILED_DATASETS else "historical_recovery",
        "output_promptbank": str(final_path),
    }
    write_json(work / "rebuild_certificate.json", certificate)
    print(json.dumps(certificate, indent=2), flush=True)

    from core.dag_responses import collect_responses
    collect_responses(config_path=args.config, prompt_path=inputs["query_bank"],
                      image_paths=base_images, cache_dir=cache_dir, device=args.device,
                      dataset=args.dataset, stage="base", calibration_split=base_split,
                      clean_only=args.dataset == "loveda")
    stage1_dir = work / "stage1_base_dag"
    base_bank_path = work / "selected_base_promptbank.txt"
    if args.dataset == "loveda":
        import torch
        from core.dag_evidence_search import run_search
        torch.set_num_threads(1)
        run_search(cache_dir, pool_path, inputs["evidence_candidates"], inputs["classes"],
                   stage1_dir, base_bank_path)
    else:
        from core.dag_base_search import run_search
        protocol = SEARCH_PROTOCOL[args.dataset]
        run_search(SimpleNamespace(
            cache_dir=cache_dir, pool=pool_path, partitions=base_partitions_path,
            candidates=inputs["candidates"], classes_json=inputs["classes"],
            base_bank_candidates=inputs["base_bank_candidates"],
            query_budget=protocol["query_budget"],
            proposal_min_consensus=protocol["base_consensus"],
            calibration_split=base_split, out_dir=stage1_dir, prompt_out=base_bank_path))
    base_rows = read_rows(base_bank_path)
    base_candidates = json.loads(inputs["base_bank_candidates"].read_text(encoding="utf-8"))["banks"]
    normalized_candidates = [[list(row) for row in bank] for bank in base_candidates]
    configured_native_selected = args.dataset in DEFAULT_NATIVE_REFERENCES and base_rows == read_rows(inputs["native_source"])
    if (base_rows not in normalized_candidates and not configured_native_selected
            and args.dataset not in RECOMPILED_DATASETS + ("loveda",)):
        certificate.update(status="stage1_rejected", selected_base_sha256=sha256(base_bank_path))
        write_json(work / "rebuild_certificate.json", certificate)
        raise RuntimeError(
            "Stage-I DAG did not retain one of the recorded base-bank candidate states; "
            "Stage-II is not run and no PromptBank is installed."
        )

    if args.dataset == "isaid":
        certificate.update(schema="rose_sam3_base_dag_recovery_v1",
                           stage1="response-scored concept DAG and complete-bank consensus",
                           stage2=None, stage1_search_summary=str(stage1_dir / "search_summary.json"),
                           stage2_selection_certificate=None,
                           calibration_image_count=len(base_images))
        return finish_compilation(base_bank_path, final_path, work, certificate, expected_sha)

    stage2_dir = work / "stage2_anchor_protection_dag"
    selected_path = work / "selected_promptbank.txt"
    stage2_cache_dir = work / "stage2_response_cache"
    stage2_query_bank = write_stage2_queries(
        inputs["candidate_dag"], base_bank_path, work / "stage2_response_queries.txt"
    )
    collect_responses(config_path=args.config, prompt_path=stage2_query_bank,
                      image_paths=images, cache_dir=stage2_cache_dir, device=args.device,
                      dataset=args.dataset, stage="residual")
    from core.dag_residual_search import run_search
    bank, _ = run_search(stage2_cache_dir, inputs["candidate_dag"], base_bank_path,
                         partitions_path, stage2_dir)
    selected_path.write_bytes(("\n".join(",".join(row) for row in bank) + "\n").encode("utf-8"))
    certificate.update(
        stage1=("recorded LoveDA clean-response evidence gates" if args.dataset == "loveda" else
                "response-guided selection of complete base-bank proposals against the recorded native reference"),
        stage2="per-prompt SAM3 responses with recorded per-dataset RGB preprocessing and anchor-protected residual DAG",
        calibration_image_count=len(images),
        stage1_search_summary=str(stage1_dir / "search_summary.json"),
        stage2_selection_certificate=str(stage2_dir / "selection_certificate.json"))
    return finish_compilation(selected_path, final_path, work, certificate, expected_sha)
