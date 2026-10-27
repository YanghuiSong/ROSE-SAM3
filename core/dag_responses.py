#!/usr/bin/env python3
"""Cache per-prompt semantic, instance, presence, and fused SAM 3 responses."""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


RESPONSE_SIZE = 256
PROCESSOR_RESOLUTION = 1008
IMAGE_MAX_SIDES = {
    "openearthmap": None, "udd5": None, "vaihingen": 1008, "vdd": 1008,
    "potsdam": 1008,
    "uavid": None, "isaid": 1008,
    "loveda": 1008,
}


def _resize_map(value: torch.Tensor, size: int | tuple[int, int]) -> torch.Tensor:
    value = value.squeeze()
    if value.ndim != 2:
        value = value.reshape(value.shape[-2], value.shape[-1])
    target = (size, size) if isinstance(size, int) else tuple(size)
    if tuple(value.shape) != target:
        value = F.interpolate(
            value[None, None].float(), size=target, mode="bilinear", align_corners=False
        )[0, 0]
    return value


def load_calibration_image(path: Path, max_side: int | None = None) -> Image.Image:
    # Dataset-specific pre-resizing precedes the processor's square resize.
    with Image.open(path) as source:
        image = source.convert("RGB")
    if max_side is not None and max(image.size) > max_side:
        scale = max_side / float(max(image.size))
        size = tuple(max(1, int(round(side * scale))) for side in image.size)
        image = image.resize(size, Image.Resampling.BILINEAR)
    return image


def _extract_prompt(processor, state: dict, prompt: str, image_size: tuple[int, int], device: torch.device, response_size: int = RESPONSE_SIZE) -> dict:
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(state=state, prompt=prompt)
    width, height = image_size
    semantic = torch.zeros((height, width), device=device, dtype=torch.float32)
    instance = torch.zeros_like(semantic)

    masks = state.get("masks_logits")
    scores = state.get("object_score")
    if masks is not None and masks.shape[0]:
        for index in range(masks.shape[0]):
            mask = _resize_map(masks[index], (height, width)).to(device=device)
            score = float(scores[index]) if scores is not None else 1.0
            instance = torch.maximum(instance, mask * score)

    semantic_logits = state.get("semantic_mask_logits")
    if semantic_logits is not None:
        semantic = _resize_map(semantic_logits, (height, width)).to(device=device)

    presence = torch.as_tensor(state.get("presence_score", 1.0), device=device, dtype=torch.float32)
    if presence.numel() != 1:
        presence = presence.max()
    presence = presence.reshape(()).clamp(0.0, 1.0)
    fused = torch.maximum(semantic, instance) * presence

    return {
        "semantic": _resize_map(semantic, response_size).half().cpu(),
        "instance": _resize_map(instance, response_size).half().cpu(),
        "fused": _resize_map(fused, response_size).half().cpu(),
        "presence": presence.half().cpu(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_processor(config_path: Path, device: torch.device, confidence: float | None = None):
    """Construct the FP32 processor and record model/runtime hashes."""
    from mmengine.config import Config
    import PIL
    import sam3
    from sam3 import build_sam3_image_model
    from core.dag_response_processor import (
        DAGResponseProcessor, check_sam3_runtime, configure_recorded_numerics,
    )

    config = Config.fromfile(str(config_path.resolve()))
    if confidence is None:
        confidence = float(config.model.get("confidence_threshold", 0.5))
    checkpoint = Path(os.environ.get("SAM3_CHECKPOINT", "/data/public/sam3/sam3.pt"))
    bpe_path = Path(os.environ.get(
        "SAM3_BPE_PATH", "/data/public/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    ))
    for path in (checkpoint, bpe_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    compatibility = check_sam3_runtime()
    model = build_sam3_image_model(
        bpe_path=str(bpe_path), checkpoint_path=str(checkpoint),
        device=device, compile=False,
    ).to(device).eval()
    compatibility.update(configure_recorded_numerics(model))
    processor = DAGResponseProcessor(
        model, resolution=PROCESSOR_RESOLUTION,
        confidence_threshold=confidence, device=device,
    )
    source_root = Path(sam3.__file__).resolve().parent
    source_hash = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        source_hash.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        source_hash.update(bytes.fromhex(_sha256(path)))
    metadata = {
        "processor_resolution": PROCESSOR_RESOLUTION,
        "image_pre_resize": "none; RGB at original resolution",
        "map_downsampling": "original-resolution response then bilinear, align_corners=False",
        "confidence_threshold": confidence,
        "checkpoint_sha256": _sha256(checkpoint),
        "bpe_sha256": _sha256(bpe_path),
        "sam3_python_source_sha256": source_hash.hexdigest(),
        "collector_sha256": _sha256(Path(__file__)),
        "response_adapter_sha256": _sha256(Path(__file__).resolve().parents[1] / "core/dag_response_processor.py"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "pillow": PIL.__version__,
        "autocast": False,
        **compatibility,
    }
    return processor, metadata


def collect_responses(*, config_path, prompt_path, image_paths, cache_dir, device,
                      dataset, stage, calibration_split="train", clean_only=False):
    """Collect the recorded base or residual responses without changing numerics."""
    if stage not in ("base", "residual"):
        raise ValueError(f"Unknown response stage: {stage}")
    paths = [Path(path) for path in image_paths]
    if len(paths) < 5:
        raise ValueError("DAG response extraction requires at least five calibration images")
    prompts = list(dict.fromkeys(
        phrase.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines()
        for phrase in line.split(",") if phrase.strip()))
    if not prompts:
        raise ValueError("Prompt list is empty")
    base = stage == "base"
    device = torch.device(device)
    processor, runtime = build_processor(config_path, device, confidence=0.5 if base else None)
    response_size = 128 if base else RESPONSE_SIZE
    max_side = None if base else IMAGE_MAX_SIDES[dataset]
    cache_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index, path in enumerate(paths):
            if not path.is_file():
                raise FileNotFoundError(path)
            image = load_calibration_image(path, max_side=max_side)

            def run_view(view):
                state = processor.set_image(view)
                return {prompt: _extract_prompt(processor, state, prompt, view.size, device,
                                                 response_size=response_size)
                        for prompt in prompts}

            responses = run_view(image)
            record = {"image": str(path), "prompts": prompts, "labels_used": False,
                      "clean" if base else "responses": responses}
            if base and not clean_only:
                flip_raw = run_view(ImageOps.mirror(image))
                record["flip_aligned"] = {
                    prompt: {key: value.flip(-1) if key != "presence" else value
                             for key, value in branches.items()}
                    for prompt, branches in flip_raw.items()}
            digits = 4 if base else 5
            torch.save(record, cache_dir / f"{index:0{digits}d}.pt")
            if (index + 1) % 10 == 0 or index + 1 == len(paths):
                print(f"cached {index + 1}/{len(paths)} {stage} images; prompts={len(prompts)}", flush=True)

    manifest = {
        "schema": "rose_sam3_offline_dag_cache_v1" if base else "rose_sam3_anchor_protection_response_cache_v1",
        "image_count": len(paths), "prompt_count": len(prompts), "prompts": prompts,
        "images": [str(path) for path in paths], "image_sha256": [_sha256(path) for path in paths],
        **runtime, "labels_used": False, "validation_used": base and calibration_split == "val",
        "test_images_used": False,
    }
    if base:
        manifest.update(map_size=128, response_output_size=[128, 128],
                        validation_images_used=calibration_split == "val", validation_labels_used=False,
                        calibration_split=calibration_split,
                        flip_is_offline_stability_probe=not clean_only, clean_only=clean_only)
    else:
        manifest.update(dataset=dataset, response_size=RESPONSE_SIZE, image_max_side=max_side,
                        image_pre_resize="none" if max_side is None else f"bilinear longest side <= {max_side}",
                        tta_used=False)
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest

