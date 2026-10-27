"""Class-concept metrics from clean and aligned-flip response caches."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import torch


CLASSES: list[str] = []
EPS = 1e-6


@dataclass
class ConceptMetrics:
    pseudo_reference_fidelity: float
    domain_matchability: float
    visual_concept_agreement: float
    native_core_preservation: float
    competitor_protection: Dict[str, float]
    min_competitor_protection: float
    partition_robustness: float = 0.0
    safety_pass: bool = False
    own_gain: float = 0.0
    non_target_damage: float = 0.0
    cache_hit: bool = True

    def to_dict(self):
        return asdict(self)


def load_response_cache(
    cache_dir: str, n: int
) -> Tuple[List[str], Dict[str, Dict[str, torch.Tensor]]]:
    """Stack frozen responses in one pass, spatially reducing before stacking."""
    root = Path(cache_dir)
    files = []
    for index in range(n):
        four_digit = root / f"{index:04d}.pt"
        five_digit = root / f"{index:05d}.pt"
        path = four_digit if four_digit.is_file() else five_digit
        if not path.is_file():
            raise FileNotFoundError(f"Missing response cache record at index {index}: {path}")
        files.append(path)

    first = torch.load(files[0], map_location="cpu", weights_only=False)
    prompts = list(first["prompts"])
    keys = ("semantic", "instance", "fused", "presence")
    out: Dict[str, Dict[str, Dict[str, list[torch.Tensor]]]] = {
        prompt: {view: {key: [] for key in keys} for view in ("clean", "flip_aligned")}
        for prompt in prompts
    }
    for index, path in enumerate(files):
        record = first if index == 0 else torch.load(path, map_location="cpu", weights_only=False)
        if list(record.get("prompts", [])) != prompts:
            raise RuntimeError(f"Prompt order mismatch in response cache record: {path}")
        for prompt in prompts:
            for view in ("clean", "flip_aligned"):
                for key in keys:
                    value = record[view][prompt][key].float()
                    if key != "presence":
                        value = value[..., ::4, ::4]
                    out[prompt][view][key].append(value)
        del record

    stacked: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
    for prompt in prompts:
        stacked[prompt] = {}
        for view in ("clean", "flip_aligned"):
            stacked[prompt][view] = {
                key: torch.stack(values) for key, values in out[prompt][view].items()
            }
    return prompts, stacked


def aggregate(response, phrases: Sequence[str], view: str) -> Dict[str, torch.Tensor]:
    vals = [response[p][view] for p in phrases]
    return {k: torch.stack([v[k] for v in vals]).amax(0) for k in ("semantic", "instance", "fused")}


def class_maps(response, bank: Sequence[Sequence[str]], view: str) -> List[Dict[str, torch.Tensor]]:
    return [aggregate(response, phrases, view) for phrases in bank]


def teacher(class_maps_: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    fused = torch.stack([x["fused"] for x in class_maps_], 1)
    k = min(2, fused.shape[1])
    top = torch.topk(fused, k=k, dim=1).values
    margin = top[:, 0] - (top[:, 1] if k > 1 else torch.zeros_like(top[:, 0]))
    return {"scores": fused, "pred": fused.argmax(1), "margin": margin}


def _branch_stack(maps: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([m[k] for m in maps], 1) for k in ("semantic", "instance")}


def prepare_native_partition(response, native_bank, indices):
    idx = torch.as_tensor(indices, dtype=torch.long)
    nc = [{k: v[idx] for k, v in m.items()} for m in class_maps(response, native_bank, "clean")]
    nf = [{k: v[idx] for k, v in m.items()} for m in class_maps(response, native_bank, "flip_aligned")]
    return {"native_clean": nc, "native_flip": nf, "teacher_clean": teacher(nc),
            "teacher_flip": teacher(nf), "branch_clean": _branch_stack(nc),
            "branch_flip": _branch_stack(nf), "indices": idx}


def native_core(native_clean, native_flip, branch_clean, branch_flip, class_id: int):
    tc, tf = native_clean["pred"], native_flip["pred"]
    stable = (tc == class_id) & (tf == class_id)
    positive = (native_clean["margin"] > 0) & (native_flip["margin"] > 0)
    # Empty instance responses leave semantic support as the branch weight.
    sem = torch.minimum(branch_clean["semantic"][:, class_id], branch_flip["semantic"][:, class_id])
    inst = torch.minimum(branch_clean["instance"][:, class_id], branch_flip["instance"][:, class_id])
    inst_available = float(branch_clean["instance"].abs().max()) > EPS or float(branch_flip["instance"].abs().max()) > EPS
    support = torch.minimum(sem, inst) if inst_available else sem
    weight = torch.sqrt(native_clean["margin"].clamp_min(0) * native_flip["margin"].clamp_min(0)) * support
    return stable & positive, weight


def _soft_iou(x, core, weight):
    x = x.float().clamp_min(0); core = core.float().clamp_min(0); weight = weight.float().clamp_min(0)
    num = torch.minimum(x, core * (weight / (weight.amax().clamp_min(EPS)))).sum()
    den = torch.maximum(x, core * (weight / (weight.amax().clamp_min(EPS)))).sum()
    return float((num / (den + EPS)).item())


def _cosine(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())


def evaluate_candidate(response, native_bank, phrase: str, class_id: int,
                       prepared) -> ConceptMetrics:
    """Evaluate one concept while replacing only its own class state."""
    native_c, native_f = prepared['native_clean'], prepared['native_flip']
    tc, tf = prepared['teacher_clean'], prepared['teacher_flip']
    bc, bf = prepared['branch_clean'], prepared['branch_flip']
    idx = prepared['indices']
    candidate_c = {k: response[phrase]["clean"][k][idx] for k in ("semantic", "instance", "fused")}
    candidate_f = {k: response[phrase]["flip_aligned"][k][idx] for k in ("semantic", "instance", "fused")}
    student_c = list(native_c)
    student_f = list(native_f)
    student_c[class_id] = candidate_c; student_f[class_id] = candidate_f
    sc, sf = teacher(student_c), teacher(student_f)
    candidate_support = torch.minimum(candidate_c["semantic"], candidate_f["semantic"])
    stable, weight = native_core(tc, tf, bc, bf, class_id)
    core = (tc["pred"] == class_id).float() * weight
    fidelity = _soft_iou(candidate_support, core, weight)
    preserve_den = float((weight * stable.float()).sum().item())
    preserve = float((candidate_support * stable.float()).sum().item() / (preserve_den + EPS))
    native_presence = torch.stack([response[p]["clean"]["presence"][idx] for p in native_bank[class_id]]).amax(0)
    mask = native_presence >= torch.median(native_presence)
    domain = float(candidate_c.get("presence", response[phrase]["clean"]["presence"][idx])[mask].mean().item() / (native_presence[mask].mean().item() + EPS))
    domain = max(0.0, min(domain, 1.5)) / 1.5
    visual = max(0.0, min(1.0, 0.5 * (_cosine(candidate_c["fused"], native_c[class_id]["fused"]) + _cosine(candidate_f["fused"], native_f[class_id]["fused"])) / 2.0 + 0.5))
    protections = {}
    damages = []
    for j, cls in enumerate(CLASSES):
        if j == class_id:
            continue
        s_core, s_weight = native_core(tc, tf, bc, bf, j)
        union = (tc["pred"] == j) | (tf["pred"] == j)
        den = (s_weight * union.float()).sum()
        kept = (sc["pred"] == j) & (sf["pred"] == j)
        protections[cls] = float((s_weight * union.float() * kept.float()).sum().item() / (den.item() + EPS))
        damages.append(1.0 - protections[cls])
    # Stable native self-protection is the reference floor for this partition.
    refs = []
    for j in range(len(CLASSES)):
        s_core, s_weight = native_core(tc, tf, bc, bf, j)
        union = (tc["pred"] == j) | (tf["pred"] == j)
        refs.append(float((s_weight * union.float() * s_core.float()).sum().item() / ((s_weight * union.float()).sum().item() + EPS)))
    min_prot = min(protections.values()) if protections else 1.0
    rescue = ((tc["pred"] != class_id) | (tf["pred"] != class_id)) & (sc["pred"] == class_id) & (sf["pred"] == class_id)
    drop = stable & ~((sc["pred"] == class_id) & (sf["pred"] == class_id))
    own_gain = float((candidate_support * rescue.float()).sum().item() / (candidate_support.numel() + EPS) - (weight * drop.float()).sum().item() / (weight.numel() + EPS))
    safety = all(protections[c] >= refs[j] - 1e-6 for j, c in enumerate(CLASSES) if j != class_id)
    return ConceptMetrics(fidelity, domain, visual, preserve, protections, min_prot,
                          safety_pass=bool(safety), own_gain=own_gain,
                          non_target_damage=float(sum(damages) / max(1, len(damages))),
                          cache_hit=True)
