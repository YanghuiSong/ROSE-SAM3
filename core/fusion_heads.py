"""Semantic--instance fusion used by the frozen ROSE-SAM3 pipeline."""
from __future__ import annotations

import torch


def semantic_instance_cooperative_fusion(
    semantic_probs,
    instance_probs,
    presence_scores,
    max_instance_scores,
    class_instance_weights,
):
    """Apply SICH's bounded, role-aware instance residual to semantic evidence."""
    chunk_size = 4
    eps = 1e-4
    query_count = semantic_probs.shape[0]
    fused_output = torch.empty_like(semantic_probs)
    weights = class_instance_weights.to(
        device=semantic_probs.device,
        dtype=semantic_probs.dtype,
    ).clamp(0.0, 1.0)

    for start in range(0, query_count, chunk_size):
        end = min(start + chunk_size, query_count)
        semantic = semantic_probs[start:end].clamp(0.0, 1.0)
        instance = instance_probs[start:end].clamp(0.0, 1.0)
        presence = presence_scores[start:end].view(-1, 1, 1).clamp(0.0, 1.0)
        instance_score = max_instance_scores[start:end].view(-1, 1, 1).clamp(0.0, 1.0)
        class_gate = weights[start:end].view(-1, 1, 1)

        agreement = (1.0 - (semantic - instance).abs()).clamp(0.0, 1.0)
        uncertainty = (1.0 - (semantic - 0.5).abs() * 2.0).clamp(0.0, 1.0)
        residual = (instance - semantic).clamp_min(0.0)

        pgrf_gate = (presence * instance_score * agreement).clamp(0.0, 1.0)
        pgrf_anchor = semantic + pgrf_gate * residual

        instance_gate = presence * instance_score * class_gate
        residual_gate = instance_gate * (0.35 + 0.65 * uncertainty) * (
            0.35 + 0.65 * agreement
        )

        conflict = (1.0 - agreement) * instance_gate * (1.0 - uncertainty)
        suppression = 0.04 * conflict * (semantic - instance).clamp_min(0.0)

        fused = semantic + residual_gate * residual - suppression
        fused = pgrf_anchor + class_gate * (fused - pgrf_anchor)
        fused_output[start:end] = fused.clamp(0.0, 1.0 - eps)

    return fused_output
