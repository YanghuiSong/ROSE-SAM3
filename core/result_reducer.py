"""Reduce frozen SAM 3 outputs to the semantic and instance responses used by ROSE-SAM3."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class GroundingResultReducer:
    """Build per-prompt semantic maps, instance maps, and presence scores."""

    def __init__(
        self,
        device,
        confidence_threshold: float = 0.5,
    ):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.confidence_threshold = float(confidence_threshold)
        self.mask_query_chunk_size = 32
        self.slot_chunk_size = 4

    def reduce_slot_outputs(self, outputs, slot_count, output_size):

        height, width = (int(output_size[0]), int(output_size[1]))
        semantic_logits = outputs.get("semantic_seg")
        if semantic_logits is None:
            raise KeyError("Frozen SAM 3 output is missing semantic_seg")
        if semantic_logits.ndim == 3:
            semantic_logits = semantic_logits.unsqueeze(1)
        if semantic_logits.ndim != 4 or semantic_logits.shape[0] != slot_count:
            raise RuntimeError(f"Unexpected semantic_seg shape: {tuple(semantic_logits.shape)}")
        if semantic_logits.shape[-2:] != (height, width):
            semantic_logits = F.interpolate(
                semantic_logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        semantic_response = semantic_logits[:, 0].sigmoid().float()

        presence_logits = outputs.get("presence_logit_dec")
        if presence_logits is None:
            presence_scores = torch.ones(slot_count, device=self.device)
        else:
            presence_scores = presence_logits.sigmoid().reshape(slot_count, -1)[:, 0]

        instance_response = torch.zeros(
            (slot_count, height, width), device=self.device, dtype=torch.float32
        )
        max_instance_scores = torch.zeros(slot_count, device=self.device, dtype=torch.float32)
        masks = outputs.get("pred_masks")
        object_logits = outputs.get("pred_logits")
        if masks is not None and object_logits is not None and masks.shape[1] > 0:
            if masks.ndim == 5 and masks.shape[2] == 1:
                masks = masks.squeeze(2)
            if masks.ndim != 4 or masks.shape[0] != slot_count:
                raise RuntimeError(f"Unexpected pred_masks shape: {tuple(masks.shape)}")
            object_scores = object_logits.sigmoid().float()
            if object_scores.ndim == 3 and object_scores.shape[-1] == 1:
                object_scores = object_scores.squeeze(-1)
            if object_scores.ndim != 2 or object_scores.shape[0] != slot_count:
                raise RuntimeError(f"Unexpected pred_logits shape: {tuple(object_logits.shape)}")
            if object_scores.shape[1] != masks.shape[1]:
                raise RuntimeError("SAM 3 pred_masks and pred_logits query counts differ")

            # SAM 3's decoder score is conditioned on its prompt-presence score.
            object_scores = object_scores * presence_scores.view(-1, 1)
            max_instance_scores = object_scores.amax(dim=1)
            valid_scores = object_scores > self.confidence_threshold

            for slot_start in range(0, slot_count, self.slot_chunk_size):
                slot_end = min(slot_start + self.slot_chunk_size, slot_count)
                chunk_masks = masks[slot_start:slot_end].float()
                chunk_scores = object_scores[slot_start:slot_end]
                chunk_valid = valid_scores[slot_start:slot_end]
                chunk_best = torch.zeros(
                    (slot_end - slot_start, height, width),
                    device=self.device,
                    dtype=torch.float32,
                )

                for query_start in range(0, masks.shape[1], self.mask_query_chunk_size):
                    query_end = min(query_start + self.mask_query_chunk_size, masks.shape[1])
                    valid_window = chunk_valid[:, query_start:query_end]
                    if not valid_window.any():
                        continue
                    active_queries = torch.nonzero(valid_window.any(dim=0), as_tuple=False).flatten()
                    sub_masks = chunk_masks[:, query_start:query_end].index_select(1, active_queries)
                    valid_window = valid_window.index_select(1, active_queries)
                    if sub_masks.shape[-2:] != (height, width):
                        sub_masks = F.interpolate(
                            sub_masks.reshape(-1, 1, *sub_masks.shape[-2:]),
                            size=(height, width),
                            mode="bilinear",
                            align_corners=False,
                        ).reshape(slot_end - slot_start, active_queries.numel(), height, width)
                    sub_masks = sub_masks.sigmoid()
                    score_window = chunk_scores[:, query_start:query_end].index_select(1, active_queries)
                    sub_scores = score_window * valid_window
                    chunk_best = torch.maximum(
                        chunk_best,
                        (sub_masks * sub_scores.unsqueeze(-1).unsqueeze(-1)).amax(dim=1),
                    )

                instance_response[slot_start:slot_end] = chunk_best

        # Semantic and object responses both include prompt presence.
        semantic_response = semantic_response * presence_scores.view(-1, 1, 1)
        return {
            "semantic_logits": semantic_response,
            "instance_best_logits": instance_response,
            "presence_scores": presence_scores,
            "max_instance_scores": max_instance_scores,
        }
