"""Load the fixed class PromptBank and cache frozen SAM 3 text features."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch


class PromptBank:
    """A fixed, ordered prompt list with its precomputed SAM 3 text features."""

    _text_embedding_cache: dict[str, dict[str, torch.Tensor | None]] = {}

    def __init__(
        self,
        processor,
        classname_path,
        device,
    ):
        self.processor = processor
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.query_words, class_indices = self.get_cls_idx(classname_path)
        if not self.query_words:
            raise ValueError(f"PromptBank contains no prompts: {classname_path}")
        self.query_idx = torch.tensor(class_indices, dtype=torch.long, device=self.device)
        self.num_queries = len(self.query_words)
        self.num_cls = int(self.query_idx.max().item()) + 1

        content_key = "\n".join(self.query_words).encode("utf-8")
        digest = hashlib.sha256(content_key).hexdigest()
        self.cache_key = f"{digest}:{self.device}"
        if self.cache_key in self._text_embedding_cache:
            self.text_embeddings = self._text_embedding_cache[self.cache_key]
        else:
            self.text_embeddings = self._compute_text_embeddings()
            self._text_embedding_cache[self.cache_key] = {
                key: value.detach() if isinstance(value, torch.Tensor) else value
                for key, value in self.text_embeddings.items()
            }

    @staticmethod
    def get_cls_idx(path):
        words: list[str] = []
        class_indices: list[int] = []
        for class_index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
            prompts = [part.strip() for part in line.split(",") if part.strip()]
            if not prompts:
                raise ValueError(f"PromptBank has an empty class row at line {class_index + 1}: {path}")
            words.extend(prompts)
            class_indices.extend([class_index] * len(prompts))
        return words, class_indices


    def _compute_text_embeddings(self):
        with torch.no_grad():
            outputs = self.processor.model.backbone.forward_text(
                self.query_words,
                device=self.device,
            )
        language_features = outputs.get("language_features")
        language_mask = outputs.get("language_mask")
        if language_features is None or language_mask is None:
            raise RuntimeError(
                "SAM 3 forward_text did not return required language_features and language_mask"
            )
        return {
            "language_features": language_features,
            "language_mask": language_mask,
            "language_embeds": outputs.get("language_embeds"),
        }

    def build_backbone_out_by_indices(self, backbone_out_image_only, query_indices):
        query_indices = query_indices.to(device=self.device, dtype=torch.long)

        backbone_out = dict(backbone_out_image_only)
        language_features = self.text_embeddings["language_features"].index_select(1, query_indices)
        language_mask = self.text_embeddings["language_mask"].index_select(0, query_indices)
        language_embeds = self.text_embeddings.get("language_embeds")

        if language_embeds is not None:
            language_embeds = language_embeds.index_select(1, query_indices)

        backbone_out["language_features"] = language_features
        backbone_out["language_mask"] = language_mask
        if language_embeds is not None:
            backbone_out["language_embeds"] = language_embeds
        return backbone_out
