import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from core.prompt_bank import PromptBank


class PromptBankTests(unittest.TestCase):
    def test_cached_text_features_and_prompt_indexing(self):
        calls = []

        def forward_text(words, device):
            calls.append(list(words))
            count = len(words)
            return {
                "language_features": torch.arange(8 * count).reshape(2, count, 4).float(),
                "language_mask": torch.zeros((count, 2), dtype=torch.bool),
                "language_embeds": torch.arange(8 * count).reshape(2, count, 4).float() + 100,
            }

        processor = types.SimpleNamespace(
            model=types.SimpleNamespace(backbone=types.SimpleNamespace(forward_text=forward_text))
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            PromptBank._text_embedding_cache, {}, clear=True
        ):
            path = Path(directory) / "bank.txt"
            path.write_text("building,roof\nroad\n", encoding="utf-8")
            bank = PromptBank(processor, path, "cpu")
            cached = PromptBank(processor, path, "cpu")
            self.assertEqual(len(calls), 1)
            self.assertEqual(bank.query_words, ["building", "roof", "road"])
            self.assertEqual(bank.query_idx.tolist(), [0, 0, 1])
            self.assertEqual(cached.num_cls, 2)

            indices = torch.tensor([2, 0])
            image_features = torch.zeros((1, 4, 2, 2))
            output = bank.build_backbone_out_by_indices({"image": image_features}, indices)
            self.assertIs(output["image"], image_features)
            for key in ("language_features", "language_embeds"):
                self.assertTrue(torch.equal(output[key], bank.text_embeddings[key][:, indices]))
            self.assertTrue(torch.equal(output["language_mask"], bank.text_embeddings["language_mask"][indices]))


if __name__ == "__main__":
    unittest.main()
