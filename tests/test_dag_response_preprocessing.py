"""Protect the response protocol that determines the discrete DAG decisions."""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
from PIL import Image

from core.dag_responses import (
    _extract_prompt,
    IMAGE_MAX_SIDES,
    load_calibration_image,
)


class DAGResponsePreprocessingTests(unittest.TestCase):
    def test_coordinates_are_normalized_on_cpu_before_device_transfer(self):
        from types import SimpleNamespace
        from core.dag_response_processor import configure_recorded_numerics

        calls = []
        class Decoder(torch.nn.Module):
            compilable_stored_size = (72, 72)
            compilable_cord_cache = (torch.full((72,), -1.0),) * 2
            coord_cache = {"stale": True}
            clamp_presence_logits = True
            clamp_presence_logit_max_val = 10.0

            def forward(self, value):
                return (value, value, torch.tensor([-20.0, 20.0]), value)

            @staticmethod
            def _get_coords(height, width, device):
                calls.append(device)
                return (torch.arange(height, dtype=torch.float32, device=device) / height,
                        torch.arange(width, dtype=torch.float32, device=device) / width)

        model = torch.nn.Module()
        model.register_parameter("weight", torch.nn.Parameter(torch.zeros(())))
        model.transformer = SimpleNamespace(decoder=Decoder())
        configure_recorded_numerics(model)
        self.assertEqual(calls, ["cpu"])
        self.assertTrue(torch.equal(model.transformer.decoder.compilable_cord_cache[0],
                                    torch.arange(72, dtype=torch.float32) / 72))
        self.assertEqual(model.transformer.decoder.coord_cache, {})
        result = model.transformer.decoder(torch.zeros(()))
        self.assertTrue(torch.equal(result[2], torch.tensor([-10.0, 10.0])))

    def test_later_fused_image_mlp_is_rejected_before_response_collection(self):
        from core.dag_response_processor import check_sam3_runtime

        self.assertIn("image_mlp", check_sam3_runtime())
        unsupported = type("Mlp", (), {"__module__": "sam3.model.vitdet"})
        with patch("sam3.model.vitdet.Mlp", unsupported):
            with self.assertRaisesRegex(RuntimeError, "Unsupported SAM 3 image MLP"):
                check_sam3_runtime()

    def test_adapter_exposes_all_branches_without_an_external_processor_patch(self):
        from core.dag_response_processor import DAGResponseProcessor

        class Model:
            def forward_grounding(self, **kwargs):
                return {
                    "pred_logits": torch.tensor([[[3.0], [-3.0]]]),
                    "pred_masks": torch.zeros((1, 2, 2, 3)),
                    "presence_logit_dec": torch.tensor([[3.0]]),
                    "semantic_seg": torch.zeros((1, 1, 2, 3)),
                }

        processor = object.__new__(DAGResponseProcessor)
        processor.model = Model()
        processor.find_stage = None
        processor.confidence_threshold = 0.5
        output = processor._forward_grounding({
            "backbone_out": {}, "geometric_prompt": None,
            "original_height": 6, "original_width": 9,
        })
        self.assertEqual(tuple(output["semantic_mask_logits"].shape), (1, 1, 6, 9))
        self.assertEqual(tuple(output["masks_logits"].shape), (1, 1, 6, 9))
        self.assertEqual(output["object_score"].numel(), 1)
        self.assertAlmostEqual(float(output["presence_score"]), float(torch.sigmoid(torch.tensor(3.0))))

    def test_large_calibration_image_reaches_processor_at_original_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.png"
            original = Image.new("RGB", (1920, 1080), (15, 73, 142))
            original.putpixel((1201, 699), (251, 11, 41))
            original.save(path)

            loaded = load_calibration_image(path)

            self.assertEqual(loaded.size, original.size)
            self.assertEqual(loaded.tobytes(), original.tobytes())

    def test_stage2_uses_the_recorded_dataset_resize_protocol(self):
        self.assertIsNone(IMAGE_MAX_SIDES["udd5"])
        self.assertIsNone(IMAGE_MAX_SIDES["openearthmap"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.png"
            original = Image.new("RGB", (1920, 1080), (15, 73, 142))
            original.save(path)
            loaded = load_calibration_image(path, IMAGE_MAX_SIDES["vdd"])
            expected = original.resize((1008, 567), Image.Resampling.BILINEAR)
            self.assertEqual(loaded.tobytes(), expected.tobytes())

    def test_raw_response_uses_mask_scores_and_one_presence_factor(self):
        class Processor:
            def reset_all_prompts(self, state):
                pass

            def set_text_prompt(self, state, prompt):
                return {
                    "semantic_mask_logits": torch.full((1, 1, 2, 3), 0.2),
                    "masks_logits": torch.full((1, 1, 2, 3), 0.8),
                    "object_score": torch.tensor([0.5]),
                    "presence_score": torch.tensor(0.75),
                }

        output = _extract_prompt(
            Processor(), {}, "building", (9, 6), torch.device("cpu"),
            response_size=128,
        )

        self.assertEqual(tuple(output["fused"].shape), (128, 128))
        self.assertTrue(torch.allclose(output["semantic"].float(), torch.full((128, 128), 0.2), atol=1e-4))
        self.assertTrue(torch.allclose(output["instance"].float(), torch.full((128, 128), 0.4), atol=1e-4))
        self.assertTrue(torch.allclose(output["fused"].float(), torch.full((128, 128), 0.3), atol=1e-4))


if __name__ == "__main__":
    unittest.main()
