import unittest
import sys
import types
from unittest.mock import patch

import torch

def _segmentor_import_stubs():
    """Keep this unit test runnable without the full OpenMMLab stack."""
    mmengine = types.ModuleType("mmengine")
    structures = types.ModuleType("mmengine.structures")
    structures.PixelData = type("PixelData", (), {})
    mmengine.structures = structures
    mmseg = types.ModuleType("mmseg")
    models = types.ModuleType("mmseg.models")
    segmentors = types.ModuleType("mmseg.models.segmentors")
    segmentors.BaseSegmentor = type("BaseSegmentor", (), {})
    registry = types.ModuleType("mmseg.registry")

    class Registry:
        @staticmethod
        def register_module():
            return lambda cls: cls

    registry.MODELS = Registry()
    mmseg.models = models
    models.segmentors = segmentors
    mmseg.registry = registry
    stubs = {
        "mmengine": mmengine,
        "mmengine.structures": structures,
        "mmseg": mmseg,
        "mmseg.models": models,
        "mmseg.models.segmentors": segmentors,
        "mmseg.registry": registry,
    }

    sam3 = types.ModuleType("sam3")
    sam3_model = types.ModuleType("sam3.model")
    data_misc = types.ModuleType("sam3.model.data_misc")

    class FindStage:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    data_misc.FindStage = FindStage
    sam3.model = sam3_model
    sam3_model.data_misc = data_misc
    stubs.update({
        "sam3": sam3,
        "sam3.model": sam3_model,
        "sam3.model.data_misc": data_misc,
    })
    return stubs


with patch.dict(sys.modules, _segmentor_import_stubs()):
    from rose_sam3_segmentor import ROSESAM3Segmentor
    from core.grounding_execution_engine import GroundingExecutionEngine


class SICHInferenceInterfaceTests(unittest.TestCase):
    def test_segmentor_calls_the_sich_fusion_interface(self):
        segmentor = object.__new__(ROSESAM3Segmentor)
        segmentor.num_queries = 1
        segmentor.query_idx = torch.tensor([0])
        segmentor.query_words = ["building"]
        outputs = {
            "semantic_logits": torch.full((1, 4, 4), 0.6),
            "instance_best_logits": torch.full((1, 4, 4), 0.8),
            "presence_scores": torch.tensor([0.9]),
            "max_instance_scores": torch.tensor([0.7]),
        }

        result = ROSESAM3Segmentor._sich_prompt_fusion(segmentor, outputs)

        self.assertEqual(tuple(result.shape), (1, 4, 4))
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(((result >= 0) & (result < 1)).all())

    def test_engine_returns_all_prompt_responses_at_original_resolution(self):
        from PIL import Image

        class Model:
            num_feature_levels = 1

            def _get_dummy_prompt(self, num_prompts):
                return None

            def forward_grounding(self, **kwargs):
                slots = kwargs["find_input"].text_ids.numel()
                return {"semantic_seg": torch.zeros((slots, 1, 3, 5)),
                        "presence_logit_dec": torch.ones((slots, 1))}

        class Bank:
            num_queries = 2

            @staticmethod
            def build_backbone_out_by_indices(**kwargs):
                return {}

        processor = types.SimpleNamespace(
            model=Model(),
            set_image=lambda image: {"backbone_out": {
                "backbone_fpn": [torch.zeros((1, 2, 3, 5))],
                "vision_pos_enc": [torch.zeros((1, 2, 3, 5))]}})
        engine = GroundingExecutionEngine(processor, Bank(), "cpu", prompt_batch_size=2)
        output = engine.infer_image(Image.new("RGB", (12, 8)))
        self.assertEqual(set(output), {"semantic_logits", "instance_best_logits",
                                      "presence_scores", "max_instance_scores"})
        self.assertEqual(tuple(output["semantic_logits"].shape), (2, 8, 12))
        self.assertEqual(tuple(output["instance_best_logits"].shape), (2, 8, 12))
        self.assertTrue(torch.equal(output["instance_best_logits"], torch.zeros((2, 8, 12))))
        expected = .5 * torch.sigmoid(torch.tensor(1.0))
        self.assertTrue(torch.equal(output["semantic_logits"], torch.full((2, 8, 12), expected)))


if __name__ == "__main__":
    unittest.main()
