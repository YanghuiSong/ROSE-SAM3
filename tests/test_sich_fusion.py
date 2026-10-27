import unittest

import torch

from core.fusion_heads import semantic_instance_cooperative_fusion


class SICHFusionTests(unittest.TestCase):
    def test_response_is_independent_of_neighboring_pixels(self):
        semantic = torch.full((1, 5, 5), 0.4)
        isolated = torch.zeros_like(semantic)
        isolated[:, 2, 2] = 0.8
        uniform = torch.full_like(semantic, 0.8)
        score = torch.ones(1)
        isolated_response = semantic_instance_cooperative_fusion(
            semantic, isolated, score, score, score)
        uniform_response = semantic_instance_cooperative_fusion(
            semantic, uniform, score, score, score)
        torch.testing.assert_close(
            isolated_response[:, 2, 2], uniform_response[:, 2, 2],
            rtol=0, atol=0)

    def test_disagreement_suppresses_confident_semantic_response(self):
        semantic = torch.full((1, 1, 1), 0.9)
        instance = torch.full_like(semantic, 0.1)
        score = torch.ones(1)
        result = semantic_instance_cooperative_fusion(
            semantic, instance, score, score, score)
        self.assertLess(result.item(), semantic.item())
        torch.testing.assert_close(
            result, torch.full_like(result, 0.87952), rtol=0, atol=1e-7)

    def test_matches_pointwise_equation_across_query_chunks(self):
        generator = torch.Generator().manual_seed(42)
        semantic = torch.rand((13, 7, 9), generator=generator)
        instance = torch.rand((13, 7, 9), generator=generator)
        presence = torch.rand(13, generator=generator)
        instance_scores = torch.rand(13, generator=generator)
        weights = torch.tensor([0.0, 0.12, 0.18, 0.25, 0.30, 0.45, 1.0,
                                0.12, 0.30, 0.45, 1.0, 0.0, 0.25])
        agreement = (1.0 - (semantic - instance).abs()).clamp(0.0, 1.0)
        uncertainty = (1.0 - (semantic - 0.5).abs() * 2.0).clamp(0.0, 1.0)
        positive = (instance - semantic).clamp_min(0.0)
        p = presence[:, None, None]
        m = instance_scores[:, None, None]
        w = weights[:, None, None]
        anchor = semantic + (p * m * agreement).clamp(0.0, 1.0) * positive
        gate = p * m * w
        residual_gate = gate * (0.35 + 0.65 * uncertainty) * (0.35 + 0.65 * agreement)
        conflict = (1.0 - agreement) * gate * (1.0 - uncertainty)
        suppression = 0.04 * conflict * (semantic - instance).clamp_min(0.0)
        refined = semantic + residual_gate * positive - suppression
        expected = (anchor + w * (refined - anchor)).clamp(0.0, 1.0 - 1e-4)
        actual = semantic_instance_cooperative_fusion(
            semantic, instance, presence, instance_scores, weights)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
