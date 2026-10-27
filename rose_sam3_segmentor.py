"""Frozen SAM 3 inference with semantic-instance cooperative fusion."""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from PIL import Image
from mmengine.structures import PixelData
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS

from core.fusion_heads import semantic_instance_cooperative_fusion


@MODELS.register_module()
class ROSESAM3Segmentor(BaseSegmentor):
    """The fixed ROSE-SAM3 inference pipeline."""

    def __init__(
        self,
        classname_path,
        device="cuda",
        prob_thd=0.0,
        bg_idx=0,
        confidence_threshold=0.5,
        prompt_batch_size=15,
    ):
        super().__init__()
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.prob_thd = float(prob_thd)
        self.bg_idx = int(bg_idx)
        self.confidence_threshold = float(confidence_threshold)
        self.prompt_batch_size = max(1, int(prompt_batch_size))

        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from core.dag_response_processor import check_sam3_runtime, configure_recorded_numerics
        from core.prompt_bank import PromptBank
        from core.grounding_execution_engine import GroundingExecutionEngine

        check_sam3_runtime()

        sam3_checkpoint = os.environ.get(
            "SAM3_CHECKPOINT", "/data/public/sam3/sam3.pt"
        )
        sam3_bpe_path = os.environ.get(
            "SAM3_BPE_PATH", "/data/public/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        )
        if not os.path.isfile(sam3_checkpoint):
            raise FileNotFoundError(f"SAM 3 checkpoint not found: {sam3_checkpoint}")
        if not os.path.isfile(sam3_bpe_path):
            raise FileNotFoundError(f"SAM 3 BPE vocabulary not found: {sam3_bpe_path}")

        model = build_sam3_image_model(
            bpe_path=sam3_bpe_path,
            checkpoint_path=sam3_checkpoint,
            compile=False,
        ).to(self.device)
        self.model = model
        configure_recorded_numerics(model)
        self.processor = Sam3Processor(
            model,
            resolution=1008,
            confidence_threshold=self.confidence_threshold,
            device=self.device,
        )
        self.prompt_bank = PromptBank(
            processor=self.processor,
            classname_path=classname_path,
            device=self.device,
        )
        self.query_words = self.prompt_bank.query_words
        self.query_idx = self.prompt_bank.query_idx
        self.num_queries = self.prompt_bank.num_queries
        self.num_cls = self.prompt_bank.num_cls

        # SICH is applied to the frozen SAM 3 semantic and instance responses.
        # The reducer uses MAX only to expose these two response branches.
        self.execution_engine = GroundingExecutionEngine(
            processor=self.processor,
            prompt_bank=self.prompt_bank,
            device=self.device,
            prompt_batch_size=self.prompt_batch_size,
            confidence_threshold=self.confidence_threshold,
        )
        print(
            f"ROSE-SAM3: {self.num_queries} prompts across {self.num_cls} classes; "
            "frozen SAM 3, SICH, within-class MAX"
        )

    def _infer_components(self, image):
        with torch.no_grad():
            return self.execution_engine.infer_image(image)

    def _sich_query_weights(self, reference):
        """Fixed class-role prior for the instance contribution in SICH."""
        weights = torch.empty(self.num_queries, device=reference.device, dtype=reference.dtype)
        query_classes = self.query_idx.detach().cpu().tolist()
        class_phrases = {
            class_index: " ".join(
                str(self.query_words[index]).lower().replace("_", " ")
                for index, query_class in enumerate(query_classes)
                if int(query_class) == int(class_index)
            )
            for class_index in set(int(value) for value in query_classes)
        }
        for query_index, phrase in enumerate(self.query_words):
            text = str(phrase).lower().replace("_", " ")
            joined = f"{text} {class_phrases[int(query_classes[query_index])]}"
            if any(key in joined for key in (
                "car", "vehicle", "truck", "bus", "ship", "boat", "plane",
                "aircraft", "helicopter", "tank", "storage tank", "baseball",
                "tennis", "basketball", "roundabout", "pool",
            )):
                weight = 1.0
            elif any(key in joined for key in ("building", "roof", "house", "facade")):
                weight = 0.45
            elif any(key in joined for key in ("road", "bridge", "track", "harbor")):
                weight = 0.30
            elif any(key in joined for key in (
                "tree", "forest", "vegetation", "grass", "lawn", "water",
                "barren", "bareland", "cropland", "agricultural",
            )):
                weight = 0.12
            elif any(key in joined for key in ("background", "clutter", "other", "misc")):
                weight = 0.18
            else:
                weight = 0.25
            weights[query_index] = weight
        return weights

    def _sich_prompt_fusion(self, outputs):
        semantic = outputs["semantic_logits"].float().clamp(0.0, 1.0)
        instance = outputs["instance_best_logits"].float().clamp(0.0, 1.0)
        presence = outputs["presence_scores"]
        max_instance = outputs["max_instance_scores"]
        return semantic_instance_cooperative_fusion(
            semantic_probs=semantic,
            instance_probs=instance,
            presence_scores=presence.to(device=semantic.device, dtype=semantic.dtype).clamp(0.0, 1.0),
            max_instance_scores=max_instance.to(device=semantic.device, dtype=semantic.dtype).clamp(0.0, 1.0),
            class_instance_weights=self._sich_query_weights(semantic),
        )

    def _aggregate_to_class_logits(self, query_logits):
        class_logits = torch.full(
            (self.num_cls, *query_logits.shape[-2:]),
            float("-inf"),
            device=query_logits.device,
            dtype=query_logits.dtype,
        )
        query_logits = query_logits.float().clamp(0.0, 1.0)
        for class_index in range(self.num_cls):
            prompt_mask = self.query_idx == class_index
            if prompt_mask.any():
                class_logits[class_index] = query_logits[prompt_mask].amax(dim=0)
        return class_logits.clamp(0.0, 1.0)

    def _inference_single_view(self, image):
        outputs = self._infer_components(image)
        prompt_logits = self._sich_prompt_fusion(outputs)
        return self._aggregate_to_class_logits(prompt_logits)

    def _build_prediction_outputs(self, class_logits):
        seg_pred = torch.argmax(class_logits, dim=0)
        max_values = class_logits.max(dim=0).values
        seg_pred[max_values < self.prob_thd] = self.bg_idx

        return seg_pred

    def predict(self, inputs, data_samples):
        if data_samples is None:
            raise ValueError("ROSE-SAM3 evaluation requires image metadata with source paths")
        for sample in data_samples:
            meta = sample.metainfo
            with Image.open(meta["img_path"]) as source:
                image = source.convert("RGB")
            class_logits = self._inference_single_view(image)
            ori_shape = meta["ori_shape"]
            if isinstance(ori_shape, (list, tuple)) and len(ori_shape) > 2:
                ori_shape = ori_shape[:2]
            if class_logits.shape[-2:] != tuple(ori_shape):
                class_logits = F.interpolate(
                    class_logits.unsqueeze(0),
                    size=ori_shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            seg_pred = self._build_prediction_outputs(class_logits)
            sample.set_data({
                "seg_logits": PixelData(**{"data": class_logits}),
                "pred_sem_seg": PixelData(**{"data": seg_pred.unsqueeze(0)}),
            })
        return data_samples

    def _forward(self, inputs, data_samples=None):
        raise NotImplementedError("ROSE-SAM3 uses the prediction-only evaluation path")


    def encode_decode(self, inputs, batch_img_metas):
        raise NotImplementedError("ROSE-SAM3 uses the prediction-only evaluation path")

    def extract_feat(self, inputs):
        raise NotImplementedError("ROSE-SAM3 has no trainable feature-extraction path")

    def loss(self, inputs, data_samples):
        raise NotImplementedError("ROSE-SAM3 is training-free")
