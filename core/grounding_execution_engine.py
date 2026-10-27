"""Execute the fixed frozen SAM 3 response path for one image."""
from contextlib import nullcontext
import torch
from sam3.model.data_misc import FindStage
from core.result_reducer import GroundingResultReducer


class GroundingExecutionEngine:
    """Return the four per-prompt responses required by SICH."""

    def __init__(self, processor, prompt_bank, device, prompt_batch_size=15, confidence_threshold=0.5):
        self.processor = processor
        self.model = processor.model
        self.prompt_bank = prompt_bank
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.num_queries = int(prompt_bank.num_queries)
        self.prompt_batch_size = max(1, min(int(prompt_batch_size), self.num_queries))
        self.reducer = GroundingResultReducer(self.device, confidence_threshold)
        self._find_stage_cache = {}
        self._dummy_prompt_cache = {}
        self._effective_prompt_batch_cache = {}

    def infer_image(self, image):
        with torch.no_grad():
            autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
            with autocast:
                inference_state = self.processor.set_image(image)
            backbone_out = self._float_backbone(inference_state["backbone_out"])
            self._prepare_cached_image_features(backbone_out)
            output = self._run_prompt_batches(backbone_out, (image.height, image.width))
        return {key: value[0] for key, value in output.items()}

    def _run_prompt_batches(self, backbone_out, output_size):
        height, width = (int(output_size[0]), int(output_size[1]))
        query_indices = torch.arange(self.num_queries, device=self.device, dtype=torch.long)
        effective_batch = min(self.num_queries, self.prompt_batch_size)
        cache_key = (height, width, self.num_queries)
        effective_batch = min(effective_batch, self._effective_prompt_batch_cache.get(cache_key, effective_batch))
        while True:
            result = {
                "semantic_logits": torch.zeros((1, self.num_queries, height, width), device=self.device),
                "instance_best_logits": torch.zeros((1, self.num_queries, height, width), device=self.device),
                "presence_scores": torch.zeros((1, self.num_queries), device=self.device),
                "max_instance_scores": torch.zeros((1, self.num_queries), device=self.device),
            }
            try:
                for start in range(0, self.num_queries, effective_batch):
                    end = min(start + effective_batch, self.num_queries)
                    prompt_indices = query_indices[start:end]
                    find_input, geometric_prompt = self._prompt_context(end - start)
                    prompt_backbone = self.prompt_bank.build_backbone_out_by_indices(
                        backbone_out_image_only=backbone_out, query_indices=prompt_indices)
                    outputs = self.model.forward_grounding(
                        backbone_out=prompt_backbone, find_input=find_input,
                        geometric_prompt=geometric_prompt, find_target=None)
                    reduced = self.reducer.reduce_slot_outputs(outputs, end - start, (height, width))
                    for key, value in reduced.items():
                        result[key][:, start:end] = value.unsqueeze(0).to(self.device)
                return result
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
                if not is_oom or effective_batch <= 1:
                    raise
                torch.cuda.empty_cache()
                effective_batch = max(1, effective_batch // 2)
                self._effective_prompt_batch_cache[cache_key] = effective_batch
                print(f"SAM 3 prompt batch reduced to {effective_batch} after CUDA out-of-memory")

    def _prompt_context(self, prompt_count):
        key = int(prompt_count)
        if key not in self._find_stage_cache:
            self._find_stage_cache[key] = FindStage(
                img_ids=torch.zeros(key, device=self.device, dtype=torch.long),
                text_ids=torch.arange(key, device=self.device, dtype=torch.long),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )
        if key not in self._dummy_prompt_cache:
            self._dummy_prompt_cache[key] = self.model._get_dummy_prompt(num_prompts=key)
        return self._find_stage_cache[key], self._dummy_prompt_cache[key]

    def _prepare_cached_image_features(self, backbone_out):
        num_levels = self.model.num_feature_levels
        vis_feats = backbone_out["backbone_fpn"][-num_levels:]
        vis_pos_enc = backbone_out["vision_pos_enc"][-num_levels:]
        backbone_out["_cached_img_feats"] = [
            feat.flatten(2).permute(2, 0, 1).contiguous() for feat in vis_feats
        ]
        backbone_out["_cached_img_pos_embeds"] = [
            pos.flatten(2).permute(2, 0, 1).contiguous() for pos in vis_pos_enc
        ]
        backbone_out["_cached_vis_feat_sizes"] = [pos.shape[-2:] for pos in vis_pos_enc]

    def _float_backbone(self, value):
        if isinstance(value, dict):
            return {key: self._float_backbone(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._float_backbone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._float_backbone(item) for item in value)
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.to(dtype=torch.float32)
        return value
