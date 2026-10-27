"""SAM 3 branch decoding and recorded numerical conventions.

Output decoding follows Meta's Sam3Processor (SAM 3 license applies).
"""
import torch

from sam3.model.data_misc import interpolate
from sam3.model.sam3_image_processor import Sam3Processor


SAM3_REFERENCE_COMMIT = "a13e358df479a80938c8bfed0e0754bea8fdb277"


def _recorded_presence_clamp(decoder, inputs, outputs):
    # Apply the decoder's presence-logit clipping before sigmoid.
    if decoder.clamp_presence_logits and outputs[2] is not None:
        limit = decoder.clamp_presence_logit_max_val
        return (*outputs[:2], outputs[2].clamp(min=-limit, max=limit), *outputs[3:])
    return outputs


def configure_recorded_numerics(model):
    """Initialize normalized coordinates on CPU and enable presence clipping."""
    decoder = model.transformer.decoder
    size = decoder.compilable_stored_size
    if size is None:
        raise RuntimeError("SAM 3 decoder has no recorded positional grid")
    device = next(model.parameters()).device
    coordinates = decoder._get_coords(int(size[0]), int(size[1]), device="cpu")
    decoder.compilable_cord_cache = tuple(value.to(device) for value in coordinates)
    decoder.coord_cache = {}
    decoder.register_forward_hook(_recorded_presence_clamp)
    return {
        "box_relative_coordinates": "FP32 CPU normalization, then device transfer",
        "presence_logit_clamp": "apply decoder-configured clipping before sigmoid",
    }


def check_sam3_runtime():
    """Reject the later image MLP that unconditionally casts inputs to BF16."""
    from sam3.model.vitdet import Mlp

    if not Mlp.__module__.startswith("timm."):
        raise RuntimeError(
            "Unsupported SAM 3 image MLP for released-response reproduction: "
            f"{Mlp.__module__}.{Mlp.__name__}. Install the official SAM 3 source "
            f"at commit {SAM3_REFERENCE_COMMIT}; newer fused BF16 image paths "
            "do not implement the recorded response protocol."
        )
    return {
        "sam3_reference_commit": SAM3_REFERENCE_COMMIT,
        "image_mlp": f"{Mlp.__module__}.{Mlp.__name__}",
    }


class DAGResponseProcessor(Sam3Processor):
    @torch.inference_mode()
    def _forward_grounding(self, state):
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )
        if "semantic_seg" not in outputs:
            raise RuntimeError("This SAM 3 model does not expose its semantic_seg branch")
        presence = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        scores = (outputs["pred_logits"].sigmoid() * presence).squeeze(-1)
        keep = scores > self.confidence_threshold
        size = (state["original_height"], state["original_width"])
        masks = interpolate(
            outputs["pred_masks"][keep].unsqueeze(1), size,
            mode="bilinear", align_corners=False,
        ).sigmoid()
        semantic = interpolate(
            outputs["semantic_seg"], size,
            mode="bilinear", align_corners=False,
        ).sigmoid()
        state["masks_logits"] = masks
        state["masks"] = masks > 0.5
        state["scores"] = scores[keep]
        state["object_score"] = scores[keep]
        state["semantic_mask_logits"] = semantic
        state["presence_score"] = presence.squeeze()
        return state
