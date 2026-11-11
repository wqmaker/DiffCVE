import torch
from torch import nn

from diffcve.utils import instantiate_from_config, resolve_path


class FrozenOpenCLIPEmbedder(nn.Module):
    def __init__(self, cond_model_path, arch="ViT-H-14", layer="penultimate", device="cpu"):
        super().__init__()
        if layer not in ("last", "penultimate"):
            raise ValueError("OpenCLIP layer must be last or penultimate")
        checkpoint_path = resolve_path(cond_model_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        import open_clip

        model = open_clip.create_model(arch, device=torch.device("cpu"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
        del model.visual
        self.model = model.eval().requires_grad_(False).to(device)
        self.layer_idx = 0 if layer == "last" else 1
        self.tokenize = open_clip.tokenize

    @torch.no_grad()
    def forward(self, text):
        tokens = self.tokenize(text).to(self.model.token_embedding.weight.device)
        x = self.model.token_embedding(tokens) + self.model.positional_embedding
        x = x.permute(1, 0, 2)
        blocks = self.model.transformer.resblocks
        for index, block in enumerate(blocks):
            if index == len(blocks) - self.layer_idx:
                break
            x = block(x, attn_mask=self.model.attn_mask)
        return self.model.ln_final(x.permute(1, 0, 2))

    def encode(self, text):
        return self(text)


def qp_interpolation_weights(qp, anchors):
    anchors = sorted(anchors)
    if not anchors:
        raise ValueError("At least one text anchor is required")
    if qp in anchors:
        return {qp: 1.0}
    if qp < anchors[0]:
        return {anchors[0]: 1.0}
    if qp > anchors[-1]:
        return {anchors[-1]: 1.0}
    for lower, upper in zip(anchors, anchors[1:]):
        if lower < qp < upper:
            weight = (qp - lower) / (upper - lower)
            return {lower: 1.0 - weight, upper: weight}


def interpolate_text_conditions(anchors, qps):
    conditions = {}
    for qp in sorted(set(qps)):
        weights = qp_interpolation_weights(qp, anchors)
        reference = anchors[next(iter(weights))]
        if len(weights) == 1:
            conditions[qp] = reference
        else:
            combined = sum(anchors[key].float() * weight for key, weight in weights.items())
            if not torch.isfinite(combined).all():
                raise ValueError(f"QP {qp}: nonfinite interpolated text condition")
            conditions[qp] = combined.to(dtype=reference.dtype)
    return conditions


def load_text_conditions(path, encoder_config=None, prompts=None, qps=None):
    if encoder_config is None:
        if prompts is not None:
            raise ValueError("text_prompts requires cond_stage_config")
        conditions = torch.load(resolve_path(path), map_location="cpu")
    else:
        if not prompts or any(not isinstance(text, str) or not text.strip() for text in prompts.values()):
            raise ValueError("text_prompts must map QPs to nonempty text strings")
        encoder = instantiate_from_config(encoder_config)
        with torch.no_grad():
            conditions = {qp: encoder.encode([text]).detach().cpu() for qp, text in prompts.items()}
    if not isinstance(conditions, dict) or not conditions:
        raise ValueError("Text conditions must be a nonempty QP-to-embedding dictionary")
    result = {}
    for qp, embedding in conditions.items():
        key = int(qp)
        if key in result:
            raise ValueError(f"Duplicate text condition for QP {key}")
        if (not isinstance(embedding, torch.Tensor) or embedding.shape != (1, 77, 1024)
                or not embedding.is_floating_point() or not torch.isfinite(embedding).all()):
            raise ValueError(f"QP {key}: expected a finite floating-point embedding of shape [1, 77, 1024]")
        result[key] = embedding
    return result if qps is None else interpolate_text_conditions(result, qps)
