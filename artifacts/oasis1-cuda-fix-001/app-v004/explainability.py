"""Exploratory Grad-CAM for the frozen OASIS-1 ImageClassifier2D."""
from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from inference import ImageClassifier2D, preprocess_image

TARGET_LAYER_NAME = "features[6]"

def _target_layer(model: ImageClassifier2D) -> torch.nn.Module:
    # features = conv/relu/pool repeated three times, then deterministic pool.
    # features[6] is the final 3rd convolution and retains a spatial map.
    return model.features[6]

def gradcam(model: ImageClassifier2D, image: Image.Image, target_class: int = 1) -> tuple[np.ndarray, float]:
    """Return a normalized HxW map and the real positive-class probability.

    This is a visualization-only forward/backward pass. It does not alter the
    model weights, prediction threshold, or tensors used by subject inference.
    """
    model.eval()
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    layer = _target_layer(model)
    handle_forward = layer.register_forward_hook(lambda _, __, out: activations.append(out))
    handle_gradient = layer.register_full_backward_hook(lambda _, grad_in, grad_out: gradients.append(grad_out[0]))
    try:
        tensor = preprocess_image(image).unsqueeze(0).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        probability = float(logits.softmax(dim=1)[0, 1].detach().cpu())
        if target_class not in (0, 1): raise ValueError("target_class must be 0 or 1")
        logits[0, target_class].backward()
        if not activations or not gradients: raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")
        feature = activations[-1].detach()[0]
        gradient = gradients[-1].detach()[0]
        weights = gradient.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * feature).sum(dim=0))
        cam = torch.nn.functional.interpolate(cam[None, None], size=(128, 128), mode="bilinear", align_corners=False)[0, 0]
        cam = cam.cpu().numpy()
        scale = float(cam.max())
        if scale > 0: cam = cam / scale
        else: cam = np.zeros_like(cam)
        return cam.astype(np.float32), probability
    finally:
        handle_forward.remove(); handle_gradient.remove(); model.zero_grad(set_to_none=True)

def heatmap_rgb(heatmap: np.ndarray) -> Image.Image:
    """Small dependency-free cyan-to-magenta heatmap renderer."""
    values = np.clip(heatmap, 0, 1)
    red = (255 * values).astype(np.uint8)
    green = (255 * (1 - np.abs(values - .5) * 2)).astype(np.uint8)
    blue = (255 * (1 - values)).astype(np.uint8)
    return Image.fromarray(np.stack([red, green, blue], axis=-1), mode="RGB")

def overlay_rgb(image: Image.Image, heatmap: np.ndarray, opacity: float) -> Image.Image:
    base = image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    layer = heatmap_rgb(heatmap)
    return Image.blend(base, layer, float(np.clip(opacity, 0, 1)))
