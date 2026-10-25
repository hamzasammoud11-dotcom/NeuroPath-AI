"""Validated Grad-CAM implementation for the frozen v003 CNN."""
from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from inference import ImageClassifier2D, preprocess_image

TARGET_LAYER_NAME = "features[6]"

def gradcam(model: ImageClassifier2D, image: Image.Image, target_class: int = 1) -> tuple[np.ndarray, float]:
    """Compute a finite normalized 128x128 map for the real positive-class pass."""
    if target_class not in (0, 1): raise ValueError("target_class must be 0 or 1")
    model.eval(); activations=[]; gradients=[]; layer=model.features[6]
    h1=layer.register_forward_hook(lambda _,__,out: activations.append(out))
    h2=layer.register_full_backward_hook(lambda _,__,out: gradients.append(out[0]))
    try:
        x=preprocess_image(image).unsqueeze(0).requires_grad_(True); model.zero_grad(set_to_none=True)
        logits=model(x); probability=float(logits.softmax(1)[0,1].detach().cpu()); logits[0,target_class].backward()
        if not activations or not gradients: raise RuntimeError("Grad-CAM hooks captured no tensors")
        weights=gradients[-1].detach()[0].mean((1,2),keepdim=True); cam=torch.relu((weights*activations[-1].detach()[0]).sum(0))
        cam=torch.nn.functional.interpolate(cam[None,None],size=(128,128),mode="bilinear",align_corners=False)[0,0].cpu().numpy()
        maximum=float(cam.max()); cam=cam/maximum if maximum>0 else np.zeros_like(cam)
        if not np.isfinite(cam).all(): raise RuntimeError("Grad-CAM produced non-finite values")
        return cam.astype(np.float32), probability
    finally:
        h1.remove(); h2.remove(); model.zero_grad(set_to_none=True)

def heatmap_rgb(cam: np.ndarray) -> Image.Image:
    x=np.clip(cam,0,1); return Image.fromarray(np.stack([(255*x).astype(np.uint8),(255*(1-np.abs(x-.5)*2)).astype(np.uint8),(255*(1-x)).astype(np.uint8)],axis=-1),mode="RGB")

def overlay_rgb(image: Image.Image, cam: np.ndarray, opacity: float) -> Image.Image:
    return Image.blend(image.convert("RGB").resize((128,128),Image.Resampling.BILINEAR),heatmap_rgb(cam),float(np.clip(opacity,0,1)))
