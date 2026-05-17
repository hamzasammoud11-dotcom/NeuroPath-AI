from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - optional dependency
    nib = None


class GradCAM3D:
    """Grad-CAM for 3D CNN backbones."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def _forward_hook(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self.activations = output.detach()

        def _backward_hook(_: nn.Module, __: tuple[torch.Tensor, ...], grad_output: tuple[torch.Tensor, ...]) -> None:
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(_forward_hook)
        self.target_layer.register_full_backward_hook(_backward_hook)

    def generate(
        self,
        volume: torch.Tensor,
        target_class: int | None = None,
        extra_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self._forward(volume, extra_features)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[:, target_class].sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients")

        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = self._normalize(cam)
        return cam

    def generate_upsampled(
        self,
        volume: torch.Tensor,
        target_shape: tuple[int, int, int],
        target_class: int | None = None,
        extra_features: torch.Tensor | None = None,
        threshold: float | None = 0.6,
    ) -> torch.Tensor:
        cam = self.generate(volume, target_class=target_class, extra_features=extra_features)
        if threshold is not None:
            cam = torch.where(cam >= threshold, cam, torch.zeros_like(cam))
        cam = cam.unsqueeze(1)
        cam = self._safe_interpolate_3d(cam, target_shape)
        return cam.squeeze(1)

    def _forward(self, volume: torch.Tensor, extra_features: torch.Tensor | None) -> torch.Tensor:
        if extra_features is None:
            return self.model(volume)
        return self.model(volume, extra_features)

    @staticmethod
    def _normalize(cam: torch.Tensor) -> torch.Tensor:
        cam_min = cam.flatten(1).min(dim=1, keepdim=True).values
        cam_max = cam.flatten(1).max(dim=1, keepdim=True).values
        normalized = (cam - cam_min[:, None, None, None]) / (cam_max[:, None, None, None] - cam_min[:, None, None, None] + 1e-6)
        return normalized

    @staticmethod
    def _safe_interpolate_3d(cam: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        if cam.dim() != 5:
            raise ValueError(f"Expected 5D tensor for interpolation, got {cam.dim()}D")
        if len(target_shape) != 3:
            raise ValueError("target_shape must be a 3-tuple for 3D interpolation")
        return F.interpolate(cam, size=target_shape, mode="trilinear", align_corners=False)


def find_default_target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "layer4"):
        layer4 = getattr(model, "layer4")
        if isinstance(layer4, (nn.Sequential, list)) and len(layer4) > 0:
            block = layer4[-1]
            if hasattr(block, "conv3"):
                return block.conv3
            if hasattr(block, "conv2"):
                return block.conv2
            return block
        return layer4
    if hasattr(model, "encoder"):
        encoder = getattr(model, "encoder")
        if isinstance(encoder, nn.Sequential):
            for layer in reversed(encoder):
                if isinstance(layer, nn.Conv3d):
                    return layer
            return encoder[-1]
    raise ValueError("Unable to infer target layer for Grad-CAM")


def export_cam_to_nifti(
    cam: torch.Tensor,
    reference_path: str,
    output_path: str,
) -> None:
    if nib is None:
        raise ImportError("nibabel is required to export NIfTI files")
    reference = nib.load(reference_path)
    data = cam.detach().cpu().numpy()
    if data.ndim == 4:
        data = data[0]
    nifti = nib.Nifti1Image(data.astype("float32"), reference.affine)
    nib.save(nifti, output_path)
