"""Image-only baseline with shared training and inference preprocessing."""
import numpy as np
from PIL import Image
import torch
from torch import nn

PREPROCESSING = {"size": 128, "mode": "L", "scale": 255.0, "mean": 0.5, "std": 0.5}
CLASS_MAPPING = {"NonDemented": 0, "VeryMildDemented": 1,
                 "MildDemented": 1, "ModerateDemented": 1}


def preprocess_image(image):
    size = PREPROCESSING["size"]
    image = image.convert(PREPROCESSING["mode"]).resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32).copy() / PREPROCESSING["scale"]
    return torch.from_numpy((array - PREPROCESSING["mean"]) / PREPROCESSING["std"]).unsqueeze(0)


class ImageClassifier2D(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for cin, cout in ((1, 16), (16, 32), (32, 64)):
            layers.extend([nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)])
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((4, 4)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(1024, 2))

    def forward(self, image):
        return self.classifier(self.features(image))


def load_image_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if (payload.get("architecture") != "image2d_v1"
            or payload.get("preprocessing") != PREPROCESSING
            or payload.get("class_mapping") != CLASS_MAPPING):
        raise ValueError("Incompatible 2D checkpoint metadata")
    state = payload["model_state_dict"]
    if any(not torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("Checkpoint contains non-finite weights")
    model = ImageClassifier2D().to(device)
    model.load_state_dict(state, strict=True)
    model.patient_separation_verified = (payload.get("patient_separation_verified") is True
                                         and payload.get("split_unit") == "patient"
                                         and bool(payload.get("patient_id_source")))
    return model.eval()
