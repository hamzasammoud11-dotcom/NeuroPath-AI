"""Pure UI helpers; inference remains delegated to inference-v003."""
from __future__ import annotations
import json
from datetime import datetime, timezone

def display_report(result: dict, subject_id: str) -> dict:
    return {
        "subject_id": subject_id,
        "cdr_score": result["cdr_score"],
        "prediction": result["prediction"],
        "threshold": result["threshold"],
        "slice_positive_probabilities": result["slice_positive_probabilities"],
        "model_scope": "Exploratory non-lateralized CDR classification only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "warning": "Research use only — not for medical diagnosis.",
    }

def report_bytes(result: dict, subject_id: str) -> bytes:
    return json.dumps(display_report(result, subject_id), indent=2, ensure_ascii=False).encode("utf-8")

def preview_image(image, brightness: float, contrast: float):
    """Adjust only a display copy; the original image is never passed here."""
    from PIL import ImageEnhance
    return ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(brightness)).enhance(contrast)
