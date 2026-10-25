"""Pure helpers for explainability reports."""
from __future__ import annotations
import json
from datetime import datetime, timezone

def report_bytes(result: dict, subject_id: str) -> bytes:
    report = {"subject_id": subject_id, "cdr_score": result["cdr_score"], "prediction": result["prediction"], "threshold": result["threshold"], "slice_positive_probabilities": result["slice_positive_probabilities"], "explainability": "Grad-CAM regions that most influenced the model prediction; not a validated lesion localization", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "warning": "Research use only — not for medical diagnosis."}
    return json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")
