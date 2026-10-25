"""State/report helpers independent of Streamlit."""
from __future__ import annotations
import json
from datetime import datetime, timezone

def state_signature(ordered): return tuple(name for name,_ in ordered) if ordered else None
def report_bytes(result: dict, subject_id: str) -> bytes:
    payload={"subject_id":subject_id,"cdr_score":result["cdr_score"],"prediction":result["prediction"],"threshold":result["threshold"],"slice_positive_probabilities":result["slice_positive_probabilities"],"scope":"Exploratory non-lateralized CDR classification only","generated_at_utc":datetime.now(timezone.utc).isoformat(),"warning":"Research use only — not for medical diagnosis."}
    return json.dumps(payload,indent=2,ensure_ascii=False).encode()
