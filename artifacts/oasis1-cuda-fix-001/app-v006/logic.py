from __future__ import annotations
import json
from datetime import datetime,timezone
def report_bytes(result,subject_id):
    return json.dumps({"subject_id":subject_id,"cdr_score":result["cdr_score"],"prediction":result["prediction"],"threshold":result["threshold"],"slice_positive_probabilities":result["slice_positive_probabilities"],"scope":"Exploratory non-lateralized CDR classification only","generated_at_utc":datetime.now(timezone.utc).isoformat(),"warning":"Research use only — not for medical diagnosis."},indent=2,ensure_ascii=False).encode()
def signature(ordered): return tuple(name for name,_ in ordered) if ordered else None
