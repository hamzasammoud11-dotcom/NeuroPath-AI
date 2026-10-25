"""Read-only discovery of real per-epoch training artifacts."""
from __future__ import annotations
import json, re
from pathlib import Path

def _number(value, *keys):
    if isinstance(value, dict):
        for key in keys:
            if isinstance(value.get(key), (int,float)): return float(value[key])
    return None

def load_curves(root: Path, variant: str = "horizontal_flip") -> dict:
    """Return only metrics found in summary/history/validation JSON files."""
    candidates=[]
    for summary_path in root.rglob("summary.json"):
        try: summary=json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): continue
        if summary.get("variant") not in (None, variant): continue
        files=sorted(summary_path.parent.glob("validation-*.json"))
        history_path=summary_path.parent/"history.json"
        candidates.append((summary_path,summary,files,history_path))
    if not candidates: return {"available":False,"rows":[],"summary":None,"path":None}
    summary_path,summary,files,history_path=candidates[0]
    rows=[]
    if history_path.is_file():
        try:
            history=json.loads(history_path.read_text(encoding="utf-8"))
            for item in history:
                metrics=item.get("validation",item.get("metrics",{})); rows.append({"epoch":item.get("epoch"),"training_loss":item.get("train_loss",item.get("training_loss")),"validation_auroc":_number(metrics,"auroc","validation_auroc"),"validation_balanced_accuracy":_number(metrics,"balanced_accuracy","validation_balanced_accuracy")})
        except (OSError, json.JSONDecodeError): pass
    for path in files:
        try: payload=json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): continue
        metrics=payload.get("original_metrics",payload.get("metrics",payload.get("validation",payload)))
        match=re.search(r"validation-(\d+)",path.name); epoch=payload.get("epoch",int(match.group(1)) if match else None)
        rows.append({"epoch":epoch,"training_loss":payload.get("train_loss",payload.get("training_loss")),"validation_auroc":_number(metrics,"auroc","validation_auroc"),"validation_balanced_accuracy":_number(metrics,"balanced_accuracy","validation_balanced_accuracy")})
    dedup={row.get("epoch"):row for row in rows if row.get("epoch") is not None}; rows=sorted(dedup.values(),key=lambda row:row["epoch"])
    return {"available":bool(rows),"rows":rows,"summary":summary,"path":str(summary_path.parent)}
