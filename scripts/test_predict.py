import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import requests

API_URL = "http://127.0.0.1:8000/predict"

# Create a small synthetic NIfTI volume
vol = np.random.rand(64, 64, 64).astype("float32")
img = nib.Nifti1Image(vol, affine=np.eye(4))
with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
    tmp_path = Path(f.name)
    img.to_filename(str(tmp_path))

payload = {
    # Match server's default TABULAR_DIM (14) to avoid shape mismatch during testing
    "clinical_features": [65.0, 26.0, 0.0] + [0.0] * 11,
    "gender": "female",
}

files = {"nifti_file": (tmp_path.name, open(tmp_path, "rb"))}
data = {"clinical_json": json.dumps(payload)}

print("Posting to", API_URL)
try:
    r = requests.post(API_URL, files=files, data=data, timeout=120)
    print("Status", r.status_code)
    try:
        print(r.json())
    except Exception:
        print(r.text[:1000])
except Exception as e:
    print("Request failed:", e)
finally:
    try:
        files["nifti_file"][1].close()
    except Exception:
        pass
    try:
        tmp_path.unlink()
    except Exception:
        pass
