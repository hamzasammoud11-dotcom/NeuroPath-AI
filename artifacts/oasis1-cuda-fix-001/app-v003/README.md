# NeuroPath AI · Futuristic dashboard v003

This local Streamlit application is a visual redesign of app-v002. It preserves the validated `inference-v003` loader, checkpoint, preprocessing, and training-partition guard. It never trains, modifies weights, opens dataset images, sends uploads externally, or evaluates held-out test patients.

## Run

```powershell
cd "C:\neuropath ai\artifacts\oasis1-cuda-fix-001\app-v003"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest -v test_app.py
streamlit run app.py
```

The verified checkpoint remains in `../inference-v003/neuropath-oasis1-cnn-v003.pt` and must have SHA-256 `349ccc59ec3ba8dc834fdca7f426eb50fc8d1e5be38cb8d0bd58ab7a30c73e60`.

The UI supports a sidebar workspace, guarded nine-slice upload, sequence validation, 3×3 gallery, slice explorer, display-only brightness/contrast controls, real checkpoint loading, subject-level score, slice output chart, model metadata, limitations, reset-by-rerun, and a JSON research report containing only verified model outputs. It intentionally has no Grad-CAM or clinical explanation.
