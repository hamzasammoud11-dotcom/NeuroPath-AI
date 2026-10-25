# NeuroPath AI · app-v006

App-v006 preserves app-v005 and its inference/Grad-CAM behavior. Overview now reads real per-epoch artifacts (`summary.json`, `history.json`, and `validation-*.json`) and displays available training loss, validation AUROC, and validation balanced accuracy as dark Plotly curves. Missing artifacts produce a clear message; no metric is fabricated. The selected model summary remains CNN v003, `horizontal_flip`, epoch 18, best validation AUROC 0.6975.

Explainability now displays all nine original/Grad-CAM overlay pairs simultaneously in a compact 3×3 grid, with filename and real slice-level score. The shared opacity slider changes visualization only. Results and Analytics remain available, and navigation preserves uploaded images, prediction, and heatmaps in Streamlit session state.

## Launch

```powershell
cd "C:\neuropath ai\artifacts\oasis1-cuda-fix-001\app-v006"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest -v test_app.py
streamlit run app.py
```

The verified checkpoint remains at `../inference-v003/neuropath-oasis1-cnn-v003.pt`; its SHA-256 is checked by the existing loader. No model, preprocessing, study artifact, or held-out patient is modified or evaluated.
