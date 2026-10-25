# NeuroPath AI · app-v005

App-v005 preserves the futuristic v003 visual language and keeps app-v003/app-v004 unchanged. It reuses the verified `inference-v003` checkpoint and frozen preprocessing, and imports the existing training-partition guard from `app-v002`. It adds real sidebar pages **Analytics** and **Explainability**.

Analytics uses the nine real positive-class outputs, shows filenames and slice indices, draws a dark Plotly curve with the fixed 0.5 threshold, and reports the subject-level arithmetic mean separately. Results contains no slice chart.

Explainability uses Grad-CAM target layer `features[6]`, the final Conv2d layer in `ImageClassifier2D` before the third max-pool and deterministic grid pool. It targets the positive-class output (class 1), uses real activations and gradients, verifies finite 128×128 maps, and checks that its per-slice forward scores match the original prediction pass. Heatmaps mean regions that influenced the CNN prediction; they are not validated lesion localizations and make no laterality or clinical claim.

Session state preserves uploaded slices, prediction, and heatmaps while switching pages. Inference runs only from the explicit **Run verified CNN** button and is cached thereafter.

## Launch

```powershell
cd "C:\neuropath ai\artifacts\oasis1-cuda-fix-001\app-v005"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest -v test_app.py
streamlit run app.py
```

The checkpoint remains at `../inference-v003/neuropath-oasis1-cnn-v003.pt` with expected SHA-256 `349ccc59ec3ba8dc834fdca7f426eb50fc8d1e5be38cb8d0bd58ab7a30c73e60`. No held-out images are opened or evaluated.
