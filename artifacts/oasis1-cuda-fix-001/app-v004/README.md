# NeuroPath AI · app-v004 exploratory explainability

This local Streamlit app preserves app-v003 and reuses the verified `../inference-v003/inference.py` loader, checkpoint, preprocessing, and subject-level prediction. It adds an exploratory Grad-CAM display only; it does not retrain, alter weights, modify prediction tensors, access held-out patients, upload images, or implement Grad-CAM as a clinical explanation.

## Explainability method

Grad-CAM targets `features[6]`, the final `Conv2d` layer in `ImageClassifier2D`, before the third max-pool and deterministic grid pool. The map is calculated from a separate real forward/backward pass for class 1 (CDR > 0), normalized for display, and resized to 128×128. The existing nine-slice prediction remains untouched; app-v004 checks that each Grad-CAM pass reproduces the corresponding positive-class score.

Interpret maps only as “regions that most influenced the model prediction” and an “exploratory model explainability map”. They are “not a validated lesion localization”. Anatomical laterality is UNVERIFIED, and no clinical diagnosis claim is supported.

## Run

```powershell
cd "C:\neuropath ai\artifacts\oasis1-cuda-fix-001\app-v004"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest -v test_app.py
streamlit run app.py
```

Upload exactly nine 208×208 grayscale PNGs from one frozen training subject. The app provides Original, Heatmap, and Overlay tabs, an opacity slider, slice browsing, a real slice-score bar chart, and a JSON research report. Display brightness/contrast controls are deliberately omitted from the inference path; the explainability pass uses the original uploaded images and frozen preprocessing.

Known limitations: Grad-CAM is resolution-limited and can highlight shortcuts or diffuse regions; it does not establish causality, anatomy, lesions, or clinical validity. Browser startup was not exercised in the current environment unless Streamlit is installed.
