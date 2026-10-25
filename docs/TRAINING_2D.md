# Train the Kaggle image baseline

This replaces the image trainer's repeated-slice pseudo-3D inputs with a small
2D CNN trained from scratch. It predicts NonDemented (0) versus VeryMildDemented,
MildDemented and ModerateDemented (1). It has no clinical or genetic branch.
It is a baseline; improved accuracy is not guaranteed.

## Data handling

Use `OriginalDataset` from
https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset.
The publisher attributes the originals to tourist55's four-class dataset;
the underlying clinical provenance and patient identities are unverified.

All four class folders are required. The trainer rejects the augmented folder,
checks image readability, removes exact decoded-pixel duplicates, and rejects
duplicate pixels with conflicting severity labels or patient identities.
By default it requires a patient manifest and assigns whole patients to
approximately 70% train, 15% validation, and 15% test, choosing grouped partitions
close to source-class proportions. Patient grouping takes priority over ratios.
All images and visits from a patient remain in one split. Both binary classes
must occur in every split; insufficient patient diversity causes a clear error.
The explicit exploratory override stratifies individual images instead. Rotation is
applied only while loading training images. No pre-augmented images are used.
In exploratory mode, near duplicates and slices of the same patient can remain
across splits. These are image-level results, not verified patient-level results.

Each run saves file paths, dimensions, pixel hashes, split membership, seed,
label mapping and preprocessing in `manifest.json`. Patient IDs are retained
from the supplied mapping or null in exploratory mode. Class weights come only from the training split. Validation balanced
accuracy selects the best checkpoint and controls early stopping. The test
split is evaluated once, after loading that checkpoint. Do not tune experiments
against test results. Reusing this test set for model selection invalidates it
as a held-out evaluation.

## Cloud run

1. Run `python scripts/package_cloud.py` locally.
2. Open `notebooks/train_cloud.ipynb` in Google Colab and select a GPU runtime.
3. Configure the first code cell: run name, split mode and patient-ID provenance.
   Patient mode requires a genuine `patients.csv`; for the current Kaggle data
   without patient IDs, explicitly choose `SPLIT_MODE='image'` and
   `ALLOW_IMAGE_LEVEL_EXPLORATION=True` only for exploratory training.
4. Run the cells, authorize Drive mounting, and upload
   `artifacts/cloud/neuropath-code.zip` when prompted.
5. Keep results in `MyDrive/NeuroPathAI/<run-name>/results` and download the ZIP.

The notebook installs only the training dependencies and downloads Kaggle data
directly in the cloud. Your PC does not run the training. If Kaggle requires
authentication, complete its supported login flow in the notebook; do not
commit credentials. Cloud training and the real dataset have not been tested
by the local fixture checks. The notebook refuses CPU training. Data is read
from the cloud VM; only source code, patient mapping, checkpoints and reports
are persisted to Drive. The resolved Kaggle dataset version is recorded for
subsequent resumes. GPU availability and runtime limits depend on Colab.

`best_model.pt` is the selected inference checkpoint. `last_model.pt` saves the
model, optimizer, completed epoch, early-stopping state, history and RNG states
after every epoch, using temporary files before replacement. A mid-epoch
interruption repeats that epoch. To resume, set `RESUME=True`, retain the run
name/settings/code, reconnect to a GPU runtime and rerun the notebook.
Increasing the total epoch target is allowed; changed data, mapping, training
code or core hyperparameters are rejected. Keep the original code bundle for
resuming that run. Exact numerical reproducibility is not guaranteed across
different GPU/runtime versions. Package versions and runtime metadata are saved
in `requirements-frozen.txt` and `environment.json`; resume metadata is separate.

References: [Colab FAQ](https://research.google.com/colaboratory/faq.html) and
[KaggleHub versioned downloads](https://github.com/Kaggle/kagglehub#download-dataset).

## Patient manifest

Create a UTF-8 CSV with `path,patient_id` columns. Paths are relative to
`OriginalDataset`, for example `NonDemented/scan.jpg`. IDs must come from a
documented source mapping, not image UUIDs or fabricated identities. Every
image must be covered. Blank IDs, duplicate paths, conflicting pixel-to-patient
assignments, and missing files are rejected. Supply `--patient-id-source` with
the metadata export/reference. A verified split means IDs in that supplied
mapping are disjoint; the identity mapping is not independently authenticated.
Patients can have changing labels across visits; grouping preserves them all.
Reports remain per-image metrics, even when splitting by patient.

## Local commands

```powershell
python -m pip install -r requirements-training.txt
python scripts/train_images.py --data-dir data/raw/OriginalDataset --patient-manifest data/patients.csv --patient-id-source "source metadata export" --output-dir artifacts/data-check --prepare-only
python scripts/train_images.py --data-dir data/raw/OriginalDataset --patient-manifest data/patients.csv --patient-id-source "source metadata export" --output-dir artifacts/image2d-run1 --epochs 30
# Explicit exploratory mode when genuine patient IDs are unavailable:
python scripts/train_images.py --data-dir data/raw/OriginalDataset --allow-image-split --output-dir artifacts/exploratory-run1 --epochs 30
```

Output directories must be new unless `--resume` is supplied.
Each epoch processes the entire training split. Reports include sensitivity,
specificity, balanced accuracy, F1, ROC-AUC, Brier score and confusion matrix.
Brier score is reported for inspection; probabilities are not calibrated.

## Use the checkpoint in the existing app

Extract the result ZIP under `artifacts/image2d-run1`, then start a new API process:

```powershell
$env:NEUROPATH_MODEL = 'artifacts/image2d-run1/best_model.pt'
python -m uvicorn neuro_path_ai.api.app:app --host 127.0.0.1 --port 8000
```

The existing dashboard sends uploads to `/predict`. With this checkpoint,
upload a single MRI slice as JPG/PNG/BMP. Full NIfTI/DICOM volumes are rejected.
Montages are unsupported and cannot be reliably identified from pixels alone.
Clinical inputs are ignored and the API returns an explicit warning. The 2D
model does not generate the old 3D heatmaps. Preprocessing and binary class
metadata must match the saved checkpoint or loading fails. Legacy compatible
3D checkpoints remain supported.

Local verification: `python -m unittest discover -s tests -v`.
