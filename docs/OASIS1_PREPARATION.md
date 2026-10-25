# OASIS-1 preparation only: no training or bulk extraction

The source files in this workspace are under
`src/neuro_path_ai/data/oasis1/`. They include two official disc archives, a
clinical XLSX workbook, a reliability XLSX workbook, and the source fact sheet.
The preparation tool accepts CSV or XLSX without changing either original.

Run:

```powershell
python -B scripts/prepare_oasis1.py --data-dir src/neuro_path_ai/data/oasis1 --clinical src/neuro_path_ai/data/oasis1/oasis_cross-sectional-5708aa0a98d82080.xlsx --reliability src/neuro_path_ai/data/oasis1/oasis_cross-sectional-reliability-063c8642b909ee76.xlsx --output-dir artifacts/oasis1-preparation-001 --seed 42
```

Use a new output name for another preparation. Existing directories are refused.
Source archives, metadata, the CNN, the Kaggle pipeline, and prior results are
not modified. No model instance or optimizer is created. No training runs.
`requirements-oasis1-prepare.txt` lists preparation dependencies; pandas,
openpyxl, registration software and a GPU are not required.

## Selection and clinical identity

- Read the released `ID`, `Age` and `CDR` cells directly. Preserve source values
  and their workbook row numbers. Source session IDs must match the documented
  `OAS1_xxxx_MRy` structure. Patient IDs use its documented subject component.
- Keep baseline `MR1` sessions only, age >=60, with CDR 0, 0.5, 1 or 2.
- Set label 0 for CDR 0, and label 1 for CDR 0.5/1/2. Exclude missing ages,
  missing/unexpected CDR values and reliability/repeat sessions. Do not impute
  labels or use MMSE to manufacture them. Missing MMSE does not exclude an
  otherwise eligible image-only subject.
- Match archive session directories and volume filenames exactly to the
  clinical IDs. Missing or ambiguous products are not guessed.
- This is CDR-defined dementia status in the OASIS-1 cohort, not a biomarker
  diagnosis, progression target, or a blanket relabeling of CDR 0.5 as MCI.

The two supplied discs are a **partial release**, not the whole OASIS-1 cohort.
The inventory reports eligible clinical rows absent from these discs, unmatched
archive sessions, and every exclusion reason. Only subjects whose selected
volumes pass preparation QC enter the splits.

## Archive handling and 3D-to-2D policy

The tool streams compressed archives twice: first to inventory members, then to
read only the selected Analyze pairs. Unrelated members must be traversed to
advance the gzip stream, but are never extracted to disk. No `extractall` is
used. Only derived PNG slices, contact sheets, manifests and QC reports are
written. Source archives are hashed before and after processing.

Only the brain-masked, gain-field-corrected averaged T88 product is used:
`PROCESSED/MPRAGE/T88_111/*_mpr_n[34]_anon_111_t88_masked_gfc.hdr/.img`.
RAW repetitions, unmasked images, segmentations, GIFs and MR2 repeats are not
used as extra training subjects. Unpaired or multiply matched products are
excluded.

The fixed preparation policy is:

1. Decode Analyze data with nibabel, respecting header endianness and any
   singleton fourth dimension. Require the documented 176x208x176 spatial
   shape, 1-mm isotropic voxels, finite nonnegative data and nonempty foreground.
2. Measure foreground (positive voxels) percentiles 1 and 99 **per volume**.
   Clip scaled values to [0,1], retain zero background and explicitly round to
   uint8. These statistics are not fitted on validation/test patients.
3. Select nine storage-axis-2 indices at fractions
   0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65 and 0.70 of depth minus one,
   using Python's `round`. Reject empty slices and repeated indices.
4. Apply one fixed `np.rot90` display transform, center-pad to a square, and
   save lossless PNG. Use the existing shared `preprocess_image` unchanged for
   128x128 grayscale normalized CNN inputs. Store both PNG and model-input hashes.

The transverse-plane interpretation comes from the official T88 product
description. Analyze's inferred affine is recorded, **not treated as an
independent anatomical orientation verification**. Left/right orientation,
slice plane, foreground coverage and intensity presentation require review of
the supplied volume provenance and generated contact sheets before training.
No atlas registration or invented spatial affine is introduced.

## Patient-disjoint splits

Split the unique prepared-patient table with fixed seed 42, stratifying by the
binary label: approximately 70% train and 30% remainder, then half the remainder
for validation and half for test (second seed =43). Require both classes in
each partition. All nine slices inherit their patient's assignment. No
image-level fallback, oversampling, augmentation or resampling is performed.

Report subject counts, CDR distribution and age range/mean per split. Rare
severity levels are not forced into every partition. A small two-disc cohort
can have demographic imbalance and is not an independent clinical validation.

## Outputs and review gates

- `archive_inventory.json`: source archive checksums, sessions and selected products.
- `clinical_inventory.json` / `.csv`: original metadata, matches and exclusion reasons.
- `volume_qc.json`: selected member checksums, dimensions, voxel sizes, endianness,
  foreground/intensity measurements and failures.
- `subject_manifest.json` / `subject_splits.csv`: one selected session per patient,
  clinical-row lineage and immutable subject assignments.
- `manifest.json`: train/validation/test slice records, real patient/session IDs,
  original CDR, binary labels, volume lineage, indices and image/input hashes.
- `preparation_policy.json`: versioned deterministic volume-to-slice policy.
- `qc_summary.json`: counts, split balance, exact preprocessed cross-split
  matches, exclusions, provenance fingerprint and remaining gates.
- `qc_review.html` with `qc/*_contact.png`: per-subject nine-slice contact sheets.
- `slices/<patient_id>/*.png`: new derived images only.

`ready_for_training` is deliberately false: preparation is not a leakage or
anatomical-orientation clearance. No existing training/checkpoint code is
changed to accept these new labels. Do not feed this manifest to the Kaggle
trainer or rename its target to make the old checkpoint loader accept it.

The manifest currently records absolute paths for derived slices. Audit it
locally without rebasing:

```powershell
python -B scripts/leakage_audit.py --manifest artifacts/oasis1-preparation-001/manifest.json --output-dir artifacts/oasis1-leakage-audit-001 --fail-on-findings
```

Use a new audit directory. Inspect QC first, then exact and near-duplicate
findings with the existing visual-review tool. No training should start before
those gates are reviewed. If relocating outputs to Colab, retain the original
manifest and use an explicit path-mapping adapter in the future OASIS workflow;
do not apply the Kaggle `OriginalDataset` rebasing convention to these paths.

The same source files, preparation code, dependencies and seed reproduce image
bytes and subject assignments. A new output location changes absolute paths
and the manifest fingerprint. All metadata exclusions remain inspectable.

## Results for the supplied two-disc preparation

The executed run is in `artifacts/oasis1-preparation-001/`. Both source archive
checksums and the metadata checksums were rechecked before final manifests
were published. No original Analyze pair was written to disk.

- Clinical workbook: 436 session rows; reliability workbook: 20 rows.
- Supplied discs: 77 sessions, all matched to the clinical table; this includes
  a repeat session. The primary cohort uses MR1 only.
- 198 clinical rows meet the baseline age/CDR rule across the whole workbook;
  163 are absent from the two supplied discs. Other exclusions are 218 rows
  under age 60 and 20 repeat/reliability rows.
- Prepared cohort: 35 subjects, 15 with CDR 0 and 20 with CDR 0.5 or 1.
  No CDR 2 subject is present in this prepared subset.

| Split | Subjects | CDR 0 | CDR >0 | Slices |
|---|---:|---:|---:|---:|
| Train | 24 | 10 | 14 | 216 |
| Validation | 5 | 2 | 3 | 45 |
| Test | 6 | 3 | 3 | 54 |

Validation mean age is 82.2 years versus 74.5 for training and 74.0 for test.
This descriptive imbalance and the very small validation/test patient counts
must be considered before future experiments. The fixed splits were not
changed after reviewing these statistics.

Additional new report directories:

- `artifacts/oasis1-leakage-audit-001/`: complete configured audit of all 315
  slices; zero exact file/pixel/input overlaps and zero repeated patient IDs.
  There are 60 pHash/SSIM candidates, none at the configured high priority.
  These are review candidates, not confirmed duplicate subjects.
- `artifacts/oasis1-leakage-review-001/review.html`: all 60 candidates,
  explicitly including lower-priority pairs; 34 train/test, 21 train/validation,
  and 5 validation/test. No preview hash mismatches occurred.
- `artifacts/oasis1-verification-001/verification.json`: independent record
  checks, runtime/package versions, output fingerprints and confirmation that
  protected Kaggle code matches the pre-existing training ZIP.

The prepared cohort remains `ready_for_training=false`. Orientation review,
candidate review and the future OASIS-specific training/evaluation path are
still required. No training has started.

## Synthetic tests

```powershell
python -B -m unittest discover -s tests -p test_oasis1_prepare.py -v
```

Tests create small generated Analyze pairs and metadata with explicitly
synthetic IDs. They cover big-endian decoding, clinical exclusions, CSV/XLSX
reading, patient grouping, reproducibility, archive traversal protection,
source immutability and prevention of bulk extraction. They never train.
