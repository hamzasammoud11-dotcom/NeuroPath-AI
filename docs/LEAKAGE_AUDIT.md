# Standalone leakage audit: no training

The audit reads an existing `manifest.json` and its listed images. It never
imports the trainer, loads checkpoints, creates splits, or changes input files.
Reports go only into a **new** output directory outside the dataset. Existing
directories are refused; there is no overwrite or delete option.

If no manifest exists, stop here: this tool cannot audit splits that have not
been defined. Do not run the training notebook merely to generate an audit.
No dataset or real run manifest was available in the local workspace when this
tool was implemented; verification uses generated images only.

## What is checked

- Every pair of splits: train/validation, train/test, validation/test.
- SHA-256 of file bytes, decoded grayscale pixels plus dimensions (matching the
  training manifest), and the actual shared preprocessed float32 model input.
  Exact groups list all members. Pair counts refer to manifest records; counts
  for different hash types overlap and must not be added together.
- Repeated resolved paths and patient IDs across splits. Missing patient IDs,
  absent provenance, and unverifiable identity claims remain explicit risks.
- Stored manifest pixel hashes versus current image contents; missing,
  unreadable or changed images make the audit incomplete.
- Near-duplicate candidates: a deterministic BK-tree performs exact Hamming
  search over 64-bit DCT perceptual hashes (DC bit excluded). Default radius:
  **6 bits**. SSIM is computed for non-exact cross-split pHash candidates on
  shared 128x128 grayscale inputs converted to intensity [0,1]. Default high
  review priority: **SSIM >= 0.95**. All evaluated candidates are exported,
  including those below this threshold. Exact matches are reported separately.

Thresholds are starting points for candidate review, not validated medical
identity rules. SSIM has no registration or rotation/crop alignment. Different
slices, transformations and repeat visits can evade detection. Similarity does
not establish patient identity. No matches does not certify absence of leakage.
Choose thresholds before inspecting model performance, and review candidates
without using test accuracy to guide dataset changes.

The default maximum is 200,000 cross-split pHash-neighbor pairs examined,
including pairs already represented in exact groups. Reaching this limit
marks the audit **incomplete**; it never silently reports a completed search.
Increase the limit in a new run if needed. Extremely similar images may cause
many candidates and long runtimes. No GPU is required.

## Local execution

```powershell
python -m pip install -r requirements-audit.txt
python -B scripts/leakage_audit.py --manifest artifacts/image2d-run1/manifest.json --dataset-root data/raw/OriginalDataset --output-dir artifacts/leakage-audit-001 --phash-distance 6 --ssim-threshold 0.95 --max-candidates 200000 --fail-on-findings
```

`--dataset-root` must point directly to the current `OriginalDataset` directory.
It relocates old Windows or Colab paths by preserving everything after their
`OriginalDataset` segment. Class-relative paths are also accepted. Absolute
paths without that segment require their original locations (omit the option).
Without the option, relative paths are resolved against the current working
directory. Missing or ambiguous rebasing is reported, not guessed by basename.
The manifest's preprocessing must exactly match the shared pipeline settings.

Exit codes: **0** completed (findings permitted unless requested otherwise);
**1** invalid configuration/manifest or execution error; **2** completed with
findings when `--fail-on-findings` is set; **3** incomplete audit. Missing patient
identity remains a reported risk even when the command exits 0.

## Exact Google Colab steps

These are standalone cells. **Do not run the training cell.** The existing
training notebook and its source ZIP do not need to be changed.

1. On your PC create the separate audit-only bundle:

```powershell
python scripts/package_leakage_audit.py
```

Upload `artifacts/cloud/neuropath-leakage-audit.zip` to the root of My Drive.
The packager refuses an existing archive; use a new `--output` filename for a
future version rather than overwriting one. This bundle contains the tool,
shared preprocessing module, requirements and these instructions, not weights
or training code.

2. Open a new Colab notebook; a CPU runtime is sufficient. Run:

```python
from google.colab import drive
from pathlib import Path
from zipfile import ZipFile
import json, subprocess, sys

drive.mount('/content/drive')
bundle = Path('/content/drive/MyDrive/neuropath-leakage-audit.zip')
project = Path('/content/neuropath-audit-001')
assert bundle.is_file(), 'Adjust bundle to the ZIP location in Drive'
assert not project.exists(), 'Choose a new project directory; do not delete an existing one'
project.mkdir()
with ZipFile(bundle) as archive:
    for item in archive.infolist():
        assert (project / item.filename).resolve().is_relative_to(project.resolve())
    archive.extractall(project)
```

3. Install audit dependencies in the cloud runtime:

```python
%pip install -r /content/neuropath-audit-001/requirements-audit.txt kagglehub==1.0.2
```

4. Point to your **existing** run manifest and retrieve the same dataset version:

```python
import kagglehub

experiment = Path('/content/drive/MyDrive/NeuroPathAI/image2d-run1')
manifest = experiment / 'results/manifest.json'
assert manifest.is_file(), 'An existing run manifest is required; do not start training'
dataset_info = experiment / 'dataset.json'
assert dataset_info.is_file(), 'Locate the original run dataset version before auditing'
handle = json.loads(dataset_info.read_text())['versioned_handle']
dataset = Path(kagglehub.dataset_download(handle))
roots = list(dataset.rglob('OriginalDataset'))
assert len(roots) == 1, roots
dataset_root = roots[0]
print('Manifest:', manifest)
print('Dataset version:', handle)
print('Images:', dataset_root)
```

If the images are already available in the cloud VM, use that `OriginalDataset`
path as `dataset_root` instead of downloading. If Kaggle requests authentication,
use its interactive `kagglehub.login()` flow and rerun the download cell; do not
save tokens in notebook cells. The tool itself does not download anything.

5. Audit and save reports in a new Drive directory:

```python
output = experiment / 'leakage-audit-001'
assert not output.exists(), 'Choose a new audit output name'
result = subprocess.run([
    sys.executable, '-B', str(project / 'scripts/leakage_audit.py'),
    '--manifest', str(manifest), '--dataset-root', str(dataset_root),
    '--output-dir', str(output), '--phash-distance', '6',
    '--ssim-threshold', '0.95', '--max-candidates', '200000',
    '--fail-on-findings',
], check=False)
print('Exit code:', result.returncode)
if result.returncode not in (0, 2, 3):
    raise RuntimeError('Audit failed; inspect the error above')
report = json.loads((output / 'audit.json').read_text())
print('Status:', report['status'])
print('Overlap summary:', json.dumps(report['split_pair_summary'], indent=2))
print('Identity risks:', report['unresolved_patient_identity_risks'])
print('Issues:', report['issues'])
```

Change `image2d-run1` if your saved experiment has another name. An incomplete
result is not a pass. Review exact groups and high-priority candidates first;
also inspect lower-priority candidates and sample outside the search threshold
before drawing conclusions. Do not change the original splits during review.

## Reports and reproducibility

For side-by-side, read-only review of high-priority near candidates, see
[Visual leakage review](LEAKAGE_REVIEW.md). It generates a separate offline HTML
report with filters and exportable annotations. It never changes this audit.

- `audit.json`: full report, input manifest SHA-256, tool and preprocessing code
  SHA-256, package/Python versions, thresholds, metadata, inventory and findings.
- `inventory.csv`: each record's paths, IDs, labels, hashes and read status.
- `exact_groups.csv`: cross-split exact group memberships; join member IDs to inventory.
- `near_candidates.csv`: split pairs, image paths, pHash distances, SSIM and review priority.
- `patient_overlaps.csv`: patient IDs repeated across splits and member records.
- `split_summary.csv`: counts for all three split pairs.
- `identity_risks.csv`: unresolved patient-identity caveats.
- `issues.csv`: missing/corrupt images, manifest mismatches and truncation.

The audit checks only manifest-listed records. Unlisted images and removed
duplicate aliases cannot be reconstructed from this manifest. File hashes
are rechecked at the end to detect changes during execution; keep the dataset
quiescent while auditing. Reports contain local paths and supplied patient IDs.

Use the same input bytes, dataset version, tool/preprocessing code, thresholds,
and recorded package versions to reproduce results. No random sampling is used;
JSON and CSV are deterministic in an unchanged environment. Paths and numeric
SSIM/pHash results can differ after relocation or library changes. Freeze and
review the audit before any later, separately authorized split corrections.

Run only the lightweight audit tests (not the training tests):

```powershell
python -B -m unittest discover -s tests -p test_leakage_audit.py -v
```
