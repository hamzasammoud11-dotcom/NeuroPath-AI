# Visual leakage review: offline HTML, no training

This separate tool reads an existing `audit.json` and the corresponding images.
It writes only `review.html` and `review_index.json` into a **new** directory,
separate from the dataset and original audit directory. Existing outputs are
refused. There is no overwrite, delete, training, split-editing or checkpoint
operation. The audit JSON, model and dataset are unchanged.

## What the page shows

The default view contains only candidates marked `review_priority="high"` by
the existing audit. Sections are ordered **train/test first**, train/validation,
then validation/test. Inside each section, SSIM descends and pHash distance
breaks ties. Each pair displays both images, split names, SSIM, pHash distance,
binary labels, source severity classes, patient IDs when supplied, and paths.
Binary/severity conflicts are highlighted separately.

Filters cover split group, minimum SSIM, maximum pHash distance, annotation
decision, binary-label conflict and text search across paths, labels, IDs and
notes. Results are paginated in groups of 20. The default report limit is
500 eligible pairs, prioritizing train/test; omitted counts are explicit. Use
`--max-pairs` to include more or generate additional views from the same audit.
`--include-lower-priority` includes lower-priority candidates too.

Previews are grayscale thumbnails, preserving aspect ratio, without contrast
enhancement or registration. Only images matching the audit's file SHA-256
are embedded. Missing, changed or unresolvable files show an error placeholder;
they are never silently displayed as the audited image. Metrics are copied
from the original audit; they are not recomputed from display thumbnails.

An incomplete source audit remains visibly incomplete. Exact-duplicate groups
are not shown in this near-candidate view; consult the original audit reports.
No candidates or no matching filters is not evidence of patient independence.
Similarity and visual inspection cannot authenticate patient identity.

## Local command

The generator needs Python and Pillow only; it does not import training code
or load the model module.

```powershell
python -m pip install "Pillow>=10,<13"
python -B scripts/leakage_review.py --audit artifacts/leakage-audit-001/audit.json --dataset-root data/raw/OriginalDataset --output-dir artifacts/leakage-review-001 --max-pairs 500
```

`--dataset-root` is the current `OriginalDataset` folder. Paths from previous
Windows or Colab runtimes are rebased using their `OriginalDataset` segment;
class-relative paths also work. Otherwise omit this option only if the saved
resolved paths still exist. Invalid relocation produces unavailable previews,
not guesses based on image basenames. The report embeds the preview bytes, so
the resulting HTML works offline without access to the original paths.

Exit codes: 0 = generation completed with all included previews available;
1 = invalid input/configuration or execution failure;
2 = report generated but one or more included previews unavailable.
Exit 0 does not mean the source audit was complete or the dataset passed.
Output includes audit/code/template fingerprints and Python/Pillow versions.
Unchanged inputs and environment produce identical HTML and index files.

## Reviewer annotations

1. Enter your name or initials, select a decision, and enter notes for each pair.
2. Decisions are **Unreviewed**, **Suspected related images**, **Visually distinct**,
   and **Uncertain**. These are review judgments, not confirmed patient identities.
3. Click **Export annotations JSON** before refreshing or closing the tab. Notes
   are held in tab memory; there is no automatic persistent storage or server.
   Keep the downloaded JSON and optionally upload it to Drive yourself.
4. Open the same report later and use **Import saved annotations** to resume.
   Imports must match the original audit SHA-256 and contain only pair IDs
   available in this report. Conflicting in-memory annotations require explicit
   browser confirmation before replacement. Export a merged JSON after import.
5. CSV export is available for external review, but JSON is required to resume.
   CSV cells with spreadsheet formula prefixes are escaped; JSON preserves notes.

Annotations include the audit fingerprint, stable pair IDs, reviewer, decision,
notes and timestamps. They never modify the HTML, input audit, dataset or splits.
Any later split corrections require a separate decision and workflow. The page
loads no external scripts, fonts, analytics or images and makes no network calls.
The exported report embeds images and may contain supplied patient IDs: share
it only where the underlying data may be shared.

## Exact Google Colab instructions

Use a new CPU notebook. **Do not run any training cell.** An existing real-data
audit is required; the source audit and dataset version must already be known.

1. On your PC, make a new bundle (the old audit ZIP is left intact):

```powershell
python scripts/package_leakage_audit.py --output artifacts/cloud/neuropath-leakage-review-v1.zip
```

Upload `artifacts/cloud/neuropath-leakage-review-v1.zip` to the root of My Drive.
This archive is also provided prebuilt. If it already exists locally, use the
provided file or choose a new filename; do not overwrite it.

2. In Colab mount Drive and unpack into a new temporary project directory:

```python
from google.colab import drive, files
from pathlib import Path
from zipfile import ZipFile
import json, subprocess, sys

drive.mount('/content/drive')
bundle = Path('/content/drive/MyDrive/neuropath-leakage-review-v1.zip')
project = Path('/content/neuropath-review-001')
assert bundle.is_file(), 'Adjust the ZIP path to its location in Drive'
assert not project.exists(), 'Choose a new project directory; do not delete anything'
project.mkdir()
with ZipFile(bundle) as archive:
    for item in archive.infolist():
        assert (project / item.filename).resolve().is_relative_to(project.resolve())
    archive.extractall(project)
```

3. Install visual-review dependencies and point to the existing audit:

```python
%pip install "Pillow>=10,<13" kagglehub==1.0.2

experiment = Path('/content/drive/MyDrive/NeuroPathAI/image2d-run1')
audit = experiment / 'leakage-audit-001/audit.json'
assert audit.is_file(), 'Locate your existing audit.json; do not start training'
```

4. Use the same dataset version as the audited experiment. If `dataset_root`
already points to the correct images in this runtime, skip this download:

```python
import kagglehub

dataset_info = experiment / 'dataset.json'
assert dataset_info.is_file(), 'Locate the audited dataset version before proceeding'
handle = json.loads(dataset_info.read_text())['versioned_handle']
dataset = Path(kagglehub.dataset_download(handle))
roots = list(dataset.rglob('OriginalDataset'))
assert len(roots) == 1, roots
dataset_root = roots[0]
```

If Kaggle requests authentication, use `kagglehub.login()` interactively and
retry; do not save tokens in notebook cells. Downloading does not train a model.

5. Generate a new visual report alongside, not inside, the audit directory:

```python
output = experiment / 'leakage-review-001'
assert not output.exists(), 'Choose a new review output directory'
result = subprocess.run([
    sys.executable, '-B', str(project / 'scripts/leakage_review.py'),
    '--audit', str(audit), '--dataset-root', str(dataset_root),
    '--output-dir', str(output), '--max-pairs', '500',
], check=False)
if result.returncode not in (0, 2):
    raise RuntimeError('Review generation failed; inspect the error above')
index = json.loads((output / 'review_index.json').read_text())
print('Source audit status:', index['audit_status'])
print('Included / omitted:', index['displayed_pairs'], index['omitted_pairs'])
print('Unavailable previews:', index['preview_errors'])
files.download(str(output / 'review.html'))
```

Open the downloaded `review.html` in Chrome, Edge or Firefox. Drive's preview
may not execute the interactive controls; use the downloaded file in your
browser instead. No local web server is needed. Begin with train/test and
export annotation JSON regularly. Reports stay in Drive; annotation downloads
are saved by your browser and must be retained separately.

Change experiment/audit folder names to match your existing run. A truncated
review never covers the omitted pairs. Increasing the limit preserves stable
pair IDs for the same audit, but importing notes into a smaller view will reject
annotations for pairs absent from that view.

## Synthetic verification

```powershell
python -B -m unittest discover -s tests -p test_leakage_review.py -v
```

The tests use generated images and check source immutability, path relocation,
hash validation, priority/limits, missing images, incomplete audits, safe JSON
embedding and reproducible reports. An optional headless Edge check (when Edge
is installed) exercises filters, annotations, JSON export/import and rejecting
annotations from a different audit. It uses a temporary browser profile and no
training artifacts. No real-data audit or Colab session has been executed here.
