"""Read-only dataset/manifest leakage audit. Only a NEW report directory is written."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
from pathlib import Path, PurePosixPath
import platform
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from PIL import Image
from scipy.fft import dctn
from skimage.metrics import structural_similarity
from neuro_path_ai.models.image2d import PREPROCESSING, preprocess_image

SPLITS = ("train", "validation", "test")
HASH_FIELDS = ("file_sha256", "decoded_sha256", "input_sha256")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def pixel_hash(image):
    # Deliberately matches images2d.discover_originals, including its dimension prefix.
    gray = image.convert("L")
    return sha256(str(gray.size).encode() + gray.tobytes())


def perceptual_hash(image):
    """64-bit DCT pHash; DC bit is forced to zero, median excludes DC."""
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    low = dctn(np.asarray(gray, dtype=np.float64), norm="ortho")[:8, :8].flatten()
    bits = low > np.median(low[1:])
    bits[0] = False
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def model_input(image):
    return np.asarray(preprocess_image(image).cpu().numpy(), dtype="<f4", order="C")


def resolve_image(raw_path, dataset_root):
    if dataset_root is None:
        return Path(raw_path).expanduser().resolve()
    # Rebase a manifest from Windows or a previous Colab VM without flattening names.
    parts = PurePosixPath(raw_path.replace("\\", "/")).parts
    anchors = [i for i, part in enumerate(parts) if part == "OriginalDataset"]
    if anchors:
        parts = parts[anchors[-1] + 1:]
    elif PurePosixPath(raw_path.replace("\\", "/")).is_absolute() or (parts and ":" in parts[0]):
        raise ValueError("Absolute path has no OriginalDataset anchor; use class-relative manifest paths or omit --dataset-root")
    resolved = dataset_root.joinpath(*parts).resolve()
    if not resolved.is_relative_to(dataset_root):
        raise ValueError("Image path escapes --dataset-root")
    return resolved


class BKTree:
    """Exact Hamming-radius search; no random sampling or approximate neighbors."""
    def __init__(self):
        self.root = None

    def insert(self, value, index):
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]

    def query(self, value, radius):
        pending = [self.root] if self.root is not None else []
        matches = []
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                matches.extend((index, distance) for index in node[1])
            pending.extend(child for edge, child in sorted(node[2].items())
                           if distance - radius <= edge <= distance + radius)
        return sorted(matches)


def cross_groups(records, field):
    groups = defaultdict(list)
    for record in records:
        if record.get(field) is not None and record.get(field) != "":
            groups[record[field]].append(record)
    output = []
    for value, members in sorted(groups.items(), key=lambda item: str(item[0])):
        counts = Counter(r["split"] for r in members)
        if len(counts) > 1:
            pairs = {f"{a}:{b}": counts[a] * counts[b]
                     for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]}
            output.append(dict(value=value, member_ids=[r["id"] for r in members],
                               split_counts=dict(sorted(counts.items())), cross_split_pairs=pairs,
                               source_class_conflict=len({r.get("source_class") for r in members}) > 1,
                               binary_label_conflict=len({r.get("label") for r in members}) > 1,
                               multiple_patient_ids=len({r.get("patient_id") for r in members if r.get("patient_id")}) > 1))
    return output


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit(manifest_path, output_dir, dataset_root=None, phash_distance=6,
          ssim_threshold=0.95, max_candidates=200000):
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    dataset_root = Path(dataset_root).resolve() if dataset_root is not None else None
    if output_dir.exists():
        raise ValueError("Output directory already exists; choose a NEW separate directory")
    if dataset_root is not None and output_dir.is_relative_to(dataset_root):
        raise ValueError("Output directory must be outside the dataset")
    if not 0 <= phash_distance <= 64 or not 0 <= ssim_threshold <= 1 or max_candidates < 1:
        raise ValueError("Invalid pHash radius, SSIM threshold or candidate limit")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object")
    if manifest.get("preprocessing") != PREPROCESSING:
        raise ValueError("Manifest preprocessing does not match the pipeline; audit cannot reproduce model inputs")
    splits = manifest.get("splits", {})
    if not isinstance(splits, dict) or set(splits) != set(SPLITS) or any(not isinstance(splits[s], list) or not splits[s] for s in SPLITS):
        raise ValueError("Manifest must contain nonempty train, validation and test lists")
    # Reject reports under image directories even without --dataset-root.
    records, issues = [], []
    for split in SPLITS:
        for index, row in enumerate(splits[split]):
            if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not row["path"]:
                raise ValueError(f"Invalid image record: {split}:{index}")
            patient = row.get("patient_id")
            if patient is not None and (not isinstance(patient, str) or not patient.strip()):
                raise ValueError("patient_id must be null or a nonblank string")
            record = dict(id=f"{split}:{index}", split=split, manifest_path=row["path"],
                          label=row.get("label"), source_class=row.get("source_class"),
                          patient_id=patient, manifest_decoded_sha256=row.get("sha256"),
                          status="error")
            try:
                path = resolve_image(row["path"], dataset_root)
                if output_dir.is_relative_to(path.parent):
                    raise ValueError("Output directory must be outside image directories")
                record["resolved_path"] = str(path)
            except ValueError as exc:
                if "Output directory" in str(exc):
                    raise
                record["error"] = str(exc)
                issues.append(dict(record_id=record["id"], kind="unresolvable_path", detail=str(exc)))
            records.append(record)
    for record in records:
        if "resolved_path" not in record:
            continue
        try:
            path = Path(record["resolved_path"])
            before = path.stat()
            data = path.read_bytes()
            # Decode the exact bytes that were hashed, avoiding a second file read.
            from io import BytesIO
            with Image.open(BytesIO(data)) as image:
                image.load()
                tensor = model_input(image)
                record.update(file_sha256=sha256(data), decoded_sha256=pixel_hash(image),
                              input_sha256=sha256(str(tensor.shape).encode() + tensor.tobytes()),
                              phash=f"{perceptual_hash(image):016x}", width=image.width, height=image.height)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Image changed while being audited")
            record["status"] = "ok"
            expected = record["manifest_decoded_sha256"]
            if not expected or expected != record["decoded_sha256"]:
                issues.append(dict(record_id=record["id"], kind="manifest_hash_mismatch",
                                   detail="Stored decoded-pixel hash is absent or differs from current image"))
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            record["error"] = str(exc)
            issues.append(dict(record_id=record["id"], kind="unreadable_or_changed_image", detail=str(exc)))
    valid = [r for r in records if r["status"] == "ok"]
    exact = {field: cross_groups(valid, field) for field in HASH_FIELDS}
    path_groups = cross_groups(records, "resolved_path")
    patients = cross_groups(records, "patient_id")
    risks = []
    missing_ids = sum(r["patient_id"] is None for r in records)
    if missing_ids:
        risks.append(f"{missing_ids}/{len(records)} image records lack patient IDs; same-patient slices or visits may cross splits")
    if not manifest.get("patient_id_source"):
        risks.append("No patient-ID provenance supplied in the manifest")
    risks.append("Patient identities cannot be authenticated from images, UUID filenames or a manifest flag; supplied mapping requires source verification")
    risks.append("Perceptual similarity is not proof of shared patient identity; undetected related slices may remain")
    if manifest.get("patient_separation_verified") and (missing_ids or patients):
        issues.append(dict(record_id="", kind="patient_separation_claim_contradicted",
                           detail="Manifest claims verified separation but IDs are missing or overlapping"))

    @lru_cache(maxsize=128)
    def comparison_image(path, expected_hash):
        data = Path(path).read_bytes()
        if sha256(data) != expected_hash:
            raise ValueError("Image changed before SSIM comparison")
        from io import BytesIO
        with Image.open(BytesIO(data)) as image:
            # SSIM luminance is evaluated on nonnegative intensity, not z-centered values.
            return model_input(image)[0] * PREPROCESSING["std"] + PREPROCESSING["mean"]

    tree, near, examined, ssim_evaluated = BKTree(), [], 0, 0
    near_complete = True
    for index, current in enumerate(valid):
        for previous_index, distance in tree.query(int(current["phash"], 16), phash_distance):
            previous = valid[previous_index]
            if previous["split"] == current["split"]:
                continue
            if examined >= max_candidates:
                near_complete = False
                break
            examined += 1
            if any(previous[field] == current[field] for field in HASH_FIELDS):
                continue  # Already reported by exact groups, not counted as near duplicates.
            try:
                score = float(structural_similarity(
                    comparison_image(previous["resolved_path"], previous["file_sha256"]),
                    comparison_image(current["resolved_path"], current["file_sha256"]), data_range=1.0))
                ssim_evaluated += 1
                near.append(dict(left_id=previous["id"], right_id=current["id"],
                                 left_split=previous["split"], right_split=current["split"],
                                 left_path=previous["manifest_path"], right_path=current["manifest_path"],
                                 phash_distance=distance, ssim=score,
                                 review_priority="high" if score >= ssim_threshold else "lower",
                                 requires_review=True))
            except (OSError, ValueError) as exc:
                issues.append(dict(record_id=current["id"], kind="ssim_error", detail=str(exc)))
        if not near_complete:
            break
        tree.insert(int(current["phash"], 16), index)
    if not near_complete:
        issues.append(dict(record_id="", kind="candidate_limit_reached",
                           detail="Near-duplicate search truncated; increase --max-candidates and rerun into a NEW directory"))
    for record in valid:
        try:
            if sha256(Path(record["resolved_path"]).read_bytes()) != record["file_sha256"]:
                raise ValueError("Image changed during the audit")
        except (OSError, ValueError) as exc:
            issues.append(dict(record_id=record["id"], kind="image_changed_during_audit", detail=str(exc)))
    if manifest_path.read_bytes() != manifest_bytes:
        issues.append(dict(record_id="", kind="manifest_changed_during_audit", detail="Manifest changed during the audit"))
    pair_summary = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            key = f"{a}:{b}"
            pair_summary[key] = dict(
                **{field + "_pairs": sum(group["cross_split_pairs"][key] for group in exact[field]) for field in HASH_FIELDS},
                repeated_path_pairs=sum(group["cross_split_pairs"][key] for group in path_groups),
                overlapping_patient_ids=sum(a in g["split_counts"] and b in g["split_counts"] for g in patients),
                near_candidates=sum({r["left_split"], r["right_split"]} == {a, b} for r in near),
                high_priority_candidates=sum({r["left_split"], r["right_split"]} == {a, b} and r["review_priority"] == "high" for r in near))
    detected = bool(any(exact.values()) or patients or path_groups or near)
    report = dict(schema_version=1, status="incomplete" if issues else ("findings" if detected else "no_cross_split_matches_detected"),
                  manifest_sha256=sha256(manifest_bytes), tool_sha256=sha256(Path(__file__).read_bytes()),
                  manifest_metadata={key: manifest.get(key) for key in
                                     ("seed", "source", "split_unit", "patient_separation_verified", "patient_id_source", "class_mapping")},
                  preprocessing_code_sha256=sha256((Path(__file__).resolve().parents[1] / "src/neuro_path_ai/models/image2d.py").read_bytes()),
                  environment=dict(python=platform.python_version(), **{p: version(p) for p in ("numpy", "Pillow", "scipy", "scikit-image", "torch")}),
                  parameters=dict(phash_distance=phash_distance, ssim_threshold=ssim_threshold,
                                  max_candidates=max_candidates, ssim_alignment="none; shared grayscale resize, intensity [0,1]",
                                  phash_algorithm="DCT32 low8x8 median excluding DC; DC bit zero"),
                  counts=dict(records=len(records), readable=len(valid), missing_patient_ids=missing_ids,
                              cross_split_phash_pairs_examined=examined, ssim_evaluated=ssim_evaluated),
                  split_counts=dict(Counter(r["split"] for r in records)), split_pair_summary=pair_summary,
                  exact_groups=exact, path_overlap_groups=path_groups, patient_overlap_groups=patients,
                  near_search_complete=near_complete, near_candidates=near, issues=issues,
                  patient_identity_authenticated=False,
                  unresolved_patient_identity_risks=risks,
                  limitations=["Near candidates require human review; pHash radius and SSIM thresholds do not certify absence of leakage",
                               "Rotations, crops, adjacent slices and contrast changes may evade this unaligned search",
                               "Only manifest-listed images are audited; omitted dataset files are not scanned",
                               "Exact pair counts for different hash types overlap and must not be summed"],
                  inventory=records)
    # No overwrite option: all exported files belong to this newly created directory.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(output_dir / "inventory.csv", records,
              ["id", "split", "manifest_path", "resolved_path", "patient_id", "label", "source_class", "status", "error",
               "width", "height", "manifest_decoded_sha256", *HASH_FIELDS, "phash"])
    memberships = [dict(kind=field, group_hash=group["value"], member_id=member)
                   for field, groups in exact.items() for group in groups for member in group["member_ids"]]
    write_csv(output_dir / "exact_groups.csv", memberships, ["kind", "group_hash", "member_id"])
    write_csv(output_dir / "near_candidates.csv", near,
              ["left_id", "right_id", "left_split", "right_split", "left_path", "right_path", "phash_distance", "ssim", "review_priority", "requires_review"])
    write_csv(output_dir / "patient_overlaps.csv",
              [dict(patient_id=g["value"], member_id=member) for g in patients for member in g["member_ids"]],
              ["patient_id", "member_id"])
    write_csv(output_dir / "issues.csv", issues, ["record_id", "kind", "detail"])
    write_csv(output_dir / "split_summary.csv", [dict(split_pair=pair, **values) for pair, values in pair_summary.items()],
              ["split_pair", *next(iter(pair_summary.values())).keys()])
    write_csv(output_dir / "identity_risks.csv", [dict(risk=risk) for risk in risks], ["risk"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, help="Current OriginalDataset directory; rebases stale manifest paths")
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW directory outside the dataset")
    parser.add_argument("--phash-distance", type=int, default=6)
    parser.add_argument("--ssim-threshold", type=float, default=0.95)
    parser.add_argument("--max-candidates", type=int, default=200000)
    parser.add_argument("--fail-on-findings", action="store_true")
    args = parser.parse_args()
    try:
        report = audit(args.manifest, args.output_dir, args.dataset_root,
                       args.phash_distance, args.ssim_threshold, args.max_candidates)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Audit error: {exc}\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"],
                      "split_pair_summary": report["split_pair_summary"], "reports": str(args.output_dir)}, indent=2))
    if report["status"] == "incomplete":
        return 3
    return 2 if args.fail_on_findings and report["status"] == "findings" else 0


if __name__ == "__main__":
    raise SystemExit(main())
