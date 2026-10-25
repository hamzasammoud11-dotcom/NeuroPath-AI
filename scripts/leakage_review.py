"""Generate an offline, read-only visual review from an existing leakage audit."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import math
from importlib.metadata import version
import platform
from pathlib import Path, PurePosixPath
import sys

sys.dont_write_bytecode = True
from PIL import Image

GROUPS = (("train", "test"), ("train", "validation"), ("validation", "test"))


def resolve_source(record, dataset_root):
    if dataset_root is None:
        raw = record.get("resolved_path") or record.get("manifest_path")
        if not isinstance(raw, str) or not raw:
            raise ValueError("No image path in audit inventory")
        return Path(raw).expanduser().resolve()
    raw = record.get("manifest_path")
    if not isinstance(raw, str) or not raw:
        raise ValueError("No manifest path for dataset relocation")
    parts = PurePosixPath(raw.replace("\\", "/")).parts
    anchors = [i for i, part in enumerate(parts) if part == "OriginalDataset"]
    if anchors:
        parts = parts[anchors[-1] + 1:]
    elif PurePosixPath(raw.replace("\\", "/")).is_absolute() or (parts and ":" in parts[0]):
        raise ValueError("Absolute path lacks an OriginalDataset anchor; cannot rebase")
    path = dataset_root.joinpath(*parts).resolve()
    if not path.is_relative_to(dataset_root):
        raise ValueError("Image path escapes dataset root")
    return path


def thumbnail(record, dataset_root, output_dir, size):
    result = {key: record.get(key) for key in ("id", "split", "label", "source_class", "patient_id", "manifest_path", "file_sha256")}
    try:
        path = resolve_source(record, dataset_root)
        if output_dir.is_relative_to(path.parent):
            raise ValueError("Output directory must be outside image directories")
        result["current_path"] = str(path)
        data = path.read_bytes()
        expected = record.get("file_sha256")
        if not expected or sha256(data).hexdigest() != expected:
            raise ValueError("Image bytes do not match the audit file hash; preview withheld")
        with Image.open(BytesIO(data)) as image:
            image = image.convert("L")
            result["native_size"] = list(image.size)
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
        result.update(preview_status="verified", data_uri="data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        if "Output directory" in str(exc):
            raise
        result.update(preview_status="unavailable", preview_error=str(exc), data_uri=None)
    return result


def build_review(audit_path, output_dir, dataset_root=None, max_pairs=500,
                 thumbnail_size=384, include_lower_priority=False):
    audit_path = Path(audit_path).resolve()
    output_dir = Path(output_dir).resolve()
    dataset_root = Path(dataset_root).resolve() if dataset_root else None
    if output_dir.exists():
        raise ValueError("Output directory exists; choose a NEW directory")
    if output_dir.is_relative_to(audit_path.parent):
        raise ValueError("Output directory must be separate from the original audit directory")
    if dataset_root and output_dir.is_relative_to(dataset_root):
        raise ValueError("Output directory must be outside the dataset")
    if max_pairs < 1 or not 32 <= thumbnail_size <= 1024:
        raise ValueError("max-pairs must be positive; thumbnail-size must be 32..1024")
    audit_bytes = audit_path.read_bytes()
    fingerprint = sha256(audit_bytes).hexdigest()
    audit = json.loads(audit_bytes)
    if not isinstance(audit, dict) or audit.get("schema_version") != 1:
        raise ValueError("Expected a version-1 leakage audit JSON")
    inventory = audit.get("inventory")
    candidates = audit.get("near_candidates")
    if not isinstance(inventory, list) or not isinstance(candidates, list):
        raise ValueError("Audit must contain inventory and near_candidates lists")
    records = {}
    for record in inventory:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or record["id"] in records:
            raise ValueError("Inventory IDs must be unique strings")
        records[record["id"]] = record
    selected, seen, duplicates = [], set(), 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Invalid near-candidate record")
        if not include_lower_priority and candidate.get("review_priority") != "high":
            continue
        left, right = records.get(candidate.get("left_id")), records.get(candidate.get("right_id"))
        if left is None or right is None:
            raise ValueError("Candidate refers to an unknown inventory ID")
        if left.get("split") != candidate.get("left_split") or right.get("split") != candidate.get("right_split"):
            raise ValueError("Candidate and inventory split names disagree")
        group = next((group for group in GROUPS if {left["split"], right["split"]} == set(group)), None)
        if group is None:
            raise ValueError("Candidate must connect two different supported splits")
        score, distance = candidate.get("ssim"), candidate.get("phash_distance")
        if (not isinstance(score, (int, float)) or not math.isfinite(score) or not -1 <= score <= 1
                or type(distance) is not int or not 0 <= distance <= 64):
            raise ValueError("Invalid SSIM or pHash distance")
        if left["split"] != group[0]:
            left, right = right, left
        identity = tuple(sorted((left["id"], right["id"])))
        if identity in seen:
            duplicates += 1
            continue
        seen.add(identity)
        selected.append(dict(pair_id=sha256(json.dumps([fingerprint, *identity]).encode()).hexdigest(),
                             group="/".join(group), left_id=left["id"], right_id=right["id"],
                             ssim=score, phash_distance=distance, priority=candidate.get("review_priority"),
                             label_conflict=left.get("label") != right.get("label"),
                             source_class_conflict=left.get("source_class") != right.get("source_class")))
    group_order = {"/".join(group): index for index, group in enumerate(GROUPS)}
    selected.sort(key=lambda pair: (group_order[pair["group"]], -pair["ssim"], pair["phash_distance"], pair["pair_id"]))
    eligible_count = len(selected)
    selected = selected[:max_pairs]
    image_ids = sorted({pair[key] for pair in selected for key in ("left_id", "right_id")})
    images = {identity: thumbnail(records[identity], dataset_root, output_dir, thumbnail_size) for identity in image_ids}
    payload = dict(schema_version=1, audit_sha256=fingerprint,
                   environment=dict(python=platform.python_version(), Pillow=version("Pillow")),
                   parameters=dict(max_pairs=max_pairs, thumbnail_size=thumbnail_size),
                   tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                   template_sha256=sha256((Path(__file__).parent / "leakage_review_template.html").read_bytes()).hexdigest(),
                   audit_status=audit.get("status"), near_search_complete=audit.get("near_search_complete"),
                   identity_risks=audit.get("unresolved_patient_identity_risks", []),
                   audit_issues=audit.get("issues", []), audit_parameters=audit.get("parameters", {}),
                   eligible_pairs=eligible_count, displayed_pairs=len(selected), omitted_pairs=eligible_count - len(selected),
                   duplicate_candidate_records=duplicates, include_lower_priority=include_lower_priority,
                   preview_errors=sum(item["preview_status"] != "verified" for item in images.values()),
                   group_counts=dict(Counter(pair["group"] for pair in selected)),
                   pairs=selected, images=images)
    template = (Path(__file__).parent / "leakage_review_template.html").read_text(encoding="utf-8")
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = template.replace("__REVIEW_JSON__", encoded)
    # No inputs are rewritten; all previews are embedded in this new offline report.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "review.html").write_text(html, encoding="utf-8")
    index = {**payload, "images": {key: {k: v for k, v in value.items() if k != "data_uri"} for key, value in images.items()}}
    (output_dir / "review_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True, help="Existing audit.json")
    parser.add_argument("--dataset-root", type=Path, help="Current OriginalDataset directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW directory separate from input audit/dataset")
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--thumbnail-size", type=int, default=384)
    parser.add_argument("--include-lower-priority", action="store_true")
    args = parser.parse_args()
    try:
        index = build_review(args.audit, args.output_dir, args.dataset_root, args.max_pairs,
                             args.thumbnail_size, args.include_lower_priority)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Review error: {exc}\n")
    print(json.dumps({key: index[key] for key in ("displayed_pairs", "omitted_pairs", "preview_errors", "group_counts")}, indent=2))
    print(args.output_dir / "review.html")
    return 2 if index["preview_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
