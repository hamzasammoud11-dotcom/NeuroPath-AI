"""Synthetic audit-only tests: no training, model loading or checkpoint writes."""
import importlib.util
import json
from pathlib import Path
import random
import sys
import subprocess
import tempfile
import unittest

import numpy as np
from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("leakage_audit", ROOT / "scripts/leakage_audit.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class LeakageAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.data = self.base / "OriginalDataset"
        self.data.mkdir()
        rng = np.random.default_rng(23)
        self.paths = {}
        for split in tool.SPLITS:
            path = self.data / f"{split}.png"
            Image.fromarray(rng.integers(30, 190, (64, 64), dtype=np.uint8)).save(path)
            self.paths[split] = path
        self.manifest = self.base / "manifest.json"
        self.write_manifest()

    def write_manifest(self, patient_ids=None):
        splits = {}
        for split, path in self.paths.items():
            with Image.open(path) as image:
                digest = tool.pixel_hash(image)
            splits[split] = [dict(path=str(path), sha256=digest, label=0,
                                  source_class="NonDemented",
                                  patient_id=(patient_ids or {}).get(split))]
        payload = dict(preprocessing=tool.PREPROCESSING, splits=splits)
        self.manifest.write_text(json.dumps(payload), encoding="utf-8")

    def run_audit(self, name="audit", **kwargs):
        return tool.audit(self.manifest, self.base / name, self.data, **kwargs)

    def test_exact_hashes_and_all_three_split_pairs(self):
        original = self.paths["train"].read_bytes()
        for path in self.paths.values():
            path.write_bytes(original)
        self.write_manifest()
        report = self.run_audit()
        for field in tool.HASH_FIELDS:
            self.assertEqual(len(report["exact_groups"][field]), 1)
            for pair in report["split_pair_summary"].values():
                self.assertEqual(pair[field + "_pairs"], 1)
        self.assertEqual(report["status"], "findings")

    def test_metadata_only_change_matches_pixels_but_not_file_hash(self):
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("synthetic", "different metadata")
        with Image.open(self.paths["train"]) as image:
            image.save(self.paths["validation"], pnginfo=metadata)
        self.write_manifest()
        report = self.run_audit()
        self.assertEqual(report["exact_groups"]["file_sha256"], [])
        self.assertEqual(len(report["exact_groups"]["decoded_sha256"]), 1)

    def test_different_native_dimensions_same_preprocessed_input(self):
        Image.new("L", (32, 32), 100).save(self.paths["train"])
        Image.new("L", (64, 64), 100).save(self.paths["validation"])
        self.write_manifest()
        report = self.run_audit()
        self.assertFalse(report["exact_groups"]["decoded_sha256"])
        self.assertEqual(len(report["exact_groups"]["input_sha256"]), 1)

    def test_brightness_variant_is_near_candidate_with_ssim(self):
        with Image.open(self.paths["train"]) as image:
            array = np.asarray(image).copy()
        Image.fromarray(array + 5).save(self.paths["validation"])
        self.write_manifest()
        report = self.run_audit()
        pair = next(r for r in report["near_candidates"] if r["right_split"] == "validation")
        self.assertLessEqual(pair["phash_distance"], 6)
        self.assertGreater(pair["ssim"], 0.95)
        self.assertTrue(pair["requires_review"])

    def test_patient_overlap_and_missing_identity_risks(self):
        self.write_manifest({"train": "patient-a", "test": "patient-a"})
        report = self.run_audit()
        self.assertEqual(report["patient_overlap_groups"][0]["value"], "patient-a")
        self.assertEqual(report["counts"]["missing_patient_ids"], 1)
        self.assertTrue(report["unresolved_patient_identity_risks"])

    def test_unreadable_file_and_manifest_drift_make_audit_incomplete(self):
        self.paths["test"].write_bytes(b"not an image")
        Image.new("L", (64, 64), 12).save(self.paths["validation"])
        report = self.run_audit()
        self.assertEqual(report["status"], "incomplete")
        self.assertTrue({"unreadable_or_changed_image", "manifest_hash_mismatch"}.issubset({i["kind"] for i in report["issues"]}))

    def test_source_files_unchanged_and_output_never_overwritten(self):
        before = {str(p): p.read_bytes() for p in [self.manifest, *self.paths.values()]}
        self.run_audit()
        self.assertEqual(before, {str(p): p.read_bytes() for p in [self.manifest, *self.paths.values()]})
        output = self.base / "audit" / "audit.json"
        existing = output.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_audit()
        self.assertEqual(existing, output.read_bytes())
        with self.assertRaisesRegex(ValueError, "outside the dataset"):
            tool.audit(self.manifest, self.data / "audit", self.data)

    def test_windows_path_relocation(self):
        path = tool.resolve_image(r"C:\old\OriginalDataset\train.png", self.data)
        self.assertEqual(path, self.paths["train"])
        with self.assertRaisesRegex(ValueError, "escapes"):
            tool.resolve_image("../outside.png", self.data)

    def test_candidate_limit_is_reported_as_incomplete(self):
        for index, path in enumerate(self.paths.values()):
            Image.new("L", (64, 64), 100 + index).save(path)
        self.write_manifest()
        report = self.run_audit(max_candidates=1)
        self.assertFalse(report["near_search_complete"])
        self.assertEqual(report["status"], "incomplete")

    def test_unknown_preprocessing_refused_without_reports(self):
        value = json.loads(self.manifest.read_text())
        value["preprocessing"]["size"] = 96
        self.manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "preprocessing"):
            self.run_audit()
        self.assertFalse((self.base / "audit").exists())

    def test_cli_reports_incomplete_with_exit_three(self):
        self.paths["test"].write_bytes(b"invalid image")
        output = self.base / "cli-audit"
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/leakage_audit.py"),
                                 "--manifest", str(self.manifest), "--dataset-root", str(self.data),
                                 "--output-dir", str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertEqual(json.loads((output / "audit.json").read_text())["status"], "incomplete")

    def test_reports_are_reproducible(self):
        first = self.run_audit("first")
        second = self.run_audit("second")
        self.assertEqual(first, second)
        for path in (self.base / "first").iterdir():
            self.assertEqual(path.read_bytes(), (self.base / "second" / path.name).read_bytes())

    def test_bktree_matches_exhaustive_hamming_search(self):
        rng = random.Random(42)
        values = [rng.getrandbits(64) for _ in range(60)] + [7, 7, 15]
        tree = tool.BKTree()
        for index, value in enumerate(values):
            tree.insert(value, index)
        for query in values[:4] + [7]:
            for radius in (0, 6, 64):
                expected = sorted((i, (query ^ v).bit_count()) for i, v in enumerate(values) if (query ^ v).bit_count() <= radius)
                self.assertEqual(tree.query(query, radius), expected)


if __name__ == "__main__":
    unittest.main()
