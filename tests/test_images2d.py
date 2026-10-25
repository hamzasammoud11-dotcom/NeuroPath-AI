"""Small real image fixtures exercise splitting, preprocessing and training."""
from contextlib import redirect_stdout
import importlib.util
import io
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neuro_path_ai.data.images2d import discover_originals, split_originals, ImageDataset2D, validate_split_isolation
from neuro_path_ai.models.image2d import CLASS_MAPPING, PREPROCESSING, load_image_checkpoint, preprocess_image


class ImagePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "OriginalDataset"
        rng = np.random.default_rng(42)
        for name in CLASS_MAPPING:
            folder = self.root / name
            folder.mkdir(parents=True)
            for index in range(20):
                Image.fromarray(rng.integers(0, 256, (16, 16), dtype=np.uint8)).save(folder / f"{index}.png")

    def patient_manifest(self):
        manifest = self.root.parent / "patients.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "patient_id"])
            for folder in sorted(self.root.iterdir()):
                for path in sorted(folder.glob("*.png")):
                    # Two images per known fixture patient.
                    writer.writerow([path.relative_to(self.root).as_posix(), f"{folder.name}-{int(path.stem) // 2}"])
        return manifest

    def test_patient_split_keeps_multiple_images_together(self):
        records, _ = discover_originals(self.root, self.patient_manifest())
        splits = split_originals(records)
        self.assertEqual(splits, split_originals(records))
        validate_split_isolation(splits, require_patients=True)
        self.assertEqual(sum(map(len, splits.values())), 80)
        self.assertTrue(all({r["label"] for r in rows} == {0, 1} for rows in splits.values()))
        for rows in splits.values():
            counts = {}
            for row in rows:
                counts[row["patient_id"]] = counts.get(row["patient_id"], 0) + 1
            self.assertTrue(all(count == 2 for count in counts.values()))

    def test_missing_ids_require_explicit_exploratory_override(self):
        records, _ = discover_originals(self.root)
        with self.assertRaisesRegex(ValueError, "Patient IDs are required"):
            split_originals(records)
        records[0]["patient_id"] = "one-known-patient"
        with self.assertRaisesRegex(ValueError, "All images"):
            split_originals(records, allow_image_split=True)

    def test_patient_manifest_rejects_missing_duplicate_and_blank_rows(self):
        manifest = self.patient_manifest()
        lines = manifest.read_text().splitlines()
        for contents, message in [(lines[:-1], "Missing patient ID"),
                                  (lines + [lines[1]], "Duplicate patient manifest"),
                                  ([lines[0], lines[1].split(",")[0] + ","], "blank")]:
            with self.subTest(message=message):
                manifest.write_text("\n".join(contents), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    discover_originals(self.root, manifest)

    def test_same_pixels_cannot_belong_to_different_patients(self):
        folder = self.root / "NonDemented"
        (folder / "2.png").write_bytes((folder / "0.png").read_bytes())
        with self.assertRaisesRegex(ValueError, "different patients"):
            discover_originals(self.root, self.patient_manifest())

    def test_overlap_is_rejected(self):
        records, _ = discover_originals(self.root, self.patient_manifest())
        splits = split_originals(records)
        splits["test"].append(splits["train"][0])
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate_split_isolation(splits, require_patients=True)

    def test_too_few_patients_fail_without_image_fallback(self):
        records, _ = discover_originals(self.root, self.patient_manifest())
        for record in records:
            record["patient_id"] = str(record["label"])
        with self.assertRaisesRegex(ValueError, "More patients"):
            split_originals(records)

    def test_duplicates_removed_and_splits_disjoint_reproducible(self):
        folder = self.root / "NonDemented"
        (folder / "copy.png").write_bytes((folder / "0.png").read_bytes())
        records, duplicates = discover_originals(self.root)
        self.assertEqual((len(records), duplicates), (80, 1))
        splits = split_originals(records, allow_image_split=True)
        self.assertEqual(splits, split_originals(records, allow_image_split=True))
        sets = [set(row["sha256"] for row in rows) for rows in splits.values()]
        self.assertEqual(sum(map(len, sets)), len(set.union(*sets)))
        self.assertTrue(all(set(CLASS_MAPPING) == {r["source_class"] for r in rows} for rows in splits.values()))

    def test_conflicting_labels_rejected(self):
        (self.root / "MildDemented" / "conflict.png").write_bytes((self.root / "NonDemented" / "0.png").read_bytes())
        with self.assertRaisesRegex(ValueError, "conflicting"):
            discover_originals(self.root)

    def test_augmented_folder_rejected(self):
        with self.assertRaisesRegex(ValueError, "OriginalDataset"):
            discover_originals(self.root.parent / "AugmentedAlzheimerDataset")

    def test_dataset_parent_with_augmented_name_finds_originals(self):
        parent = self.root.parent / "augmented-alzheimer-mri-dataset"
        parent.mkdir()
        original = parent / "OriginalDataset"
        self.root.rename(original)
        records, _ = discover_originals(parent)
        self.assertEqual(len(records), 80)

    def test_preprocessing_shared_and_eval_deterministic(self):
        records, _ = discover_originals(self.root)
        dataset = ImageDataset2D(records)
        actual, _ = dataset[0]
        with Image.open(records[0]["path"]) as image:
            expected = preprocess_image(image)
        self.assertEqual(tuple(actual.shape), (1, 128, 128))
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual, dataset[0][0]))

    def test_one_epoch_training_checkpoint_and_heldout_outputs(self):
        spec = importlib.util.spec_from_file_location("train_images", ROOT / "scripts/train_images.py")
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        output = self.root.parent / "run"
        manifest_path = self.patient_manifest()
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with patch.object(sys, "argv", ["train_images", "--data-dir", str(self.root),
                                       "--output-dir", str(output), "--epochs", "1",
                                       "--patient-manifest", str(manifest_path),
                                       "--patient-id-source", "generated test fixture"]), redirect_stdout(io.StringIO()):
            trainer.main()
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertTrue(manifest["patient_separation_verified"])
        self.assertEqual(manifest["split_unit"], "patient")
        validate_split_isolation(manifest["splits"], require_patients=True)
        self.assertTrue((output / "test_predictions.csv").is_file())
        metrics = json.loads((output / "test_metrics.json").read_text())
        self.assertIn("specificity", metrics)
        model = load_image_checkpoint(output / "best_model.pt")
        self.assertTrue(model.patient_separation_verified)
        image, _ = ImageDataset2D(manifest["splits"]["test"])[0]
        with torch.no_grad():
            self.assertEqual(tuple(model(image.unsqueeze(0)).shape), (1, 2))
        payload = torch.load(output / "best_model.pt", weights_only=True)
        payload["preprocessing"] = {**PREPROCESSING, "size": 96}
        torch.save(payload, output / "invalid.pt")
        with self.assertRaisesRegex(ValueError, "metadata"):
            load_image_checkpoint(output / "invalid.pt")


    def test_resume_matches_uninterrupted_training_and_rejects_changed_settings(self):
        spec = importlib.util.spec_from_file_location("train_images_resume", ROOT / "scripts/train_images.py")
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        manifest = self.patient_manifest()
        continuous = self.root.parent / "continuous"
        interrupted = self.root.parent / "interrupted"

        def run(output, epochs, extra=()):
            command = ["train_images", "--data-dir", str(self.root), "--output-dir", str(output),
                       "--epochs", str(epochs), "--patient-manifest", str(manifest),
                       "--patient-id-source", "fixture", *extra]
            with patch.object(sys, "argv", command), redirect_stdout(io.StringIO()):
                trainer.main()

        run(continuous, 2)
        run(interrupted, 1)
        run(interrupted, 2, ["--resume"])
        full = torch.load(continuous / "last_model.pt", weights_only=True)
        resumed = torch.load(interrupted / "last_model.pt", weights_only=True)
        self.assertEqual(resumed["epoch"], 2)
        self.assertEqual(full["history"], resumed["history"])
        for key, value in full["model_state_dict"].items():
            self.assertTrue(torch.equal(value, resumed["model_state_dict"][key]), key)
        self.assertTrue((interrupted / "requirements-frozen.txt").is_file())
        self.assertTrue((interrupted / "environment.json").is_file())
        with redirect_stdout(io.StringIO()), patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                run(interrupted, 3, ["--resume", "--batch-size", "8"])


if __name__ == "__main__":
    unittest.main()
