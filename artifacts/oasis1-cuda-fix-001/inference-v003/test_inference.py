import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).parent))
from inference import (CLASS_MAPPING, EXPECTED_CHECKPOINT_NAME, ImageClassifier2D,
                       PREPROCESSING, load_model, predict_subject, predict_subject_paths)


class InferenceTests(unittest.TestCase):
    def checkpoint(self, path: Path) -> str:
        payload = {"architecture": "image2d_v1", "preprocessing": PREPROCESSING,
                   "class_mapping": CLASS_MAPPING, "variant": "horizontal_flip",
                   "epoch": 18, "protocol": {"threshold": 0.5,
                   "aggregation": "arithmetic mean of slice positive probabilities per subject"},
                   "model_state_dict": ImageClassifier2D().state_dict()}
        torch.save(payload, path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_synthetic_nine_slice_aggregation(self):
        model = ImageClassifier2D().eval()
        result = predict_subject(model, [Image.fromarray(np.full((208, 208), i, dtype=np.uint8)) for i in range(9)])
        self.assertEqual(result["slice_count"], 9)
        self.assertAlmostEqual(result["cdr_score"], np.mean(result["slice_positive_probabilities"]))
        self.assertIn(result["prediction"], (0, 1))

    def test_hash_pin_and_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / EXPECTED_CHECKPOINT_NAME
            digest = self.checkpoint(good)
            self.assertTrue(load_model(good, expected_sha256=digest).training is False)
            with self.assertRaises(ValueError): load_model(good, expected_sha256="0" * 64)
            with self.assertRaises(ValueError): load_model(good.with_name("other.pt"), expected_sha256=digest)

    def test_exactly_nine_paths(self):
        with self.assertRaises(ValueError): predict_subject_paths(ImageClassifier2D(), [])


if __name__ == "__main__":
    unittest.main()
