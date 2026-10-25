"""Isolated runtime checks; require torch but no running API or Streamlit UI.

Load the relevant functions from source to avoid importing UI startup code
and optional imaging/web dependencies in this small regression suite.
"""
from __future__ import annotations

import ast
import asyncio
import logging
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neuro_path_ai.models.fusion import FusionModel
from neuro_path_ai.models.imaging import Simple3DCNN
from neuro_path_ai.models.image2d import ImageClassifier2D, load_image_checkpoint, preprocess_image


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail


def load_functions(relative_path, names, namespace):
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8-sig"))
    functions = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name in names]
    for node in functions:
        node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), relative_path, "exec"), namespace)
    return namespace


class PredictionReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkpoint = Path(self.temp.name) / "model.pt"
        self.ns = load_functions("src/neuro_path_ai/api/app.py", {
            "_load_checkpoint_state", "_infer_backbone", "_infer_tabular_dim",
            "_resolve_backbone", "_resolve_tabular_dim", "_load_model",
            "_startup", "health", "predict", "_predict_image_2d",
        }, dict(Path=Path, logging=logging, torch=torch, FusionModel=FusionModel,
                Simple3DCNN=Simple3DCNN, HTTPException=HTTPException,
                ImageClassifier2D=ImageClassifier2D, load_image_checkpoint=load_image_checkpoint,
                preprocess_image=preprocess_image,
                File=lambda *a: None, Form=lambda *a: None,
                MODEL_CHECKPOINT=self.checkpoint, BACKBONE_STRATEGY="auto",
                TABULAR_DIM_ENV=None, _device=None, _model=None,
                _target_layer=None, _model_error="Model not initialized",
                find_default_target_layer=lambda model: model.encoder[0]))

    def state(self):
        return FusionModel(Simple3DCNN(), tabular_dim=14, imaging_dim=128).state_dict()

    def assert_unavailable(self):
        with self.assertLogs(level="ERROR"):
            self.ns["_startup"]()
        self.assertIsNone(self.ns["_model"])
        with self.assertRaises(HTTPException) as health_error:
            self.ns["health"]()
        self.assertEqual(health_error.exception.status_code, 503)
        # Refuse before parsing clinical data or reading/writing the upload.
        upload = types.SimpleNamespace(read=Mock(side_effect=AssertionError("read upload")))
        with self.assertRaises(HTTPException) as prediction_error:
            asyncio.run(self.ns["predict"](upload, "invalid json"))
        self.assertEqual(prediction_error.exception.status_code, 503)
        upload.read.assert_not_called()

    def test_missing_checkpoint(self):
        self.assert_unavailable()

    def test_corrupt_checkpoint(self):
        self.checkpoint.write_bytes(b"not a checkpoint")
        self.assert_unavailable()

    def test_invalid_checkpoint_payloads(self):
        for payload in ({}, {"model_state_dict": {}}, {"weight": "invalid"}):
            with self.subTest(payload=payload):
                torch.save(payload, self.checkpoint)
                self.assert_unavailable()

    def test_incompatible_checkpoint(self):
        state = self.state()
        del state["head.classifier.bias"]
        torch.save(state, self.checkpoint)
        self.assert_unavailable()

    def test_nonfinite_checkpoint(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                state = self.state()
                state["head.classifier.bias"][0] = value
                torch.save(state, self.checkpoint)
                self.assert_unavailable()

    def test_valid_raw_and_wrapped_checkpoints(self):
        state = self.state()
        for payload in (state, {"model_state_dict": state, "epoch": 2}):
            with self.subTest(wrapped="model_state_dict" in payload):
                torch.save(payload, self.checkpoint)
                self.ns["_startup"]()
                self.assertEqual(self.ns["health"](), {"status": "ok"})
                self.assertFalse(self.ns["_model"].training)
                for key, value in self.ns["_model"].state_dict().items():
                    self.assertTrue(torch.equal(value.cpu(), state[key]))
        # A subsequent failed initialization must clear the previously loaded model.
        self.checkpoint.unlink()
        self.assert_unavailable()

    def test_dashboard_always_calls_api(self):
        result = Mock(status_code=200)
        result.json.return_value = {"probability": 0.73, "prediction": 1}
        requests = types.SimpleNamespace(post=Mock(return_value=result))
        import json
        ns = load_functions("src/neuro_path_ai/ui/dashboard.py", {"_call_api"}, dict(
            json=json, requests=requests, st=Mock(),
            _normalize_api_url=lambda url: url,
            _resolve_api_urls=lambda url: ("base", url, "health", "host", 8000),
            _check_api_health=lambda url: True,
        ))
        upload = types.SimpleNamespace(name="reference.png", getvalue=lambda: b"image")
        response = ns["_call_api"]("http://localhost:8000/predict", upload, {"clinical_features": [65]})
        self.assertEqual(response, result.json.return_value)
        requests.post.assert_called_once()

    def test_2d_checkpoint_api_prediction_and_rejected_volume(self):
        from io import BytesIO
        from PIL import Image
        from neuro_path_ai.models.image2d import CLASS_MAPPING, PREPROCESSING
        torch.save(dict(architecture="image2d_v1", model_state_dict=ImageClassifier2D().state_dict(),
                        preprocessing=PREPROCESSING, class_mapping=CLASS_MAPPING), self.checkpoint)
        self.ns["_startup"]()
        self.assertEqual(self.ns["health"](), {"status": "ok"})
        data = BytesIO()
        Image.new("L", (32, 32), 100).save(data, format="PNG")
        async def read():
            return data.getvalue()
        upload = types.SimpleNamespace(filename="slice.png", read=read)
        response = asyncio.run(self.ns["predict"](upload, "{}"))
        self.assertTrue(0 <= response["probability"] <= 1)
        self.assertEqual(response["model_type"], "image2d_v1")
        self.assertIn("clinical/genetic inputs are not used", response["warning"])
        upload.filename = "scan.nii.gz"
        with self.assertRaises(HTTPException) as error:
            asyncio.run(self.ns["predict"](upload, "{}"))
        self.assertEqual(error.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
