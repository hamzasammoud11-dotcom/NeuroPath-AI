"""Small synthetic Analyze archives/clinical tables; no training or real IDs inferred."""
from contextlib import redirect_stdout
import csv
import importlib.util
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neuro_path_ai.data import oasis1_prepare as tool


class OasisPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.clinical = self.data / "clinical.csv"
        self.shape = (16, 20, 32)
        rng = np.random.default_rng(42)
        grid = np.indices(self.shape)
        mask = sum(((grid[i] - (self.shape[i] - 1) / 2) / (self.shape[i] * .45)) ** 2 for i in range(3)) < 1
        self.volume = np.where(mask, rng.integers(50, 250, self.shape), 0).astype(np.int16)
        self.header, self.image = self.analyze_bytes(self.volume)
        rows = []
        for index in range(1, 13):
            rows.append(dict(ID=f"OAS1_{index:04d}_MR1", Age=65 + index, CDR=0 if index <= 6 else .5,
                             **{"M/F": "F" if index % 2 else "M", "MMSE": ""}))
        rows += [dict(ID="OAS1_0013_MR1", Age=45, CDR=0),
                 dict(ID="OAS1_0014_MR1", Age=70, CDR=""),
                 dict(ID="OAS1_0001_MR2", Age=66, CDR=0)]
        self.write_clinical(rows)
        self.archive = self.data / "disc-test.tar.gz"
        with tarfile.open(self.archive, "w:gz") as archive:
            for row in rows:
                identity = row["ID"]
                base = f"disc/{identity}/PROCESSED/MPRAGE/T88_111/{identity}_mpr_n4_anon_111_t88_masked_gfc"
                self.add_member(archive, base + ".img", self.image)
                self.add_member(archive, base + ".hdr", self.header)
            self.add_member(archive, "disc/RAW/never-extract.txt", b"must remain in tar")

    def write_clinical(self, rows):
        with self.clinical.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "Age", "CDR", "M/F", "MMSE"])
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def add_member(archive, name, data):
        info = tarfile.TarInfo(name)
        info.size = len(data)
        archive.addfile(info, BytesIO(data))

    @staticmethod
    def analyze_bytes(volume):
        header = nib.AnalyzeHeader(endianness=">")
        header.set_data_dtype(np.int16)
        image = nib.AnalyzeImage(volume, np.diag([-1, 1, 1, 1]), header=header)
        holders = {key: nib.FileHolder(fileobj=BytesIO()) for key in ("header", "image")}
        image.to_file_map(holders)
        return holders["header"].fileobj.getvalue(), holders["image"].fileobj.getvalue()

    def test_metadata_age_cdr_and_repeat_rules(self):
        records, _ = tool.clinical_records(self.clinical)
        self.assertEqual(sum(r["clinical_eligible"] for r in records), 12)
        self.assertEqual(records[6]["label"], 1)
        self.assertEqual(records[6]["patient_id"], "OAS1_0007")
        self.assertEqual(records[12]["exclusion_reason"], "age_below_60")
        self.assertEqual(records[13]["exclusion_reason"], "missing_or_invalid_cdr")
        self.assertEqual(records[14]["exclusion_reason"], "repeat_or_reliability_session_excluded")
        self.assertIsNone(records[0]["mmse"])

    def test_no_imputation_or_unknown_cdr_labels(self):
        self.write_clinical([dict(ID="OAS1_0001_MR1", Age=60, CDR=3),
                             dict(ID="invented", Age=70, CDR=0),
                             dict(ID="OAS1_0002_MR1", Age="", CDR=0)])
        records, _ = tool.clinical_records(self.clinical)
        self.assertTrue(all(row["label"] is None for row in records))

    def test_duplicate_clinical_session_refused(self):
        self.write_clinical([dict(ID="OAS1_0001_MR1", Age=60, CDR=0)] * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate clinical"):
            tool.clinical_records(self.clinical)

    def test_age_boundary_and_all_documented_cdr_labels(self):
        self.write_clinical([dict(ID=f"OAS1_{i:04d}_MR1", Age=60, CDR=cdr)
                             for i, cdr in enumerate((0, .5, 1, 2), start=1)])
        records, _ = tool.clinical_records(self.clinical)
        self.assertEqual([r["label"] for r in records], [0, 1, 1, 1])
        self.assertTrue(all(r["age"] == 60 for r in records))

    def test_xlsx_reader_preserves_sparse_cells_and_source_ids(self):
        path = self.data / "table.xlsx"
        xml = '''<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
        <row r="1"><c r="A1" t="inlineStr"><is><t>ID</t></is></c><c r="B1" t="inlineStr"><is><t>Age</t></is></c><c r="C1" t="inlineStr"><is><t>CDR</t></is></c></row>
        <row r="2"><c r="A2" t="inlineStr"><is><t>OAS1_0001_MR1</t></is></c><c r="B2"><v>70</v></c><c r="C2"><v>0.5</v></c></row>
        <row r="3"><c r="A3" t="inlineStr"><is><t>OAS1_0002_MR1</t></is></c><c r="B3"><v>70</v></c></row>
        </sheetData></worksheet>'''
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("xl/worksheets/sheet1.xml", xml)
        records, _ = tool.clinical_records(path)
        self.assertEqual(records[0]["cdr"], .5)
        self.assertEqual(records[1]["exclusion_reason"], "missing_or_invalid_cdr")

    def test_analyze_endianness_shape_and_deterministic_slices(self):
        with patch.object(tool, "EXPECTED_SHAPE", self.shape):
            volume, qc = tool.load_analyze(self.header, self.image)
        self.assertTrue(np.array_equal(volume, self.volume))
        self.assertEqual(qc["header_endianness"], ">")
        planes = tool.volume_slices(volume, qc)
        second = tool.volume_slices(volume, qc)
        self.assertEqual(len(planes), 9)
        self.assertEqual(len({p[0] for p in planes}), 9)
        self.assertTrue(all(a[2].tobytes() == b[2].tobytes() for a, b in zip(planes, second)))
        self.assertTrue(all(p[2].width == p[2].height for p in planes))
        self.assertFalse(qc["anatomical_orientation_independently_verified"])
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            tool.load_analyze(self.header, self.image)

    def test_group_split_reproducibility(self):
        records, _ = tool.clinical_records(self.clinical)
        eligible = [r for r in records if r["clinical_eligible"]]
        self.assertEqual(tool.subject_splits(eligible, 42), tool.subject_splits(eligible, 42))
        with self.assertRaisesRegex(ValueError, "one session"):
            tool.subject_splits(eligible + [eligible[0]], 42)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            tool.subject_splits(eligible[:2], 42)

    def test_singleton_fourth_dimension_and_nonfinite_volume_qc(self):
        header, image = self.analyze_bytes(self.volume[..., None])
        with patch.object(tool, "EXPECTED_SHAPE", self.shape):
            volume, qc = tool.load_analyze(header, image)
        self.assertEqual(tuple(volume.shape), self.shape)
        self.assertEqual(qc["original_shape"], [*self.shape, 1])
        constant = np.ones(self.shape, dtype=np.int16)
        header, image = self.analyze_bytes(constant)
        with patch.object(tool, "EXPECTED_SHAPE", self.shape), self.assertRaisesRegex(ValueError, "Degenerate"):
            tool.load_analyze(header, image)

    def test_streamed_preparation_immutable_sources_and_no_bulk_extraction(self):
        before = {p: tool.file_hash(p) for p in self.data.iterdir()}
        output = self.root / "prepared"
        with patch.object(tool, "EXPECTED_SHAPE", self.shape), redirect_stdout(StringIO()):
            result = tool.prepare(self.data, self.clinical, output)
        self.assertEqual(before, {p: tool.file_hash(p) for p in self.data.iterdir()})
        self.assertEqual(result["prepared_subjects"], 12)
        self.assertEqual(result["prepared_slices"], 108)
        self.assertFalse(result["training_started"])
        self.assertFalse(result["ready_for_training"])
        self.assertFalse(list(output.rglob("*.hdr")))
        self.assertFalse(list(output.rglob("*.img")))
        self.assertFalse(list(output.rglob("never-extract.txt")))
        manifest = json.loads((output / "manifest.json").read_text())
        seen = set()
        for rows in manifest["splits"].values():
            ids = {row["patient_id"] for row in rows}
            self.assertFalse(seen & ids)
            seen.update(ids)
            self.assertTrue(all(Path(r["path"]).is_file() for r in rows))
        with self.assertRaisesRegex(ValueError, "NEW output"):
            tool.prepare(self.data, self.clinical, output)

    def test_unsafe_tar_path_refused_without_extraction(self):
        path = self.root / "unsafe.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            self.add_member(archive, "../escape.txt", b"fixture")
        with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, "Unsafe archive"):
            tool.archive_index([path])
        self.assertFalse((self.root / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()
