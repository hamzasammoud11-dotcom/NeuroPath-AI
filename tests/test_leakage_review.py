"""Visual-review fixtures only; no training and no checkpoint access."""
import importlib.util
import json
from hashlib import sha256
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("leakage_review", ROOT / "scripts/leakage_review.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class LeakageReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.dataset = self.base / "OriginalDataset"
        self.dataset.mkdir()
        self.audit_dir = self.base / "audit"
        self.audit_dir.mkdir()
        self.audit = self.audit_dir / "audit.json"
        inventory = []
        for index, split in enumerate(("train", "validation", "test")):
            path = self.dataset / f"{split}.png"
            Image.new("L", (64, 32), 100 + index).save(path)
            inventory.append(dict(id=f"{split}:0", split=split, label=index % 2,
                                  source_class="NonDemented" if index % 2 == 0 else "MildDemented",
                                  patient_id=None, manifest_path=str(path), resolved_path=str(path),
                                  file_sha256=sha256(path.read_bytes()).hexdigest()))
        self.payload = dict(schema_version=1, status="findings", near_search_complete=True,
                            inventory=inventory, unresolved_patient_identity_risks=["Identity unavailable"],
                            parameters={"ssim_threshold": 0.95}, issues=[], near_candidates=[
            dict(left_id="validation:0", right_id="test:0", left_split="validation", right_split="test", ssim=0.999, phash_distance=1, review_priority="high"),
            dict(left_id="validation:0", right_id="train:0", left_split="validation", right_split="train", ssim=0.998, phash_distance=2, review_priority="high"),
            dict(left_id="test:0", right_id="train:0", left_split="test", right_split="train", ssim=0.970, phash_distance=4, review_priority="high"),
            dict(left_id="train:0", right_id="test:0", left_split="train", right_split="test", ssim=0.9, phash_distance=5, review_priority="lower"),
        ])
        self.save_audit()

    def save_audit(self):
        self.audit.write_text(json.dumps(self.payload), encoding="utf-8")

    def build(self, name="review", **kwargs):
        return tool.build_review(self.audit, self.base / name, self.dataset, **kwargs)

    def test_priority_group_order_metrics_labels_and_embedded_images(self):
        result = self.build()
        self.assertEqual([p["group"] for p in result["pairs"]], ["train/test", "train/validation", "validation/test"])
        self.assertEqual(result["pairs"][0]["left_id"], "train:0")
        self.assertEqual(result["pairs"][0]["ssim"], 0.970)
        self.assertEqual(result["images"]["validation:0"]["label"], 1)
        html = (self.base / "review/review.html").read_text(encoding="utf-8")
        self.assertIn("data:image/png;base64,", html)
        self.assertNotIn("__REVIEW_JSON__", html)
        self.assertNotIn('src="https:', html)
        self.assertNotIn('src="file:', html)
        self.assertEqual(result["preview_errors"], 0)

    def test_unchanged_inputs_and_exclusive_output(self):
        before = {p: p.read_bytes() for p in [self.audit, *self.dataset.iterdir()]}
        self.build()
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        existing = (self.base / "review/review.html").read_bytes()
        with self.assertRaisesRegex(ValueError, "exists"):
            self.build()
        self.assertEqual(existing, (self.base / "review/review.html").read_bytes())
        for output in (self.dataset / "report", self.audit_dir / "report"):
            with self.assertRaises(ValueError):
                tool.build_review(self.audit, output, self.dataset)

    def test_missing_or_modified_image_withholds_preview(self):
        (self.dataset / "train.png").write_bytes(b"changed after audit")
        # Missing is simulated by changing the inventory path, without deleting files.
        self.payload["inventory"][2]["manifest_path"] = str(self.dataset / "missing.png")
        self.save_audit()
        result = self.build()
        self.assertEqual(result["preview_errors"], 2)
        self.assertEqual(result["images"]["train:0"]["preview_status"], "unavailable")
        html = (self.base / "review/review.html").read_text()
        self.assertIn("preview withheld", html)

    def test_limit_prioritizes_train_test_and_reports_omissions(self):
        result = self.build(max_pairs=1)
        self.assertEqual(result["displayed_pairs"], 1)
        self.assertEqual(result["omitted_pairs"], 2)
        self.assertEqual(result["pairs"][0]["group"], "train/test")

    def test_incomplete_source_and_zero_candidates_remain_explicit(self):
        self.payload.update(status="incomplete", near_search_complete=False, near_candidates=[])
        self.save_audit()
        result = self.build()
        self.assertEqual(result["audit_status"], "incomplete")
        self.assertFalse(result["near_search_complete"])
        self.assertEqual(result["displayed_pairs"], 0)

    def test_script_injection_is_inert_json(self):
        attack = '</script><script>alert("fixture")</script>'
        self.payload["inventory"][0]["source_class"] = attack
        self.save_audit()
        self.build()
        html = (self.base / "review/review.html").read_text(encoding="utf-8")
        self.assertNotIn(attack, html)
        raw = re.search(r'<script type="application/json" id="reviewData">(.*?)</script>', html, re.S).group(1)
        self.assertEqual(json.loads(raw)["images"]["train:0"]["source_class"], attack)

    def test_relocation_and_invalid_candidate_ids(self):
        for record in self.payload["inventory"]:
            record["manifest_path"] = "C:/old/OriginalDataset/" + record["split"] + ".png"
        self.save_audit()
        self.assertEqual(self.build()["preview_errors"], 0)
        self.payload["near_candidates"][0]["left_id"] = "does-not-exist"
        self.save_audit()
        with self.assertRaisesRegex(ValueError, "unknown inventory"):
            self.build("invalid")

    def test_deterministic_html_and_index(self):
        self.assertEqual(self.build("first"), self.build("second"))
        for name in ("review.html", "review_index.json"):
            self.assertEqual((self.base / "first" / name).read_bytes(), (self.base / "second" / name).read_bytes())

    @unittest.skipUnless(Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe").exists(), "Optional browser check requires installed Edge")
    def test_browser_filters_annotations_export_and_import(self):
        self.build()
        html = (self.base / "review/review.html").read_text(encoding="utf-8")
        checks = r'''<script>
(async()=>{const result=document.createElement('pre');result.id='browser-test-result';document.body.append(result);try{
const check=(value,message)=>{if(!value)throw new Error(message);};
check(document.querySelectorAll('article.pair').length===3,'initial cards');
$('group').value='train/test';$('group').dispatchEvent(new Event('input'));
check(document.querySelectorAll('article.pair').length===1,'group filter');
$('minSsim').value='0.99';$('minSsim').dispatchEvent(new Event('input'));
check(document.querySelectorAll('article.pair').length===0,'SSIM filter');
$('minSsim').value='-1';$('minSsim').dispatchEvent(new Event('input'));
$('reviewer').value='Synthetic reviewer';
const textarea=document.querySelector('article textarea');textarea.value='Review fixture';textarea.dispatchEvent(new Event('input'));
const select=document.querySelector('article select');select.value='uncertain';select.dispatchEvent(new Event('change'));
let exported;download=(name,text,type)=>{exported={name,text,type};};$('exportJson').click();
const saved=JSON.parse(exported.text);check(saved.annotations.length===1,'export count');check(saved.annotations[0].notes==='Review fixture','export notes');
annotations={};const transfer=new DataTransfer();transfer.items.add(new File([JSON.stringify(saved)],'notes.json',{type:'application/json'}));
$('importJson').files=transfer.files;$('importJson').dispatchEvent(new Event('change'));
await new Promise(resolve=>setTimeout(resolve,200));check(Object.values(annotations)[0].decision==='uncertain','import decision');
$('decision').value='uncertain';$('decision').dispatchEvent(new Event('input'));
check(document.querySelectorAll('article.pair').length===1,'decision filter');
const wrong={...saved,audit_sha256:'wrong'};const bad=new DataTransfer();bad.items.add(new File([JSON.stringify(wrong)],'bad.json'));$('importJson').files=bad.files;$('importJson').dispatchEvent(new Event('change'));
await new Promise(resolve=>setTimeout(resolve,200));check($('message').textContent.includes('Import failed'),'foreign audit import rejected');
result.textContent='PASS';setDirty(false);
}catch(error){result.textContent='FAIL: '+error.message;setDirty(false);}})();
</script>'''
        test_page = self.base / "browser-check.html"
        test_page.write_text(html.replace("</body>", checks + "</body>"), encoding="utf-8")
        browser = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        process = subprocess.run([browser, "--headless", "--disable-gpu", "--no-first-run",
                                  "--disable-background-networking", "--disable-extensions",
                                  "--user-data-dir=" + str(self.base / "browser-profile"),
                                  "--virtual-time-budget=3000", "--dump-dom", test_page.as_uri()],
                                 capture_output=True, timeout=40)
        output = process.stdout.decode("utf-8", errors="replace")
        self.assertIn('<pre id="browser-test-result">PASS</pre>', output,
                      process.stderr.decode("utf-8", errors="replace")[-1000:] + output[-1200:])


if __name__ == "__main__":
    unittest.main()
