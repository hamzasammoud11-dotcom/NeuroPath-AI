import unittest
from pathlib import Path
import tempfile,json
import numpy as np
from PIL import Image
import sys
sys.path.insert(0,str(Path(__file__).parent.parent/'inference-v003'))
from inference import ImageClassifier2D,predict_subject
from explainability import gradcam
from training_curves import load_curves
class CurveTests(unittest.TestCase):
    def test_missing_artifacts_are_graceful(self):
        with tempfile.TemporaryDirectory() as d: self.assertFalse(load_curves(Path(d))["available"])
    def test_history_and_validation_are_read(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); run=root/'horizontal_flip'; run.mkdir(); (run/'summary.json').write_text(json.dumps({'variant':'horizontal_flip','epoch':18,'validation_auroc':.6975})); (run/'history.json').write_text(json.dumps([{'epoch':1,'train_loss':.8,'validation':{'auroc':.61,'balanced_accuracy':.55}},{'epoch':18,'train_loss':.3,'validation':{'auroc':.6975,'balanced_accuracy':.64}}])); result=load_curves(root); self.assertTrue(result['available']); self.assertEqual(len(result['rows']),2); self.assertAlmostEqual(result['rows'][-1]['validation_auroc'],.6975)
    def test_synthetic_upload_navigation_inference_results_state(self):
        model=ImageClassifier2D().eval(); images=[Image.fromarray(np.full((208,208),i*10,dtype=np.uint8),mode='L') for i in range(1,10)]
        state={'uploads':images,'result':None,'cams':None}; state['result']=predict_subject(model,images,subject_id='synthetic-train'); state['cams']=[gradcam(model,image) for image in images]
        self.assertEqual(len(state['uploads']),9); self.assertEqual(len(state['cams']),9); self.assertIn('cdr_score',state['result']); self.assertTrue(np.isfinite(state['result']['cdr_score']))
if __name__=='__main__': unittest.main()
