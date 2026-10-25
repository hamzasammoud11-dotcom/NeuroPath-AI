import io, sys, unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,str(Path(__file__).parent)); sys.path.insert(0,str(Path(__file__).parent.parent/'inference-v003')); sys.path.insert(0,str(Path(__file__).parent.parent/'app-v002'))
from inference import ImageClassifier2D, predict_subject
from explainability import TARGET_LAYER_NAME, gradcam
from validation import EXPECTED_Z, validate_uploads

class ExplainabilityTests(unittest.TestCase):
    def image(self, value=80): return Image.fromarray(np.full((208,208),value,dtype=np.uint8),mode='L')
    def test_gradcam_shape_and_probability(self):
        model=ImageClassifier2D().eval(); heat,p=gradcam(model,self.image()); self.assertEqual(heat.shape,(128,128)); self.assertTrue(0<=p<=1); self.assertEqual(TARGET_LAYER_NAME,'features[6]')
    def test_explainability_does_not_change_subject_logic(self):
        model=ImageClassifier2D().eval(); images=[self.image(i*10) for i in range(1,10)]; before=predict_subject(model,images); gradcam(model,images[0]); after=predict_subject(model,images); self.assertEqual(before['prediction'],after['prediction']); self.assertAlmostEqual(before['cdr_score'],after['cdr_score'],places=7)
    def test_synthetic_training_upload(self):
        class Upload:
            def __init__(self,name,image):
                self.name=name; b=io.BytesIO(); image.save(b,format='PNG'); self.data=b.getvalue()
            def getvalue(self): return self.data
        files=[Upload(f'OAS1_0013_MR1_z{z:03d}.png',self.image()) for z in EXPECTED_Z]
        self.assertEqual(len(validate_uploads(files)),9)

if __name__=='__main__': unittest.main()
