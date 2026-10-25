import io,sys,unittest
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parent)); sys.path.insert(0,str(Path(__file__).parent.parent/'inference-v003')); sys.path.insert(0,str(Path(__file__).parent.parent/'app-v002'))
from inference import ImageClassifier2D,predict_subject
from explainability import TARGET_LAYER_NAME,gradcam
from validation import EXPECTED_Z,validate_uploads
from logic import state_signature

class Upload:
    def __init__(self,name,image): self.name=name; b=io.BytesIO(); image.save(b,format='PNG'); self.data=b.getvalue()
    def getvalue(self): return self.data

class V005Tests(unittest.TestCase):
    def img(self,v=80): return Image.fromarray(np.full((208,208),v,dtype=np.uint8),mode='L')
    def files(self,subject='OAS1_0013'): return [Upload(f'{subject}_MR1_z{z:03d}.png',self.img()) for z in EXPECTED_Z]
    def test_gradcam_finite_and_size(self):
        h,p=gradcam(ImageClassifier2D().eval(),self.img()); self.assertEqual(h.shape,(128,128)); self.assertTrue(np.isfinite(h).all()); self.assertTrue(0<=p<=1); self.assertEqual(TARGET_LAYER_NAME,'features[6]')
    def test_gradcam_preserves_prediction_logic(self):
        model=ImageClassifier2D().eval(); images=[self.img(i*15) for i in range(1,10)]; before=predict_subject(model,images); gradcam(model,images[0]); after=predict_subject(model,images); self.assertEqual(before['prediction'],after['prediction']); self.assertAlmostEqual(before['cdr_score'],after['cdr_score'],places=7)
    def test_training_guard_and_session_signature(self):
        ordered=validate_uploads(list(reversed(self.files()))); self.assertEqual(len(ordered),9); self.assertEqual(state_signature(ordered),state_signature(ordered)); test_files=self.files('OAS1_0342');
        with self.assertRaises(ValueError): validate_uploads(test_files)

if __name__=='__main__': unittest.main()
