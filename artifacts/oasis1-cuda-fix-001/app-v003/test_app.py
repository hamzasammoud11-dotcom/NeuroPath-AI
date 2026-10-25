import io, sys, unittest
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent/'app-v002'))
from validation import EXPECTED_Z, validate_uploads
from logic import display_report, preview_image

class Upload:
    def __init__(self,name,image): self.name=name; b=io.BytesIO(); image.save(b,format='PNG'); self.data=b.getvalue()
    def getvalue(self): return self.data

class AppV003Tests(unittest.TestCase):
    def files(self):
        image=Image.fromarray(np.zeros((208,208),dtype=np.uint8), mode='L')
        return [Upload(f'OAS1_0013_MR1_z{z:03d}.png',image) for z in EXPECTED_Z]
    def test_training_sequence_and_preview(self):
        ordered=validate_uploads(list(reversed(self.files())))
        self.assertEqual(len(ordered),9); self.assertEqual(preview_image(ordered[0][1],1.2,0.8).size,(208,208))
    def test_result_report_uses_real_fields(self):
        result={'cdr_score':.61,'prediction':1,'threshold':.5,'slice_positive_probabilities':[.61]*9}
        report=display_report(result,'OAS1_0013'); self.assertEqual(report['prediction'],1); self.assertNotIn('confidence',report)
    def test_held_out_subject_blocked(self):
        files=self.files(); image=Image.fromarray(np.zeros((208,208),dtype=np.uint8),mode='L'); b=Upload('OAS1_0342_MR1_z052.png',image); files[0]=b
        with self.assertRaises(ValueError): validate_uploads(files)

if __name__=='__main__': unittest.main()
