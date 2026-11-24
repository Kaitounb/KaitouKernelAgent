"""CPU target-gate regressions, not evidence of actual GPU or production qualification."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from kernel_agent import gemm_qualification as q

class GemmTargetBindingTests(unittest.TestCase):
 def test_development_store_unchanged(self):
  with tempfile.TemporaryDirectory() as d,patch.object(q,'check_target') as check:
   r=q.registry(Path(d)/'store');self.assertFalse(r.initialize()['production_release']);check.assert_not_called()
 def test_missing_qualified_fails_before_commands(self):
  with tempfile.TemporaryDirectory() as d,patch.object(q.store,'DATA',Path(d)/'absent'),patch.object(q,'command_result') as command:
   with self.assertRaisesRegex(ValueError,'not packaged'):q.registry(Path(d)/'store',channel='qualified')
   command.assert_not_called()
 def test_real_binding_policy_rejects_driver_gpu_and_image(self):
  e={'image_id':q.IMAGE};t={'gpu_uuid':'GPU-test','driver_version':'591.74'};image={'exit':0,'stdout':q.IMAGE+'\n'}
  with patch.object(q,'command_result',side_effect=[image,{'exit':0,'stdout':'GPU-test, 591.74, 100.00, 54\n'}]):q.check_target(e,t)
  for row in ['GPU-other, 591.74, 100.00, 54\n','GPU-test, other, 100.00, 54\n','GPU-test, 591.74, 100.00, 54\nGPU-test, 591.74, 100.00, 54\n']:
   with patch.object(q,'command_result',side_effect=[image,{'exit':0,'stdout':row}]),self.assertRaisesRegex(ValueError,'binding differs'):q.check_target(e,t)
  with patch.object(q,'command_result',side_effect=[{'exit':0,'stdout':'wrong-image'}, {'exit':0,'stdout':'GPU-test, 591.74'}]),self.assertRaisesRegex(ValueError,'binding differs'):q.check_target(e,t)

 def test_unqualified_power_temperature_and_malformed_telemetry_rejected(self):
  e={'image_id':q.IMAGE};t={'gpu_uuid':'GPU-test','driver_version':'591.74'};image={'exit':0,'stdout':q.IMAGE+'\n'}
  for telemetry in ['240.00, 54','100.00, 83','100.00, nan','100.00, N/A','99.00, 54','N/A, 54']:
   with patch.object(q,'command_result',side_effect=[image,{'exit':0,'stdout':'GPU-test, 591.74, '+telemetry}]),self.assertRaisesRegex(ValueError,'requires100W'):q.check_target(e,t)

if __name__=='__main__':unittest.main()
