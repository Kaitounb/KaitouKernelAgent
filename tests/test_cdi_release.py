"""CPU-only RTX3060 release/admission negatives; not production evidence."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from kernel_agent import cdi_qualification as q
ROOT=Path(__file__).resolve().parents[1]
class CdiReleaseTests(unittest.TestCase):
 def test_original_runtime_reused_except_fixed_binding(self):
  original=(ROOT/'kernel_agent/runner.py').read_text();actual=(ROOT/'kernel_agent/cdi_runner.py').read_text();actual=actual.replace('"""Original bounded DockerRunner, fixed reviewed RTX3060 image/nativeCDI only."""','"""Bounded Docker adapter for reviewed GEMM artifacts. No model or promotion logic."""').replace(q.IMAGE,'sha256:ce00f6dc2eea43dc508b90b548f624646d643a92d363aa715fdb5a3b4f49290a').replace("if gpu:cmd+=['--runtime=runc','--device','nvidia.com/gpu=0','--env','NVIDIA_VISIBLE_DEVICES=void']", "if gpu:cmd+=['--gpus','device=0']");self.assertEqual(actual,original)
 def test_no_certificate_stops_before_environment(self):
  with tempfile.TemporaryDirectory() as td,patch.object(q,'DATA',Path(td)),patch.object(q,'check_environment') as check:
   with self.assertRaisesRegex(ValueError,'not packaged'):q.registry(Path(td)/'store',channel='qualified')
   check.assert_not_called()
 def test_development_store_and_original_catalog_unchanged(self):
  with tempfile.TemporaryDirectory() as td,patch.object(q,'check_environment'):
   r=q.registry(Path(td)/'store');s=r.initialize();self.assertFalse(s['production_release']);self.assertEqual(s['selected_version'],'sm86-cublas-container');r.activate('sm86-cublas-original');self.assertEqual(r.rollback()['selected_version'],'sm86-cublas-container')
  self.assertNotEqual(json.loads((ROOT/'kernel_agent/data/releases.json').read_text())['versions']['sm86-cublas-container']['image_id'],q.IMAGE)
 def test_gpu_driver_image_and_profile_mismatch(self):
  image={'exit':0,'stdout':q.IMAGE};ok=q.TARGET['gpu_uuid']+', '+q.TARGET['driver_version']+', 170.00, 45'
  with patch.object(q,'command_result',side_effect=[image,{'exit':0,'stdout':ok}]):q.check_environment()
  for value in [ok.replace(q.TARGET['gpu_uuid'],'GPU-other'),ok.replace(q.TARGET['driver_version'],'other'),ok+'\n'+ok]:
   with patch.object(q,'command_result',side_effect=[image,{'exit':0,'stdout':value}]),self.assertRaisesRegex(ValueError,'binding differs'):q.check_environment()
  for tail in ['160, 45','170, 83','170, nan','170, N/A']:
   with patch.object(q,'command_result',side_effect=[image,{'exit':0,'stdout':q.TARGET['gpu_uuid']+', '+q.TARGET['driver_version']+', '+tail}]),self.assertRaisesRegex(ValueError,'requires170W'):q.check_environment()
  with patch.object(q,'command_result',side_effect=[{'exit':0,'stdout':'other'},{'exit':0,'stdout':ok}]),self.assertRaisesRegex(ValueError,'binding differs'):q.check_environment()
 def test_qualified_certificate_runtime_and_evidence_flags(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);(d/'gemm_contract_v1.json').write_bytes((q.DATA/'gemm_contract_v1.json').read_bytes());(d/'gemm_deployment_policy_v2.json').write_bytes((q.DATA/'gemm_deployment_policy_v2.json').read_bytes());catalog=q.index();c={'schema_version':1,'status':'QUALIFIED','scope':q.SCOPE,'production_release':True,'source_execution_enabled':False,'environment':q.ENVIRONMENT,'target_binding':q.TARGET,'operating_profile':q.PROFILE,'gpu_binding':'nativeCDIdevice0/runc/visiblevoid','binary_sha256':sorted(q.BINARIES),'runtime_files':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in q.RUNTIME_FILES},'deployment_policy_sha256':q.sha(q.DATA/'gemm_deployment_policy_v2.json')}
   for name in ['independent_correctness_pass','stability_pass','nonregression_pass','installed_runtime_pass','boundary_pass','recovery_pass','rollback_pass','raw_audit_pass']:c[name]=True
   for meta in catalog['versions'].values():meta['production_release']=True
   def write(value):
    raw=json.dumps(value).encode();(d/'gemm_qualification.json').write_bytes(raw);catalog['qualification_sha256']=hashlib.sha256(raw).hexdigest();(d/'qualified_releases.json').write_text(json.dumps(catalog))
   write(c)
   with patch.object(q,'DATA',d):self.assertEqual(q.index('qualified')['default'],'sm86-cublas-container')
   for change in [{'recovery_pass':False},{'target_binding':{**q.TARGET,'driver_version':'other'}},{'runtime_files':{}},{'binary_sha256':[{}]}]:
    bad=copy.deepcopy(c);bad.update(change);write(bad)
    with patch.object(q,'DATA',d),self.assertRaises(ValueError):q.index('qualified')
if __name__=='__main__':unittest.main()
