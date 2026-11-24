"""CPU rejection/admission policy regressions; not evidence of GPU qualification."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from kernel_agent.operators import qualification as q
from kernel_agent.operators.runtime_contract import specifications,validate_arrays
from kernel_agent.operators.cli import approved_runner
import numpy as np

class OperatorQualificationTests(unittest.TestCase):
    def test_absent_certificate_rejected_without_commands(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(q,'DATA',Path(folder)),patch.object(q,'command_result') as command:
            with self.assertRaisesRegex(ValueError,'not packaged'):q.certificate()
            command.assert_not_called()
    def test_unlisted_python_startup_and_bytecode_rejected(self):
        import hashlib
        from kernel_agent.operators.runner import OperatorRunner
        for name in ['sitecustomize.py','__pycache__/safe.cpython-312.pyc']:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);(root/'safe.py').write_bytes(b'# CPU fixture\n');body={'files':{'safe.py':hashlib.sha256((root/'safe.py').read_bytes()).hexdigest()}};bid=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest();(root/'aot-manifest.json').write_text(json.dumps({**body,'bundle_id':bid}));OperatorRunner(root,bid)
                extra=root/name;extra.parent.mkdir(exist_ok=True);extra.write_bytes(b'# unlisted CPU-only fixture\n')
                with self.assertRaisesRegex(ValueError,'unlisted'):OperatorRunner(root,bid)
    def test_sdk_symlink_entry_rejected(self):
        import hashlib
        from kernel_agent.operators.runner import OperatorRunner
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'safe.py').write_bytes(b'# CPU fixture\n');body={'files':{'safe.py':hashlib.sha256((root/'safe.py').read_bytes()).hexdigest()}};bid=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest();(root/'aot-manifest.json').write_text(json.dumps({**body,'bundle_id':bid}));(root/'extra').symlink_to(root/'safe.py')
            with self.assertRaisesRegex(ValueError,'filesystem entry'):OperatorRunner(root,bid)
    def test_nonobject_manifest_rejected_before_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'aot-manifest.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'must be an object'):approved_runner(root)
    def test_unknown_channel_rejected_before_bundle_read(self):
        with self.assertRaisesRegex(ValueError,'unknown operator'):approved_runner('/missing','custom')
    def test_incomplete_certificate_rejected(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);b=json.dumps({'status':'QUALIFIED','production_release':True}).encode();(d/'qualification.json').write_bytes(b);(d/'qualified_catalog.json').write_text(json.dumps({'qualification_sha256':hashlib.sha256(b).hexdigest()}))
            with patch.object(q,'DATA',d),self.assertRaisesRegex(ValueError,'certificate invalid'):q.certificate()
    def test_device_or_image_mismatch_rejected(self):
        e={'image_id':q.IMAGE,'gpu_uuid':'GPU-test','driver_version':'591.74'}
        good=[{'exit':0,'stdout':q.IMAGE+'\n'},{'exit':0,'stdout':'GPU-test, 591.74\n'}]
        with patch.object(q,'command_result',side_effect=good):q.check_environment(e)
        for value in ['GPU-other, 591.74\n','GPU-test, 591.73\n','GPU-test, 591.74\nGPU-test, 591.74\n']:
            with patch.object(q,'command_result',side_effect=[good[0],{'exit':0,'stdout':value}]),self.assertRaisesRegex(ValueError,'binding differs'):q.check_environment(e)
    def test_default_qualified_scope_only(self):
        c={'operations':['attention','mla']};q.check_request(c,{'operation':'attention'});q.check_request(c,{'operation':'mla','page_size':16,'splits':1})
        for request in [{'operation':'softmax'},{'operation':'mla','page_size':1,'splits':1},{'operation':'mla','page_size':16,'splits':4}]:
            with self.assertRaises(ValueError):q.check_request(c,request)
    def test_contract_rejects_excessive_attention_before_array_allocation(self):
        req={'operation':'attention','bundle_id':'0'*64,'fault':None,'batch':16,'sq':8192,'sk':8192,'heads':128,'dim':128,'causal':True,'mask':False}
        with self.assertRaisesRegex(ValueError,'192MiB'):specifications(req)
    def test_mla_page_alias_and_nonfinite_rejected(self):
        req={'operation':'mla','bundle_id':'0'*64,'fault':None,'lengths':[17],'heads':16,'page_size':16,'splits':1,'num_pages':2};specs,_=specifications(req);arrays={n:np.zeros(shape,dtype=dtype) for n,(shape,dtype) in specs.items()};arrays['indptr.npy'][:]=[0,2]
        with self.assertRaisesRegex(ValueError,'nonpermutation'):validate_arrays(req,arrays)
        arrays['indices.npy'][:]=[0,1];arrays['q_abs.npy'][0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'finite-value'):validate_arrays(req,arrays)

if __name__=='__main__':unittest.main()
