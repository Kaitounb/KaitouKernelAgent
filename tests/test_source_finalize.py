"""CPU-only orchestration regressions. Load the actual function without Torch/GPU imports."""
import ast,json,tempfile,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class SourceFinalizeTests(unittest.TestCase):
    def test_missing_or_failed_sanitizer_finalizes_without_timing_or_promotion(self):
        tree=ast.parse(Path('experiments/p3_01/source_pipeline.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='finish')
        for probe_exit in [0,1]:
            with self.subTest(probe_exit=probe_exit),tempfile.TemporaryDirectory() as temp:
                calls=[]
                def command(cmd,**kw):
                    calls.append(cmd);return SimpleNamespace(returncode=probe_exit if cmd[0]=='docker' else 1,stdout='',stderr='CPU fixture')
                ns={'json':json,'threading':threading,'time':time,'subprocess':SimpleNamespace(run=command),'IMAGE':'CPU-ONLY','expected_before':lambda cases:{},'print':lambda *args:None}
                exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual source finish>','exec'),ns)
                runner=SimpleNamespace(run=Mock(return_value=(None,{'exit':1,'rejection':'CPU fixture'})))
                root=Path(temp);ns['finish'](root,{'parent_id':'old'},{'status':'BUILT_UNVALIDATED','binary_sha256':'binary','source_identity':'source'},runner,[],[],[],[],{},[],[],[],lambda:None)
                result=json.loads((root/'result.json').read_text())
                self.assertEqual(result['source_identity'],'source')
                self.assertEqual(result['verdict'],'REJECT');self.assertEqual(result['aa_status'],'NOT_RUN_SANITIZER_NOT_ACCEPTED');self.assertEqual(result['incumbent_after'],'old')
                self.assertFalse(result['production_release']);self.assertEqual(result['model_calls_this_gpu_run'],0)
                self.assertEqual(runner.run.call_count,2 if probe_exit==0 else 0)
                self.assertTrue(all(call.kwargs.get('sanitizer') in ['memcheck','synccheck'] for call in runner.run.call_args_list))


if __name__=='__main__':unittest.main()
