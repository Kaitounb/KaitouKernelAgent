"""CPU-only installed orchestration policy. No SDK/network/compiler/GPU calls."""
import ast,difflib,fcntl,json,os,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from kernel_agent.optimization.command import execute,status
from kernel_agent.optimization.resources import ASSETS,verify_bundle

REAL=Path('evidence/20261122-p3-01-deepseek-source/response.json')


def valid_response(path):
    target='kernels/gemm/wmma_sm86.cu';base=(ASSETS/target).read_text();changed=base.replace('    for (int i = 0;', '    // CPU fixture: a reviewed source identity, never compiled here.\n    for (int i = 0;',1)
    diff='diff --git a/'+target+' b/'+target+'\n'+''.join(difflib.unified_diff(base.splitlines(True),changed.splitlines(True),fromfile='a/'+target,tofile='b/'+target))
    body={'model':'deepseek-flash','choices':[{'finish_reason':'stop','message':{'content':json.dumps({'schema_version':1,'parent_id':'current','patch':diff,'reason':'CPU-only policy fixture'})}}],'usage':{'total_tokens':1}}
    path.write_text(json.dumps(body));return path


class OptimizerTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.response=valid_response(self.root/'response.json');self.run=self.root/'run'
    def tearDown(self):self.temp.cleanup()
    def pending(self,**kw):return execute(self.run,response=self.response,checks_only=True,**kw)
    def test_historical_rejection_never_resamples(self):
        callback=Mock(side_effect=AssertionError('GPU must not run'))
        state=execute(self.run,response=REAL,recount=True,evaluator=callback)
        self.assertEqual(state['status'],'REJECT');self.assertTrue(state['known_decision_reused']);callback.assert_not_called()
        self.assertEqual(state['calls_reserved'],0);self.assertEqual(state['new_external_model_calls'],0);self.assertTrue((self.run/'patch-validation.json').is_file())
    def test_strict_patch_failure_is_preserved(self):
        state=execute(self.run,response=REAL);self.assertEqual(state['status'],'REJECT');self.assertIn('invalid_proposal',state['reason']);self.assertEqual((self.run/'response.json').read_bytes(),REAL.read_bytes())
    def test_bad_json_is_rejected_not_executed(self):
        self.response.write_text('{');state=execute(self.run,response=self.response);self.assertEqual(state['status'],'REJECT');self.assertEqual(state['new_external_model_calls'],0)
    def test_empty_choices_rejected_with_a_record(self):
        self.response.write_text(json.dumps({'model':'deepseek-flash','choices':[]}))
        state=execute(self.run,response=self.response);self.assertEqual(state['status'],'REJECT')
    def test_unconfirmed_live_outcome_not_reported_as_completed(self):
        with patch.dict(os.environ,{'DEEPSEEK_API_KEY':'CPU-secret'}),patch('kernel_agent.optimization.propose.call_once',side_effect=RuntimeError('CPU fake request failure')):
            state=execute(self.run,live=True,max_calls=1)
        self.assertEqual(state['status'],'FAILED');self.assertEqual(state['calls_reserved'],1);self.assertIsNone(state['new_external_model_calls']);self.assertEqual(state['external_call_outcome'],'UNCONFIRMED')
    def test_live_budget_must_be_explicit(self):
        with patch.dict(os.environ,{'DEEPSEEK_API_KEY':'CPU-secret'}),self.assertRaises(ValueError):execute(self.run,live=True,max_calls=0)
        self.assertFalse((self.run/'run.json').exists())
    def test_review_resume_once_and_terminal_idempotency(self):
        before=self.pending();self.assertEqual(before['status'],'NEEDS_SOURCE_REVIEW')
        callback=Mock(return_value={'verdict':'CORRECTNESS_ONLY','reason':'NOT_RUN_CHECKS_ONLY','qa_checks':0,'aa_status':'NOT_RUN_CHECKS_ONLY'})
        state=execute(self.run,resume=True,checks_only=True,review_sha=before['source_sha256'],deps=self.root,evaluator=callback)
        self.assertEqual(state['status'],'CHECKS_COMPLETE');self.assertFalse(state['release_activated']);self.assertEqual(state['evidence_kind'],'CPU_MOCK')
        state2=execute(self.run,response=self.response,checks_only=True,evaluator=callback);self.assertEqual(state2,state);self.assertEqual(callback.call_count,1)
        self.assertTrue((self.run/'review.json').is_file());self.assertFalse(state['production_release'])
    def test_screening_keep_is_only_a_recommendation(self):
        before=execute(self.run,response=self.response)
        callback=Mock(return_value={'verdict':'KEEP','reason':'CPU simulated screening pass','qa_checks':0,'aa_status':'PASS','decision':{'verdict':'KEEP','runtime_activation':False}})
        state=execute(self.run,resume=True,review_sha=before['source_sha256'],deps=self.root,evaluator=callback)
        self.assertEqual(state['status'],'SCREENING_PASS');self.assertFalse(state['release_activated']);self.assertEqual(state['runtime_default'],'cublas');self.assertFalse(state['production_release']);self.assertTrue((self.run/'decision.json').is_file())
    def test_checks_only_cannot_keep(self):
        before=self.pending();callback=Mock(return_value={'verdict':'KEEP','reason':'invalid CPU fixture'})
        state=execute(self.run,resume=True,checks_only=True,review_sha=before['source_sha256'],deps=self.root,evaluator=callback)
        self.assertEqual(state['status'],'FAILED');self.assertFalse(state['release_activated'])
    def test_source_change_after_review_stops(self):
        before=self.pending();(self.run/'source/wmma_sm86.cu').write_text('tampered');callback=Mock()
        state=execute(self.run,resume=True,checks_only=True,review_sha=before['source_sha256'],deps=self.root,evaluator=callback)
        self.assertEqual(state['status'],'FAILED');callback.assert_not_called()
    def test_concurrent_lock_prevents_second_attempt(self):
        self.run.mkdir()
        with (self.run/'.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(RuntimeError):execute(self.run,response=self.response)
        self.assertFalse((self.run/'run.json').exists())
    def test_interrupted_model_reservation_never_retries(self):
        with patch.dict(os.environ,{'DEEPSEEK_API_KEY':'CPU-secret'}),patch('kernel_agent.optimization.propose.call_once',side_effect=KeyboardInterrupt) as call:
            with self.assertRaises(KeyboardInterrupt):execute(self.run,live=True,max_calls=1,provider_factory=object())
            self.assertEqual(status(self.run)['status'],'MODEL_RESERVED')
            state=execute(self.run,resume=True);self.assertEqual(state['status'],'STOPPED_UNCERTAIN_CALL');self.assertEqual(state['calls_reserved'],1);self.assertEqual(call.call_count,1)
    def test_interrupted_gpu_stage_never_restarts(self):
        before=self.pending();callback=Mock(side_effect=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt):execute(self.run,resume=True,checks_only=True,review_sha=before['source_sha256'],deps=self.root,evaluator=callback)
        self.assertEqual(status(self.run)['status'],'EVALUATING')
        state=execute(self.run,resume=True,checks_only=True,evaluator=callback);self.assertEqual(state['status'],'STOPPED_NEEDS_CLEANUP');self.assertEqual(callback.call_count,1)
    def test_failure_redacts_secret(self):
        before=self.pending();callback=Mock(side_effect=RuntimeError('failed CPU-secret'))
        with patch.dict(os.environ,{'DEEPSEEK_API_KEY':'CPU-secret'}):state=execute(self.run,resume=True,checks_only=True,review_sha=before['source_sha256'],deps=self.root,evaluator=callback)
        self.assertIn('[REDACTED]',state['reason']);self.assertNotIn('CPU-secret',(self.run/'run.json').read_text())
    def test_bundle_tampering_rejected(self):
        from kernel_agent.optimization import resources
        copied=self.root/'package';shutil.copytree(resources.PACKAGE,copied,ignore=shutil.ignore_patterns('__pycache__'))
        (copied/'assets/kernels/gemm/wmma_sm86.cu').write_text('changed')
        with patch.object(resources,'PACKAGE',copied),self.assertRaises(ValueError):verify_bundle()
    def test_reused_numerical_and_timed_functions_are_unchanged(self):
        def functions(path):return {n.name:ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
        package=ASSETS.parent
        for name in ['fixtures','source_timing','evaluate','patches']:
            self.assertEqual(functions(package/(name+'.py')),functions(Path('experiments/p3_01')/(name+'.py')),name)


if __name__=='__main__':unittest.main()
