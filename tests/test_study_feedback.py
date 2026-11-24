"""Critical feedback-boundary/regression checks, entirely CPU fixtures."""
import copy,hashlib,json,unittest
from kernel_agent.optimization.study_feedback import prepare


class StudyFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.task={'source':'fixed CUDA','source_sha256':hashlib.sha256(b'fixed CUDA').hexdigest(),'contract':'FP16 core','gpu':'SM86','allowed_changes':'fixed CUDA diff','model':'deepseek-flash','temperature':.2,'max_tokens':768};self.execution={'candidate_id':'a'*64,'compile_status':'PASS','compile_error':None,'correctness':'PASS','latency_ms':[.01,.011],'evaluation_wall_seconds':1.};self.outcome={'step':0,'candidate_id':'a'*64,'verdict':'REJECT','reason':'aa_unstable','source_change':{'source_sha256':'b'*64,'patch':'own fixed-file diff'},'evaluation_scope':'exact_cache'};self.profile={'facts':[{'id':'F1','metric':'launch__registers_per_thread','value':40.,'unit':'register/thread','status':'observed'}],'provenance':[{'csv_sha256':'c'*64,'binary_sha256':'d'*64,'upstream_profiler_sha256':'e'*64}],'profile_wall_seconds':4.}

    def test_b_uses_safe_verdict_baseline_and_patch_without_loading_profile(self):
        def poison():raise AssertionError('B profile read')
        result=prepare('B','gemm',self.task,self.execution,[self.execution],[self.outcome],profile_loader=poison);feedback=result['payload']['feedback'];self.assertEqual(feedback['outcomes'][0]['reason'],'aa_unstable');self.assertEqual(feedback['outcomes'][0]['source_change']['patch'],'own fixed-file diff');self.assertEqual(feedback['baseline_execution']['latency_ms'],[.01,.011]);self.assertNotIn('profile',feedback);self.assertEqual(result['profile_loader_calls'],0)

    def test_bc_same_objective_history_and_only_c_profile(self):
        b=prepare('B','gemm',self.task,self.execution,[self.execution],[self.outcome]);c=prepare('C','gemm',self.task,self.execution,[self.execution],[self.outcome],profile_loader=lambda:self.profile);self.assertEqual(b['common_sha256'],c['common_sha256']);self.assertEqual(b['payload']['objective'],c['payload']['objective']);self.assertEqual(b['payload']['feedback']['outcomes'],c['payload']['feedback']['outcomes']);self.assertEqual(c['profile_wall_seconds'],4.)

    def test_resource_log_and_cross_candidate_history_rejected(self):
        for change in [lambda x:x.update(raw_ptxas_stdout='register sentinel'),lambda x:x.update(candidate_id='f'*64),lambda x:x.update(step=2)]:
            item=copy.deepcopy(self.outcome);change(item)
            with self.subTest(item=item),self.assertRaises(ValueError):prepare('B','gemm',self.task,self.execution,[self.execution],[item])

    def test_invalid_sdk_values_never_enter_history(self):
        item=copy.deepcopy(self.outcome);item['source_change']={'page_size':16,'splits':2};prepare('B','mla',self.task,self.execution,[self.execution],[item]);item['source_change']['splits']=True
        with self.assertRaises(ValueError):prepare('B','mla',self.task,self.execution,[self.execution],[item])

    def test_context_loader_is_c_only_and_binds_every_fact(self):
        def poison():raise AssertionError('B context read')
        prepare('B','gemm',self.task,self.execution,[],[],profile_context_loader=poison)
        context=[{'kind':'parent','source_sha256':self.task['source_sha256'],'binary_sha256':'d'*64,'fact_ids':['F1'],'kernel':'gemm_wmma<4>','scope':'native kernel only'}]
        prepare('C','gemm',self.task,self.execution,[],[],profile_loader=lambda:self.profile,profile_context_loader=lambda:context)
        context[0]['fact_ids']=['F2']
        with self.assertRaises(ValueError):prepare('C','gemm',self.task,self.execution,[],[],profile_loader=lambda:self.profile,profile_context_loader=lambda:context)


if __name__=='__main__':unittest.main()
