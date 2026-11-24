"""CPU checks of B/C isolation and prompt-binding failures, not agent outcome tests."""
import copy,hashlib,json,unittest
from kernel_agent.optimization.feedback_channels import prepare_channel


class ChannelTests(unittest.TestCase):
    def setUp(self):
        self.task={'source':'fixed CUDA source','source_sha256':hashlib.sha256(b'fixed CUDA source').hexdigest(),'contract':'public FP16','gpu':'SM86','allowed_changes':'same bounded CUDA body','model':'deepseek-flash','temperature':.2,'max_tokens':768}
        self.execution={'candidate_id':'a'*64,'compile_status':'PASS','compile_error':None,'correctness':'PASS','latency_ms':[.01,.02],'evaluation_wall_seconds':3.}
        self.profile={'facts':[{'id':'F1','metric':'sm__cycles_active.avg','value':17.,'unit':'cycle','status':'observed'}],'provenance':[{'csv_sha256':'b'*64,'binary_sha256':'c'*64,'upstream_profiler_sha256':'d'*64}],'profile_wall_seconds':4.}

    def test_ab_do_not_read_poison_loader(self):
        def poison():raise AssertionError('A/B read private profile')
        a=prepare_channel('A',self.task,[self.execution],profile_loader=poison);b=prepare_channel('B',self.task,[self.execution],profile_loader=poison)
        self.assertEqual(a['payload'],b['payload']);self.assertEqual(b['profile_loader_calls'],0);self.assertNotIn('sm__',json.dumps(b))

    def test_bc_only_profile_differs_and_cost_is_counted(self):
        b=prepare_channel('B',self.task,[self.execution]);c=prepare_channel('C',self.task,[self.execution],profile_loader=lambda:self.profile)
        self.assertEqual(b['common_sha256'],c['common_sha256']);self.assertEqual(b['payload']['feedback']['executions'],c['payload']['feedback']['executions']);self.assertEqual(c['profile_wall_seconds'],4.);self.assertEqual(c['profile_loader_calls'],1)

    def test_reject_nested_resource_log_leaks(self):
        for key in ['ncu','metrics','stdout','stderr','profile_path']:
            value=copy.deepcopy(self.execution);value[key]='hardware sentinel'
            with self.subTest(key=key),self.assertRaises(ValueError):prepare_channel('B',self.task,[value])

    def test_missing_zero_and_nonfinite_remain_distinct(self):
        value=copy.deepcopy(self.profile);value['facts'][0].update(status='unavailable',value=0)
        with self.assertRaises(ValueError):prepare_channel('C',self.task,[],profile_loader=lambda:value)
        value['facts'][0]['value']=None;prepare_channel('C',self.task,[],profile_loader=lambda:value)
        value['facts'][0].update(status='observed',value=float('nan'))
        with self.assertRaises(ValueError):prepare_channel('C',self.task,[],profile_loader=lambda:value)

    def test_source_and_provenance_binding(self):
        value=copy.deepcopy(self.task);value['source']+=' change'
        with self.assertRaises(ValueError):prepare_channel('B',value,[])
        value=copy.deepcopy(self.profile);value['provenance'][0]['csv_sha256']='invented'
        with self.assertRaises(ValueError):prepare_channel('C',self.task,[],profile_loader=lambda:value)

    def test_legacy_profile_cost_unknown_is_not_filled_with_zero(self):
        value=copy.deepcopy(self.profile);value['profile_wall_seconds']=None
        result=prepare_channel('C',self.task,[],profile_loader=lambda:value)
        self.assertIsNone(result['profile_wall_seconds'])


if __name__=='__main__':unittest.main()
