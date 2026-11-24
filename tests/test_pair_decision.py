"""CPU simulations of the dual-binary gate/controller. No hardware timing or speedup claims."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock
from pair_fixtures import fixture
from kernel_agent.optimization.pair_decision import decide_pair
from kernel_agent.optimization.pair_control import collect_batches


class PairDecisionTests(unittest.TestCase):
    def setUp(self):self.aa,self.ab,self.ledgers,self.descriptor,self.contract,self.workloads=fixture()
    def judge(self):return decide_pair(self.aa,self.ab,self.ledgers,self.descriptor,evidence_kind='CPU_SIMULATION')
    def test_clear_gain_simulation_no_activation(self):
        r=self.judge();self.assertEqual(r['verdict'],'KEEP');self.assertEqual(r['evidence_kind'],'CPU_SIMULATION');self.assertFalse(r['runtime_activation']);self.assertFalse(r['production_release']);self.assertFalse(r['frozen_best_updated'])
    def test_no_gain_rejected(self):
        self.aa,self.ab,self.ledgers,self.descriptor,_,_=fixture((1,1,1));self.assertEqual(self.judge()['verdict'],'REJECT')
    def test_session_uncertainty_rejected(self):
        self.aa,self.ab,self.ledgers,self.descriptor,_,_=fixture((1.02,1.08,1.12));self.assertEqual(self.judge()['verdict'],'REJECT')
    def test_protection_regression_cannot_be_hidden_by_anchors(self):
        for r in self.ab:
            row=next(x for x in r['report']['measurements'] if x['workload']=='dev-02' and x['input_count']==16)
            for s in row['segments']:s['elapsed_ms']=10 if s['label']=='incumbent' else 10/.95
        self.assertEqual(self.judge()['verdict'],'REJECT')
    def test_role_specific_graph_output_binding(self):
        r=self.ab[0]['report']['measurements'][0];r['output_hashes']['candidate']=r['output_hashes']['incumbent'];self.assertEqual(self.judge()['verdict'],'REJECT')
    def test_diagnostic_backend_cannot_be_promoted(self):
        self.ab[0]['report']['backends']['candidate']='cublas';self.assertEqual(self.judge()['verdict'],'REJECT')
    def test_missing_aa_or_ab_rejected(self):
        self.aa.pop();self.assertEqual(self.judge()['verdict'],'REJECT');self.aa,self.ab,self.ledgers,self.descriptor,_,_=fixture();self.ab.pop();self.assertEqual(self.judge()['verdict'],'REJECT')
    def test_duplicate_session_and_wrong_order_rejected(self):
        self.ab[1]['report']['session']=0;self.assertEqual(self.judge()['verdict'],'REJECT');self.setUp();self.ab[0]['report']['measurements'][0]['capture_order'].reverse();self.assertEqual(self.judge()['verdict'],'REJECT')
    def test_artifact_protocol_worker_and_ledger_tampering(self):
        def binary():self.ab[0]['report']['incumbent_binary_sha256']='e'*64
        def protocol():self.descriptor['protocol_sha256']='e'*64
        def worker():self.ab[0]['record']['protected_hashes']['artifact/pair_worker.py']='e'*64
        def oracle():self.ledgers['incumbent'].pop(next(iter(self.ledgers['incumbent'])))
        def alias():self.ab[0]['report']['function_identity_distinct']=False
        def cleanup():self.ab[0]['record']['cleanup']['after']={'exit':1,'stderr':'daemon unreachable'}
        def nan():self.ab[0]['report']['measurements'][0]['segments'][0]['elapsed_ms']=float('nan')
        def post():self.ab[0]['report']['checks_after']['candidate'].pop()
        for mutation in [binary,protocol,worker,oracle,alias,cleanup,nan,post]:
            with self.subTest(case=mutation.__name__):self.setUp();mutation();self.assertEqual(self.judge()['verdict'],'REJECT')
    def test_unstable_aa_cannot_be_overridden_by_fast_ab(self):
        self.aa[0]['report']['candidate_aa'][0]['segments'][0]['elapsed_ms']=20;r=self.judge();self.assertEqual(r['reason'],'candidate_aa_unstable')
    def control(self,runner,idle):
        with tempfile.TemporaryDirectory() as root:return collect_batches(runner,Path(root),self.contract,self.workloads,self.ledgers,self.descriptor,idle,evidence_kind='CPU_SIMULATION')
    def test_busy_before_aa_does_not_start_gpu(self):
        runner=Mock();r=self.control(runner,lambda:False);runner.run.assert_not_called();self.assertEqual(r['aa_status'],'SKIPPED_GPU_BUSY')
    def test_busy_before_ab_preserves_aa_without_retry(self):
        runner=Mock();runner.run.side_effect=[(r['report'],r['record']) for r in self.aa];idle=Mock(side_effect=[True,False]);r=self.control(runner,idle);self.assertEqual(runner.run.call_count,3);self.assertEqual(r['ab_status'],'SKIPPED_GPU_BUSY')
    def test_aa_unstable_does_not_start_ab(self):
        self.aa[1]['report']['candidate_aa'][1]['segments'][0]['elapsed_ms']=20;runner=Mock();runner.run.side_effect=[(r['report'],r['record']) for r in self.aa];r=self.control(runner,lambda:True);self.assertEqual(runner.run.call_count,3);self.assertEqual(r['ab_status'],'NOT_RUN_AA_FAILED')
    def test_invalid_witness_or_cleanup_stops_immediately(self):
        runner=Mock();bad=copy.deepcopy(self.aa[0]);bad['record']['cleanup']['removed']=False;runner.run.return_value=(bad['report'],bad['record']);r=self.control(runner,lambda:True);self.assertEqual(runner.run.call_count,1);self.assertEqual(r['aa_status'],'INVALID_WITNESS')
    def test_stable_order_runs_exactly_one_batch_each(self):
        runner=Mock();runner.run.side_effect=[(r['report'],r['record']) for r in self.aa+self.ab];r=self.control(runner,lambda:True);self.assertEqual(r['verdict'],'KEEP');self.assertEqual([c.args[0]['mode'] for c in runner.run.call_args_list],['aa']*3+['ab']*3)
    def test_known_rejection_prevents_any_gpu_batch(self):
        from kernel_agent.optimization.identity import ROOT
        known=next(iter(json.loads((ROOT/'known-rejections.json').read_text()).values()))
        self.descriptor['candidate']['files']['kernels/gemm/wmma_sm86.cu']='2ec37a08a96b374eac5038af49e81c987ad0dd74f4822b264f5503f9a74773d2';self.descriptor['candidate']['id']=known['candidate_id'];self.descriptor['candidate_source_sha256']['wmma_sm86.cu']=self.descriptor['candidate']['files']['kernels/gemm/wmma_sm86.cu']
        runner=Mock();r=self.control(runner,lambda:True);self.assertEqual(r['reason'],'known_candidate_aa_unstable');runner.run.assert_not_called()
    def test_duplicate_source_prevents_any_gpu_batch(self):
        self.descriptor['candidate']=copy.deepcopy(self.descriptor['parent']);self.descriptor['candidate_source_sha256']={name:self.descriptor['parent']['files']['kernels/gemm/'+name] for name in self.descriptor['candidate_source_sha256']}
        runner=Mock();r=self.control(runner,lambda:True);self.assertEqual(r['reason'],'duplicate');runner.run.assert_not_called()
    def test_statistics_match_original_frozen_gate(self):
        import importlib.util
        from kernel_agent.optimization.identity import ROOT
        spec=importlib.util.spec_from_file_location('legacy_frozen_gate',ROOT/'evaluators/gemm_decision.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
        for speeds in [(1.06,1.06,1.06),(1,1,1),(1.02,1.08,1.12)]:
            self.aa,self.ab,self.ledgers,self.descriptor,_,_=fixture(speeds)
            parent=legacy.identity(4);candidate=legacy.identity(2);guard=legacy.protected_hashes();reports=[]
            for session in range(3):
                checks=[{**row,'input_immutable':True} for row in self.aa[session]['report']['checks_before'][:28]]
                measurements=copy.deepcopy(self.ab[session]['report']['measurements']);aa=copy.deepcopy(self.aa[session]['report']['candidate_aa'])
                for row in measurements+aa:row['cpu_reference_and_inputs_pass']=True
                reports.append({'status':'PASS','evidence_mode':'GPU','session':session,'parent':parent,'candidate':candidate,'harness_before':guard,'harness_after':guard,'protocol_sha256':legacy.sha(ROOT/'workloads/gemm_measurement_v1.json'),'environment':{'capability':[8,6]},'graph_launches':32,'replays':12,'binary':{'sha256':self.descriptor['binary_sha256']['incumbent']},'correctness_before_pass':True,'correctness_after_pass':True,'checks_before':checks,'checks_after':copy.deepcopy(checks),'measurements':measurements,'candidate_aa':aa})
            old=legacy.decide(reports,parent,candidate,guard);new=self.judge()
            self.assertEqual(new['policies'],old['policies'])
    def test_malformed_descriptor_does_not_change_actual_parent(self):
        expected=self.descriptor['parent']['id'];self.descriptor['parent']['id']='wrong';self.assertEqual(self.judge()['incumbent_after'],expected)
        r=decide_pair([],[],{},None,evidence_kind='CPU_SIMULATION');self.assertEqual(r['verdict'],'REJECT');self.assertEqual(r['incumbent_after'],expected)


if __name__=='__main__':unittest.main()
