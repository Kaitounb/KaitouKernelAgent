"""CPU fixtures for proposal/evidence gates; these are not GPU or model runs."""
import copy
import json
import unittest

from evaluators.gemm_decision import ROOT,BOUNDARIES,identity,validate_proposal,decide,protected_hashes,sha


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.parent=identity(4);self.candidate=identity(2);self.guard=protected_hashes()
        self.dev=json.loads((ROOT/'workloads/gemm_development_v1.json').read_text())['workloads']

    def proposal(self,**extra):
        return dict(schema_version=1,parent_id=self.parent['id'],warps=2,reason='test',**extra)

    def reports(self,speedups=(1.06,1.06,1.06)):
        checks=[{'shape_mnk':w['shape_mnk'],'seed':seed,'distribution':'random','offset_guard':False,'input_immutable':True} for w in self.dev for seed in (171,172)]
        checks += [{'shape_mnk':list(shape),'seed':seed,'distribution':dist,'offset_guard':offset,'input_immutable':True} for shape,dist,offset in BOUNDARIES for seed in (171,172)]
        reports=[]
        for session,speedup in enumerate(speedups):
            rows=[];aa=[]
            for w in self.dev:
                for count in (1,16):
                    segments=[]
                    for pair in range(9):
                        labels=['incumbent','candidate'] if (pair+session)%2==0 else ['candidate','incumbent']
                        segments.extend({'pair':pair,'order':pos,'label':label,'elapsed_ms':10 if label=='incumbent' else 10/speedup} for pos,label in enumerate(labels))
                    case={'workload':w['id'],'shape_mnk':w['shape_mnk'],'input_count':count,'capture_order':['incumbent','candidate'] if session%2==0 else ['candidate','incumbent'],'segments':segments,'cpu_reference_and_inputs_pass':True}
                    rows.append(case);negative=copy.deepcopy(case)
                    for r in negative['segments']:r['elapsed_ms']=10
                    aa.append(negative)
            reports.append({'status':'PASS','evidence_mode':'GPU','session':session,'parent':self.parent,'candidate':self.candidate,
                            'harness_before':self.guard,'harness_after':self.guard,'protocol_sha256':sha(ROOT/'workloads/gemm_measurement_v1.json'),
                            'environment':{'capability':[8,6]},'graph_launches':32,'replays':12,
                            'binary':{'sha256':json.loads((ROOT/'workloads/gemm_measurement_v1.json').read_text())['initial_template']['binary_sha256']},
                            'correctness_before_pass':True,'correctness_after_pass':True,'checks_before':copy.deepcopy(checks),'checks_after':copy.deepcopy(checks),
                            'measurements':rows,'candidate_aa':aa})
        return reports

    def judge(self,reports):
        return decide(reports,self.parent,self.candidate,self.guard)

    def test_valid_proposal(self):
        self.assertEqual(validate_proposal(json.dumps(self.proposal()),self.parent['id'])['warps'],2)

    def test_illegal_proposals(self):
        for value in [True,2.0,'2',0,16,'__import__("os")']:
            d=self.proposal();d['warps']=value
            with self.subTest(value=value),self.assertRaises(ValueError):validate_proposal(json.dumps(d),self.parent['id'])
        for field in ['source','atol','command','workloads']:
            with self.subTest(field=field),self.assertRaises(ValueError):validate_proposal(json.dumps(self.proposal(**{field:'forbidden'})),self.parent['id'])

    def test_stale_parent(self):
        with self.assertRaises(ValueError):validate_proposal(json.dumps(self.proposal()),'wrong')

    def test_duplicate_and_nan_json(self):
        for text in ['{"warps":1,"warps":2}','{"warps":NaN}']:
            with self.assertRaises(ValueError):validate_proposal(text,self.parent['id'])

    def test_clear_gain_simulation(self):
        self.assertEqual(self.judge(self.reports())['verdict'],'KEEP')

    def test_no_gain_simulation(self):
        d=self.judge(self.reports((1,1,1)));self.assertEqual(d['verdict'],'REJECT');self.assertEqual(d['incumbent_after'],self.parent['id'])

    def test_uncertain_gain(self):
        self.assertEqual(self.judge(self.reports((1.02,1.08,1.12)))['verdict'],'REJECT')

    def test_guard_regression(self):
        data=self.reports()
        for report in data:
            row=next(r for r in report['measurements'] if r['workload']=='dev-02' and r['input_count']==16)
            for r in row['segments']:r['elapsed_ms']=10 if r['label']=='incumbent' else 10/.95
        self.assertEqual(self.judge(data)['verdict'],'REJECT')

    def test_missing_and_tampered_evidence(self):
        def failed(d):d[0]['status']='FAIL'
        def drift(d):d[0]['harness_after']={}
        def post(d):d[0]['correctness_after_pass']=False
        def missing(d):d[0]['measurements'].pop()
        def duplicate(d):d[0]['measurements'][0]=d[0]['measurements'][1]
        def parent(d):d[0]['parent']=self.candidate
        def binary(d):d[0]['binary']['sha256']='untrusted'
        def coverage(d):d[0]['checks_after'][-1]=d[0]['checks_after'][0]
        def shape(d):d[0]['measurements'][0]['shape_mnk']=[1,1,1]
        def nan(d):d[0]['measurements'][0]['segments'][0]['elapsed_ms']=float('nan')
        def order(d):d[0]['measurements'][0]['capture_order'].reverse()
        for modify in [failed,drift,post,missing,duplicate,parent,binary,coverage,shape,nan,order]:
            with self.subTest(mutation=modify.__name__):
                d=self.reports();modify(d);verdict=self.judge(d)
                self.assertEqual(verdict['verdict'],'REJECT');self.assertEqual(verdict['incumbent_after'],self.parent['id'])

    def test_aa_noise_rejected(self):
        d=self.reports();d[0]['candidate_aa'][0]['segments'][0]['elapsed_ms']=20
        self.assertEqual(self.judge(d)['verdict'],'REJECT')


if __name__=='__main__':unittest.main()
