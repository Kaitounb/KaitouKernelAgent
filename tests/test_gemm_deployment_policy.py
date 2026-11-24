"""CPU policy fixtures only, no qualification certificate or actual GPU acceptance."""
import copy,json,unittest
from pathlib import Path
from experiments.full_delivery.gemm_deployment_gate import assess_deployment
from kernel_agent.operators.measurement import assess

class GemmDeploymentPolicyTests(unittest.TestCase):
 def setUp(self):
  self.policy=json.loads((Path(__file__).resolve().parents[1]/'workloads/gemm_deployment_stability_v2.json').read_text());self.workloads=[{'id':'w'+str(i)} for i in range(8)]
 def reports(self,mode):
  reports=[]
  for session in range(3):
   rows=[]
   for w in self.workloads:
    for count in [1,16]:
     segments=[]
     for pair in range(9):
      order=['incumbent','candidate'] if (pair+session)%2==0 else ['candidate','incumbent']
      for pos,label in enumerate(order):segments.append({'pair':pair,'label':label,'order':pos,'elapsed_ms':1000.})
     rows.append({'workload':w['id'],'input_count':count,'segments':segments})
   reports.append({'status':'PASS','mode':mode,'session':session,'rows':rows})
  return reports
 def test_three_aa_required(self):self.assertEqual(assess_deployment(self.reports('aa')[:2],[],self.workloads,self.policy)['aa_status'],'INCOMPLETE')
 def test_deployment_3percent_does_not_change_research_2percent(self):
  aa=self.reports('aa');next(x for x in aa[0]['rows'][0]['segments'] if x['pair']==0 and x['label']=='incumbent')['elapsed_ms']=1029.
  self.assertEqual(assess_deployment(aa,self.reports('ab'),self.workloads,self.policy)['verdict'],'PASS');self.assertEqual(assess(aa,[],self.workloads,['w0'])['reason'],'candidate_aa_unstable')
 def test_pair_3percent_exclusive(self):
  aa=self.reports('aa');next(x for x in aa[0]['rows'][0]['segments'] if x['pair']==0 and x['label']=='incumbent')['elapsed_ms']=1030.
  self.assertEqual(assess_deployment(aa,[],self.workloads,self.policy)['aa_status'],'REJECT')
 def test_group_1point5percent_retained(self):
  aa=self.reports('aa')
  for x in aa[0]['rows'][0]['segments']:
   if x['label']=='incumbent':x['elapsed_ms']=1016.
  self.assertEqual(assess_deployment(aa,[],self.workloads,self.policy)['aa_status'],'REJECT')
 def test_nonregression_retained(self):
  ab=self.reports('ab')
  for report in ab:
   for x in report['rows'][0]['segments']:
    if x['label']=='incumbent':x['elapsed_ms']=969.
  self.assertEqual(assess_deployment(self.reports('aa'),ab,self.workloads,self.policy)['verdict'],'REJECT')

if __name__=='__main__':unittest.main()
