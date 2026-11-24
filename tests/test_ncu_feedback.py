"""CPU schema/binding tests; no model/network/GPU access."""
import copy,json,unittest
from experiments.p1_04.propose import validate_feedback


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.parent={'id':'local-parent'};self.facts={'facts':[{'id':'F1','metric':'sm__cycles_active.avg'}]}
        self.value={'proposal':{'schema_version':1,'parent_id':'current','warps':2,'reason':'bounded test'},
                    'fact_ids':['F1'],'hypothesis':'latency may limit issue','competing_explanations':['insufficient grid work'],
                    'expected_changes':[{'metric':'sm__cycles_active.avg','direction':'decrease'}]}
    def test_local_binding(self):
        d=validate_feedback(json.dumps(self.value),self.parent,self.facts)
        self.assertEqual(d['proposal']['parent_id'],'local-parent')
    def test_reject_uncited_or_unknown_metrics(self):
        for field,value in [('fact_ids',['invented']),('expected_changes',[{'metric':'invented','direction':'decrease'}]),('expected_changes',[{'metric':'sm__cycles_active.avg','direction':'promote'}])]:
            d=copy.deepcopy(self.value);d[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):validate_feedback(json.dumps(d),self.parent,self.facts)
    def test_reject_source_or_rule_changes(self):
        for field,value in [('source','arbitrary'),('atol',1),('warps',True)]:
            d=copy.deepcopy(self.value);d['proposal'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_feedback(json.dumps(d),self.parent,self.facts)


if __name__=='__main__':unittest.main()
