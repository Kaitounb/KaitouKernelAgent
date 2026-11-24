"""Adversarial raw-evidence checks; CPU fixtures are never GPU acceptance."""
import copy,unittest
from kernel_agent.operators.measurement import assess


def reports(mode,ratio=1.):
    results=[]
    for session in range(3):
        rows=[]
        for w in ['a','b']:
            for n in [1,16]:
                segments=[]
                for pair in range(9):
                    order=['incumbent','candidate'] if (session+pair)%2==0 else ['candidate','incumbent']
                    segments.extend({'pair':pair,'label':role,'order':i,'elapsed_ms':10. if role=='incumbent' else 10./ratio} for i,role in enumerate(order))
                rows.append({'workload':w,'input_count':n,'segments':segments})
        results.append({'session':session,'status':'PASS','mode':mode,'rows':rows})
    return results


class MeasurementTests(unittest.TestCase):
    def gate(self,aa=None,ab=None):return assess(reports('aa') if aa is None else aa,reports('ab',1.1) if ab is None else ab,[{'id':'a'},{'id':'b'}],['a'])

    def test_valid_mock_and_guard_regression(self):
        self.assertEqual(self.gate()['verdict'],'KEEP')
        ab=reports('ab',1.1)
        for r in ab:
            for row in r['rows']:
                if row['workload']=='b':
                    for x in row['segments']:
                        if x['label']=='candidate':x['elapsed_ms']=11.
        self.assertEqual(self.gate(ab=ab)['reason'],'insufficient_gain_or_regression')

    def test_duplicate_missing_wrong_session_and_mode(self):
        for mutate in [lambda a:a[0]['rows'].append(copy.deepcopy(a[0]['rows'][0])),lambda a:a[0]['rows'].pop(),lambda a:a[0].update(session=1),lambda a:a[0].update(mode='ab')]:
            aa=reports('aa');mutate(aa);self.assertEqual(self.gate(aa=aa)['reason'],'invalid_evidence')

    def test_nonfinite_reordered_or_truncated_segments(self):
        for mutate in [lambda r:r['segments'][0].update(elapsed_ms=float('nan')),lambda r:r['segments'][0].update(order=1),lambda r:r['segments'].pop()]:
            aa=reports('aa');mutate(aa[0]['rows'][0]);self.assertEqual(self.gate(aa=aa)['reason'],'invalid_evidence')

    def test_unstable_or_missing_ab_never_kept(self):
        self.assertEqual(self.gate(aa=reports('aa',1.021))['reason'],'candidate_aa_unstable');self.assertEqual(self.gate(ab=[])['reason'],'AA_PASS_AB_NOT_RUN')


if __name__=='__main__':unittest.main()
