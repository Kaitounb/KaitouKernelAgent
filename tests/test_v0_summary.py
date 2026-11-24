"""CPU tests of final report statistics; no model or GPU execution."""
import copy,math,unittest
from experiments.p1_05.summarize import summarize_group


def groups(ratios):
    output=[]
    for session,ratio in enumerate(ratios):
        segments=[]
        for pair in range(9):
            order=['incumbent','candidate'] if (pair+session)%2==0 else ['candidate','incumbent']
            for pos,label in enumerate(order):segments.append({'pair':pair,'order':pos,'label':label,'elapsed_ms':12 if label=='incumbent' else 12/ratio})
        output.append((session,{'segments':segments}))
    return output


class SummaryTests(unittest.TestCase):
    def test_same_identity_noise_control(self):
        d=summarize_group(groups([1,1,1]))
        self.assertEqual(d['geomean'],1);self.assertEqual(d['exploratory_session_interval'],[1,1])
        self.assertEqual(d['raw_segments'],54);self.assertEqual(d['baseline_median_us'],31.25)
    def test_geometric_equal_session_weight(self):
        d=summarize_group(groups([1,2,8]))
        self.assertAlmostEqual(d['geomean'],16**(1/3));self.assertLess(d['exploratory_session_interval'][0],1)
    def test_constant_gap(self):
        d=summarize_group(groups([.5,.5,.5]));self.assertEqual(d['geomean'],.5)
        self.assertEqual(d['candidate_median_us'],62.5)
    def test_long_segment_call_count(self):
        data=groups([1,1,1])
        for _,row in data:row['calls_per_segment']=32*48
        result=summarize_group(data)
        self.assertEqual(result['baseline_median_us'],7.8125)
        self.assertEqual(result['geomean'],1)
    def test_reject_missing_duplicate_sessions(self):
        d=groups([1,1,1]);d[1]=(0,d[1][1])
        with self.assertRaises(ValueError):summarize_group(d)
    def test_reject_nonfinite_and_missing_pair(self):
        for corrupt in ['nan','missing','order']:
            d=groups([1,1,1])
            if corrupt=='nan':d[0][1]['segments'][0]['elapsed_ms']=float('nan')
            elif corrupt=='missing':d[0][1]['segments'].pop()
            else:d[0][1]['segments'][0]['order']=1
            with self.subTest(corrupt=corrupt),self.assertRaises(ValueError):summarize_group(d)


if __name__=='__main__':unittest.main()
