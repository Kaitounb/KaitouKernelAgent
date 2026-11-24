"""CPU negative checks for the new physical-environment receipt gate."""
import unittest
from experiments.full_delivery.cublas_locked_qualification_host import validate_clock_evidence

class ClockQualificationTest(unittest.TestCase):
    def fixture(self,sm=1200,memory=6800,temp=36,count=40):
        samples=[{'wall':10+i,'exit':0,'stdout':f'timestamp, 95, {sm}, {memory}, {temp}, 57'} for i in range(count)]
        return samples,[{'phase':'aa','started_wall':0,'ended_wall':100}]
    def test_expected_loaded_clocks(self):
        samples,records=self.fixture();self.assertEqual(validate_clock_evidence(samples,records)['loaded_samples'],40)
    def test_wrong_sm_rejected(self):
        with self.assertRaises(ValueError):validate_clock_evidence(*self.fixture(sm=1500))
    def test_wrong_memory_or_hot_rejected(self):
        for kwargs in [{'memory':7000},{'temp':80}]:
            with self.subTest(**kwargs),self.assertRaises(ValueError):validate_clock_evidence(*self.fixture(**kwargs))
    def test_missing_evidence_rejected(self):
        with self.assertRaises(ValueError):validate_clock_evidence(*self.fixture(count=29))
if __name__=='__main__':unittest.main()
