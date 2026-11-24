"""CPU synthetic CSV tests of the actual pinned KernelAgent parser plus CUDA sharding."""
import csv
from pathlib import Path
import tempfile
import unittest
from experiments.p1_04.ncu_adapter import parse_profile,select_metrics


class ParserTests(unittest.TestCase):
    metrics=['sm__cycles_active.avg','launch__registers_per_thread','sm__warps_active.avg.pct_of_peak_sustained_active','missing__metric']
    def make_csv(self,path,duplicate=False,wrong=False):
        with path.open('w',newline='') as f:
            f.write('==WARNING== synthetic warning before header\n')
            w=csv.writer(f);w.writerow(['ID','Kernel Name',*self.metrics[:-1]])
            w.writerow(['','','cycle','register/thread','%'])
            for i in range(3):w.writerow([0 if duplicate else i,'void gemm_wmma<8>()' if wrong else 'void gemm_wmma<4>()','1,234.5' if i==0 else 'nan' if i==1 else 20,32,50])

    def test_all_launches_units_and_unavailable(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'data.csv';self.make_csv(path);d=parse_profile(path,self.metrics,4)
        self.assertEqual([r['id'] for r in d['launches']],[0,1,2])
        self.assertEqual(d['launches'][0]['metrics'][self.metrics[0]]['value'],1234.5)
        self.assertIsNone(d['launches'][1]['metrics'][self.metrics[0]]['value'])
        self.assertEqual(d['launches'][0]['metrics'][self.metrics[1]]['unit'],'register/thread')
        self.assertTrue(all(r['metrics']['missing__metric']['status']=='unavailable' and r['metrics']['missing__metric']['value'] is None for r in d['launches']))
        self.assertEqual(len(next(iter(d['upstream_prompt_metrics'].values()))),3)

    def test_reject_wrong_kernel_and_duplicate_ids(self):
        for duplicate,wrong in [(True,False),(False,True)]:
            with self.subTest(duplicate=duplicate,wrong=wrong),tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'data.csv';self.make_csv(path,duplicate,wrong)
                with self.assertRaises(ValueError):parse_profile(path,self.metrics,4)

    def test_missing_header(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'data.csv';path.write_text('bad header\n')
            with self.assertRaises(ValueError):parse_profile(path,self.metrics,4)

    def test_query_missing_metric_never_becomes_zero(self):
        selected,missing=select_metrics('sm__cycles_active.avg')
        self.assertIn('sm__cycles_active.avg',selected)
        self.assertNotIn('dram__bytes_read.sum',selected)
        self.assertIn('dram__bytes_read.sum',missing)
        self.assertIn('launch__registers_per_thread',selected)

    def test_mla_kernel_identity_is_explicit_and_gemm_default_stays_strict(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'data.csv';self.make_csv(path);path.write_text(path.read_text().replace('void gemm_wmma<4>()','void BatchMLAPagedAttentionKernel<fixed>()'))
            with self.assertRaises(ValueError):parse_profile(path,self.metrics,4)
            result=parse_profile(path,self.metrics,None,expected_kernel='BatchMLAPagedAttentionKernel')
            self.assertEqual(len(result['launches']),3)
            self.assertTrue(all('BatchMLAPagedAttentionKernel' in row['kernel'] for row in result['launches']))


if __name__=='__main__':unittest.main()
