"""CPU policy tests using mocked workers. They do not simulate a real Docker/GPU outage."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('numpy'),'NumPy unavailable; run in the declared target environment')
class DeploymentPolicyTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from kernel_agent.registry import Registry
        self.np=np;self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'store';Registry(self.root).initialize();self.inputs=np.ones((2,2),np.float16)
        self.before=(self.root/'current.json').read_bytes()
    def tearDown(self):self.temp.cleanup()
    def test_unconfirmed_cleanup_stops_gpu_attempts(self):
        from kernel_agent.deployment import execute
        with patch('kernel_agent.deployment.DockerRunner') as worker:
            worker.return_value.run.return_value=(None,{'status':'REJECT','cleanup':{'removed':False},'rejection':'unknown cleanup'})
            with self.assertRaises(RuntimeError):execute(self.root,self.inputs,self.inputs)
            self.assertEqual(worker.call_count,1)
        result=json.loads(next((self.root/'runs').glob('*/execution.json')).read_text());self.assertEqual(result['fallback_reason'],'cleanup_unconfirmed');self.assertEqual(len(result['attempts']),1)
        self.assertEqual((self.root/'current.json').read_bytes(),self.before)
    def test_clean_worker_failure_uses_one_fallback(self):
        from kernel_agent.deployment import execute
        with patch('kernel_agent.deployment.DockerRunner') as worker:
            worker.return_value.run.side_effect=[(None,{'status':'REJECT','cleanup':{'removed':True},'rejection':'fixture'}),(self.inputs.astype(self.np.float32),{'status':'OUTPUT_VALID','cleanup':{'removed':True}})]
            _,result=execute(self.root,self.inputs,self.inputs)
            self.assertEqual(worker.call_count,2);self.assertTrue(result['fallback_used']);self.assertEqual(result['executed_version'],'sm86-cublas-original')
    def test_two_worker_failures_are_bounded(self):
        from kernel_agent.deployment import execute
        with patch('kernel_agent.deployment.DockerRunner') as worker:
            worker.return_value.run.side_effect=[(None,{'status':'REJECT','cleanup':{'removed':True}}),(None,{'status':'REJECT','cleanup':{'removed':True}})]
            with self.assertRaises(RuntimeError):execute(self.root,self.inputs,self.inputs)
            self.assertEqual(worker.call_count,2)
    def test_invalid_input_never_starts_worker(self):
        from kernel_agent.deployment import execute
        bad=self.inputs.copy();bad[0,0]=self.np.nan
        with patch('kernel_agent.deployment.DockerRunner') as worker:
            with self.assertRaises(ValueError):execute(self.root,bad,self.inputs)
            worker.assert_not_called()


if __name__=='__main__':unittest.main()
