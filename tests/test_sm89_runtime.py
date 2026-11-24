"""CPU allowlist/registry/boundary regressions; no GPU or real qualification certificate."""
import copy,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from kernel_agent.sm89 import catalog
from kernel_agent.sm89.runner import validate_boundary
from kernel_agent.deployment import execute


class SM89RegistryTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'store'
    def tearDown(self):self.temp.cleanup()
    def test_sm89_versions_real_rollback(self):
        store=catalog.registry(self.root);self.assertEqual(store.initialize()['selected_version'],'sm89-cublas-current');self.assertEqual(store.rollback()['selected_version'],'sm89-cublas-baseline');self.assertEqual(store.rollback()['selected_version'],'sm89-cublas-current')
    def test_corrupt_active_only_selects_sm89_fallback(self):
        store=catalog.registry(self.root);store.initialize();binary,_=store.release('sm89-cublas-current');binary.chmod(0o644);binary.write_bytes(b'CPU tampering fixture');_,meta,reason=store.select();self.assertEqual(meta['version'],'sm89-cublas-baseline');self.assertTrue(reason)
    def test_missing_qualification_does_not_create_store(self):
        data=Path(self.temp.name)/'without-certificate';shutil.copytree(catalog.DATA,data)
        for name in ['qualified_releases.json','gemm_qualification.json']:(data/name).unlink(missing_ok=True)
        with patch.object(catalog,'DATA',data),self.assertRaisesRegex(ValueError,'not packaged'):catalog.registry(self.root,channel='qualified').initialize()
        self.assertFalse(self.root.exists())
    def test_catalog_path_injection_fails_before_store(self):
        data=Path(self.temp.name)/'data';shutil.copytree(catalog.DATA,data);value=json.loads((data/'releases.json').read_text());value['versions']['../../escape']=value['versions'].pop('sm89-cublas-current');(data/'releases.json').write_text(json.dumps(value))
        with patch.object(catalog,'DATA',data),self.assertRaisesRegex(ValueError,'catalog invalid'):catalog.registry(self.root)
        self.assertFalse(self.root.exists())
    def test_invalid_input_does_not_start_native_worker(self):
        catalog.registry(self.root).initialize();bad=np.full((2,2),np.nan,np.float16)
        with patch('kernel_agent.sm89.runner.NativeRunner') as runner:
            with self.assertRaises(ValueError):execute(self.root,bad,bad,registry_factory=catalog.registry,runner_factory=runner)
            runner.assert_not_called()
    def test_unknown_cleanup_stops_sm89_fallback(self):
        catalog.registry(self.root).initialize();inputs=np.ones((2,2),np.float16)
        with patch('kernel_agent.sm89.runner.NativeRunner') as runner:
            runner.return_value.run.return_value=(None,{'status':'REJECT','cleanup':{'removed':False}})
            with self.assertRaises(RuntimeError):execute(self.root,inputs,inputs,registry_factory=catalog.registry,runner_factory=runner)
            self.assertEqual(runner.call_count,1)
        result=json.loads(next((self.root/'runs').glob('*/execution.json')).read_text());self.assertEqual(result['fallback_reason'],'cleanup_unconfirmed')


class SM89BoundaryTests(unittest.TestCase):
    def test_root_or_incomplete_claim_is_rejected(self):
        with self.assertRaises(ValueError):validate_boundary({'uid':0})
        with self.assertRaises(ValueError):validate_boundary({'uid':65534})
    def test_integer_true_does_not_forge_boundary(self):
        # Explicit CPU metadata fixture, not actual boundary evidence.
        meta={'uid':65534,'gid':65534,'capabilities_zero':True,'no_new_privs':True,'ipv4_socket_denied':True,'ipv6_socket_denied':True,'unix_ipc_available':True,'private_read_denied':True,'artifact_write_denied':True,'input_write_denied':True,'controller_signal_denied':True,'control_canary_in_env':False,'credentials_in_env':False,'torch':'2.7.1+cu126','cuda':'12.6','device':'sm89','execution':'reviewed-native','input_immutable':True,'producer_consumer_stream':True,'fresh_output':True}
        validate_boundary(meta);meta['ipv4_socket_denied']=1
        with self.assertRaises(ValueError):validate_boundary(meta)


if __name__=='__main__':unittest.main()
