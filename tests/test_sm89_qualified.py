"""Qualified catalog regression using only the real evidence-backed certificate, in temp stores."""
import hashlib,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from kernel_agent.sm89 import catalog


class SM89QualifiedTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.data=self.root/'data';shutil.copytree(catalog.DATA,self.data);self.patcher=patch.object(catalog,'DATA',self.data);self.patcher.start()
    def tearDown(self):self.patcher.stop();self.temp.cleanup()
    def test_real_certificate_catalog_and_version_rollback(self):
        store=catalog.registry(self.root/'qualified',channel='qualified');self.assertTrue(store.initialize()['production_release']);self.assertEqual(store.rollback()['selected_version'],'sm89-cublas-baseline')
    def test_certificate_edit_rejected_before_registry_creation(self):
        p=self.data/'gemm_qualification.json';p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'certificate invalid'):catalog.registry(self.root/'wrong',channel='qualified')
        self.assertFalse((self.root/'wrong').exists())
    def test_changed_runtime_fingerprint_rejected_even_with_rehashed_catalog(self):
        p=self.data/'gemm_qualification.json';cert=json.loads(p.read_text());cert['runtime_files']['kernel_agent/protocol.py']='0'*64;p.write_text(json.dumps(cert));index=json.loads((self.data/'qualified_releases.json').read_text());index['qualification_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();(self.data/'qualified_releases.json').write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError,'runtime code differs'):catalog.registry(self.root/'wrong',channel='qualified')


if __name__=='__main__':unittest.main()
