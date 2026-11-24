"""Real filesystem integrity/atomic pointer checks. No GPU, Docker, NumPy or model."""
import json,tempfile,unittest
from pathlib import Path
from kernel_agent.registry import Registry,atomic_json


class RegistryTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'store';self.registry=Registry(self.root);self.registry.initialize()
    def tearDown(self):self.temp.cleanup()
    def test_default_cublas_development_only(self):
        status=self.registry.status();self.assertEqual(status['selected_version'],'sm86-cublas-container');self.assertFalse(status['production_release'])
    def test_activate_and_real_rollback(self):
        self.registry.activate('sm86-cublas-original');self.assertEqual(self.registry.pointer()['previous'],'sm86-cublas-container');self.registry.rollback();self.assertEqual(self.registry.pointer()['active'],'sm86-cublas-container')
    def test_corrupt_binary_fallback(self):
        binary,_=self.registry.release('sm86-cublas-container');binary.chmod(0o644);binary.write_bytes(b'bad');self.assertIn('mismatch',self.registry.select()[2])
    def test_manifest_mismatch_fallback(self):
        path=self.root/'versions/sm86-cublas-container/manifest.json';path.chmod(0o644);path.write_text('{}');self.assertIn('metadata',self.registry.select()[2])
    def test_missing_active_fallback(self):
        binary,_=self.registry.release('sm86-cublas-container');binary.unlink();self.assertEqual(self.registry.select()[1]['version'],'sm86-cublas-original')
    def test_invalid_pointer_no_traversal(self):
        atomic_json(self.root/'current.json',{'active':'../../secret','previous':'sm86-cublas-original'});self.assertIsNotNone(self.registry.select()[2])
    def test_corrupt_fallback_fails_closed(self):
        (self.root/'current.json').write_text('bad');path=self.root/'fallback.so';path.chmod(0o644);path.write_bytes(b'bad')
        with self.assertRaises(ValueError):self.registry.select()
    def test_binary_symlink_rejected(self):
        path,_=self.registry.release('sm86-cublas-container');path.unlink();path.symlink_to(self.root/'fallback.so');self.assertIsNotNone(self.registry.select()[2])
    def test_unapproved_activation_keeps_pointer(self):
        before=(self.root/'current.json').read_bytes()
        with self.assertRaises(ValueError):self.registry.activate('../anything')
        self.assertEqual((self.root/'current.json').read_bytes(),before)
    def test_existing_init_rejected(self):
        with self.assertRaises(FileExistsError):self.registry.initialize()


if __name__=='__main__':unittest.main()
