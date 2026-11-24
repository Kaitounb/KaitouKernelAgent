"""CPU fixtures only: no real release certificate, GPU execution or promotion."""
import hashlib,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import kernel_agent.registry as registry


class QualifiedChannelTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.data=self.root/'data';self.data.mkdir();self.original=registry.DATA
        for n in ['gemm_contract_v1.json','releases.json']:shutil.copyfile(self.original/n,self.data/n)
        shutil.copytree(self.original/'artifacts',self.data/'artifacts');self.legacy=(self.data/'releases.json').read_bytes();self.patch=patch.object(registry,'DATA',self.data);self.patch.start()
    def tearDown(self):self.assertEqual((self.data/'releases.json').read_bytes(),self.legacy);self.patch.stop();self.temp.cleanup()
    def fixture(self):
        catalog=json.loads(self.legacy);meta=catalog['versions'][catalog['default']];certificate={'status':'QUALIFIED','backend':'cublas','production_release':True,'scope':'SM86_FIXED_IMAGE_GEMM_CUBLAS','independent_correctness_pass':True,'recovery_pass':True,'final_pairing_pass':True,'raw_audit_pass':True,'environment':{k:meta[k] for k in ['image_id','torch','cuda','device']},'binary_sha256':[v['binary_sha256'] for v in catalog['versions'].values()],'test_scope':'CPU_FIXTURE_NOT_RELEASE_ACCEPTANCE'}
        raw=json.dumps(certificate).encode();(self.data/'gemm_qualification.json').write_bytes(raw);catalog['qualification_sha256']=hashlib.sha256(raw).hexdigest()
        for m in catalog['versions'].values():m['production_release']=True;m['status']='CPU_FIXTURE_NOT_RELEASE_ACCEPTANCE'
        (self.data/'qualified_releases.json').write_text(json.dumps(catalog));return catalog,certificate
    def test_missing_certificate_fails_closed_legacy_still_works(self):
        with self.assertRaisesRegex(ValueError,'not packaged'):registry.index('qualified')
        self.assertEqual(registry.index(),json.loads(self.legacy));self.assertFalse(registry.Registry(self.root/'legacy').initialize()['production_release'])
    def test_wrong_certificate_digest_rejected(self):
        self.fixture();p=self.data/'gemm_qualification.json';p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'certificate invalid'):registry.index('qualified')
    def test_unqualified_artifact_and_environment_rejected(self):
        catalog,_=self.fixture();key=catalog['default'];catalog['versions'][key]['binary_sha256']='0'*64;(self.data/'qualified_releases.json').write_text(json.dumps(catalog))
        with self.assertRaisesRegex(ValueError,'artifact/environment'):registry.index('qualified')
    def test_malformed_certificate_fields_fail_before_registry_use(self):
        catalog,certificate=self.fixture();certificate['binary_sha256']=[[],[]];raw=json.dumps(certificate).encode();(self.data/'gemm_qualification.json').write_bytes(raw);catalog['qualification_sha256']=hashlib.sha256(raw).hexdigest();(self.data/'qualified_releases.json').write_text(json.dumps(catalog))
        with self.assertRaisesRegex(ValueError,'schema invalid'):registry.index('qualified')
        catalog,_=self.fixture();catalog['versions']=[];(self.data/'qualified_releases.json').write_text(json.dumps(catalog))
        with self.assertRaisesRegex(ValueError,'catalog invalid'):registry.index('qualified')
        catalog,_=self.fixture();catalog['versions'][catalog['default']]['torch']='wrong';(self.data/'qualified_releases.json').write_text(json.dumps(catalog))
        with self.assertRaisesRegex(ValueError,'artifact/environment'):registry.index('qualified')
    def test_fixture_rollback_and_corrupt_active_select_only_listed_fallback(self):
        catalog,_=self.fixture();store=registry.Registry(self.root/'store',channel='qualified');result=store.initialize();self.assertEqual(result['selected_version'],catalog['default']);self.assertEqual(store.rollback()['selected_version'],catalog['fallback']);self.assertEqual(store.rollback()['selected_version'],catalog['default']);active=store.root/'versions'/catalog['default']/'nka_gemm_v1.so';active.chmod(0o644);active.write_bytes(b'CPU corruption fixture');_,meta,reason=store.select();self.assertEqual(meta['binary_sha256'],catalog['versions'][catalog['fallback']]['binary_sha256']);self.assertTrue(reason)


if __name__=='__main__':unittest.main()
