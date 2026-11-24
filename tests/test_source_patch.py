"""Real Git patch guard checks. Simulated model responses, no compiler/GPU/API calls."""
import difflib,json,re,unittest
from pathlib import Path
from experiments.p3_01.patches import TARGET,apply_reviewable_patch

BASE=Path(TARGET).read_bytes()


def response(changed,**extra):
    patch='diff --git a/'+TARGET+' b/'+TARGET+'\n'+''.join(difflib.unified_diff(BASE.decode().splitlines(True),changed.splitlines(True),fromfile='a/'+TARGET,tofile='b/'+TARGET))
    return json.dumps({'schema_version':1,'parent_id':'current','patch':patch,'reason':'CPU patch guard fixture',**extra})


class SourcePatchTests(unittest.TestCase):
    def valid(self):return BASE.decode().replace('i += 16','i += 32')
    def test_fixed_cuda_change_applies_in_temp_only(self):
        before=Path(TARGET).read_bytes();result=apply_reviewable_patch(response(self.valid()),BASE)
        self.assertNotEqual(result['updated_source'],BASE);self.assertEqual(Path(TARGET).read_bytes(),before);self.assertEqual(result['status'],'PATCH_APPLIED_NOT_EXECUTED')
    def test_explicit_recount_only_recovers_header_counts(self):
        data=json.loads(response(self.valid()));data['patch']=re.sub(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@',r'@@ -\1,999 +\2,999 @@',data['patch'])
        with self.assertRaises(ValueError):apply_reviewable_patch(json.dumps(data),BASE)
        result=apply_reviewable_patch(json.dumps(data),BASE,recount=True)
        self.assertEqual(result['updated_source'],self.valid().encode());self.assertTrue(result['patch_recounted'])
    def test_protected_path_rejected(self):
        raw=json.loads(response(self.valid()));raw['patch']=raw['patch'].replace(TARGET,'evaluators/gemm_decision.py')
        with self.assertRaises(ValueError):apply_reviewable_patch(json.dumps(raw),BASE)
    def test_traversal_rejected(self):
        raw=json.loads(response(self.valid()));raw['patch']=raw['patch'].replace(TARGET,'../../outside.cu')
        with self.assertRaises(ValueError):apply_reviewable_patch(json.dumps(raw),BASE)
    def test_mode_change_rejected(self):
        raw=json.loads(response(self.valid()));raw['patch']=raw['patch'].replace('\n---','\nold mode 100644\nnew mode 120000\n---',1)
        with self.assertRaises(ValueError):apply_reviewable_patch(json.dumps(raw),BASE)
    def test_host_launcher_change_rejected(self):
        with self.assertRaises(ValueError):apply_reviewable_patch(response(BASE.decode().replace('return out;','return torch::zeros_like(out);')),BASE)
    def test_license_change_rejected(self):
        with self.assertRaises(ValueError):apply_reviewable_patch(response(BASE.decode().replace('2022, NVIDIA','2023, NVIDIA')),BASE)
    def test_extra_response_key_rejected(self):
        with self.assertRaises(ValueError):apply_reviewable_patch(response(self.valid(),score=100),BASE)
    def test_wrong_parent_rejected(self):
        with self.assertRaises(ValueError):apply_reviewable_patch(response(self.valid(),parent_id='other'),BASE)
    def test_noop_rejected(self):
        with self.assertRaises(ValueError):apply_reviewable_patch(response(BASE.decode()),BASE)


if __name__=='__main__':unittest.main()
