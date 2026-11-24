"""CPU-only interrupted-call/stop/budget guards, using real public frozen receipts."""
import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.full_delivery.formal_study import prepare,propose,verify
ROOT=Path(__file__).resolve().parents[1]
G=ROOT/'evidence/20261123-formal-parent-profiles/archive/remote'
M=ROOT/'evidence/20261123-formal-mla-profiles-frozen-aot/archive/remote'


@unittest.skipUnless((M/'profiles/mla/mla-dev-08/2/profile.json').exists(),'public frozen profile receipts unavailable')
class FormalStateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'study'
        with contextlib.redirect_stdout(io.StringIO()):prepare(self.root,G,M)
        self.ledger=(ROOT/'evidence/20261122-full-project-model-calls/ledger.json').read_bytes()
    def tearDown(self):
        self.assertEqual((ROOT/'evidence/20261122-full-project-model-calls/ledger.json').read_bytes(),self.ledger);self.temp.cleanup()

    def test_a_stops_on_finite_space_without_dispatch(self):
        with patch('experiments.full_delivery.formal_study.dispatch',side_effect=AssertionError('A dispatch')):
            for _ in range(4):self.assertEqual(propose(self.root,'gemm','dev-01',0,'A')['status'],'RESOLVED_EXACT_CACHE')
            closed=propose(self.root,'gemm','dev-01',0,'A');self.assertEqual(closed['reason'],'finite_parameter_space_exhausted');self.assertEqual(closed['proposals'],4)

    def test_interrupted_step_is_not_dispatched_twice(self):
        folder=self.root/'searches/gemm/dev-01/0/B/0';folder.mkdir()
        with patch('experiments.full_delivery.formal_study.dispatch',side_effect=AssertionError('repeat dispatch')):
            self.assertEqual(propose(self.root,'gemm','dev-01',0,'B')['status'],'PENDING_EXISTING_PROPOSAL')
        self.assertFalse((folder.parent/'1').exists())

    def test_protocol_drift_rejected_before_dispatch(self):
        path=self.root/'protocol.json';path.write_text(path.read_text()+' ')
        with self.assertRaises(AssertionError):verify(self.root)

    def test_three_mocked_keep_parent_actions_close_search(self):
        # Mock dispatch exercises state transitions only, not API/GPU acceptance.
        response={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'schema_version':1,'parent_id':'current','action':'keep_parent','patch':'','reason':'CPU fixture'})}}]}
        with patch('experiments.full_delivery.formal_study.dispatch',return_value=(response,{'usage':{}})) as mocked:
            for _ in range(3):self.assertEqual(propose(self.root,'gemm','dev-01',0,'B')['status'],'RESOLVED_EXACT_CACHE')
            self.assertEqual(propose(self.root,'gemm','dev-01',0,'B')['reason'],'three_consecutive_duplicate_or_invalid');self.assertEqual(mocked.call_count,3)

    def test_repeated_cached_rejection_counts_as_duplicate(self):
        patch_text=json.loads((ROOT/'evidence/20261123-mini-trial-v2/gemm/B/0/proposal.json').read_text())['proposal']['patch'];response={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'schema_version':1,'parent_id':'current','action':'modify','patch':patch_text,'reason':'CPU fixture'})}}]}
        def fake_dispatch(request,folder,*args):
            (folder/'response.json').write_text(json.dumps(response));return response,{'usage':{}}
        with patch('experiments.full_delivery.formal_study.dispatch',side_effect=fake_dispatch) as mocked:
            for _ in range(4):self.assertEqual(propose(self.root,'gemm','dev-01',0,'B')['status'],'RESOLVED_EXACT_CACHE')
            br=self.root/'searches/gemm/dev-01/0/B';first=json.loads((br/'0/decision.json').read_text());repeat=json.loads((br/'1/decision.json').read_text());self.assertEqual(first['outcome']['reason'],'aa_unstable');self.assertEqual(repeat['outcome']['reason'],'duplicate');self.assertEqual(repeat['underlying_cache_reason'],'aa_unstable');self.assertEqual(propose(self.root,'gemm','dev-01',0,'B')['proposals'],4);self.assertEqual(mocked.call_count,4)


if __name__=='__main__':unittest.main()
