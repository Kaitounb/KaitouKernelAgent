"""Mocked provider budget/parent/response tests. No API, SDK, credential file, compiler or GPU."""
import difflib,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from experiments.p3_01.prepare import prepare
from experiments.p3_01.propose import call_once
from experiments.p3_01.patches import TARGET


class FakeProvider:
    calls=0;fail=False;finish='stop'
    def __init__(self,**kwargs):self.client=SimpleNamespace(_client=SimpleNamespace(event_hooks={}),with_options=lambda **kw:self.client)
    def is_available(self):return True
    def get_response(self,*args,**kwargs):
        type(self).calls+=1
        if self.fail:raise RuntimeError('fake-key-provider-error')
        text=Path(TARGET).read_text();changed=text.replace('i += 16','i += 32');diff='diff --git a/'+TARGET+' b/'+TARGET+'\n'+''.join(difflib.unified_diff(text.splitlines(True),changed.splitlines(True),fromfile='a/'+TARGET,tofile='b/'+TARGET))
        content=json.dumps({'schema_version':1,'parent_id':'current','patch':diff,'reason':'CPU mocked proposal'})
        body={'choices':[{'finish_reason':self.finish,'message':{'content':content}}],'usage':{'total_tokens':1}}
        for hook in self.client._client.event_hooks['response']:hook(SimpleNamespace(read=lambda:None,json=lambda:body))
        return SimpleNamespace(content=content,usage=body['usage'])


class SourceProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.prepared=self.root/'prepared';prepare(self.prepared);FakeProvider.calls=0;FakeProvider.fail=False;FakeProvider.finish='stop';self.env=patch.dict(os.environ,{'DEEPSEEK_API_KEY':'fake-key-provider'});self.env.start()
    def tearDown(self):self.env.stop();self.temp.cleanup()
    def test_one_mocked_call_no_execution(self):
        output=self.root/'run';result=call_once(self.prepared,output,provider_factory=FakeProvider);self.assertEqual(FakeProvider.calls,1);self.assertEqual(result['execution_status'],'NOT_EXECUTED')
        self.assertFalse(json.loads((output/'usage.json').read_text())['real_external_model_call'])
    def test_failure_no_retry_and_redacted(self):
        FakeProvider.fail=True;output=self.root/'run'
        with self.assertRaises(RuntimeError):call_once(self.prepared,output,provider_factory=FakeProvider)
        self.assertEqual(FakeProvider.calls,1);self.assertNotIn('fake-key-provider',(output/'usage.json').read_text())
    def test_incomplete_output_rejected(self):
        FakeProvider.finish='length';output=self.root/'run'
        with self.assertRaises(RuntimeError):call_once(self.prepared,output,provider_factory=FakeProvider)
        self.assertEqual(FakeProvider.calls,1);self.assertFalse((output/'proposal.json').exists())
    def test_budget_change_rejected_before_call(self):
        request=json.loads((self.prepared/'request.json').read_text());request['max_retries']=3;(self.prepared/'request.json').write_text(json.dumps(request))
        with self.assertRaises(ValueError):call_once(self.prepared,self.root/'run',provider_factory=FakeProvider)
        self.assertEqual(FakeProvider.calls,0)


if __name__=='__main__':unittest.main()
