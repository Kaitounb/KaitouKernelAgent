import json,tempfile,unittest
from pathlib import Path
from kernel_agent.optimization.call_budget import CallBudget


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.auth=self.root/'authorization.json';self.auth.write_text(json.dumps({'status':'USER_AUTHORIZED','maximum_new_calls':2}));self.budget=CallBudget(self.root/'ledger',self.auth)
    def tearDown(self):self.temp.cleanup()
    def test_uncertain_and_failure_consume_total_cap(self):
        self.assertTrue(self.budget.reserve('one','a'*64)['dispatch_authorized']);self.budget.reserve('two','b'*64);self.budget.finish('two',status='FAILED')
        with self.assertRaises(ValueError):self.budget.reserve('three','c'*64)
    def test_reopen_never_reauthorizes_dispatch(self):
        self.budget.reserve('one','a'*64);new=CallBudget(self.root/'ledger',self.auth);self.assertFalse(new.reserve('one','a'*64)['dispatch_authorized'])
        with self.assertRaises(ValueError):new.reserve('one','b'*64)
    def test_terminal_usage_and_authorization_immutable(self):
        self.budget.reserve('one','a'*64)
        with self.assertRaises(ValueError):self.budget.finish('one',status='RECEIVED',usage={'prompt_tokens':1,'completion_tokens':2,'total_tokens':9})
        self.budget.finish('one',status='RECEIVED',usage={'prompt_tokens':1,'completion_tokens':2,'total_tokens':3})
        with self.assertRaises(ValueError):self.budget.finish('one',status='FAILED')
        self.auth.write_text(json.dumps({'status':'USER_AUTHORIZED','maximum_new_calls':3}))
        with self.assertRaises(ValueError):CallBudget(self.root/'ledger',self.auth).reserve('two','b'*64)


if __name__=='__main__':unittest.main()
