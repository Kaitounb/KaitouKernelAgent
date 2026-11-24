"""Durable total-call reservations around the existing KernelAgent provider.

Uncertain/failed/duplicate proposals consume a slot. Reopening a reservation never
authorizes a second network call. There are no credentials in this ledger.
"""
import fcntl,hashlib,json,os,re,uuid
from pathlib import Path


def _write(path,value):
    temp=path.with_name('.'+path.name+'-'+uuid.uuid4().hex)
    try:
        with temp.open('x') as out:json.dump(value,out,indent=2,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
        os.replace(temp,path);fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:temp.unlink(missing_ok=True)


class CallBudget:
    def __init__(self,root,authorization):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.authorization=Path(authorization);raw=self.authorization.read_bytes();self.auth=json.loads(raw);self.auth_sha=hashlib.sha256(raw).hexdigest()
        if self.auth.get('status')!='USER_AUTHORIZED' or type(self.auth.get('maximum_new_calls')) is not int or self.auth['maximum_new_calls']<1:raise ValueError('explicit total-call authorization missing')

    def _mutate(self,fn):
        with (self.root/'lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:
                path=self.root/'ledger.json';state=json.loads(path.read_text()) if path.exists() else {'authorization_sha256':self.auth_sha,'maximum_calls':self.auth['maximum_new_calls'],'calls':{}}
                if state['authorization_sha256']!=self.auth_sha or state['maximum_calls']!=self.auth['maximum_new_calls']:raise ValueError('budget authorization changed')
                result=fn(state);_write(path,state);return result
            finally:fcntl.flock(lock,fcntl.LOCK_UN)

    def reserve(self,key,request_sha256):
        if not re.fullmatch('[A-Za-z0-9_-]{1,120}',key) or not re.fullmatch('[0-9a-f]{64}',request_sha256):raise ValueError('invalid reservation binding')
        def update(state):
            if key in state['calls']:
                row=state['calls'][key]
                if row['request_sha256']!=request_sha256:raise ValueError('reservation already bound to another request')
                return {'dispatch_authorized':False,**row}
            if len(state['calls'])>=state['maximum_calls']:raise ValueError('total model-call cap exhausted')
            row={'ordinal':len(state['calls'])+1,'request_sha256':request_sha256,'status':'RESERVED_UNCERTAIN','usage':None};state['calls'][key]=row
            return {'dispatch_authorized':True,**row}
        return self._mutate(update)

    def finish(self,key,*,status,usage=None):
        if status not in ['RECEIVED','FAILED','UNKNOWN']:raise ValueError('invalid call outcome')
        if usage is not None:
            if not isinstance(usage,dict) or any(k not in ['prompt_tokens','completion_tokens','total_tokens'] or type(v) is not int or v<0 for k,v in usage.items()):raise ValueError('usage must be nonnegative integer token counts')
            if set(usage)=={'prompt_tokens','completion_tokens','total_tokens'} and usage['prompt_tokens']+usage['completion_tokens']!=usage['total_tokens']:raise ValueError('token usage does not add up')
        def update(state):
            row=state['calls'][key]
            if row['status']!='RESERVED_UNCERTAIN':raise ValueError('terminal call outcome cannot be overwritten')
            row.update(status=status,usage=usage);return row.copy()
        return self._mutate(update)
