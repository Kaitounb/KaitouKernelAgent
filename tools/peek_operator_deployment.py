"""Read-only progress for the two exact preregistered public operator batches."""
import argparse,json,re
from pathlib import Path
from tools.ssh_3070 import ROOT,execute_python


def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence',choices=['evidence/20261123-sm86-operator-deployment-stability','evidence/20261123-sm86-operator-deployment-shared-input'],default='evidence/20261123-sm86-operator-deployment-shared-input');a=p.parse_args();out=ROOT/a.evidence;root=json.loads((out/'location.json').read_text())['remote_root'];assert re.fullmatch('/tmp/nka-sm86-ops-(stability|shared)-[A-Za-z0-9_-]+',root)
 payload='ROOT='+repr(root)+'\n'+r'''
import json
from pathlib import Path
root=Path(ROOT);v={'sessions':json.loads((root/'progress.json').read_text()) if (root/'progress.json').exists() else []}
for p in sorted(root.glob('*.progress.json')):
 x=json.loads(p.read_text());v[p.name]={'rows':len(x['rows']),'last_workload':x['rows'][-1]['workload'] if x['rows'] else None}
if (root/'result.json').exists():
 x=json.loads((root/'result.json').read_text());v['result']={k:y for k,y in x.items() if k not in ['records','decision']}
if (root/'host.stderr').exists():v['host_stderr_tail']=(root/'host.stderr').read_text()[-1000:]
print(json.dumps(v))
'''
 r=execute_python(payload,45);assert r.returncode==0;v=json.loads(r.stdout);(out/'latest-progress.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
