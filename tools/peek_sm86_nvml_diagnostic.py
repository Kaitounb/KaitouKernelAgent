"""Read saved3070 diagnostic progress, without any GPU/NVML query."""
import json
from tools.ssh_3070 import ROOT,execute_python

def main():
 out=ROOT/'evidence/20261124-sm86-nvml-mechanism-utf8';plan=json.loads((out/'plan.json').read_text());payload='HOST_SHA='+repr(plan['host_sha256'])+'\n'+r'''
import json
from pathlib import Path
paths=[p for p in Path('/tmp').glob('nka-cublas-final-nvml-*') if (p/'plan.json').exists() and json.loads((p/'plan.json').read_text()).get('host_sha256')==HOST_SHA];assert len(paths)==1;root=paths[0];v={'root':str(root)}
for name in ['progress.json','result.json','authorized-process-cleanup.json']:
 if (root/name).exists():v[name]=json.loads((root/name).read_text())
for name in ['failure.log','parent.stderr']:
 if (root/name).exists():v[name]=(root/name).read_text()[-3000:]
print(json.dumps(v))
'''
 r=execute_python(payload,60);(out/'peek.stderr').write_text(r.stderr);assert r.returncode==0;v=json.loads(r.stdout);(out/'latest-progress.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
