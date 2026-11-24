"""Resume transport only: inventory first; never replay a started GPU validation."""
import argparse,json,uuid
from tools.ssh_3060 import ROOT,execute_python

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['preview','qualified'],required=True);a=p.parse_args();version='0.1.3rc2' if a.phase=='preview' else '0.1.3';out=ROOT/('evidence/20261124-3060-'+version+'-installed');source=(out/'driver.py').read_text();wheel_sha=json.loads((ROOT/('evidence/20261124-3060-build-'+version+'/result.json')).read_text())['wheel_sha256'];attempt=out/('transport-resume-'+uuid.uuid4().hex[:8]);attempt.mkdir()
 payload='VERSION='+repr(version)+'\nEXPECTED='+repr(wheel_sha)+'\nSOURCE='+repr(source)+'\n'+r'''
import json
from pathlib import Path
roots=list((Path.home()/'nka-persistent').glob('gemm-release-'+VERSION+'-*'));assert len(roots)<=1,'multiple release roots: inspect all, never select by result'
if not roots:exec(compile(SOURCE,'original-frozen-install-driver.py','exec'))
else:
 root=roots[0];result=root/'result.json'
 if (root/'location.json').exists():assert json.loads((root/'location.json').read_text())['wheel_sha256']==EXPECTED
 if result.exists():print(result.read_text())
 else:print(json.dumps({'status':'EXISTING_INCOMPLETE_INSTALL_NOT_REPLAYED','root':str(root),'validation_reserved':(root/'validation-started.json').exists(),'production_release':False}))
'''
 (attempt/'driver.py').write_text(payload);r=execute_python(payload,1500);(attempt/'transport.stderr').write_text(r.stderr);(attempt/'transport.stdout').write_text(r.stdout);assert r.returncode==0;v=json.loads(r.stdout);(attempt/'result.json').write_text(json.dumps(v,indent=2))
 if 'location' in v:
  (out/'result.json').write_text(json.dumps(v,indent=2));(out/'location.json').write_text(json.dumps(v['location'],indent=2))
 print(json.dumps({k:v[k] for k in ['status','phase','version','location','root','validation_reserved','production_release'] if k in v}))
if __name__=='__main__':main()
