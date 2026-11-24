"""Freeze fresh server-local synthetic Attention acceptance after the RC is immutable."""
import base64,hashlib,json
from pathlib import Path
from tools.ssh_3070 import ROOT,execute_python


def main():
 out=ROOT/'evidence/20261123-attention-sm86-independent';out.mkdir(exist_ok=False);private=ROOT/'.private/attention-sm86-rc1-independent-v1';private.mkdir(mode=0o700,exist_ok=False);installed=json.loads((ROOT/'evidence/20261123-sm86-operators-rc1/finish-result.json').read_text())['persistent_root'];source=(ROOT/'experiments/full_delivery/attention_independent_host.py').read_bytes()
 payload='SOURCE='+repr(base64.b64encode(source).decode())+'\nINSTALLED='+repr(installed)+'\n'+r'''
import base64,json,subprocess,tempfile,uuid
from pathlib import Path
host=Path(tempfile.mkdtemp(prefix='nka-attention-private-host-'));host.chmod(0o700);script=host/'host.py';script.write_bytes(base64.b64decode(SOURCE));private=Path('/home/dministrator/cuda-lab/nka-persistent')/('attention-independent-'+uuid.uuid4().hex);cmd=['/home/dministrator/cuda-lab/.venv/bin/python','-B',str(script),'--root',str(private),'--install-root',INSTALLED,'--phase','prepare'];r=subprocess.run(cmd,capture_output=True,text=True,timeout=30);assert r.returncode==0
print(json.dumps({'remote_private_root':str(private),'remote_host_script':str(script),'installed_root':INSTALLED,'freeze':json.loads(r.stdout)}))
'''
 (out/'prepare-driver.py').write_text(payload);r=execute_python(payload,60);(out/'prepare.stderr').write_text(r.stderr);assert r.returncode==0;v=json.loads(r.stdout);(private/'location.json').write_text(json.dumps(v,indent=2));(private/'location.json').chmod(0o600);(out/'freeze.json').write_text(json.dumps(v['freeze'],indent=2));print(json.dumps(v['freeze']))
if __name__=='__main__':main()
