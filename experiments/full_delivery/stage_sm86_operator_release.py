"""Stage reviewed runtime sources and an identical public AOT copy, no GPU execution."""
import base64,hashlib,json
from pathlib import Path
from tools.ssh_3070 import ROOT,execute_python

def main():
 out=ROOT/'evidence/20261123-sm86-operators-rc1';out.mkdir(exist_ok=False)
 names=[p for p in (ROOT/'kernel_agent').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='.DS_Store']+[ROOT/'pyproject.toml',ROOT/'experiments/full_delivery/sm86_operator_release_validate.py']
 names +=[ROOT/'tests'/n for n in ['test_registry.py','test_deployment_policy.py','test_qualified_channel.py','test_sm89_runtime.py','test_sm89_qualified.py','test_operator_qualification.py']]
 files={str(p.relative_to(ROOT)):p.read_bytes() for p in names};(out/'source-hashes.json').write_text(json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in files.items()},indent=2))
 payload='FILES='+repr({n:base64.b64encode(b).decode() for n,b in files.items()})+'\n'+r'''
import base64,hashlib,json,shutil,stat,tempfile
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-sm86-operators-release-'));root.chmod(0o700)
for n,v in FILES.items():p=root/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v))
old=Path('/home/dministrator/cuda-lab/nka-persistent/operators-cfae38ce5c0f');mode=stat.S_IMODE(old.stat().st_mode);m=json.loads((old/'aot-manifest.json').read_text());sdk=root/'sdk';sdk.mkdir()
for n,h in m['files'].items():
 source=old/n;assert source.is_file() and not source.is_symlink() and hashlib.sha256(source.read_bytes()).hexdigest()==h
 target=sdk/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);target.chmod(0o444)
shutil.copyfile(old/'aot-manifest.json',sdk/'aot-manifest.json');(sdk/'aot-manifest.json').chmod(0o444)
for p in [sdk,*sdk.rglob('*')]:
 if p.is_dir():p.chmod(0o555)
assert stat.S_IMODE(old.stat().st_mode)==mode==0o700
value={'status':'STAGED','remote_root':str(root),'source':str(root/'source'),'sdk':str(sdk),'bundle_id':m['bundle_id'],'aot_files_verified':len(m['files']),'old_bundle_permissions_unchanged':True,'new_gpu_calls':0,'new_model_calls':0};(root/'location.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))
'''
 (out/'stage-driver.py').write_text(payload);r=execute_python(payload,180);(out/'stage.stderr').write_text(r.stderr);(out/'stage.stdout').write_text(r.stdout);assert r.returncode==0;v=json.loads(r.stdout);(out/'location.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
