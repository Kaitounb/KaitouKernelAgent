"""Verify installed SDK admission hardening and unchanged numerical/execution bodies."""
import ast,hashlib,json,re,shutil,zipfile
from pathlib import Path
import numpy as np
from kernel_agent.operators.qualification import RUNTIME_FILES
from experiments.full_delivery.audit_sm86_operator_release import oracle
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def function(path,name):
 tree=ast.parse(Path(path).read_text());nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name];assert len(nodes)==1;return ast.dump(nodes[0],include_attributes=False)

def main():
 out=ROOT/'evidence/20261123-sm86-operators-rc2';archive=out/'raw-archive';raw=archive/'remote';c=json.loads((archive/'collection.json').read_text());assert sha(archive/'raw.zip')==c['archive_sha256']
 with zipfile.ZipFile(archive/'raw.zip') as z:
  manifest=json.loads(z.read('manifest.json'))
  for n,m in manifest.items():assert sha(raw/n)==m['sha256'] and (raw/n).stat().st_size==m['bytes']
 r=json.loads((raw/'result.json').read_text());assert r['status']=='PASS' and r['rejections']==['startup','bytecode','symlink','nonobject'];log=(raw/'cpu-tests.stderr').read_text();cpu=int(re.search(r'Ran (\d+) tests',log)[1]);assert cpu==40 and log.rstrip().endswith('OK');smoke=json.loads((raw/'installed-smoke/result.json').read_text());assert smoke['status']=='PASS' and smoke['gpu_outputs']==smoke['gpu_attempts']==2
 for label in r['rejections']:
  actual='nonobject' if label=='nonobject' else label;record=next(x for x in r['records'] if x['label']=='reject-'+actual);assert record['exit']!=0 and 'inspect' in record['command'];assert 'unlisted' in (raw/'reject-startup.stderr').read_text() and 'unlisted' in (raw/'reject-bytecode.stderr').read_text()
 parent=ROOT/'evidence/20261123-sm86-operators-rc1/raw-archive/remote/prefix';prefix=raw/'prefix';assert function(parent/'kernel_agent/operators/runner.py','run')==function(prefix/'kernel_agent/operators/runner.py','run') and function(parent/'kernel_agent/operators/cli.py','execute')==function(prefix/'kernel_agent/operators/cli.py','execute')
 for name in ['pipe_worker.py','flashinfer_backend.py','runtime_contract.py','partition.py','array_io.py']:
  assert sha(parent/'kernel_agent/operators'/name)==sha(prefix/'kernel_agent/operators'/name)
 checked=0;max_error=0.
 for case in sorted((raw/'installed-smoke').iterdir()):
  if not (case/'outputs').is_dir():continue
  request=json.loads((case/'request.json').read_text());arrays={p.name:np.load(p,allow_pickle=False) for p in (case/'inputs').glob('*.npy')};expected=oracle(request,arrays);execution=json.loads((case/'run/execution.json').read_text());assert len(execution['attempts'])==1 and not execution['production_release'];rec=execution['attempts'][0];assert rec['cleanup']['removed'] and rec['protected_unchanged'] and rec['status']=='OUTPUT_VALID'
  for n,h in rec['protected_hashes'].items():assert sha(case/'run/attempt-0'/n)==h
  for n,target in expected.items():
   actual=np.load(case/'outputs'/n,allow_pickle=False);finite=np.isfinite(target);delta=np.abs(actual.astype(np.float64)[finite]-target[finite]);assert np.array_equal(np.isneginf(actual),np.isneginf(target)) and np.all(delta<=.002+.002*np.abs(target[finite]));max_error=max(max_error,float(delta.max()) if delta.size else 0.)
  checked+=1
 wheel=next((raw/'dist').glob('*.whl'));assert sha(wheel)==r['wheel_sha256']
 with zipfile.ZipFile(wheel) as z:
  for n in z.namelist():
   if n.startswith('kernel_agent/') and not n.endswith('/'):assert z.read(n)==(prefix/n).read_bytes()
 fingerprints={n:sha(prefix/n) for n in RUNTIME_FILES};assert all(sha(ROOT/n)==h for n,h in fingerprints.items());target=ROOT/'artifacts/wheels'/wheel.name
 if target.exists():assert sha(target)==sha(wheel)
 else:shutil.copyfile(wheel,target);target.chmod(0o444)
 audit={'status':'AUDIT_PASS_HARDENED_RC','version':'0.1.1rc2','wheel_sha256':sha(wheel),'archive_sha256':c['archive_sha256'],'raw_files_verified':len(manifest),'installed_cpu_tests':cpu,'installed_gpu_outputs':checked,'installed_gpu_attempts':2,'new_sdk_admission_rejections':4,'maximum_absolute_error':max_error,'primary_execution_unchanged_from_rc1':True,'runtime_files':fingerprints,'new_cuda_builds':0,'new_model_calls':0,'new_event_samples':0,'production_release':False};(out/'verification.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v for k,v in audit.items() if k!='runtime_files'}))

if __name__=='__main__':main()
