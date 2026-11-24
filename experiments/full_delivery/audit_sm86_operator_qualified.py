"""Final GPU-free acceptance of the real SM86 qualified wheel, frames and package rollback."""
import hashlib,json,re,shutil,zipfile
from pathlib import Path
import numpy as np
from kernel_agent.protocol import decode_frame
from kernel_agent.optimization.source_packets import unpack
from kernel_agent.operators.runtime_contract import specifications
from experiments.full_delivery.audit_sm86_operator_release import oracle
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 out=ROOT/'evidence/20261123-sm86-operators-qualified-installed';archive=out/'raw-archive';raw=archive/'remote';col=json.loads((archive/'collection.json').read_text());assert sha(archive/'raw.zip')==col['archive_sha256']
 with zipfile.ZipFile(archive/'raw.zip') as z:
  manifest=json.loads(z.read('manifest.json'))
  for n,m in manifest.items():assert sha(raw/n)==m['sha256'] and (raw/n).stat().st_size==m['bytes']
 result=json.loads((raw/'result.json').read_text());v=json.loads((raw/'validation/result.json').read_text());assert result['status']==v['status']=='PASS' and v['production_release'] and v['gpu_outputs']==9 and v['gpu_attempts']==11 and v['qualified_outputs']==7 and len(v['real_qualification_negative_checks'])==5
 tests=(raw/'validation/cpu-tests.stderr').read_text();cpu=int(re.search(r'Ran (\d+) tests',tests)[1]);assert cpu==40 and tests.rstrip().endswith('OK')
 prefix=raw/'prefix';certificate=json.loads((prefix/'kernel_agent/operators/qualification.json').read_text());assert sha(prefix/'kernel_agent/operators/qualification.json')==json.loads((prefix/'kernel_agent/operators/qualified_catalog.json').read_text())['qualification_sha256'];assert certificate['status']=='QUALIFIED' and certificate['scope']=='SM86_REVIEWED_FIXED_ATTENTION_MLA' and certificate['production_release'] is True
 parent=ROOT/'evidence/20261123-sm86-operators-rc2/raw-archive/remote/prefix'
 for n,h in certificate['runtime_files'].items():assert sha(prefix/n)==h and sha(parent/n)==h
 for n,h in certificate['evidence'].items():assert sha(ROOT/n)==h
 sdk=json.loads((raw/'sdk/aot-manifest.json').read_text());assert certificate['bundle_id']==sdk['bundle_id'] and all(sha(raw/'sdk'/n)==h for n,h in sdk['files'].items())
 wheel=next((raw/'dist').glob('*.whl'));assert sha(wheel)==result['wheel_sha256']
 with zipfile.ZipFile(wheel) as z:
  for n in z.namelist():
   if n.startswith('kernel_agent/') and not n.endswith('/'):assert z.read(n)==(prefix/n).read_bytes()
 attempts=list((raw/'validation').rglob('record.json'));assert len(attempts)==11;valid=0
 for p in attempts:
  rec=json.loads(p.read_text());assert rec['cleanup']['removed'] and rec['protected_unchanged']
  for n,h in rec['protected_hashes'].items():assert sha(p.parent/n)==h
  cmd=rec['command'];mounts=[cmd[i+1] for i,x in enumerate(cmd) if x=='--mount'];assert len(mounts)==3 and all(x.endswith(',readonly') for x in mounts) and '--network=none' in cmd and '--read-only' in cmd and '--cap-drop=ALL' in cmd and '--security-opt=no-new-privileges' in cmd and '--memory=2g' in cmd and '--pids-limit=32' in cmd
  if rec['status']!='OUTPUT_VALID':continue
  valid+=1;m=rec['metadata'];assert m['uid']==m['boundary_gid']==65534 and m['reference_available'] is False and m['aot_prebuilt'] and m['guards_ok'] and m['producer_consumer_stream'] and m['output_fresh_contiguous'];assert all(m['boundary_'+k] is True for k in ['cap_eff_zero','cap_bnd_zero','no_new_privs','network_only_loopback','root_readonly','mounted_files_readonly','host_control_absent','credentials_env_absent'])
  req=json.loads((p.parent/'inputs/request.json').read_text());ins,outs=specifications(req);arrays={n:np.load(p.parent/'inputs'/n,allow_pickle=False) for n in ins};assert m['input_hashes']==[hashlib.sha256(arrays[n].tobytes()).hexdigest() for n in sorted(arrays)];assert m['loaded_native_sha256'] and all(h in set(sdk['files'].values()) for h in m['loaded_native_sha256']);limits={n:int(np.prod(shape))*np.dtype(dtype).itemsize+16384 for n,(shape,dtype) in outs.items()};limit=sum(limits.values())+1024*1024+(len(limits)+1)*512;meta,payload=decode_frame(p.parent/'capture/stdout.bin',limit);assert meta==m;report,files=unpack(payload,limits);assert report=={'format':'reviewed-native-operator-v1','output_names':sorted(outs)}
  for n,b in files.items():assert b==(p.parent/n).read_bytes()
 assert valid==9;max_error=0.;checks=[]
 for folder in sorted((raw/'validation').iterdir()):
  if not (folder/'outputs').is_dir():continue
  req=json.loads((folder/'request.json').read_text());ins,_=specifications(req);arrays={n:np.load(folder/'inputs'/n,allow_pickle=False) for n in ins};targets=oracle(req,arrays);errors={}
  for n,target in targets.items():
   actual=np.load(folder/'outputs'/n,allow_pickle=False);finite=np.isfinite(target);delta=np.abs(actual.astype(np.float64)[finite]-target[finite]);assert np.array_equal(np.isneginf(actual),np.isneginf(target)) and np.all(delta<=.002+.002*np.abs(target[finite]));errors[n]=float(delta.max()) if delta.size else 0.;max_error=max(max_error,errors[n])
  execution=json.loads((folder/'run/execution.json').read_text());checks.append({'case':folder.name,'production_release':execution['production_release'],'max_abs_errors':errors})
 assert len(checks)==9 and sum(c['production_release'] for c in checks)==7
 post=json.loads((out/'postflight.json').read_text());assert post['qualified_inspect']['production_release'] is True and post['qualified_inspect']['scope']==certificate['scope'] and post['current_target']==result['location']['prefix'] and not post['native_containers'] and not post['unexpected_jit_libraries'];assert post['launcher_sha256']==sha(raw/'kernel-agent-operators')
 target=ROOT/'artifacts/wheels'/wheel.name
 if target.exists():assert sha(target)==sha(wheel)
 else:shutil.copyfile(wheel,target);target.chmod(0o444)
 audit={'status':'AUDIT_PASS_QUALIFIED_DEPLOYMENT','version':'0.1.1','scope':certificate['scope'],'wheel_sha256':sha(wheel),'qualification_sha256':sha(prefix/'kernel_agent/operators/qualification.json'),'archive_sha256':col['archive_sha256'],'raw_files_verified':len(manifest),'runtime_files_identical_to_installed_rc2':len(certificate['runtime_files']),'installed_cpu_tests':cpu,'gpu_outputs_numpy_fp64_verified':9,'gpu_attempts':11,'qualified_outputs':7,'explicit_preview_rollback_outputs':2,'real_qualification_negative_checks':5,'maximum_absolute_error':max_error,'case_checks':checks,'new_model_calls':0,'new_cuda_builds':0,'new_event_samples':0,'old_sealed_reexecuted':False,'production_release':True};(out/'verification.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:y for k,y in audit.items() if k!='case_checks'}))

if __name__=='__main__':main()
