"""CPU replay of installed GEMM preview/final outputs, wheel, boundaries and package rollback."""
import argparse,hashlib,io,json,re,zipfile
from pathlib import Path
import numpy as np
from kernel_agent.protocol import decode_frame
from kernel_agent.optimization.source_packets import unpack
from kernel_agent.operators.runtime_contract import specifications
from experiments.full_delivery.audit_sm86_operator_release import oracle
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(folder):
 archive=folder/'raw-archive';raw=archive/'remote';receipt=json.loads((archive/'collection.json').read_text());assert sha(archive/'raw.zip')==receipt['archive_sha256']
 with zipfile.ZipFile(archive/'raw.zip') as z:manifest=json.loads(z.read('manifest.json'))
 for n,m in manifest.items():assert (raw/n).stat().st_size==m['bytes'] and sha(raw/n)==m['sha256']
 result=json.loads((raw/'result.json').read_text());v=json.loads((raw/'validation/result.json').read_text());assert result['status']==v['status']=='PASS' and result['validation']==v
 remote=Path(result['location']['remote_root']);work=raw/'validation';production=result['phase']=='qualified';version='0.1.2' if production else '0.1.2rc1';assert result['version']==v['version']==version and result['production_release'] is production
 wheel=raw/Path(result['wheel']).relative_to(remote);assert sha(wheel)==result['wheel_sha256'];prefix=raw/'prefix'
 with zipfile.ZipFile(wheel) as z:
  members={n:z.read(n) for n in z.namelist() if n.startswith('kernel_agent/') and not n.endswith('/')}
 for n,b in members.items():assert (prefix/n).read_bytes()==b
 for n,h in json.loads((folder/'source-hashes.json').read_text()).items():assert sha(raw/'source'/n)==h
 if not production:assert 'kernel_agent/data/gemm_qualification.json' not in members and 'kernel_agent/data/qualified_releases.json' not in members
 expected=json.loads(members['kernel_agent/data/releases.json']);digests={x['binary_sha256'] for x in expected['versions'].values()}
 for h in digests:assert hashlib.sha256(members['kernel_agent/data/artifacts/'+h+'.so']).hexdigest()==h
 tests=(work/'cpu-tests.stderr').read_text();cpu=int(re.search(r'Ran (\d+) tests',tests)[1]);assert cpu==49 and tests.rstrip().endswith('OK')
 operator_certificate=json.loads(members['kernel_agent/operators/qualification.json']);parent=ROOT/'evidence/20261123-sm86-operators-qualified-installed/raw-archive/remote';sdk=json.loads((parent/'sdk/aot-manifest.json').read_text());assert members['kernel_agent/operators/qualification.json']==(parent/'prefix/kernel_agent/operators/qualification.json').read_bytes()
 for n,h in operator_certificate['runtime_files'].items():assert hashlib.sha256(members[n]).hexdigest()==h
 outputs=0;attempts=0;qualified=0;max_error=0.
 def gemm_record(path):
  nonlocal max_error
  rec=json.loads((path/'record.json').read_text());assert rec['cleanup']['removed'] and rec['protected_files_unchanged']
  for n,h in rec['protected_sha256'].items():assert sha(path/n)==h
  cmd=rec['command'];mounts=[cmd[i+1] for i,x in enumerate(cmd) if x=='--mount'];assert len(mounts)==2 and all(x.endswith(',readonly') for x in mounts)
  for flag in ['--network=none','--sig-proxy=false','--read-only','--log-driver=none','--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=32','--memory=2g','--memory-swap=2g','--cpus=2']:assert cmd.count(flag)==1
  assert cmd[cmd.index('--user')+1]=='65534:65534'
  for name in ['pipe_worker.py','protocol.py']:assert (path/'artifact'/name).read_bytes()==members['kernel_agent/'+name]
  for stream in ['stdout','stderr']:assert (path/'capture'/(stream+'.bin')).stat().st_size==rec['stored_bytes'][stream]<=rec['limits'][stream]
  if rec['status']=='REJECT':
   assert rec['case'] in ['crash','timeout']
   if rec['case']=='timeout':assert rec['reason']=='timeout' and b'NKA_FINITE_CUDA_READY' in (path/'capture/stderr.bin').read_bytes()
   return None
  assert rec['status']=='OUTPUT_VALID' and rec['exit']==0 and rec['reason'] is None
  a=np.load(path/'inputs/a.npy',allow_pickle=False);b=np.load(path/'inputs/b.npy',allow_pickle=False);meta,payload=decode_frame(path/'capture/stdout.bin',a.shape[0]*b.shape[1]*4+16384);assert meta==rec['metadata'] and meta['uid']==65534 and meta['torch']=='2.14.0+cu130' and meta['device']=='sm86' and rec['binary_sha256'] in digests and sha(path/'artifact/nka_gemm_v1.so')==rec['binary_sha256'] and payload==(path/'result.npy').read_bytes();value=np.load(io.BytesIO(payload),allow_pickle=False);target=a.astype(np.float64)@b.astype(np.float64);delta=np.abs(value.astype(np.float64)-target);assert value.dtype==np.float32 and value.shape==target.shape and np.all(delta<=3e-5+1e-4*np.abs(target));max_error=max(max_error,float(delta.max()));return value
 for check in v['checks']:
  job=raw/Path(check['run_directory']).relative_to(remote);run=json.loads((job/'execution.json').read_text());assert run['status']=='PASS' and run['production_release'] is check['production_release'] and len(run['attempts'])==check['attempts'] and run['fallback_used']==check['fallback'];actual=np.load(work/(check['label']+'.npy'),allow_pickle=False);successful=[]
  for i in range(len(run['attempts'])):
   path=job/('attempt-'+str(i));assert json.loads((path/'record.json').read_text())=={k:v for k,v in run['attempts'][i].items() if k!='version'};value=gemm_record(path)
   if value is not None:successful.append(value)
  assert len(successful)==1 and np.array_equal(successful[0],actual);outputs+=1;attempts+=len(run['attempts']);qualified+=check['production_release']
 assert outputs==v['gemm_gpu_outputs']==8 and attempts==v['gemm_gpu_attempts']==10 and qualified==(7 if production else 0)
 boundary=gemm_record(work/'boundary');assert boundary is not None;m=json.loads((work/'boundary/record.json').read_text())['metadata'];assert all(m[k] is True for k in ['capabilities_zero','no_new_privs','artifact_write_denied','inputs_write_denied','external_connect_denied']) and m['interfaces']==['lo'] and all(m[k] is False for k in ['docker_socket_visible','control_canary_in_env','host_private_visible','host_pid_visible'])
 for op in ['attention','mla']:
  folder_op=work/('operator-'+op);req=json.loads((folder_op/'request.json').read_text());ins,outs=specifications(req);arrays={n:np.load(folder_op/'inputs'/n,allow_pickle=False) for n in ins};targets=oracle(req,arrays);execution=json.loads((folder_op/'run/execution.json').read_text());assert execution['production_release'] and execution['status']=='PASS' and len(execution['attempts'])==1
  rec=execution['attempts'][0];path=folder_op/'run/attempt-0';assert json.loads((path/'record.json').read_text())==rec and rec['protected_unchanged'] and rec['cleanup']['removed']
  for n,h in rec['protected_hashes'].items():assert sha(path/n)==h
  meta=rec['metadata'];assert meta['uid']==65534 and meta['aot_prebuilt'] and meta['guards_ok'] and meta['input_hashes']==[hashlib.sha256(arrays[n].tobytes()).hexdigest() for n in sorted(arrays)] and all(h in sdk['files'].values() for h in meta['loaded_native_sha256'])
  limits={n:int(np.prod(shape))*np.dtype(dtype).itemsize+16384 for n,(shape,dtype) in outs.items()};m,payload=decode_frame(path/'capture/stdout.bin',sum(limits.values())+1024*1024+(len(limits)+1)*512);report,files=unpack(payload,limits);assert m==meta and report['output_names']==sorted(outs)
  for n,target in targets.items():
   assert files[n]==(path/n).read_bytes()==(folder_op/'outputs'/n).read_bytes();actual=np.load(io.BytesIO(files[n]),allow_pickle=False);finite=np.isfinite(target);delta=np.abs(actual.astype(np.float64)[finite]-target[finite]);assert np.array_equal(np.isneginf(actual),np.isneginf(target)) and np.all(delta<=.002+.002*np.abs(target[finite]));max_error=max(max_error,float(delta.max()) if delta.size else 0.)
 assert (work/'version.stdout').read_text().strip()==(work/'current-version.stdout').read_text().strip()==version and (work/'previous-version.stdout').read_text().strip()==('0.1.2rc1' if production else '0.1.1') and v['current_prefix']==str(remote/'prefix')
 cert_hash=None
 if production:
  cert_raw=members['kernel_agent/data/gemm_qualification.json'];cert_hash=hashlib.sha256(cert_raw).hexdigest();cert=json.loads(cert_raw);assert json.loads(members['kernel_agent/data/qualified_releases.json'])['qualification_sha256']==cert_hash
  for n,h in cert['runtime_files'].items():assert hashlib.sha256(members[n]).hexdigest()==h
  for n,h in cert['evidence'].items():assert sha(ROOT/n)==h
  assert 'wrong driver' in v['negatives']
 else:assert 'missing certificate' in v['negatives']
 target=ROOT/'artifacts/wheels'/wheel.name
 if target.exists():assert sha(target)==sha(wheel)
 else:target.write_bytes(wheel.read_bytes());target.chmod(0o444)
 value={'status':'AUDIT_PASS_QUALIFIED' if production else 'AUDIT_PASS_PREVIEW','version':version,'installed_cpu_tests':cpu,'gemm_gpu_outputs':outputs,'gemm_gpu_attempts':attempts,'operator_gpu_outputs':2,'boundary_gpu_outputs':1,'total_gpu_outputs':11,'total_gpu_attempts':13,'qualified_gemm_outputs':qualified,'maximum_absolute_error':max_error,'wheel_sha256':sha(wheel),'qualification_sha256':cert_hash,'raw_files_verified':len(manifest),'archive_sha256':receipt['archive_sha256'],'new_event_samples':0,'new_model_calls':0,'heldout_reexecuted':False,'production_release':production};(folder/'verification.json').write_text(json.dumps(value,indent=2));return value
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();print(json.dumps(audit(a.folder.resolve())))
