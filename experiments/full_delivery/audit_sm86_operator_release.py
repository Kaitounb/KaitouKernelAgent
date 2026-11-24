"""GPU-free archive, frame and independent NumPy FP64 replay of the SM86 operator RC."""
import hashlib,json,re,shutil,zipfile
from pathlib import Path
import numpy as np
from kernel_agent.protocol import decode_frame
from kernel_agent.optimization.source_packets import unpack
from kernel_agent.operators.array_io import decode_array
from kernel_agent.operators.runtime_contract import specifications
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def oracle(request,arrays):
 if request['operation']=='attention':
  q,k,v=[np.transpose(arrays[n].astype(np.float64),(0,2,1,3)) for n in ['q.npy','k.npy','v.npy']];sq,sk=request['sq'],request['sk'];visible=np.ones((sq,sk),bool)
  if request['mask']:visible &= arrays['mask.npy']
  if request['causal']:visible &= np.arange(sk)[None,:]<=np.arange(sq)[:,None]+sk-sq
  scores=np.matmul(q,np.swapaxes(k,-1,-2))*request['dim']**-.5;scores=np.where(visible,scores,-np.inf);m=np.max(scores,axis=-1,keepdims=True);m=np.where(np.isfinite(m),m,0);weights=np.exp(scores-m);den=weights.sum(axis=-1,keepdims=True);prob=np.divide(weights,den,out=np.zeros_like(weights),where=den>0);return {'output.npy':np.matmul(prob,v).transpose(0,2,1,3)}
 z=np.zeros(arrays['q_abs.npy'].shape,np.float64);lse=np.full(arrays['q_abs.npy'].shape[:2],-np.inf,np.float64)
 for bi,n in enumerate(request['lengths']):
  if not n:continue
  ptr=arrays['indptr.npy'];indices=arrays['indices.npy'][ptr[bi]:ptr[bi+1]];c=arrays['cache.npy'][indices].reshape(-1,512)[:n].astype(np.float64);rope=arrays['rope_cache.npy'][indices].reshape(-1,64)[:n].astype(np.float64);s=(arrays['q_abs.npy'][bi].astype(np.float64)@c.T+arrays['q_rope.npy'][bi].astype(np.float64)@rope.T)*192**-.5;m=s.max(-1,keepdims=True);p=np.exp(s-m);den=p.sum(-1,keepdims=True);z[bi]=(p/den)@c;lse[bi]=m[:,0]+np.log(den[:,0])
 return {'z.npy':z,'lse.npy':lse}


def main():
 out=ROOT/'evidence/20261123-sm86-operators-rc1';archive=out/'raw-archive';raw=archive/'remote';collection=json.loads((archive/'collection.json').read_text());assert sha(archive/'raw.zip')==collection['archive_sha256']
 with zipfile.ZipFile(archive/'raw.zip') as z:
  manifest=json.loads(z.read('manifest.json'))
  for n,row in manifest.items():assert (raw/n).stat().st_size==row['bytes'] and sha(raw/n)==row['sha256']
 result=json.loads((raw/'installed/result.json').read_text());assert result['status']=='PASS' and len(result['checks'])==13 and result['gpu_attempts']==17
 log=(raw/'installed/cpu-tests.stderr').read_text();cpu=int(re.search(r'Ran (\d+) tests',log)[1]);assert cpu==37 and log.rstrip().endswith('OK')
 wheel=next((raw/'dist').glob('*.whl'));assert sha(wheel)==json.loads((out/'install-result.json').read_text())['wheel_sha256']
 with zipfile.ZipFile(wheel) as z:
  for n in z.namelist():
   if n.startswith('kernel_agent/') and not n.endswith('/'):assert z.read(n)==(raw/'prefix'/n).read_bytes()
 bundle=json.loads((raw/'sdk/aot-manifest.json').read_text());assert all(sha(raw/'sdk'/n)==h for n,h in bundle['files'].items())
 attempts=list(raw.rglob('record.json'));assert len(attempts)==21;valid=0
 for p in attempts:
  rec=json.loads(p.read_text());assert rec['cleanup']['removed'] and rec['protected_unchanged']
  for n,h in rec['protected_hashes'].items():assert sha(p.parent/n)==h
  command=rec['command'];mounts=[command[i+1] for i,x in enumerate(command) if x=='--mount'];assert len(mounts)==3 and all(x.endswith(',readonly') for x in mounts)
  assert '--network=none' in command and '--read-only' in command and '--cap-drop=ALL' in command and '--security-opt=no-new-privileges' in command and '--memory=2g' in command and '--pids-limit=32' in command
  if rec['status']!='OUTPUT_VALID':continue
  meta=rec['metadata'];assert meta['uid']==65534 and meta['reference_available'] is False and meta['guards_ok'] and meta['producer_consumer_stream'] and meta['aot_prebuilt'];valid+=1
  if 'rollback-dev6-' not in str(p):
   flags=['cap_eff_zero','cap_bnd_zero','no_new_privs','network_only_loopback','root_readonly','mounted_files_readonly','host_control_absent','credentials_env_absent'];assert all(meta['boundary_'+k] is True for k in flags) and meta['boundary_gid']==65534 and meta['output_fresh_contiguous']
  inputs=p.parent/'inputs';request=json.loads((inputs/'request.json').read_text());specs,outputs=specifications(request);arrays={n:np.load(inputs/n,allow_pickle=False) for n in specs};expected_hashes=[hashlib.sha256(arrays[n].tobytes()).hexdigest() for n in sorted(arrays)];assert meta['input_hashes']==expected_hashes
  limits={n:int(np.prod(shape))*np.dtype(dtype).itemsize+16384 for n,(shape,dtype) in outputs.items()};payload_limit=sum(limits.values())+1024*1024+(len(limits)+1)*512;m,payload=decode_frame(p.parent/'capture/stdout.bin',payload_limit);assert m==meta;report,files=unpack(payload,limits);assert report=={'format':'reviewed-native-operator-v1','output_names':sorted(outputs)}
  for n,spec in outputs.items():assert np.array_equal(decode_array(files[n],*spec),np.load(p.parent/n,allow_pickle=False))
 assert valid==15
 verified=[];max_error=0.
 for phase in ['smoke-flat-witness','installed']:
  for folder in sorted((raw/phase).iterdir()):
   if not (folder/'outputs').is_dir():continue
   request=json.loads((folder/'request.json').read_text());specs,_=specifications(request);arrays={n:np.load(folder/'inputs'/n,allow_pickle=False) for n in specs};targets=oracle(request,arrays);errors={}
   for n,target in targets.items():
    saved=np.load(folder/'references'/n,allow_pickle=False);finite=np.isfinite(target);assert np.array_equal(np.isneginf(saved),np.isneginf(target)) and np.allclose(saved[finite],target[finite],rtol=1e-10,atol=1e-10)
    actual=np.load(folder/'outputs'/n,allow_pickle=False);delta=np.abs(actual.astype(np.float64)[finite]-target[finite]);assert np.array_equal(np.isneginf(actual),np.isneginf(target)) and np.all(delta<=2e-3+2e-3*np.abs(target[finite]));errors[n]=float(delta.max()) if delta.size else 0.;max_error=max(max_error,errors[n])
   verified.append({'phase':phase,'case':folder.name,'max_abs_errors':errors})
 negatives=json.loads((raw/'admission-negatives/result.json').read_text());assert negatives['status']=='PASS' and len(negatives['checks'])==5 and all(x['no_gpu_work_created'] for x in negatives['checks'])
 assert not (raw/'prefix/kernel_agent/operators/qualification.json').exists();post=json.loads((raw/'postflight.json').read_text());assert post['docker_containers']==''
 target=ROOT/'artifacts/wheels'/wheel.name
 if target.exists():assert sha(target)==sha(wheel)
 else:shutil.copyfile(wheel,target);target.chmod(0o444)
 audit={'status':'AUDIT_PASS_INSTALLED_RC','version':'0.1.1rc1','wheel_sha256':sha(wheel),'archive_sha256':collection['archive_sha256'],'raw_files_verified':len(manifest),'installed_cpu_tests':cpu,'installed_gpu_outputs':13,'installed_gpu_attempts':17,'smoke_gpu_outputs':2,'all_observed_worker_attempts':21,'all_successful_outputs_numpy_fp64_verified':len(verified),'initial_frame_schema_failures_preserved':2,'admission_negatives_before_gpu':5,'case_checks':verified,'maximum_absolute_error':max_error,'new_cuda_builds':0,'new_model_calls':0,'new_event_samples_this_install':0,'sealed_reexecuted':False,'production_release':False,'remaining':'current SM86 operator stability and final evidence-qualified release; RC is development-only'};(out/'verification.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v for k,v in audit.items() if k!='case_checks'}))

if __name__=='__main__':main()
