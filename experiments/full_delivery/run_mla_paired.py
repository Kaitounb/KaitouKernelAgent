"""One-shot SDK calibration/Graph checks on authorized resources; lossless evidence."""
import argparse,base64,hashlib,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',choices=['3070','server'],required=True);p.add_argument('--phase',choices=['graph','calibration','evaluate'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--page-size',type=int,choices=[1,16,64],default=16);p.add_argument('--splits',type=int,choices=[1,2,4],default=1);p.add_argument('--sdk-prefix');a=p.parse_args();a.output.mkdir(exist_ok=False)
    names=['kernel_agent/operators/__init__.py','kernel_agent/operators/references.py','kernel_agent/operators/flashinfer_backend.py','kernel_agent/operators/partition.py','kernel_agent/operators/measurement.py','kernel_agent/optimization/identity.py','experiments/full_delivery/mla_graph_probe.py','experiments/full_delivery/mla_paired.py','workloads/mla_development_v1.json','workloads/mla_contract_v1.json','workloads/mla_measurement_v2.json','experiments/full_delivery/run_mla_paired.py']
    files={n:(ROOT/n).read_bytes() for n in names};files['kernel_agent/__init__.py']=b''
    hashes={n:hashlib.sha256(v).hexdigest() for n,v in files.items()};freeze={'files':hashes,'target':a.target,'phase':a.phase,'config':{'page_size':a.page_size,'splits':a.splits},'new_model_calls':0,'production_release':False,'sdk_prefix':a.sdk_prefix};freeze['id']=hashlib.sha256(json.dumps(freeze,sort_keys=True,separators=(',',':')).encode()).hexdigest();(a.output/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    files['freeze.json']=(a.output/'freeze.json').read_bytes()
    payload='FILES='+repr({n:base64.b64encode(v).decode() for n,v in files.items()})+'\n'+r'''
import base64,fcntl,hashlib,io,json,os,subprocess,tempfile,threading,time,zipfile
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='nka-mla-paired-'))
for n,v in FILES.items():
 p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v))
freeze=json.loads((root/'freeze.json').read_text());target=freeze['target'];phase=freeze['phase'];config=freeze['config'];py='/home/dministrator/cuda-lab/.venv/bin/python' if target=='3070' else '/tmp/nka-full-server-dqnrbkzs/venv/bin/python';cuda='/usr/local/cuda-13.0' if target=='3070' else '/usr/local/cuda-12.6';jit='/tmp/nka-mla-measurement-v1-'+target+'-jit';env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','LD_LIBRARY_PATH']};env.update(PYTHONPATH=str(root)+(':/tmp/nka-flashinfer-sdk-local-m0583epr/prefix-cxx20' if target=='3070' else ''),PATH=str(Path(py).parent)+':'+cuda+'/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin',CUDA_HOME=cuda,TORCH_CUDA_ARCH_LIST='8.6' if target=='3070' else '8.9',FLASHINFER_WORKSPACE_BASE=jit,FLASHINFER_CUDA_ARCH_LIST='8.6' if target=='3070' else '8.9',MAX_JOBS='2',OMP_NUM_THREADS='1')
if freeze.get('sdk_prefix'):
 assert target=='3070'
 manifest=json.loads((Path(freeze['sdk_prefix'])/'aot-manifest.json').read_text());body={k:v for k,v in manifest.items() if k!='bundle_id'};assert hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()==manifest['bundle_id']=='3e97a2b632ea2e189da2ef4000b3a6d0750ec3ef852206e13dcfc71adf3eab32';assert all(hashlib.sha256((Path(freeze['sdk_prefix'])/n).read_bytes()).hexdigest()==h for n,h in body['files'].items())
 env['PYTHONPATH']=str(root)+':'+freeze['sdk_prefix'];env['NKA_FROZEN_SDK_PREFIX']=freeze['sdk_prefix']
def source_hashes():return {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in freeze['files']}
def sdk_hashes():
 probe=subprocess.run([py,'-c','import flashinfer;print(flashinfer.__file__)'],env=env,capture_output=True,text=True,timeout=60)
 if probe.returncode:raise RuntimeError('SDK import unavailable: '+probe.stderr)
 sdk=Path(probe.stdout.strip()).parent
 required={'mla.py':'b78276b95b35f6a0a309554df52820032146b24015d5c30349c92cfcd671ae41','data/include/flashinfer/attention/mla.cuh':'8bcfdfc5d030f6c8ee60909981ececec227798b002143b23886379a3b6fd77cf'}
 assert all(hashlib.sha256((sdk/n).read_bytes()).hexdigest()==h for n,h in required.items())
 native_root=Path(freeze['sdk_prefix']) if freeze.get('sdk_prefix') else Path(jit)
 return {'source':{str(p.relative_to(sdk)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sdk.rglob('*') if p.is_file() and '__pycache__' not in p.parts},'native':{str(p.relative_to(native_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in native_root.rglob('*.so')}}
gpu_lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(gpu_lock,fcntl.LOCK_EX)
# Compile/capture in an isolated cache before freezing native binaries and before any Event.
warm=subprocess.run([py,'-m','experiments.full_delivery.mla_graph_probe','--work',str(root/'warmup-graph')],cwd=root,env=env,capture_output=True,text=True,timeout=650);(root/'warmup.stdout').write_text(warm.stdout);(root/'warmup.stderr').write_text(warm.stderr);assert warm.returncode==0 and json.loads((root/'warmup-graph/result.json').read_text())['status']=='PASS'
records=[];samples=[];stop=threading.Event();before=source_hashes();sdk_before=sdk_hashes();assert before==freeze['files'] and sdk_before['native'];(root/'sdk-before.json').write_text(json.dumps(sdk_before,indent=2))
def sample(label):
 r=subprocess.run(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader'],env=env,capture_output=True,text=True,timeout=10);row={'phase':label,'time':time.time(),'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr};samples.append(row);return row
def idle():
 count=0
 for _ in range(10):
  r=sample('idle-admission')
  try:good=r['exit']==0 and int(r['stdout'].split(',')[1].strip().split()[0])==0
  except (ValueError,IndexError):good=False
  count=count+1 if good else 0
  if count==3:return True
  time.sleep(1)
 return False
def monitor():
 while not stop.is_set():sample('active-batch');stop.wait(.5)
def run(cmd,label,timeout):
 start=time.monotonic();started=time.time()
 with (root/(label+'.stdout')).open('w') as out,(root/(label+'.stderr')).open('w') as err:
  try:r=subprocess.run(cmd,cwd=root,env=env,stdout=out,stderr=err,timeout=timeout);code=r.returncode
  except subprocess.TimeoutExpired:code='timeout'
 row={'label':label,'command':cmd,'exit':code,'wall_seconds':time.monotonic()-start,'started_at':started,'finished_at':time.time()};records.append(row);return code
status='FAIL';decision=None;t=threading.Thread(target=monitor);t.start()
try:
 if phase=='graph':
  status='PASS' if json.loads((root/'warmup-graph/result.json').read_text())['status']=='PASS' else 'FAIL'
 else:
  aa=[];ab=[];status='PASS'
  # No AB is admitted until all AA sessions and raw numeric checks pass.
  for mode in ['aa']+(['ab'] if phase=='evaluate' else []):
   for session in range(3):
    if not idle():status='GPU_BUSY_UNSAMPLED_REMAINDER';break
    cmd=[py,'-m','experiments.full_delivery.mla_paired','--root',str(root),'--mode',mode,'--session',str(session),'--page-size',str(config['page_size']),'--splits',str(config['splits'])]
    if phase=='calibration':cmd.append('--anchors-only')
    code=run(cmd,mode+'-'+str(session),1200);path=root/(mode+'-'+str(session)+'.json')
    if code!=0 or not path.exists():status='FAIL';break
    report=json.loads(path.read_text());expected_order=['incumbent','candidate'] if session%2==0 else ['candidate','incumbent'];assert report['config']==config and report['mode']==mode and report['session']==session and report['capability']==([8,6] if target=='3070' else [8,9]) and report['graph_launches']==report['replays']==32
    for row in report['rows']:
     assert row['capture_order']==expected_order and row['qa_after_pass'] and row['inputs_and_guards_unchanged']
     assert row['native_hashes'] and all(sdk_before['native'].get(n)==h for n,h in row['native_hashes'].items())
     assert all(len(row['input_hashes'][role])==row['input_count'] and len(row['output_hashes'][role])==32 and len(row['qa_before_errors'][role])==row['input_count'] for role in ['incumbent','candidate'])
     assert row['configs']=={'incumbent':config if mode=='aa' else {'page_size':16,'splits':1},'candidate':config}
    (aa if mode=='aa' else ab).append(report)
    if report['status']!='PASS':status='FAIL';break
   if status!='PASS':break
   if mode=='aa':
    # Run pure gate in the GPU venv (stdlib-only module; no model package).
    checker=root/'numeric_check.py';checker.write_text("import json\nfrom pathlib import Path\nfrom kernel_agent.operators.measurement import assess\nr=Path.cwd();d=json.loads((r/'workloads/mla_development_v1.json').read_text());f=json.loads((r/'freeze.json').read_text());w=[x for x in d['workloads'] if f['phase']!='calibration' or x['id'] in d['anchors']];aa=[json.loads((r/('aa-'+str(i)+'.json')).read_text()) for i in range(3)];ab=[json.loads(p.read_text()) for i in range(3) if (p:=r/('ab-'+str(i)+'.json')).exists()];(r/'decision.json').write_text(json.dumps(assess(aa,ab,w,d['anchors']),indent=2))\n")
    assert run([py,str(checker)],'aa-numeric-check',30)==0;decision=json.loads((root/'decision.json').read_text())
    if decision['reason']!='AA_PASS_AB_NOT_RUN':status='AA_REJECTED';break
  if status=='PASS' and phase=='evaluate':
   assert run([py,str(root/'numeric_check.py')],'ab-numeric-check',30)==0;decision=json.loads((root/'decision.json').read_text())
finally:stop.set();t.join()
after=source_hashes();sdk_after=sdk_hashes();(root/'sdk-after.json').write_text(json.dumps(sdk_after,indent=2));binding_ok=before==after==freeze['files'] and sdk_before==sdk_after
if not binding_ok:status='INVALID_SOURCE_OR_NATIVE_BINDING'
result={'status':status,'decision':decision,'records':records,'source_hashes_before':before,'source_hashes_after':after,'sdk_unchanged':sdk_before==sdk_after,'freeze_id':freeze['id'],'new_model_calls':0,'performance_claim':False,'production_release':False};(root/'result.json').write_text(json.dumps(result,indent=2));(root/'gpu-monitor.json').write_text(json.dumps(samples,indent=2))
manifest={};blobs={}
for path in root.rglob('*'):
 if path.is_file() and '__pycache__' not in path.parts:
  value=path.read_bytes();digest=hashlib.sha256(value).hexdigest();manifest[str(path.relative_to(root))]={'sha256':digest,'bytes':len(value)};blobs.setdefault(digest,value)
raw=io.BytesIO()
with zipfile.ZipFile(raw,'w',compression=zipfile.ZIP_DEFLATED) as z:
 z.writestr('manifest.json',json.dumps(manifest))
 for digest,value in blobs.items():z.writestr('blobs/'+digest,value)
print(json.dumps({'remote_root':str(root),'result':result,'archive_sha256':hashlib.sha256(raw.getvalue()).hexdigest(),'archive':base64.b64encode(raw.getvalue()).decode()}))
'''
    (a.output/'driver.py').write_text(payload)
    if a.target=='3070':from tools.ssh_3070 import execute_python
    else:from tools.ssh_server_gpu import execute_python
    r=execute_python(payload,7400);(a.output/'transport.stderr').write_text(r.stderr)
    if r.returncode:(a.output/'transport.stdout').write_text(r.stdout);raise RuntimeError('transport failed; collect existing job, never relaunch sampled phase')
    wire=json.loads(r.stdout);raw=base64.b64decode(wire.pop('archive'));assert hashlib.sha256(raw).hexdigest()==wire['archive_sha256'];(a.output/'raw.zip').write_bytes(raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        manifest=json.loads(z.read('manifest.json'))
        for rel,meta in manifest.items():
            n=Path(rel);assert not n.is_absolute() and '..' not in n.parts;v=z.read('blobs/'+meta['sha256']);assert len(v)==meta['bytes'] and hashlib.sha256(v).hexdigest()==meta['sha256'];p=a.output/'remote'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v)
    (a.output/'result.json').write_text(json.dumps(wire,indent=2)+'\n');print(json.dumps({'status':wire['result']['status'],'decision':wire['result']['decision'],'files_verified':len(manifest),'remote_root':wire['remote_root']}))


if __name__=='__main__':main()
