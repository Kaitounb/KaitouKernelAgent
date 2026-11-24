"""Reserve and run reviewed SM89 public correctness/sanitizer/recovery once; no Event trial."""
import base64,hashlib,json
from tools.ssh_server_gpu import ROOT,execute_python


def main():
    output=ROOT/'evidence/20261123-sm89-cublas-v2';reservation=output/'precheck-reservation';reservation.mkdir(exist_ok=False)
    location=json.loads((output/'location.json').read_text());build=json.loads((output/'build.json').read_text());assert build['exit']==0 and build['result']['status']=='BUILT'
    names=['experiments/full_delivery/sm89_cublas_support.py','experiments/full_delivery/sm89_cublas_host.py']
    files={n:(ROOT/n).read_bytes() for n in names}
    protocol={'scope':'Trusted fixed cuBLAS SM89 public QA/sanitizer/recovery only; no timing','build_result_sha256':hashlib.sha256(json.dumps(build['result'],indent=2).encode()).hexdigest(),'source_files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()},'policy_sha256':hashlib.sha256((ROOT/'workloads/gemm_deployment_stability_v2.json').read_bytes()).hexdigest(),'heldout_read':False,'model_calls':0,'container_isolation_verified':False,'production_release':False}
    files['precheck-protocol.json']=json.dumps(protocol,indent=2).encode()
    (reservation/'protocol.json').write_bytes(files['precheck-protocol.json'])
    for n,v in files.items():
        p=reservation/'sources'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v)
    payload='ROOT='+repr(location['remote_root'])+'\nFILES='+repr({n:base64.b64encode(v).decode() for n,v in files.items()})+'\n'+r'''
import base64,hashlib,json,os,subprocess
from pathlib import Path
root=Path(ROOT);assert not (root/'precheck-protocol.json').exists() and not (root/'precheck-result.json').exists()
for n,v in FILES.items():
 p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v))
p=json.loads((root/'precheck-protocol.json').read_text());assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in p['source_files'].items())
assert hashlib.sha256((root/'build-result.json').read_bytes()).hexdigest()==p['build_result_sha256']
py='/tmp/nka-full-server-dqnrbkzs/venv/bin/python';env={k:v for k,v in os.environ.items() if k in ['HOME','LANG','LD_LIBRARY_PATH']};env.update(PATH='/usr/local/cuda-12.6/bin:'+str(Path(py).parent)+':/usr/local/bin:/usr/bin:/bin',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
with (root/'precheck.stdout').open('w') as out,(root/'precheck.stderr').open('w') as err:r=subprocess.run([py,str(root/'experiments/full_delivery/sm89_cublas_host.py'),'--root',str(root),'--phase','precheck'],cwd=root,env=env,stdout=out,stderr=err,timeout=1800)
print(json.dumps({'remote_root':str(root),'exit':r.returncode,'result':json.loads((root/'precheck-result.json').read_text()) if (root/'precheck-result.json').exists() else None}))
'''
    (reservation/'driver.py').write_text(payload);r=execute_python(payload,2000,connection_path=ROOT/'.private/server-gpu-next-connection.json');(reservation/'transport.stderr').write_text(r.stderr);(reservation/'transport.stdout').write_text(r.stdout)
    if r.returncode:raise RuntimeError('Precheck transport failed; inspect existing reserved root, do not repeat')
    v=json.loads(r.stdout);(output/'precheck.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:v for k,v in v.items() if k!='result'}|{'result':{k:v['result'][k] for k in ['status','qa_checks','sanitizers','recovery','raw_segments','execution_scope','production_release']} if v['result'] else None}))


if __name__=='__main__':main()
