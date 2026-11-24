"""Finite trusted CUDA work followed by host timeout; no GPU-hang claim."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def gpu_check(binary):
    import torch
    spec = importlib.util.spec_from_file_location('nka_gemm_v1',binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cpu = torch.arange(32*48,dtype=torch.float32).reshape(32,48).remainder(7).sub_(3).half()
    b_cpu = torch.arange(48*16,dtype=torch.float32).reshape(48,16).remainder(5).sub_(2).half()
    expected = (cpu.double() @ b_cpu.double()).float()
    a,b = cpu.cuda(),b_cpu.cuda()
    for function in (module.gemm_cublas,lambda a,b:module.gemm_template(a,b,4)):
        torch.testing.assert_close(function(a,b).cpu(),expected,rtol=1e-4,atol=3e-5)
    torch.testing.assert_close(a.cpu(),cpu,rtol=0,atol=0)
    return {'status':'PASS','cpu_reference':True,'finite_cuda_work_completed':True}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary',type=Path,required=True)
    ap.add_argument('--sha256',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--worker',action='store_true')
    args=ap.parse_args()
    if hashlib.sha256(args.binary.read_bytes()).hexdigest()!=args.sha256:
        raise ValueError('trusted binary hash mismatch')
    if args.worker:
        checked=gpu_check(args.binary)
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'])
        def terminate(signum,frame):
            child.terminate()
            child.wait(timeout=3)
            sys.exit(128+signum)
        signal.signal(signal.SIGTERM,terminate)
        args.output.write_text(json.dumps({'pid':os.getpid(),'pgid':os.getpgrp(),'child_pid':child.pid,'gpu':checked}))
        time.sleep(120)
        return
    if args.output.exists():
        ap.error('Use a new output')
    ready=args.output.with_suffix('.ready.json')
    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--binary',str(args.binary),'--sha256',args.sha256,'--output',str(ready)]
    p=subprocess.Popen(command,start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    report={'status':'FAIL','scope':'host wait after finite CUDA; trusted same-UID process group; no isolation or GPU hang guarantee','command':command}
    try:
        deadline=time.monotonic()+30
        while not ready.exists() and p.poll() is None and time.monotonic()<deadline:
            time.sleep(.05)
        if not ready.exists():
            raise RuntimeError('worker did not reach finite CUDA check')
        state=json.loads(ready.read_text());report['ready']=state
        if state['pgid']!=p.pid:
            raise RuntimeError('worker not in dedicated process group')
        try:
            p.communicate(timeout=.3)
        except subprocess.TimeoutExpired:
            report['rejected']='host_timeout';os.killpg(p.pid,signal.SIGTERM)
        else:
            raise RuntimeError('fault exited without timing out')
        stdout,stderr=p.communicate(timeout=5)
        report.update(exit=p.returncode,stdout=stdout,stderr=stderr)
        for _ in range(30):
            remaining=[pid for pid in [state['pid'],state['child_pid']] if Path(f'/proc/{pid}').exists()]
            if not remaining: break
            time.sleep(.05)
        report['remaining_pids']=remaining
        if remaining:
            raise RuntimeError('processes remained')
        recover_cmd=[sys.executable,'-c','from pathlib import Path; import json; from timeout_recovery import gpu_check; print(json.dumps(gpu_check(Path('+repr(str(args.binary))+'))))']
        recovery=subprocess.run(recover_cmd,cwd=Path(__file__).parent,text=True,capture_output=True,timeout=30)
        report['recovery']={'command':recover_cmd,'exit':recovery.returncode,'stdout':recovery.stdout,'stderr':recovery.stderr}
        if recovery.returncode:
            raise RuntimeError('fresh process recovery failed')
        report['incumbent_binary_unchanged']=hashlib.sha256(args.binary.read_bytes()).hexdigest()==args.sha256
        if not report['incumbent_binary_unchanged']:
            raise RuntimeError('trusted binary changed')
        report['status']='PASS'
    finally:
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGKILL);p.communicate(timeout=5)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'])


if __name__=='__main__':
    main()
