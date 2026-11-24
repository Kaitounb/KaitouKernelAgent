"""Trusted P3-02 fixtures only. Receives inputs; never receives a reference or gate."""
import argparse,json,os,subprocess,sys,time,hashlib,socket
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);a=p.parse_args()
    req=json.loads(a.request.read_text());torch.set_num_threads(1)
    meta={'uid':os.getuid(),'case':req['case'],'input_files_only':True}
    if req['case']=='boundaries':
        status=Path('/proc/self/status').read_text()
        meta.update(capabilities_zero='CapEff:\t0000000000000000' in status,no_new_privs='NoNewPrivs:\t1' in status,
          interfaces=sorted(p.name for p in Path('/sys/class/net').iterdir()),docker_socket_visible=Path('/var/run/docker.sock').exists(),
          control_canary_in_env=bool(os.getenv('NKA_CONTROL_CANARY')),host_private_visible=Path(req['host_private_path']).exists(),
          host_pid_visible=Path('/proc') .joinpath(str(req['host_pid'])).exists(),
          pids_max=Path('/sys/fs/cgroup/pids.max').read_text().strip(),memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip(),
          cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip())
        try:Path('/artifact/readonly-guard').write_text('altered');meta['readonly_write_denied']=False
        except OSError:meta['readonly_write_denied']=True
        sock=socket.socket();sock.settimeout(.5)
        try:sock.connect(('198.51.100.1',9));meta['external_connect_denied']=False
        except OSError:meta['external_connect_denied']=True
        finally:sock.close()
    a_cpu=np.load('/inputs/a.npy',allow_pickle=False);b_cpu=np.load('/inputs/b.npy',allow_pickle=False)
    a_gpu=torch.from_numpy(a_cpu.copy()).cuda();b_gpu=torch.from_numpy(b_cpu.copy()).cuda()
    if req['case']=='build':
        from evaluators.gemm_regression import load_trusted
        module,model=load_trusted(4)
        out=model(a_gpu,b_gpu)
        binary=Path(module.__file__);meta.update(built_binary=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    else:
        import importlib.util
        binary=Path('/artifact/nka_gemm_v1.so')
        if hashlib.sha256(binary.read_bytes()).hexdigest()!=req['binary_sha256']:raise RuntimeError('trusted artifact hash mismatch')
        spec=importlib.util.spec_from_file_location('nka_gemm_v1',binary);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        if req['case']=='crash':raise RuntimeError('injected host exception')
        if req['case']=='pollution':a_gpu.zero_()
        out=module.gemm_template(a_gpu,b_gpu,4) if req['case']=='template' else module.gemm_cublas(a_gpu,b_gpu)
        if req['case']=='constant':out.zero_()
        if req['case']=='timeout':
            torch.cuda.synchronize();child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'])
            Path('/outputs/ready.json').write_text(json.dumps({'finite_cuda_work_completed':True,'worker_pid':os.getpid(),'child_pid':child.pid}))
            time.sleep(120)
        if req['case']=='symlink':os.symlink('/etc/passwd','/outputs/result.npy');print(json.dumps(meta));return
    np.save('/outputs/result.npy',out.cpu().numpy(),allow_pickle=False)
    Path('/outputs/worker.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta))


if __name__=='__main__':main()
