"""Fixed trusted CUDA/host fault fixtures. No references, model credentials or release state."""
import argparse,errno,hashlib,io,json,os,socket,subprocess,sys,time
from pathlib import Path
from protocol import encode_frame,HEADER,MAGIC


def emit(meta,payload=b''):
    sys.stdout.buffer.write(encode_frame(meta,payload));sys.stdout.buffer.flush()


def host_fixture(case):
    if case in ['stdout_flood','stderr_flood']:
        fd=1 if case=='stdout_flood' else 2
        for _ in range(256):os.write(fd,b'X'*65536)
        return True
    if case=='metadata_nested':
        raw=b'{"nested":{"not":"flat"}}';os.write(1,HEADER.pack(MAGIC,len(raw),0)+raw);return True
    if case=='metadata_forged':os.write(1,HEADER.pack(MAGIC,10**8,0));return True
    if case=='bad_frame':os.write(1,b'invalid frame');return True
    if case=='memory_oom':
        chunks=[]
        for _ in range(24):chunks.append(bytearray(16*1024*1024))
        emit({'oom_did_not_trigger':True});return True
    if case=='pids_limit':
        children=[];denied=False
        try:
            for _ in range(48):children.append(subprocess.Popen(['/bin/sleep','120']))
        except OSError as error:denied=error.errno==errno.EAGAIN
        finally:
            for child in children:child.terminate()
            for child in children:child.wait(timeout=5)
        emit({'pids_denied':denied,'spawned_children':len(children)});return True
    if case=='disk_limit':
        written=0;error_number=0
        try:
            for i in range(24):
                with open('/outputs/fill-'+str(i),'wb') as out:
                    for _ in range(16):out.write(b'X'*65536);written+=65536
        except OSError as error:error_number=error.errno
        emit({'write_errno':error_number,'written_bytes':written,'tmpfs_bytes':8388608});return True
    return False


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);req=json.loads(parser.parse_args().request.read_text());case=req['case']
    if host_fixture(case):return
    import numpy as np
    import torch
    torch.set_num_threads(1)
    if torch.__version__!='2.14.0+cu130' or torch.cuda.get_device_capability(0)!=(8,6):raise RuntimeError('unsupported torch/device')
    meta={'uid':os.getuid(),'case':case,'torch':str(torch.__version__),'device':'sm86'}
    if case=='boundaries':
        status=Path('/proc/self/status').read_text()
        meta.update(capabilities_zero='CapEff:\t0000000000000000' in status,no_new_privs='NoNewPrivs:\t1' in status,
         interfaces=sorted(p.name for p in Path('/sys/class/net').iterdir()),docker_socket_visible=Path('/var/run/docker.sock').exists(),
         control_canary_in_env=bool(os.getenv('NKA_CONTROL_CANARY')),host_private_visible=Path(req['host_private_path']).exists(),
         host_pid_visible=Path('/proc',str(req['host_pid'])).exists(),pids_max=Path('/sys/fs/cgroup/pids.max').read_text().strip(),
         memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip(),cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip())
        for label,path in [('artifact','/artifact/readonly-guard'),('inputs','/inputs/request.json')]:
            try:Path(path).write_text('altered');meta[label+'_write_denied']=False
            except OSError:meta[label+'_write_denied']=True
        sock=socket.socket();sock.settimeout(.3)
        try:sock.connect(('198.51.100.1',9));meta['external_connect_denied']=False
        except OSError:meta['external_connect_denied']=True
        finally:sock.close()
    binary=Path('/artifact/nka_gemm_v1.so')
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=req['binary_sha256']:raise RuntimeError('artifact hash mismatch')
    import importlib.util
    spec=importlib.util.spec_from_file_location('nka_gemm_v1',binary);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    x=np.load('/inputs/a.npy',allow_pickle=False);y=np.load('/inputs/b.npy',allow_pickle=False)
    a=torch.from_numpy(x.copy()).cuda();b=torch.from_numpy(y.copy()).cuda()
    if case=='crash':raise RuntimeError('injected host exception')
    if case=='pollution':a.zero_()
    out=module.gemm_template(a,b,4) if case=='template' else module.gemm_cublas(a,b)
    if case=='constant':out.zero_()
    if case=='timeout':
        torch.cuda.synchronize();subprocess.Popen(['/bin/sleep','120'])
        sys.stderr.write('NKA_FINITE_CUDA_READY\n');sys.stderr.flush();time.sleep(120)
    raw=io.BytesIO();np.save(raw,out.cpu().numpy(),allow_pickle=False)
    emit(meta,raw.getvalue())


if __name__=='__main__':main()
