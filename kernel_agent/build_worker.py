"""Reviewed-source builder inside a no-GPU container. No inputs, references or release state."""
import argparse,errno,hashlib,io,json,os,stat,subprocess,sys,time,zipfile
from pathlib import Path
from protocol import encode_frame
from build_artifacts import LIMITS


def send(meta,payload=b''):
    sys.stdout.buffer.write(encode_frame(meta,payload));sys.stdout.buffer.flush()


def fixture(case):
    if case in ['stdout_flood','stderr_flood']:
        for _ in range(256):os.write(1 if case=='stdout_flood' else 2,b'X'*65536)
        return True
    if case=='disk_limit':
        written=0;error_number=0
        try:
            for i in range(24):
                with open('/build/fill-'+str(i),'wb') as out:
                    for _ in range(16):out.write(b'X'*65536);written+=65536
        except OSError as error:error_number=error.errno
        send({'write_errno':error_number,'written_bytes':written});return True
    if case=='timeout':
        subprocess.Popen(['/bin/sleep','120']);sys.stderr.write('NKA_BUILD_CHILD_READY\n');sys.stderr.flush();time.sleep(120);return True
    if case=='bad_archive':send({'kind':'build-result-v1'},b'not a ZIP');return True
    if case=='boundaries':
        status=Path('/proc/self/status').read_text();req=json.loads(Path('/source/build-request.json').read_text())
        meta={'uid':os.getuid(),'capabilities_zero':'CapEff:\t0000000000000000' in status,'no_new_privs':'NoNewPrivs:\t1' in status,
              'interfaces':sorted(x.name for x in Path('/sys/class/net').iterdir()),'host_private_visible':Path(req['host_private_path']).exists(),
              'host_pid_visible':Path('/proc',str(req['host_pid'])).exists(),'control_canary_in_env':bool(os.getenv('NKA_CONTROL_CANARY')),
              'docker_socket_visible':Path('/var/run/docker.sock').exists(),'gpu_device_visible':Path('/dev/nvidia0').exists() or Path('/dev/dxg').exists(),
              'pids_max':Path('/sys/fs/cgroup/pids.max').read_text().strip(),'memory_max':Path('/sys/fs/cgroup/memory.max').read_text().strip(),
              'memory_swap_max':Path('/sys/fs/cgroup/memory.swap.max').read_text().strip(),'cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip()}
        for name,path in [('source','/source/readonly-guard'),('deps','/deps/readonly-guard')]:
            try:Path(path).write_text('altered');meta[name+'_write_denied']=False
            except OSError:meta[name+'_write_denied']=True
        import socket
        sock=socket.socket();sock.settimeout(.3)
        try:sock.connect(('198.51.100.1',9));meta['external_connect_denied']=False
        except OSError:meta['external_connect_denied']=True
        finally:sock.close()
        send(meta);return True
    return False


def read_bounded(path,limit):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as file:
        info=os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0<info.st_size<=limit:raise ValueError('invalid build artifact type/size')
        data=file.read(limit+1)
        if len(data)!=info.st_size:raise ValueError('artifact changed/truncated')
        return data


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',default='compile');args=parser.parse_args()
    if fixture(args.case):return
    if args.case!='compile':raise ValueError('unknown build fixture')
    # Preserve the only result channel, and redirect even native/compiler stdout to bounded stderr.
    result_fd=os.dup(1);os.dup2(2,1)
    from evaluators.gemm_regression import load_trusted
    import torch
    if torch.cuda.is_available():raise RuntimeError('builder unexpectedly has GPU access')
    module,_=load_trusted(4);binary=Path(module.__file__).resolve()
    if not binary.is_relative_to('/build/extensions'):raise ValueError('artifact outside bounded cache')
    files={name:read_bounded(binary.parent/name,limit) for name,limit in LIMITS.items()}
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as archive:
        for name,data in files.items():archive.writestr(name,data)
    meta={'kind':'build-result-v1','uid':os.getuid(),'gpu_available':torch.cuda.is_available(),'torch':str(torch.__version__),
          'cuda':str(torch.version.cuda),'binary_sha256':hashlib.sha256(files['nka_gemm_v1.so']).hexdigest()}
    sys.stdout.flush();os.dup2(result_fd,1);os.close(result_fd);send(meta,stream.getvalue())


if __name__=='__main__':main()
