"""Reference-free low-privilege NCU collector using the existing bounded runtime."""
import json,re,shutil,uuid
from pathlib import Path
from kernel_agent.runner import IMAGE,digest,command_result
from kernel_agent.bounded_io import capture,confirmed_removed
from kernel_agent.protocol import decode_frame,HEADER,META_LIMIT
from .source_packets import unpack


def run(binary,binary_sha,shape,metrics,work,collector_source,*,kernel_regex='.*gemm_wmma.*'):
    if digest(binary)!=binary_sha or not re.fullmatch('[0-9a-f]{64}',binary_sha):raise ValueError('profile binary differs')
    if len(shape)!=3 or any(type(x) is not int or not 16<=x<=4096 or x%16 for x in shape):raise ValueError('unsupported WMMA profile shape')
    if not metrics or any(not re.fullmatch('(sm|smsp|dram|lts|gpu|launch)__[A-Za-z0-9_.]+',x) for x in metrics):raise ValueError('invalid metric list')
    if kernel_regex not in ['.*gemm_wmma.*','__fixed_no_target__']:raise ValueError('unreviewed kernel filter')
    work=Path(work).resolve();work.mkdir(parents=True,exist_ok=False);work.chmod(0o700);artifact=work/'artifact';inputs=work/'inputs';artifact.mkdir();inputs.mkdir();pkg=Path(__file__).resolve().parents[1]
    for name in ['__init__.py','protocol.py','optimization/__init__.py','optimization/source_packets.py','optimization/profile_target.py','optimization/profile_worker.py']:
        p=artifact/'kernel_agent'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(pkg/name,p)
    for name in ['fixtures.py','source_worker.py','source_packets.py']:shutil.copyfile(pkg/'optimization'/name,artifact/name)
    shutil.copyfile(pkg/'protocol.py',artifact/'protocol.py');p=artifact/'experiments/p1_04/ncu_adapter.py';p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(collector_source,p);(artifact/'experiments/__init__.py').write_text('');(p.parent/'__init__.py').write_text('');(artifact/'candidate').mkdir();shutil.copyfile(binary,artifact/'candidate/nka_gemm_v1.so');(inputs/'request.json').write_text(json.dumps({'binary_sha256':binary_sha,'shape_mnk':shape,'metrics':metrics,'kernel_regex':kernel_regex}))
    for folder in [artifact,inputs]:
        for p in folder.rglob('*'):p.chmod(0o755 if p.is_dir() else 0o644)
        folder.chmod(0o755)
    hashes={str(p.relative_to(work)):digest(p) for f in [artifact,inputs] for p in f.rglob('*') if p.is_file()};name='nka-profile-'+uuid.uuid4().hex[:12];ncu=Path('/opt/nvidia/nsight-compute/2026.3.1');cmd=['docker','run','--name',name,'--pull=never','--network=none','--sig-proxy=false','--read-only','--log-driver=none','--cap-drop=ALL','--security-opt=no-new-privileges','--user','65534:65534','--init','--pids-limit=32','--memory=4g','--memory-swap=4g','--cpus=2','--env','HOME=/tmp','--env','PYTHONDONTWRITEBYTECODE=1','--env','PYTHONPATH=/artifact','--env','OMP_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1','--env','MKL_NUM_THREADS=1','--tmpfs','/tmp:rw,nosuid,nodev,noexec,size=256m,mode=1777','--mount',f'type=bind,src={artifact},dst=/artifact,readonly','--mount',f'type=bind,src={inputs},dst=/inputs,readonly','--mount',f'type=bind,src={ncu},dst={ncu},readonly','--gpus','device=0','--entrypoint','/opt/venv/bin/python',IMAGE,'-m','kernel_agent.optimization.profile_worker']
    def cleanup(reason):
        before=command_result(['docker','inspect','--format','{{json .State}}',name]);remove=command_result(['docker','rm','-f',name]);after=command_result(['docker','inspect',name]);return {'before':before,'remove':remove,'after':after,'removed':confirmed_removed(after)}
    limits={'execution.json':1024*1024,'metrics.csv':2*1024*1024,'report.ncu-rep':12*1024*1024,'validation.json':65536};bound=sum(limits.values())+2*1024*1024;record=capture(cmd,work/'capture',timeout=240,stdout_limit=HEADER.size+META_LIMIT+bound,stderr_limit=1024*1024,cleanup=cleanup);record.update(command=cmd,protected_hashes=hashes,protected_unchanged=all(digest(work/n)==h for n,h in hashes.items()),status='UNAVAILABLE');files=None
    if record['exit']==0 and record['reason'] is None and record['cleanup']['removed'] and record['protected_unchanged']:
        try:
            meta,packet=decode_frame(work/'capture/stdout.bin',bound);record['metadata']=meta;names=meta.get('artifact_names')
            if not isinstance(names,list) or not names or any(not isinstance(n,str) or n not in limits for n in names) or len(set(names))!=len(names):raise ValueError('profile artifact names differ')
            report,received=unpack(packet,{n:limits[n] for n in names})
            if meta.get('uid')!=65534 or meta.get('binary_sha256')!=binary_sha or meta.get('cpu_reference_available') is not False or report!={'format':'bounded-ncu-v1','files':sorted(received)}:raise ValueError('profile witness differs')
            for n,data in received.items():(work/n).write_bytes(data)
            if meta['status']=='PROFILE_READY' and set(received)==set(limits):record['status']='PROFILE_ARTIFACTS_READY';files=received
        except (ValueError,OSError,KeyError) as error:record['rejection']=str(error)[:512]
    (work/'record.json').write_text(json.dumps(record,indent=2)+'\n');return files,record
