"""No-GPU Docker build through the pinned KernelBench/Torch path, for reviewed CUDA bundles."""
import hashlib,json,os,shutil
from pathlib import Path
from .bounded_io import capture,confirmed_removed
from .protocol import HEADER,META_LIMIT,decode_frame
from .build_artifacts import ARCHIVE_LIMIT,decode_artifacts
from .runner import IMAGE,command_result,digest

CUDA_NAMES={'cublas_sm86.cu','wmma_sm86.cu','gemm_contract.cuh'}


def cuda_identity(files):
    if set(files)!=CUDA_NAMES or any(not isinstance(data,bytes) or not 0<len(data)<=256*1024 for data in files.values()):raise ValueError('exactly three bounded CUDA source files required')
    return hashlib.sha256(json.dumps({name:hashlib.sha256(data).hexdigest() for name,data in sorted(files.items())},sort_keys=True).encode()).hexdigest()


class DockerBuilder:
    def __init__(self,project_root,deps_root):
        self.root=Path(project_root).resolve();self.deps=Path(deps_root).resolve()
        if not (self.deps/'bin/ninja').is_file() or not (self.deps/'pydantic').is_dir():raise ValueError('existing isolated public build dependencies required')

    def build(self,files,work,*,reviewed_sha256,case='compile',host_private_path='/not-mounted'):
        identity=cuda_identity(files)
        if identity!=reviewed_sha256:raise ValueError('reviewed source identity mismatch')
        if case not in {'compile','boundaries','stdout_flood','stderr_flood','disk_limit','timeout','bad_archive'}:raise ValueError('unknown build fixture')
        work=Path(work).resolve();work.mkdir(parents=True,exist_ok=False);work.chmod(0o700)
        source=work/'source';source.mkdir(mode=0o755)
        upstream=self.root/'provenance/upstream-reviews/20261121-p1-02/ScalingIntelligence__KernelBench/src/kernelbench'
        fixed={'evaluators/gemm_regression.py':self.root/'evaluators/gemm_regression.py'}
        fixed.update({'upstream/kernelbench/'+path.name:path for path in upstream.glob('*.py')})
        for name in ['build_worker.py','protocol.py','build_artifacts.py']:fixed['worker/'+name]=Path(__file__).with_name(name)
        for rel,path in fixed.items():
            target=source/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        folder=source/'kernels/gemm';folder.mkdir(parents=True)
        for name,data in files.items():(folder/name).write_bytes(data)
        (source/'readonly-guard').write_text('unchanged');(source/'readonly-guard').chmod(0o666)
        (source/'build-request.json').write_text(json.dumps({'host_private_path':host_private_path,'host_pid':os.getpid()}))
        before={str(path.relative_to(source)):digest(path) for path in source.rglob('*') if path.is_file()}
        deps_before={str(path.relative_to(self.deps)):digest(path) for path in self.deps.rglob('*') if path.is_file() and '__pycache__' not in path.parts}
        import uuid
        name='nka-build-bounded-'+uuid.uuid4().hex[:12];build_size='8m' if case=='disk_limit' else '1g'
        cmd=['docker','run','--name',name,'--pull=never','--network=none','--sig-proxy=false','--read-only','--log-driver=none','--cap-drop=ALL','--security-opt=no-new-privileges','--user','65534:65534','--init',
             '--pids-limit=128','--memory=8g','--memory-swap=8g','--cpus=4','--ulimit','fsize=33554432:33554432','--ulimit','core=0:0',
             '--env','HOME=/tmp','--env','PYTHONDONTWRITEBYTECODE=1','--env','NVIDIA_VISIBLE_DEVICES=void','--env','OMP_NUM_THREADS=1','--env','MKL_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1',
             '--env','PATH=/deps/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin','--env','MAX_JOBS=1','--env','TORCH_CUDA_ARCH_LIST=8.6','--env','CUDA_HOME=/usr/local/cuda',
             '--env','TORCH_EXTENSIONS_DIR=/build/extensions','--env','PYTHONPATH=/source:/source/upstream:/deps',
             '--tmpfs','/tmp:rw,nosuid,nodev,noexec,size=512m','--tmpfs',f'/build:rw,nosuid,nodev,exec,size={build_size},mode=1777',
             '--mount',f'type=bind,src={source},dst=/source,readonly','--mount',f'type=bind,src={self.deps},dst=/deps,readonly',
             '--entrypoint','/opt/venv/bin/python',IMAGE,'/source/worker/build_worker.py','--case',case]
        def cleanup(reason):
            info={'inspect':command_result(['docker','inspect','--format','{{json .State}}',name])}
            if reason:info['top']=command_result(['docker','top',name,'-eo','pid,ppid,args'])
            info['remove']=command_result(['docker','rm','-f',name]);after=command_result(['docker','inspect',name]);info['inspect_after']=after;info['removed']=confirmed_removed(after)
            if reason:
                pids=[]
                for line in info['top']['stdout'].splitlines()[1:]:
                    try:pids.append(int(line.split()[0]))
                    except (ValueError,IndexError):pass
                info['observed_host_pids']=pids;info['host_pids_absent']=all(not Path('/proc',str(pid)).exists() for pid in pids)
            return info
        record=capture(cmd,work/'capture',timeout=2 if case=='timeout' else 1200,stdout_limit=HEADER.size+META_LIMIT+ARCHIVE_LIMIT,stderr_limit=1024*1024,cleanup=cleanup)
        record.update(command=cmd,case=case,source_identity=identity,source_hashes=before,dependency_hashes=deps_before,
          source_unchanged=all(digest(source/rel)==sha for rel,sha in before.items()),deps_unchanged=all(digest(self.deps/rel)==sha for rel,sha in deps_before.items()),status='REJECT',container=name)
        artifact=None
        if record['exit']==0 and record['reason'] is None:
            try:
                meta,payload=decode_frame(work/'capture/stdout.bin',ARCHIVE_LIMIT);record['metadata']=meta
                if meta.get('kind')!='build-result-v1' or meta.get('gpu_available') is not False or meta.get('uid')!=65534:raise ValueError('build metadata mismatch')
                decoded=decode_artifacts(payload);binary_sha=hashlib.sha256(decoded['nka_gemm_v1.so']).hexdigest()
                if binary_sha!=meta.get('binary_sha256'):raise ValueError('build artifact hash mismatch')
                result=work/'artifacts';result.mkdir()
                for rel,data in decoded.items():(result/rel).write_bytes(data)
                artifact=result/'nka_gemm_v1.so';record.update(status='BUILT_UNVALIDATED',binary_sha256=binary_sha,artifact_hashes={rel:hashlib.sha256(data).hexdigest() for rel,data in decoded.items()})
            except (OSError,ValueError,EOFError) as error:record['rejection']=str(error)[:512]
        else:record['rejection']=record['reason'] or 'builder_failure'
        if not record['cleanup']['removed'] or not record['source_unchanged'] or not record['deps_unchanged']:
            record.update(status='REJECT',rejection='lifecycle/protected data failure');artifact=None
        (work/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        return artifact,record
