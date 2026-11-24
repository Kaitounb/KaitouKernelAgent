"""Adapt the validated bounded Docker runner for reviewed native AOT operator bundles."""
import hashlib,json,shutil,uuid
from pathlib import Path
import numpy as np
from kernel_agent.bounded_io import capture,confirmed_removed
from kernel_agent.registry import read_regular
from kernel_agent.protocol import HEADER,META_LIMIT,decode_frame
from kernel_agent.runner import IMAGE,command_result,digest
from kernel_agent.optimization.source_packets import unpack
from .array_io import decode_array
from .runtime_contract import specifications,validate_arrays


class OperatorRunner:
    def __init__(self,bundle,bundle_id):
        self.bundle=Path(bundle).resolve();self.bundle_id=bundle_id;self.manifest=json.loads(read_regular(self.bundle/'aot-manifest.json',1024*1024))
        if not isinstance(self.manifest,dict) or not isinstance(self.manifest.get('files'),dict):raise ValueError('invalid AOT manifest schema')
        body={k:v for k,v in self.manifest.items() if k!='bundle_id'}
        if bundle_id!=self.manifest['bundle_id'] or hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=bundle_id:raise ValueError('AOT bundle binding differs')
        # PYTHONPATH exposes /sdk during Python startup. Reject extra files such
        # as sitecustomize.py or forged .pyc, not only changed listed files.
        actual=set()
        for path in self.bundle.rglob('*'):
            if path.is_symlink() or not (path.is_file() or path.is_dir()):raise ValueError('AOT directory contains unsupported filesystem entry')
            if path.is_file():actual.add(str(path.relative_to(self.bundle)))
        if actual!=set(self.manifest['files'])|{'aot-manifest.json'}:raise ValueError('AOT directory contains unlisted or missing files')
        for relative,sha in self.manifest['files'].items():
            path=Path(relative)
            if path.is_absolute() or '..' in path.parts or (self.bundle/path).is_symlink() or digest(self.bundle/path)!=sha:raise ValueError('AOT source/binary changed')

    def run(self,request,arrays,work):
        if request.get('bundle_id')!=self.bundle_id:raise ValueError('request bundle differs')
        outputs=validate_arrays(request,arrays);work=Path(work).resolve();work.mkdir(parents=True,exist_ok=False);work.chmod(0o700);artifact=work/'artifact';inputs=work/'inputs';artifact.mkdir();inputs.mkdir()
        package=Path(__file__).resolve().parents[1]
        names=['__init__.py','protocol.py','optimization/__init__.py','optimization/source_packets.py','operators/__init__.py','operators/array_io.py','operators/runtime_contract.py','operators/partition.py','operators/flashinfer_backend.py','operators/pipe_worker.py']
        for name in names:
            target=artifact/'kernel_agent'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(package/name,target)
        (inputs/'request.json').write_text(json.dumps(request))
        for name,array in arrays.items():np.save(inputs/name,array,allow_pickle=False)
        for folder in [artifact,inputs]:
            for path in folder.rglob('*'):path.chmod(0o755 if path.is_dir() else 0o644)
            folder.chmod(0o755)
        protected={str(path.relative_to(work)):digest(path) for folder in [artifact,inputs] for path in folder.rglob('*') if path.is_file()};name='nka-native-'+uuid.uuid4().hex[:12]
        cmd=['docker','run','--name',name,'--pull=never','--network=none','--sig-proxy=false','--read-only','--log-driver=none','--cap-drop=ALL','--security-opt=no-new-privileges','--user','65534:65534','--init','--pids-limit=32','--memory=2g','--memory-swap=2g','--cpus=2','--ulimit','fsize=16777216:16777216','--env','HOME=/tmp','--env','PYTHONDONTWRITEBYTECODE=1','--env','PYTHONPATH=/artifact:/sdk','--env','OMP_NUM_THREADS=1','--env','MKL_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1','--env','FLASHINFER_WORKSPACE_BASE=/tmp','--env','TORCH_CUDA_ARCH_LIST=8.6','--env','CUDA_HOME=/usr/local/cuda','--tmpfs','/tmp:rw,nosuid,nodev,noexec,size=128m,mode=1777','--mount',f'type=bind,src={artifact},dst=/artifact,readonly','--mount',f'type=bind,src={inputs},dst=/inputs,readonly','--mount',f'type=bind,src={self.bundle},dst=/sdk,readonly','--gpus','device=0','--entrypoint','/opt/venv/bin/python',IMAGE,'-m','kernel_agent.operators.pipe_worker']
        def cleanup(reason):
            before=command_result(['docker','inspect','--format','{{json .State}}',name]);removed=command_result(['docker','rm','-f',name]);after=command_result(['docker','inspect',name]);return {'before':before,'remove':removed,'after':after,'removed':confirmed_removed(after)}
        limits={key:int(np.prod(shape))*np.dtype(dtype).itemsize+16384 for key,(shape,dtype) in outputs.items()};payload_limit=sum(limits.values())+1024*1024+(len(limits)+1)*512;record=capture(cmd,work/'capture',timeout=8 if request['fault']=='timeout' else 90,stdout_limit=HEADER.size+META_LIMIT+payload_limit,stderr_limit=65536,cleanup=cleanup);record.update(command=cmd,container=name,bundle_id=self.bundle_id,protected_hashes=protected,protected_unchanged=all(digest(work/p)==sha for p,sha in protected.items()),status='REJECT');actual=None
        if record['exit']==0 and record['reason'] is None and record['cleanup']['removed'] and record['protected_unchanged']:
            try:
                meta,payload=decode_frame(work/'capture/stdout.bin',payload_limit);report,files=unpack(payload,limits);record['metadata']=meta;record['report']=report
                expected=[hashlib.sha256(arrays[n].tobytes()).hexdigest() for n in sorted(arrays)];gpu_expected=[hashlib.sha256(arrays[n].tobytes()).hexdigest() for n in sorted(arrays) if arrays[n].dtype.kind=='f']
                if meta.get('status')!='OUTPUT_READY' or meta.get('bundle_id')!=self.bundle_id or meta.get('operation')!=request['operation'] or meta.get('uid')!=65534 or meta.get('input_hashes')!=expected or meta.get('gpu_input_hashes')!=gpu_expected or any(meta.get(k) is not True for k in ['guards_ok','producer_consumer_stream','aot_prebuilt']) or meta.get('reference_available') is not False:raise ValueError('native worker witness differs')
                boundary={k[len('boundary_'):]:v for k,v in meta.items() if k.startswith('boundary_')}
                flags=['cap_eff_zero','cap_bnd_zero','no_new_privs','network_only_loopback','root_readonly','mounted_files_readonly','host_control_absent','credentials_env_absent']
                if set(boundary)!=set(flags)|{'uid','gid'} or boundary.get('uid')!=65534 or boundary.get('gid')!=65534 or any(boundary.get(k) is not True for k in flags) or meta.get('output_fresh_contiguous') is not True or any(meta.get(k)!=self.manifest[k] for k in ['torch','cuda']) or meta.get('capability')!='.'.join(map(str,self.manifest['capability'])):raise ValueError('native boundary/environment witness differs')
                if report!={'format':'reviewed-native-operator-v1','output_names':sorted(outputs)}:raise ValueError('native output report differs')
                native={self.manifest['files'][p] for p in [*self.manifest['aot_binaries'],self.manifest['softmax_binary']]};loaded=meta.get('loaded_native_sha256')
                if not isinstance(loaded,list) or not loaded or any(x not in native for x in loaded):raise ValueError('native AOT load witness missing')
                values={key:decode_array(files[key],*spec) for key,spec in outputs.items()}
                if any(not np.isfinite(value).all() for key,value in values.items() if key!='lse.npy'):raise ValueError('nonfinite native output')
                if 'lse.npy' in values:
                    valid=np.array(request['lengths'])>0;lse=values['lse.npy']
                    if not np.isfinite(lse[valid]).all() or not np.isneginf(lse[~valid]).all():raise ValueError('MLA LSE finite/empty contract differs')
                for key,value in files.items():(work/key).write_bytes(value)
                actual=values;record['status']='OUTPUT_VALID'
            except (OSError,ValueError,KeyError,EOFError) as error:record['rejection']=str(error)[:512]
        if actual is None:record.setdefault('rejection',record['reason'] or 'worker_failure')
        (work/'record.json').write_text(json.dumps(record,indent=2)+'\n');return actual,record
