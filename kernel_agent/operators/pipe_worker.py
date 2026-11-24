"""Reviewed native AOT operator worker. No references, model client, credentials or promotion."""
import hashlib,importlib.util,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from kernel_agent.protocol import encode_frame
from kernel_agent.optimization.source_packets import pack
from .array_io import encode_array,load_array
from .runtime_contract import specifications,validate_arrays


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boundary_witness():
    status={}
    for line in Path('/proc/self/status').read_text().splitlines():
        key,_,value=line.partition(':');status[key]=value.strip()
    readonly={}
    for name in ['/artifact/kernel_agent/operators/pipe_worker.py','/inputs/request.json','/sdk/aot-manifest.json']:
        try:
            fd=os.open(name,os.O_WRONLY)
        except OSError:readonly[name]=True
        else:os.close(fd);readonly[name]=False
    mounts=Path('/proc/self/mounts').read_text().splitlines()
    root_readonly=any(x.split()[1]=='/' and 'ro' in x.split()[3].split(',') for x in mounts)
    return {'uid':os.getuid(),'gid':os.getgid(),'cap_eff_zero':int(status['CapEff'],16)==0,'cap_bnd_zero':int(status['CapBnd'],16)==0,'no_new_privs':status['NoNewPrivs']=='1','network_only_loopback':set(os.listdir('/sys/class/net'))=={'lo'},'root_readonly':root_readonly,'mounted_files_readonly':all(readonly.values()),'host_control_absent':not Path('/home/dministrator/cuda-lab').exists(),'credentials_env_absent':not any('API_KEY' in k.upper() or 'AUTH_TOKEN' in k.upper() or k=='NKA_CONTROLLER_CANARY' for k in os.environ)}


def main():
    boundary=boundary_witness()
    if boundary['uid']!=65534 or boundary['gid']!=65534 or any(v is not True for k,v in boundary.items() if k not in ['uid','gid']):raise ValueError('operator container boundary differs')
    request_path=Path('/inputs/request.json');raw=request_path.read_bytes()
    if len(raw)>65536:raise ValueError('request exceeds limit')
    request=json.loads(raw);specs,outputs=specifications(request);sdk=Path('/sdk');manifest=json.loads((sdk/'aot-manifest.json').read_text());body={k:v for k,v in manifest.items() if k!='bundle_id'}
    if hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=manifest['bundle_id'] or request['bundle_id']!=manifest['bundle_id']:raise ValueError('native bundle identity differs')
    for relative,digest in manifest['files'].items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or (sdk/path).is_symlink() or sha(sdk/path)!=digest:raise ValueError('native bundle file differs')
    if torch.__version__!=manifest['torch'] or torch.version.cuda!=manifest['cuda'] or list(torch.cuda.get_device_capability())!=manifest['capability']:raise ValueError('unsupported AOT software/device environment')
    arrays={name:load_array('/inputs/'+name,*spec) for name,spec in specs.items()};validate_arrays(request,arrays)
    input_hashes=[hashlib.sha256(arrays[name].tobytes()).hexdigest() for name in sorted(arrays)];stream=torch.cuda.Stream();guards=[];gpu={}
    with torch.no_grad(),torch.cuda.stream(stream):
        torch.cuda._sleep(50000)
        for name,array in arrays.items():
            if array.dtype.kind!='f':continue
            host=torch.from_numpy(array);storage=torch.full((host.numel()+16,),7,dtype=host.dtype,device='cuda');value=storage[8:-8].view(host.shape);value.copy_(host);gpu[name]=value;guards.append(storage)
        if request['fault'] in ['crash','timeout']:
            torch.ones(1,device='cuda').add_(1);stream.synchronize()
            if request['fault']=='crash':raise RuntimeError('reviewed operator crash fixture')
            time.sleep(120)
        if request['operation']=='softmax':
            path=sdk/manifest['softmax_binary'];spec=importlib.util.spec_from_file_location('nka_torch_persistent_v271',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);result={'output.npy':module.softmax(gpu['x.npy'])}
        elif request['operation']=='attention':
            from .flashinfer_backend import attention
            mask=torch.from_numpy(arrays['mask.npy']).cuda() if request['mask'] else None
            result={'output.npy':attention(gpu['q.npy'],gpu['k.npy'],gpu['v.npy'],causal=request['causal'],mask=mask)}
        else:
            from .flashinfer_backend import SplitMLAPlan
            plan=SplitMLAPlan(lengths=request['lengths'],indptr=arrays['indptr.npy'].tolist(),indices=arrays['indices.npy'].tolist(),num_pages=request['num_pages'],page_size=request['page_size'],heads=request['heads'],scale=192**-.5,device=gpu['q_abs.npy'].device,splits=request['splits']);z,lse=plan.run(gpu['q_abs.npy'],gpu['q_rope.npy'],gpu['cache.npy'],gpu['rope_cache.npy']);result={'z.npy':z,'lse.npy':lse}
        if request['fault']=='constant':
            for value in result.values():value.zero_()
        if any(not value.is_contiguous() or any(value.data_ptr()==x.data_ptr() for x in gpu.values()) for value in result.values()):raise ValueError('operator output alias/layout differs')
        copied={name:value.clone().cpu().numpy() for name,value in result.items()};stream.synchronize()
    gpu_hashes={name:hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() for name,value in gpu.items()};guard_ok=all(bool((raw[:8]==7).all()) and bool((raw[-8:]==7).all()) for raw in guards)
    if not guard_ok or any(gpu_hashes[name]!=hashlib.sha256(arrays[name].tobytes()).hexdigest() for name in gpu):raise ValueError('GPU inputs or guards mutated')
    if any(value.shape!=outputs[name][0] or value.dtype!=np.dtype(outputs[name][1]) for name,value in copied.items()):raise ValueError('native output contract differs')
    loaded=[sha(path) for path in torch.ops.loaded_libraries if str(path).startswith('/sdk/flashinfer/data/aot/')]
    if request['operation']=='softmax':loaded.append(sha(sdk/manifest['softmax_binary']))
    files={name:encode_array(value) for name,value in copied.items()};meta={'status':'OUTPUT_READY','operation':request['operation'],'bundle_id':manifest['bundle_id'],'uid':os.getuid(),'input_hashes':input_hashes,'gpu_input_hashes':[gpu_hashes[name] for name in sorted(gpu)],'loaded_native_sha256':sorted(set(loaded)),'guards_ok':True,'producer_consumer_stream':True,'aot_prebuilt':True,'reference_available':False,**{'boundary_'+k:v for k,v in boundary.items()},'torch':str(torch.__version__),'cuda':str(torch.version.cuda),'capability':'.'.join(map(str,torch.cuda.get_device_capability())),'output_fresh_contiguous':True};payload=pack({'format':'reviewed-native-operator-v1','output_names':sorted(files)},files);sys.stdout.buffer.write(encode_frame(meta,payload));sys.stdout.buffer.flush()


if __name__=='__main__':main()
