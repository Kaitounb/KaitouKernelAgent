"""Two named Torch extensions, same mature importlib loader and unchanged paired Event callable.
No reference output, private ledger or promotion policy is available in this process.
"""
import argparse,hashlib,importlib.util,io,json,os,sys,types
from pathlib import Path
import numpy as np
import torch
from fixtures import make_inputs
from source_worker import check
from source_packets import pack
from protocol import encode_frame


def load_named(path,expected,role):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('pair artifact digest mismatch')
    package='_nka_'+role;parent=types.ModuleType(package);parent.__path__=[];sys.modules[package]=parent
    name=package+'.nka_gemm_v1';spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);a=p.parse_args();req=json.loads(a.request.read_text());torch.set_num_threads(1)
    if torch.__version__!='2.14.0+cu130' or torch.cuda.get_device_capability(0)!=(8,6):raise ValueError('unsupported pair runtime/device')
    digests={'incumbent':req['incumbent_sha256'],'candidate':req['candidate_sha256']}
    modules={role:load_named('/artifact/'+role+'/nka_gemm_v1.so',digest,role) for role,digest in digests.items()}
    if modules['incumbent'] is modules['candidate'] or modules['incumbent'].gemm_template is modules['candidate'].gemm_template:raise ValueError('extensions collapsed into one module/function')
    backends=req.get('diagnostic_backends',{'incumbent':'template','candidate':'template'})
    if backends not in [{'incumbent':'template','candidate':'template'},{'incumbent':'template','candidate':'cublas'}]:raise ValueError('unsupported pair backend map')
    models={role:((lambda x,y,module=module:module.gemm_cublas(x,y)) if backends[role]=='cublas' else (lambda x,y,module=module:module.gemm_template(x,y,4))) for role,module in modules.items()}
    mode=req['mode'];role=req.get('role','candidate')
    if role not in models:raise ValueError('unsupported QA role')
    report={'status':'PASS','mode':mode,'binary_sha256':digests[role] if mode=='qa' else digests['candidate'],'candidate_binary_sha256':digests['candidate'],'incumbent_binary_sha256':digests['incumbent'],'pair_sha256':req['pair_sha256'],
            'torch':str(torch.__version__),'cuda':torch.version.cuda,'capability':[8,6],'cpu_reference_available':False,'module_identity_distinct':True,'function_identity_distinct':True,
            'module_paths':{r:str(Path(m.__file__).resolve()) for r,m in modules.items()},'module_names':{r:m.__spec__.name for r,m in modules.items()},'backends':backends}
    if mode=='qa':
        report['executed_role']=role;files={};rows=[]
        for case in req['cases']:
            output,row=check(models[role],case);raw=io.BytesIO();np.save(raw,output,allow_pickle=False);files[case['id']+'.npy']=raw.getvalue();rows.append(row)
        report['checks']=rows;payload=pack(report,files)
    elif mode in ['aa','ab']:
        from source_timing import paired_witness
        report.update(session=req['session'],graph_launches=32,replays=12)
        if mode=='aa':
            report['checks_before']=[check(models['candidate'],c)[1] for c in req['cases']];report['candidate_aa']=[]
            for workload in req['workloads']:
                for count in [1,16]:
                    row=paired_witness(models['candidate'],models['candidate'],workload['shape_mnk'],count,req['session']);row['workload']=workload['id'];report['candidate_aa'].append(row)
            report['checks_after']=[check(models['candidate'],c)[1] for c in req['cases']]
        else:
            report['checks_before']={r:[check(model,c)[1] for c in req['cases']] for r,model in models.items()};report['measurements']=[]
            for workload in req['workloads']:
                for count in [1,16]:
                    row=paired_witness(models['incumbent'],models['candidate'],workload['shape_mnk'],count,req['session']);row['workload']=workload['id'];report['measurements'].append(row)
            report['checks_after']={r:[check(model,c)[1] for c in req['cases']] for r,model in models.items()}
        payload=json.dumps(report,allow_nan=False).encode()
    else:raise ValueError('unknown pair mode')
    sys.stdout.buffer.write(encode_frame({'kind':'source-eval-v1','mode':mode,'uid':os.getuid(),'binary_sha256':digests['candidate'],'incumbent_sha256':digests['incumbent'],'pair_sha256':req['pair_sha256']},payload));sys.stdout.buffer.flush()


if __name__=='__main__':main()
