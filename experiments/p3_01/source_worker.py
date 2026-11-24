"""Fixed source evaluation worker. It never receives a reference, output ledger or credentials."""
import argparse,hashlib,importlib.util,io,json,sys
from pathlib import Path
import numpy as np
import torch
from fixtures import make_inputs,guarded
from protocol import encode_frame
from source_packets import pack


def tensor_hash(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load(binary,expected):
    binary=Path(binary)
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=expected:raise ValueError('source binary mismatch')
    spec=importlib.util.spec_from_file_location('nka_gemm_v1',binary);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def check(model,case):
    shape=case['shape_mnk'];cpu=make_inputs(shape,case['seed'],case['distribution']);stream=torch.cuda.Stream();guards=[]
    with torch.no_grad(),torch.cuda.stream(stream):
        torch.cuda._sleep(200000)
        if case['offset_guard']:
            pairs=[guarded(value,'cuda:0') for value in cpu];guards=[p[0] for p in pairs];a,b=[p[1] for p in pairs]
        else:a,b=[value.cuda() for value in cpu]
        actual=model(a,b)
        if not isinstance(actual,torch.Tensor) or actual.dtype!=torch.float32 or actual.device!=a.device or list(actual.shape)!=[shape[0],shape[1]] or not actual.is_contiguous():raise ValueError('source output contract')
        if torch._C._is_alias_of(actual,a) or torch._C._is_alias_of(actual,b):raise ValueError('source output aliases input')
        consumed=actual.clone();output=consumed.cpu().contiguous().numpy()
        row={**case,'input_hashes':[tensor_hash(a),tensor_hash(b)],'guard_hashes':[tensor_hash(torch.cat((raw[:17],raw[-17:]))) for raw in guards],
             'output_sha256':hashlib.sha256(output.tobytes()).hexdigest(),'fresh_output':True,'producer_consumer_stream':True}
    stream.synchronize();return output,row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);a=parser.parse_args();req=json.loads(a.request.read_text());torch.set_num_threads(1)
    if torch.__version__!='2.14.0+cu130' or torch.cuda.get_device_capability(0)!=(8,6):raise ValueError('unsupported runtime/device')
    module=load('/artifact/candidate/nka_gemm_v1.so',req['candidate_sha256']);model=lambda x,y:module.gemm_template(x,y,4)
    report={'status':'PASS','mode':req['mode'],'binary_sha256':req['candidate_sha256'],'torch':str(torch.__version__),'cuda':torch.version.cuda,'capability':[8,6],'cpu_reference_available':False}
    if req['mode']=='qa':
        files={};rows=[]
        for case in req['cases']:
            output,row=check(model,case);raw=io.BytesIO();np.save(raw,output,allow_pickle=False);files[case['id']+'.npy']=raw.getvalue();rows.append(row)
        report['checks']=rows;payload=pack(report,files)
    elif req['mode']=='aa':
        from source_timing import paired_witness
        report['session']=req['session'];report['graph_launches']=32;report['replays']=12
        report['checks_before']=[check(model,case)[1] for case in req['cases']]
        report['candidate_aa']=[]
        for workload in req['workloads']:
            for count in [1,16]:
                row=paired_witness(model,model,workload['shape_mnk'],count,req['session']);row['workload']=workload['id'];report['candidate_aa'].append(row)
        report['checks_after']=[check(model,case)[1] for case in req['cases']]
        payload=json.dumps(report,allow_nan=False).encode()
    else:raise ValueError('unknown source mode')
    sys.stdout.buffer.write(encode_frame({'kind':'source-eval-v1','mode':req['mode'],'uid':65534,'binary_sha256':req['candidate_sha256']},payload));sys.stdout.buffer.flush()


if __name__=='__main__':main()
