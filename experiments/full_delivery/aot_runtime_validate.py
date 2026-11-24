"""Real container checks for reviewed AOT operators; independent CPU references stay here."""
import argparse,copy,hashlib,io,json,sys,time
from pathlib import Path
import numpy as np
import torch
from kernel_agent.operators import references as ref
from kernel_agent.operators.array_io import decode_array,encode_array
from kernel_agent.operators.runner import OperatorRunner
from kernel_agent.operators.runtime_contract import validate_arrays


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();a.work.mkdir(exist_ok=False);torch.set_num_threads(1);torch.manual_seed(774);manifest=json.loads((a.bundle/'aot-manifest.json').read_text());bid=manifest['bundle_id'];runner=OperatorRunner(a.bundle,bid);records=[]
    def request(op,**kw):return {'operation':op,'bundle_id':bid,'fault':None,**kw}
    def checked(label,req,arrays,expected):
        actual,record=runner.run(req,arrays,a.work/label);good=actual is not None
        errors={}
        if actual is not None:
            for name,value in actual.items():
                target=expected[name];mask=np.isfinite(target);delta=np.abs(value.astype(np.float64)[mask]-target[mask]);atol,rtol=(2e-6,2e-5) if req['operation']=='softmax' else (2e-3,2e-3);errors[name]=float(delta.max()) if delta.size else 0.;good=good and bool(np.all(delta<=atol+rtol*np.abs(target[mask]))) and bool(np.array_equal(np.isneginf(value),np.isneginf(target)))
        assert record['cleanup']['removed'] and record['protected_unchanged']
        row={'label':label,'correct':bool(good),'status':record['status'],'max_abs_errors':errors,'record':record};records.append(row);(a.work/'checks.json').write_text(json.dumps(records,indent=2)+'\n');return good,record
    cases=[]
    for dtype,width in [(np.float16,129),(np.float32,2049)]:
        x=(torch.randn(7,width)*20).numpy().astype(dtype);x[0]=0;x[1]=10000;x[2]=-10000;req=request('softmax',rows=7,width=width,dtype=str(np.dtype(dtype)));expected={'output.npy':torch.softmax(torch.from_numpy(x).double(),-1).numpy()};cases.append(('softmax-'+str(np.dtype(dtype)),req,{'x.npy':x},expected))
    cpu=[(torch.randn(shape)*.25).half() for shape in [(2,7,3,64),(2,19,3,64),(2,19,3,64)]];mask=torch.ones(7,19,dtype=torch.bool);mask[0]=False;mask[:,3::4]=False;arrays={n:x.numpy() for n,x in zip(['q.npy','k.npy','v.npy'],cpu)};arrays['mask.npy']=mask.numpy();expected={'output.npy':ref.attention(*(x.permute(0,2,1,3) for x in cpu),causal=True,mask=mask).permute(0,2,1,3).numpy()};cases.append(('attention-mask-causal',request('attention',batch=2,sq=7,sk=19,heads=3,dim=64,causal=True,mask=True),arrays,expected))
    lengths=[0,17,33];h=128;qa=(torch.randn(3,h,512)*.25).half();qr=(torch.randn(3,h,64)*.25).half();c=(torch.randn(3,33,512)*.25).half();kr=(torch.randn(3,33,64)*.25).half();pages=torch.zeros(5,16,512,dtype=torch.float16);rp=torch.zeros(5,16,64,dtype=torch.float16);ptr=[0,0,2,5];indices=[3,1,4,0,2]
    for bi,n in enumerate(lengths):
        for t in range(n):page=indices[ptr[bi]+t//16];slot=t%16;pages[page,slot]=c[bi,t];rp[page,slot]=kr[bi,t]
    arrays={name:x.numpy() for name,x in zip(['q_abs.npy','q_rope.npy','cache.npy','rope_cache.npy'],[qa,qr,pages,rp])};arrays.update({'indptr.npy':np.array(ptr,dtype=np.int32),'indices.npy':np.array(indices,dtype=np.int32)});z,lse=ref.mla_core(qa,qr,c,kr,lengths,192**-.5);req=request('mla',lengths=lengths,heads=h,page_size=16,splits=4,num_pages=5);expected={'z.npy':z.numpy(),'lse.npy':lse.numpy()};cases.append(('mla-split4-h128',req,arrays,expected))
    for label,r,arrays,expected in cases:
        good,_=checked(label,r,arrays,expected)
        if not good:raise RuntimeError(label+' failed; raw bounded evidence retained')
    # The constant fixture must be rejected by the external oracle/contract.
    wrong=copy.deepcopy(cases[-1][1]);wrong['fault']='constant';good,_=checked('mla-constant-rejected',wrong,cases[-1][2],cases[-1][3]);assert not good
    for fault in ['crash','timeout']:
        r=copy.deepcopy(cases[-1][1]);r['fault']=fault;actual,record=runner.run(r,cases[-1][2],a.work/('fault-'+fault));assert actual is None and record['cleanup']['removed'] and record['protected_unchanged'];records.append({'label':'fault-'+fault,'expected_failure':True,'record':record})
        good,_=checked('recovery-'+fault,*cases[-1][1:]);assert good
    # CPU-only malformed NPY and admission checks run without starting a GPU worker.
    negative=0
    base=np.zeros((2,3),np.float16)
    for target_shape,target_dtype in [((3,2),'float16'),((2,3),'float32')]:
        try:decode_array(encode_array(base),target_shape,target_dtype)
        except ValueError:negative+=1
        else:raise AssertionError('forged NPY contract accepted')
    bad=copy.deepcopy(cases[0][2]);bad['x.npy'][0,0]=np.nan
    try:validate_arrays(cases[0][1],bad)
    except ValueError:negative+=1
    else:raise AssertionError('nonfinite input accepted')
    result={'status':'PASS','bundle_id':bid,'successful_outputs':6,'expected_failed_attempts':3,'cpu_negative_checks':negative,'records':records,'reference_not_mounted':True,'model_credentials_not_transferred':True,'aot_native_binaries':len(manifest['aot_binaries']),'new_model_calls':0,'performance_sampling':False,'production_release':False};(a.work/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','bundle_id','successful_outputs','expected_failed_attempts','cpu_negative_checks']}))


if __name__=='__main__':main()
