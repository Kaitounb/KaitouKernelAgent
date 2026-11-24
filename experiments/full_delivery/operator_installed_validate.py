"""Independent installed CLI checks from an empty cwd; no Event timing or model calls."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from kernel_agent.operators import references as ref


def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();a.work.mkdir(exist_ok=False);empty=a.work/'empty-cwd';empty.mkdir();torch.set_num_threads(1);torch.manual_seed(882);bid=json.loads((a.bundle/'aot-manifest.json').read_text())['bundle_id'];records=[]
    import kernel_agent
    assert kernel_agent.__version__=='0.1.0.dev6' and Path(kernel_agent.__file__).is_relative_to(a.prefix)
    def command(label,args):
        start=time.monotonic();r=subprocess.run([sys.executable,'-m','kernel_agent.operators.cli',*args],cwd=empty,capture_output=True,text=True,timeout=200);(a.work/(label+'.stdout')).write_text(r.stdout);(a.work/(label+'.stderr')).write_text(r.stderr);records.append({'label':label,'exit':r.returncode,'wall_seconds':time.monotonic()-start});return r
    assert command('inspect',['inspect','--bundle',str(a.bundle)]).returncode==0
    def request(op,**kw):return {'operation':op,'bundle_id':bid,'fault':None,**kw}
    cases=[]
    x=(torch.randn(5,129)*20).half();cases.append(('softmax',request('softmax',rows=5,width=129,dtype='float16'),{'x.npy':x.numpy()},{'output.npy':torch.softmax(x.double(),-1).numpy()}))
    q,k,v=[(torch.randn(shape)*.25).half() for shape in [(1,3,2,64),(1,17,2,64),(1,17,2,64)]];mask=torch.ones(3,17,dtype=torch.bool);mask[0]=False;arrays={n:x.numpy() for n,x in zip(['q.npy','k.npy','v.npy'],[q,k,v])};arrays['mask.npy']=mask.numpy();expected=ref.attention(*(x.permute(0,2,1,3) for x in [q,k,v]),causal=True,mask=mask).permute(0,2,1,3).numpy();cases.append(('attention',request('attention',batch=1,sq=3,sk=17,heads=2,dim=64,causal=True,mask=True),arrays,{'output.npy':expected}))
    lengths=[0,17,33];h=128;qa,qr,c,kr=[(torch.randn(shape)*.25).half() for shape in [(3,h,512),(3,h,64),(3,33,512),(3,33,64)]];ptr=[0,0,2,5];indices=[3,1,4,0,2];pages=torch.zeros(5,16,512,dtype=torch.float16);rp=torch.zeros(5,16,64,dtype=torch.float16)
    for bi,n in enumerate(lengths):
        for t in range(n):physical=indices[ptr[bi]+t//16];slot=t%16;pages[physical,slot]=c[bi,t];rp[physical,slot]=kr[bi,t]
    arrays={n:x.numpy() for n,x in zip(['q_abs.npy','q_rope.npy','cache.npy','rope_cache.npy'],[qa,qr,pages,rp])};arrays.update({'indptr.npy':np.array(ptr,np.int32),'indices.npy':np.array(indices,np.int32)});z,lse=ref.mla_core(qa,qr,c,kr,lengths,192**-.5);mla=('mla',request('mla',lengths=lengths,heads=h,page_size=16,splits=4,num_pages=5),arrays,{'z.npy':z.numpy(),'lse.npy':lse.numpy()});cases.append(mla)
    checks=[]
    for index,(label,req,arrays,expected) in enumerate(cases+[mla,mla]):
        fault=[None,None,None,'crash','timeout'][index];label=label+('-'+fault if fault else '');folder=a.work/label;folder.mkdir();inputs=folder/'inputs';inputs.mkdir();(folder/'request.json').write_text(json.dumps(req))
        for name,value in arrays.items():np.save(inputs/name,value,allow_pickle=False)
        args=['run','--bundle',str(a.bundle),'--request',str(folder/'request.json'),'--inputs',str(inputs),'--output',str(folder/'outputs'),'--work',str(folder/'run')]
        if fault:args+=['--inject-failure',fault]
        r=command(label,args);assert r.returncode==0;execution=json.loads((folder/'run/execution.json').read_text());assert execution['fallback_used']==bool(fault)
        assert len(execution['attempts'])==(2 if fault else 1) and all(x['cleanup']['removed'] and x['protected_unchanged'] for x in execution['attempts']);errors={}
        for name,target in expected.items():
            actual=np.load(folder/'outputs'/name,allow_pickle=False);finite=np.isfinite(target);delta=np.abs(actual.astype(np.float64)[finite]-target[finite]);atol,rtol=(2e-6,2e-5) if req['operation']=='softmax' else (2e-3,2e-3);assert np.array_equal(np.isneginf(actual),np.isneginf(target)) and bool(np.all(delta<=atol+rtol*np.abs(target[finite])));errors[name]=float(delta.max()) if delta.size else 0.
        checks.append({'label':label,'independent_reference':'CPU FP64','max_abs_errors':errors,'attempts':len(execution['attempts']),'fallback':bool(fault)})
    result={'status':'PASS','installed_module':kernel_agent.__file__,'version':kernel_agent.__version__,'cli_commands':records,'checks':checks,'native_attempts':7,'new_model_calls':0,'performance_sampling':False,'production_release':False};(a.work/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','version','native_attempts','new_model_calls','production_release']}))


if __name__=='__main__':main()
