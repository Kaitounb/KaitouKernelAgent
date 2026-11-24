"""Trusted public installed-operator checks; FP64 references never enter the GPU worker."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from kernel_agent.operators import references as ref


def make_cases(bid):
    torch.set_num_threads(1);torch.manual_seed(1193);cases=[]
    def req(op,**kw):return {'operation':op,'bundle_id':bid,'fault':None,**kw}
    for label,b,sq,sk,h,d,causal,masked in [('attention-mask',2,7,19,3,64,True,True),('attention-prefill128',2,17,17,2,128,False,False),('attention-decode',1,1,129,4,64,False,False)]:
        q,k,v=[(torch.randn(shape)*.25).half() for shape in [(b,sq,h,d),(b,sk,h,d),(b,sk,h,d)]];mask=None;arrays={name:x.numpy() for name,x in zip(['q.npy','k.npy','v.npy'],[q,k,v])}
        if masked:mask=torch.ones(sq,sk,dtype=torch.bool);mask[0]=False;mask[:,3::4]=False;arrays['mask.npy']=mask.numpy()
        expected=ref.attention(*(x.permute(0,2,1,3) for x in [q,k,v]),causal=causal,mask=mask).permute(0,2,1,3).numpy()
        cases.append((label,req('attention',batch=b,sq=sq,sk=sk,heads=h,dim=d,causal=causal,mask=masked),arrays,{'output.npy':expected}))
    for label,lengths,h in [('mla-ragged',[0,17,33],128),('mla-long',[1025],16)]:
        b=len(lengths);n=max(lengths);qa,qr,c,kr=[(torch.randn(shape)*.25).half() for shape in [(b,h,512),(b,h,64),(b,n,512),(b,n,64)]];ptr=[0]
        for length in lengths:ptr.append(ptr[-1]+(length+15)//16)
        indices=torch.randperm(ptr[-1]).tolist();pages=torch.zeros(ptr[-1],16,512,dtype=torch.float16);ropes=torch.zeros(ptr[-1],16,64,dtype=torch.float16)
        for bi,length in enumerate(lengths):
            for t in range(length):page=indices[ptr[bi]+t//16];slot=t%16;pages[page,slot]=c[bi,t];ropes[page,slot]=kr[bi,t]
        arrays={name:x.numpy() for name,x in zip(['q_abs.npy','q_rope.npy','cache.npy','rope_cache.npy'],[qa,qr,pages,ropes])};arrays.update({'indptr.npy':np.array(ptr,np.int32),'indices.npy':np.array(indices,np.int32)});z,lse=ref.mla_core(qa,qr,c,kr,lengths,192**-.5)
        cases.append((label,req('mla',lengths=lengths,heads=h,page_size=16,splits=1,num_pages=ptr[-1]),arrays,{'z.npy':z.numpy(),'lse.npy':lse.numpy()}))
    return cases


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['smoke','installed'],required=True);p.add_argument('--label');p.add_argument('--installed-smoke',action='store_true');a=p.parse_args();root=a.root;work=root/(a.label or a.phase);work.mkdir(exist_ok=False);empty=work/'empty';empty.mkdir();prefix=root/('source' if a.phase=='smoke' and not a.installed_smoke else 'prefix');old=Path('/home/dministrator/cuda-lab/nka-persistent/dev6-d5f4bbf83dc0/prefix');bundle=root/'sdk';bid=json.loads((bundle/'aot-manifest.json').read_text())['bundle_id'];env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(PYTHONPATH=str(prefix),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NKA_CONTROLLER_CANARY='public-test-value-not-a-secret');commands=[];checks=[];cases=make_cases(bid);lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
    def command(label,args,selected=prefix,expected=0):
        e={**env,'PYTHONPATH':str(selected)};start=time.monotonic();r=subprocess.run([sys.executable,'-B',*args],cwd=empty,env=e,capture_output=True,text=True,timeout=220);(work/(label+'.stdout')).write_text(r.stdout);(work/(label+'.stderr')).write_text(r.stderr);commands.append({'label':label,'exit':r.returncode,'wall_seconds':time.monotonic()-start,'prefix':str(selected),'args':args});(work/'commands.json').write_text(json.dumps(commands,indent=2));assert (r.returncode==0)==(expected==0),label;return r
    def run_case(label,case,selected=prefix,fault=None,expected_boundary=True):
        _,request,arrays,targets=case;folder=work/label;folder.mkdir();inputs=folder/'inputs';inputs.mkdir();references=folder/'references';references.mkdir();(folder/'request.json').write_text(json.dumps(request))
        for name,value in arrays.items():np.save(inputs/name,value,allow_pickle=False)
        for name,value in targets.items():np.save(references/name,value,allow_pickle=False)
        args=['-m','kernel_agent.operators.cli','run','--bundle',str(bundle),'--request',str(folder/'request.json'),'--inputs',str(inputs),'--output',str(folder/'outputs'),'--work',str(folder/'run')]
        if fault:args+=['--inject-failure',fault]
        command(label,args,selected);execution=json.loads((folder/'run/execution.json').read_text());assert execution['status']=='PASS' and execution['fallback_used']==bool(fault) and execution['production_release'] is False
        assert len(execution['attempts'])==(2 if fault else 1)
        for rec in execution['attempts']:assert rec['cleanup']['removed'] and rec['protected_unchanged']
        if expected_boundary:
            m=execution['attempts'][-1]['metadata'];boundary={k[len('boundary_'):]:v for k,v in m.items() if k.startswith('boundary_')};assert all(v is True for k,v in boundary.items() if k not in ['uid','gid']) and boundary['uid']==65534 and m['output_fresh_contiguous']
        errors={}
        for name,target in targets.items():
            actual=np.load(folder/'outputs'/name,allow_pickle=False);finite=np.isfinite(target);delta=np.abs(actual.astype(np.float64)[finite]-target[finite]);assert np.array_equal(np.isneginf(actual),np.isneginf(target)) and bool(np.all(delta<=2e-3+2e-3*np.abs(target[finite])));errors[name]=float(delta.max()) if delta.size else 0.
        checks.append({'label':label,'operation':request['operation'],'max_abs_errors':errors,'attempts':len(execution['attempts']),'fallback':bool(fault),'prefix':str(selected)});(work/'checks.json').write_text(json.dumps(checks,indent=2))
    status='FAIL'
    try:
        command('inspect',['-m','kernel_agent.operators.cli','inspect','--bundle',str(bundle)])
        r=command('qualified-missing',['-m','kernel_agent.operators.cli','inspect','--bundle',str(bundle),'--channel','qualified'],expected=1);assert 'not packaged' in r.stderr
        if a.phase=='smoke':
            for i in [0,3]:run_case(cases[i][0],cases[i])
        else:
            modules=['test_registry','test_deployment_policy','test_qualified_channel','test_sm89_runtime','test_sm89_qualified','test_operator_qualification']
            args=['-c',"import sys,unittest;sys.path.append("+repr(str(root/'source/tests'))+");r=unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromNames("+repr(modules)+"));raise SystemExit(not r.wasSuccessful())"]
            command('cpu-tests',args)
            for case in cases:run_case(case[0],case)
            for index in [0,3]:
                for fault in ['crash','timeout']:run_case(cases[index][0]+'-'+fault,cases[index],fault=fault)
            # Test the actual application prefix switch in both directions, with real outputs.
            active=work/'current'
            for version,selected in [('dev6',old),('rc1',prefix)]:
                tmp=work/'next';tmp.symlink_to(selected,target_is_directory=True);os.replace(tmp,active)
                r=command('rollback-version-'+version,['-c','import kernel_agent;print(kernel_agent.__version__)'],active);assert r.stdout.strip()==('0.1.0.dev6' if version=='dev6' else '0.1.1rc1')
                for i in [0,3]:run_case('rollback-'+version+'-'+cases[i][0],cases[i],selected=active,expected_boundary=version!='dev6')
        status='PASS'
    finally:
        result={'status':status,'phase':a.phase,'checks':checks,'commands':commands,'gpu_outputs':len(checks),'gpu_attempts':sum(c['attempts'] for c in checks),'new_model_calls':0,'new_event_samples':0,'production_release':False,'scope':'SM86 Attention/MLA public release-runtime QA; not sealed acceptance or timing'};(work/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['status','phase','gpu_outputs','gpu_attempts','production_release']}))

if __name__=='__main__':main()
