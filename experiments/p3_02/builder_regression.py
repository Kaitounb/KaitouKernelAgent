"""Reviewed CUDA source -> no-GPU build -> independent GPU output validation -> external verdict."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from kernel_agent.builder import DockerBuilder,cuda_identity
from kernel_agent.runner import DockerRunner,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);parser.add_argument('--deps',type=Path,required=True);parser.add_argument('--work',type=Path,required=True);parser.add_argument('--fallback',type=Path,required=True);a=parser.parse_args()
    work=a.work;work.mkdir();torch.set_num_threads(1)
    control=work/'control';control.mkdir(mode=0o700);(control/'secret').write_text('decoy-no-real-key');(control/'best.json').write_text(json.dumps({'id':'f329ade329014e517bf5e3b3a5828f7bc1b1c86102ffc310a6519110e85426a6','warps':4,'production_release':False}))
    before=digest(control/'best.json');os.environ['NKA_CONTROL_CANARY']='decoy-not-a-real-key'
    base={name:(a.project/'kernels/gemm'/name).read_bytes() for name in ['cublas_sm86.cu','wmma_sm86.cu','gemm_contract.cuh']}
    fallback=DockerRunner(a.fallback,digest(a.fallback));builder=DockerBuilder(a.project,a.deps);builds=[];runs=[];trajectory=[];failures=[]
    def save():
        (work/'progress.json').write_text(json.dumps({'builds':builds,'runs':runs,'trajectory':trajectory,'failures':failures},indent=2)+'\n')
    def build(label,files,case='compile'):
        binary,row=builder.build(files,work/('build-'+label),reviewed_sha256=cuda_identity(files),case=case,host_private_path=str((control/'secret').resolve()))
        row['label']=label;row['best_unchanged']=digest(control/'best.json')==before;builds.append(row);save()
        if not row['cleanup']['removed'] or not row['source_unchanged'] or not row['deps_unchanged'] or not row['best_unchanged']:raise RuntimeError('builder lifecycle/protected data failed: '+label)
        return binary,row
    def run(label,runner,shape,case='cublas',expect_correct=True):
        m,n,k=shape;g=torch.Generator().manual_seed(171)
        x=torch.randn(m,k,generator=g).mul_(.25).half();y=torch.randn(k,n,generator=g).mul_(.25).half();expected=(x.double()@y.double()).float().numpy()
        output,row=runner.run(x.numpy(),y.numpy(),work/('run-'+label),case=case)
        correct=False
        if output is not None:
            delta=np.abs(output.astype(np.float64)-expected.astype(np.float64));row['max_abs_error']=float(delta.max());correct=bool(np.all(delta<=3e-5+1e-4*np.abs(expected)))
        row.update(label=label,correct=correct,expected_correct=expect_correct,best_unchanged=digest(control/'best.json')==before)
        runs.append(row)
        if correct!=expect_correct:failures.append(label)
        if not row['cleanup']['removed'] or not row['protected_files_unchanged'] or not row['best_unchanged']:raise RuntimeError('runtime lifecycle/protected data failed')
        save();return correct
    for case in ['boundaries','stdout_flood','stderr_flood','disk_limit','bad_archive']:
        binary,row=build(case,base,case)
        if binary is not None or row['status']!='REJECT':failures.append('escaped-'+case)
        if case=='boundaries':
            meta=row.get('metadata',{})
            if not all(meta.get(k) for k in ['capabilities_zero','no_new_privs','source_write_denied','deps_write_denied','external_connect_denied']) or meta.get('uid')!=65534 or meta.get('interfaces')!=['lo'] or any(meta.get(k,True) for k in ['host_private_visible','host_pid_visible','control_canary_in_env','docker_socket_visible','gpu_device_visible']):failures.append('builder-boundaries')
            if meta.get('pids_max')!='128' or meta.get('memory_max')!='8589934592' or meta.get('memory_swap_max')!='0' or meta.get('cpu_max')!='400000 100000':failures.append('builder-cgroups')
        elif case in ['stdout_flood','stderr_flood']:
            if row['reason']!=case.split('_')[0]+'_limit':failures.append('uncapped-'+case)
        elif case=='disk_limit' and row.get('metadata',{}).get('write_errno')!=28:failures.append('unbounded-build-tmpfs')
        run('recover-'+case,fallback,(17,19,23))
    syntax=dict(base);syntax['wmma_sm86.cu']+=b'\n#error NKA_INTENTIONAL_CUDA_COMPILE_FAILURE\n'
    binary,row=build('syntax-error',syntax)
    if binary is not None or row['exit']==0:failures.append('compile-error-accepted')
    trajectory.append({'label':'syntax-error','source':cuda_identity(syntax),'verdict':'REJECT','reason':'compile_failure','best_unchanged':True});run('recover-syntax',fallback,(17,19,23))
    abort=dict(base);abort['wmma_sm86.cu']+=b'\n#include <cstdlib>\nstatic int nka_abort_on_load=[](){std::abort();return 0;}();\n'
    binary,row=build('native-init-abort',abort)
    if binary is not None or row['exit']==0:failures.append('native-init-failure-accepted')
    trajectory.append({'label':'native-init-abort','source':cuda_identity(abort),'verdict':'REJECT','reason':'native_initialization_failure','best_unchanged':True});run('recover-native-init',fallback,(17,19,23))
    binary,row=build('baseline',base)
    if binary is None:raise RuntimeError('fresh baseline build failed')
    baseline=DockerRunner(binary,row['binary_sha256'])
    development=json.loads((a.project/'workloads/gemm_development_v1.json').read_text())['workloads']
    for case in ['cublas','template']:
        for item in development:run('baseline-'+case+'-'+item['id'],baseline,item['shape_mnk'],case)
    trajectory.append({'label':'baseline','source':cuda_identity(base),'verdict':'CORRECTNESS_ONLY','reason':'no_fresh_timing_or_formal_acceptance','best_unchanged':True})
    wrong=dict(base);needle=b'    const int m = a.size(0), n = b.size(1), k = a.size(1);'
    if wrong['wmma_sm86.cu'].count(needle)!=1:raise ValueError('reviewed source fixture location changed')
    wrong['wmma_sm86.cu']=wrong['wmma_sm86.cu'].replace(needle,b'    return torch::zeros({a.size(0),b.size(1)}, a.options().dtype(at::kFloat));\n'+needle)
    binary,row=build('wrong-output',wrong)
    if binary is None:raise RuntimeError('wrong-output fixture must actually compile')
    run('wrong-output',DockerRunner(binary,row['binary_sha256']),(512,512,512),'template',False)
    trajectory.append({'label':'wrong-output','source':cuda_identity(wrong),'verdict':'REJECT','reason':'independent_CPU_FP64_mismatch','best_unchanged':True});run('recover-wrong-output',fallback,(17,19,23))
    binary,row=build('timeout',base,'timeout')
    ready=(work/'build-timeout/capture/stderr.bin').read_bytes()
    if binary is not None or row['reason']!='timeout' or b'NKA_BUILD_CHILD_READY' not in ready or not row['cleanup'].get('observed_host_pids') or not row['cleanup'].get('host_pids_absent'):failures.append('build-timeout-child-cleanup')
    trajectory.append({'label':'timeout','source':cuda_identity(base),'verdict':'REJECT','reason':'host_timeout','best_unchanged':True})
    binary,row=build('fresh-after-timeout',base)
    if binary is None:raise RuntimeError('fresh builder recovery failed')
    recovery=DockerRunner(binary,row['binary_sha256']);run('fresh-builder-cublas',recovery,(17,19,23));run('fresh-builder-template',recovery,(512,512,512),'template')
    result={'status':'PASS' if not failures else 'FAIL','scope':'reviewed source build/run/independent correctness; no model generation, performance, promotion or production release',
            'builds':builds,'runs':runs,'trajectory':trajectory,'failures':failures,'best_unchanged':digest(control/'best.json')==before,
            'heldout_executed':False,'model_calls':0,'performance_claim':False,'production_release':False}
    (work/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'builds':len(builds),'gpu_runs':len(runs),'failures':failures}))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
