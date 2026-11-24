"""Real fixed-fixture bounded Docker/GPU checks; trusted host owns every reference and verdict."""
import argparse,errno,json,os
from pathlib import Path
import numpy as np
import torch
from kernel_agent.runner import DockerRunner,digest

FAULTS=['constant','pollution','crash','bad_frame','metadata_nested','metadata_forged','stdout_flood','stderr_flood','disk_limit','pids_limit','memory_oom','timeout']


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);p.add_argument('--sha256',required=True);args=p.parse_args()
    work=args.work;work.mkdir(exist_ok=False);torch.set_num_threads(1)
    control=work/'control';control.mkdir(mode=0o700);(control/'state.json').write_text(json.dumps({'published':False,'current':'unchanged-4','runtime_default':'cuBLAS'}))
    (control/'secret').write_text('decoy-not-a-real-credential');state=digest(control/'state.json');os.environ['NKA_CONTROL_CANARY']='decoy-not-a-real-key'
    runner=DockerRunner(args.binary,args.sha256);rows=[];failures=[]
    def one(case,shape=(17,19,23)):
        m,n,k=shape;g=torch.Generator().manual_seed(171)
        a=torch.randn(m,k,generator=g).mul_(.25).half();b=torch.randn(k,n,generator=g).mul_(.25).half()
        expected=(a.double()@b.double()).float().numpy()
        actual,row=runner.run(a.numpy(),b.numpy(),work/('job-'+str(len(rows)).zfill(2)+'-'+case),case=case,host_private_path=str((control/'secret').resolve()),host_pid=os.getpid())
        row['accepted']=False;row['control_state_unchanged']=digest(control/'state.json')==state
        if actual is not None:
            delta=np.abs(actual.astype(np.float64)-expected.astype(np.float64));row['max_abs_error']=float(delta.max())
            row['accepted']=bool(np.all(delta<=3e-5+1e-4*np.abs(expected)))
            if not row['accepted']:row['rejection']='independent_CPU_FP64_mismatch'
        rows.append(row)
        if row['accepted']!=(case in ['cublas','template','boundaries']) or not row['cleanup']['removed'] or not row['protected_files_unchanged'] or not row['control_state_unchanged']:failures.append(case)
        (work/'result.partial.json').write_text(json.dumps(rows,indent=2)+'\n')
        return row
    for case in ['cublas','template']:
        for shape in [(17,19,23),(512,512,512)]:one(case,shape)
    boundary=one('boundaries').get('metadata',{})
    if not all(boundary.get(k) for k in ['capabilities_zero','no_new_privs','artifact_write_denied','inputs_write_denied','external_connect_denied']) or boundary.get('uid')!=65534 or boundary.get('interfaces')!=['lo'] or boundary.get('pids_max')!='32' or boundary.get('memory_max')!='2147483648' or boundary.get('cpu_max')!='200000 100000':failures.append('boundary_configuration')
    if any(boundary.get(k,True) for k in ['docker_socket_visible','control_canary_in_env','host_private_visible','host_pid_visible']):failures.append('boundary_visibility')
    for fault in FAULTS:
        row=one(fault);one('cublas')
        if fault in ['stdout_flood','stderr_flood'] and row['reason']!=fault.split('_')[0]+'_limit':failures.append(fault+'_limit_not_enforced')
        if fault=='disk_limit' and row.get('metadata',{}).get('write_errno')!=errno.ENOSPC:failures.append('tmpfs_limit_not_enforced')
        if fault=='pids_limit' and not row.get('metadata',{}).get('pids_denied'):failures.append('pids_limit_not_enforced')
        if fault=='memory_oom':
            observed=json.loads(row['cleanup']['inspect']['stdout'])
            if not observed['OOMKilled'] or row['exit']!=137:failures.append('memory_limit_not_enforced')
        if fault=='timeout':
            log=next(work.glob('job-*-timeout/capture/stderr.bin')).read_bytes()
            if row['reason']!='timeout' or b'NKA_FINITE_CUDA_READY' not in log or not row['cleanup'].get('observed_host_pids') or not row['cleanup'].get('host_pids_absent_after_cleanup'):failures.append('finite_cuda_timeout_cleanup_not_proved')
    result={'status':'PASS' if not failures else 'FAIL','scope':'reviewed fixed fixtures in bounded runtime; no generated source or production release',
       'cases':rows,'failures':failures,'runtime_cases':len(rows),'faults':len(FAULTS),'recoveries':len(FAULTS),'heldout_executed':False,'model_calls':0,'performance_claim':False,'control_state_unchanged':digest(control/'state.json')==state}
    (work/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','runtime_cases','faults','recoveries','failures']}))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
