"""CPU-only binding/archive/boundary audit; numerical verdicts come from the independent host evaluator."""
import argparse,ast,errno,hashlib,json
from pathlib import Path
from kernel_agent.bounded_io import confirmed_removed
from kernel_agent.protocol import decode_frame

FAULTS=['constant','pollution','crash','bad_frame','metadata_nested','metadata_forged','stdout_flood','stderr_flood','disk_limit','pids_limit','memory_oom','timeout']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(folder):
    folder=Path(folder);remote=folder/'remote';run=remote/'run';result=json.loads((run/'result.json').read_text());rows=result['cases']
    if result['status']!='PASS' or result['failures'] or len(rows)!=29:raise ValueError('incomplete bounded runtime acceptance')
    expected=['cublas','cublas','template','template','boundaries']
    for fault in FAULTS:expected += [fault,'cublas']
    if [r['case'] for r in rows]!=expected:raise ValueError('fault/recovery order differs')
    envelope=json.loads((folder/'execution.json').read_text())
    for rel,meta in envelope['manifest'].items():
        path=Path(rel)
        if path.is_absolute() or '..' in path.parts or sha(remote/path)!=meta['sha256'] or (remote/path).stat().st_size!=meta['bytes']:raise ValueError('raw archive binding changed')
    for rel,expected_sha in json.loads((folder/'source-hashes.json').read_text()).items():
        if sha(folder/'sources'/rel)!=expected_sha:raise ValueError('staged source changed')
    for i,row in enumerate(rows):
        job=run/('job-'+str(i).zfill(2)+'-'+row['case']);cmd=row['command']
        desired=row['case'] in ['cublas','template','boundaries']
        if row['accepted']!=desired or not row['protected_files_unchanged'] or not row['control_state_unchanged'] or not row['cleanup']['removed'] or not confirmed_removed(row['cleanup']['inspect_after_cleanup']):raise ValueError('verdict/protected data/lifecycle mismatch')
        if cmd.count('--mount')!=2 or not all('readonly' in cmd[n+1] for n,x in enumerate(cmd) if x=='--mount'):raise ValueError('unexpected writable host mount')
        for flag in ['--read-only','--network=none','--sig-proxy=false','--log-driver=none','--pids-limit=32','--cap-drop=ALL','--security-opt=no-new-privileges']:
            if cmd.count(flag)!=1:raise ValueError('missing/repeated isolation flag')
        if cmd[cmd.index('--user')+1]!='65534:65534' or '/outputs:rw,nosuid,nodev,noexec,size=8m,mode=1777' not in cmd:raise ValueError('uid/tmpfs mismatch')
        for stream in ['stdout','stderr']:
            file=job/'capture'/(stream+'.bin')
            if file.stat().st_size!=row['stored_bytes'][stream] or file.stat().st_size>row['limits'][stream]:raise ValueError('pipe limit violated')
    for i,fault in enumerate(FAULTS):
        row=rows[5+2*i]
        if fault in ['stdout_flood','stderr_flood'] and row['reason']!=fault.split('_')[0]+'_limit':raise ValueError('pipe flood not rejected')
        if fault=='disk_limit' and row['metadata']['write_errno']!=errno.ENOSPC:raise ValueError('tmpfs quota not proved')
        if fault=='pids_limit' and not row['metadata']['pids_denied']:raise ValueError('PID cap not proved')
        if fault=='memory_oom' and not json.loads(row['cleanup']['inspect']['stdout'])['OOMKilled']:raise ValueError('cgroup OOM not proved')
        if fault=='timeout' and (row['reason']!='timeout' or not row['cleanup']['observed_host_pids'] or not row['cleanup']['host_pids_absent_after_cleanup']):raise ValueError('timeout child cleanup not proved')
    return {'status':'PASS','kind':'CPU-only source/archive/byte-limit/lifecycle audit; numerical verdicts from independent host FP64 evaluator',
            'runtime_cases':29,'faults_rejected':12,'fresh_cublas_recoveries':12,'host_writable_candidate_mounts':0,
            'pids_max':32,'tmpfs_output_bytes':8388608,'stderr_max_bytes':65536,'cpu_memory_oom_observed':True,
            'heldout_executed':False,'model_calls':0,'performance_claim':False,'p3_02_complete':False,'production_release':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Use a new audit output')
    result=audit(a.evidence);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
