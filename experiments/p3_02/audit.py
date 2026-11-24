"""CPU-only structural audit of fixed-fixture runtime evidence. Not performance/release acceptance."""
import argparse,hashlib,json,math
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(folder,build=False):
    folder=Path(folder);base=folder/'remote' if build else folder;run=base/'run'
    result=json.loads((run/'result.json').read_text());rows=result['cases']
    order=['cublas','cublas','template','template','boundaries']
    for name in ['constant','pollution','crash','symlink','timeout']:order += [name,'cublas']
    if [r['case'] for r in rows]!=order or result['status']!='PASS' or result['failures']:raise ValueError('incomplete fixture pass')
    for row in rows:
        if row['accepted']!=(row['case'] in ['cublas','template','boundaries']):raise ValueError('incorrect acceptance')
        for key in ['container_removed','inputs_unchanged','control_state_unchanged']:
            if not row[key]:raise ValueError('lifecycle or protected data failure')
        cmd=row['command']
        for flag in ['--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--rm','--init']: 
            if flag not in cmd:raise ValueError('missing isolation flag')
        for flag,value in [('--network=none',None),('--user','65534:65534'),('--memory=2g',None),('--pids-limit=128',None),('--cpus=2',None)]:
            if flag not in cmd or (value and cmd[cmd.index(flag)+1]!=value):raise ValueError('incorrect limits/namespace')
        if cmd.count('--mount')!=3 or len([x for x in cmd if 'dst=/artifact,readonly' in x])!=1 or len([x for x in cmd if 'dst=/inputs,readonly' in x])!=1:raise ValueError('unexpected mount set')
        if any('docker.sock' in x or 'gemm_acceptance' in x or '.private' in x for x in cmd):raise ValueError('private host mount')
        if row['accepted'] and (row['exit']!=0 or not math.isfinite(row['max_abs_error'])):raise ValueError('invalid successful worker')
    boundary=rows[4]['worker_metadata']
    if boundary['uid']!=65534 or boundary['interfaces']!=['lo'] or boundary['pids_max']!='128' or boundary['memory_max']!='2147483648' or boundary['cpu_max']!='200000 100000':raise ValueError('observed UID/network/cgroups mismatch')
    if not all(boundary[k] for k in ['capabilities_zero','no_new_privs','readonly_write_denied','external_connect_denied']):raise ValueError('boundary write/network failure')
    if any(boundary[k] for k in ['docker_socket_visible','control_canary_in_env','host_private_visible','host_pid_visible']):raise ValueError('control boundary exposed')
    timeout=rows[-2]
    if not timeout.get('timed_out') or not timeout['ready']['finite_cuda_work_completed'] or timeout['forced_cleanup']['exit']!=0 or timeout['processes_before_cleanup']['exit']!=0:raise ValueError('timeout fixture incomplete')
    if not rows[-1]['accepted']:raise ValueError('post-timeout recovery failed')
    archive=json.loads((folder/'artifacts/manifest.json').read_text())
    if archive['sha256']!=result['binary_sha256'] or sha(folder/'artifacts/nka_gemm_v1.so')!=archive['sha256']:raise ValueError('archived binary mismatch')
    execution=json.loads((folder/'execution.json').read_text())
    for entry in execution['entries']:
        if entry.get('sha256') and sha((base if build else run)/entry['path'])!=entry['sha256']:raise ValueError('archive file hash mismatch')
    if build:
        builder=json.loads((base/'build-execution.json').read_text());meta=json.loads((base/'build/build.json').read_text())
        if builder['exit']!=0 or builder['gpu_exposed'] or not builder['container_removed'] or meta['uid']!=65534 or meta['binary_sha256']!=result['binary_sha256']:raise ValueError('build boundary/artifact mismatch')
        if '--gpus' in builder['command'] or not all(flag in builder['command'] for flag in ['--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--memory=8g','--cpus=4']):raise ValueError('builder isolation command mismatch')
        if json.loads((base/'decoder-tests.json').read_text())['exit']!=0 or 'Ran 10 tests' not in (base/'decoder-tests.stderr.log').read_text() or '\nOK' not in (base/'decoder-tests.stderr.log').read_text():raise ValueError('decoder tests incomplete')
        for rel,value in builder['source_sha256'].items():
            if sha(folder/'sources'/rel)!=value:raise ValueError('build source changed')
    else:
        for rel,value in json.loads((folder/'sources.json').read_text()).items():
            if sha(folder/rel)!=value:raise ValueError('runtime source mismatch')
    return {'status':'PASS','kind':'CPU-only evidence/hash/lifecycle audit; runtime numerical verdicts from independent host FP64 evaluator',
      'runtime_cases':15,'faults_rejected':5,'recoveries_passed':5,'uid':65534,'source_build':build,'cpu_decoder_tests':10 if build else 0,'binary_sha256':result['binary_sha256'],
      'heldout_executed':False,'p3_02_complete':False,'production_release':False,'limitations':['fixed trusted fixtures only','no infinite CUDA kernel cancellation proof','no GPU memory quota','installation/release/rollback pending']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--build',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Use a new output')
    result=audit(a.evidence,a.build);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
