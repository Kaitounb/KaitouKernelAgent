"""CPU-only source/archive/artifact/lifecycle replay for the reviewed builder regression."""
import argparse,hashlib,json
from pathlib import Path
from kernel_agent.bounded_io import confirmed_removed
from kernel_agent.build_artifacts import ARCHIVE_LIMIT,decode_artifacts
from kernel_agent.protocol import HEADER,META_LIMIT,decode_frame


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(folder):
    folder=Path(folder);remote=folder/'remote';run=remote/'run';result=json.loads((run/'result.json').read_text());execution=json.loads((folder/'execution.json').read_text())
    if result['status']!='PASS' or result['failures'] or not result['best_unchanged']:raise ValueError('source pipeline did not pass')
    expected=['boundaries','stdout_flood','stderr_flood','disk_limit','bad_archive','syntax-error','native-init-abort','baseline','wrong-output','timeout','fresh-after-timeout']
    if [r['label'] for r in result['builds']]!=expected:raise ValueError('build cases incomplete')
    for rel,meta in execution['manifest'].items():
        path=Path(rel)
        if path.is_absolute() or '..' in path.parts or sha(remote/path)!=meta['sha256'] or (remote/path).stat().st_size!=meta['bytes']:raise ValueError('archive integrity failed')
    for rel,value in json.loads((folder/'source-hashes.json').read_text()).items():
        if sha(folder/'sources'/rel)!=value:raise ValueError('staged source changed')
    project=Path(__file__).resolve().parents[2]
    original=json.loads((project/'evidence/20261122-p1-05-final/bindings.json').read_text())
    fixed_pins={'evaluators/gemm_regression.py':original['harness']['evaluators/gemm_regression.py']}
    records=json.loads((project/'provenance/upstream-reviews/20261121-p1-02/source-record.json').read_text())
    for record in records:
        if record['repo']=='ScalingIntelligence/KernelBench' and record['path'] in ['src/kernelbench/eval.py','src/kernelbench/timing.py','src/kernelbench/dataset.py']:
            if record['revision']!='423217d9fda91e0c2d67e4a43bf62f96f6d104f1':raise ValueError('upstream revision changed')
            fixed_pins['upstream/kernelbench/'+Path(record['path']).name]=record['sha256']
    if len(fixed_pins)!=4:raise ValueError('fixed loader pins missing')
    if sha(project/'evidence/20261122-p1-04-feedback/freeze-current.json')!=original['freeze_sha256']:raise ValueError('actual frozen search best changed')
    built=[]
    for row in result['builds']:
        cmd=row['command'];job=run/('build-'+row['label'])
        if '--gpus' in cmd or 'NVIDIA_VISIBLE_DEVICES=void' not in cmd:raise ValueError('builder GPU access configured')
        for flag in ['--network=none','--sig-proxy=false','--read-only','--log-driver=none','--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=128','--memory=8g','--memory-swap=8g','--cpus=4']:
            if cmd.count(flag)!=1:raise ValueError('missing builder boundary')
        if cmd.count('--mount')!=2 or not all('readonly' in cmd[n+1] for n,x in enumerate(cmd) if x=='--mount'):raise ValueError('builder writable host mount')
        if cmd[cmd.index('--user')+1]!='65534:65534' or not row['source_unchanged'] or not row['deps_unchanged'] or not row['best_unchanged']:raise ValueError('protected boundary failed')
        if not row['cleanup']['removed'] or not confirmed_removed(row['cleanup']['inspect_after']):raise ValueError('builder not removed')
        for stream in ['stdout','stderr']:
            size=(job/'capture'/(stream+'.bin')).stat().st_size
            if size!=row['stored_bytes'][stream] or size>row['limits'][stream]:raise ValueError('build pipe cap violated')
        for rel,value in row['source_hashes'].items():
            if sha(job/'source'/rel)!=value:raise ValueError('actual build source hash differs')
        for rel,value in fixed_pins.items():
            if row['source_hashes'].get(rel)!=value:raise ValueError('fixed loader/harness differs from validated pin')
        source_fields={name:row['source_hashes']['kernels/gemm/'+name] for name in ['cublas_sm86.cu','wmma_sm86.cu','gemm_contract.cuh']}
        if hashlib.sha256(json.dumps(source_fields,sort_keys=True).encode()).hexdigest()!=row['source_identity']:raise ValueError('CUDA source identity differs')
        if row['status']=='BUILT_UNVALIDATED':
            meta,payload=decode_frame(job/'capture/stdout.bin',ARCHIVE_LIMIT);files=decode_artifacts(payload)
            for name,data in files.items():
                digest=hashlib.sha256(data).hexdigest()
                if digest!=row['artifact_hashes'][name] or sha(job/'artifacts'/name)!=digest:raise ValueError('artifact reparse mismatch')
            if meta['gpu_available'] is not False or meta['binary_sha256']!=row['binary_sha256']:raise ValueError('build metadata mismatch')
            built.append(row['label'])
    if built!=['baseline','wrong-output','fresh-after-timeout']:raise ValueError('unexpected build successes')
    boundary=result['builds'][0]['metadata']
    if any(boundary[k] for k in ['gpu_device_visible','host_private_visible','host_pid_visible','control_canary_in_env','docker_socket_visible']) or boundary['interfaces']!=['lo']:raise ValueError('visibility isolation failed')
    timeout=next(r for r in result['builds'] if r['label']=='timeout')
    if timeout['reason']!='timeout' or not timeout['cleanup']['observed_host_pids'] or not timeout['cleanup']['host_pids_absent']:raise ValueError('build timeout child cleanup not proved')
    runs=result['runs']
    if len(runs)!=27:raise ValueError('GPU result count differs')
    for row in runs:
        if row['correct']!=row['expected_correct'] or not row['cleanup']['removed'] or not row['protected_files_unchanged'] or not row['best_unchanged']:raise ValueError('GPU output/lifecycle verdict mismatch')
    trajectory=result['trajectory']
    if [row['verdict'] for row in trajectory]!=['REJECT','REJECT','CORRECTNESS_ONLY','REJECT','REJECT']:raise ValueError('external decision replay differs')
    if any(result[k] for k in ['heldout_executed','model_calls','performance_claim','production_release']):raise ValueError('scope changed')
    return {'status':'PASS','kind':'CPU-only source/archive/closed-ZIP/lifecycle/decision replay; numerical verdicts from independent host FP64 evaluator',
            'fixed_loader_pins_verified':4,'actual_search_freeze_unchanged':True,'builder_cases':len(result['builds']),'successful_fresh_builds':3,'gpu_runs':len(runs),'correct_outputs':sum(row['correct'] for row in runs),'wrong_output_rejected':True,
            'host_writable_builder_mounts':0,'builder_gpu_access':False,'best_unchanged':True,'heldout_executed':False,'new_model_calls':0,'new_performance_claim':False,'production_release':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Use a new output')
    result=audit(a.evidence);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
