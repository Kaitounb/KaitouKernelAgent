"""Trusted private evaluator. Never prints exact workload shapes, seeds, inputs or references."""
import argparse,hashlib,importlib.util,json,os,shutil,subprocess,sys,traceback
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;private=root/'private';private.mkdir(mode=0o700);rows=[];summary={'status':'FAIL','executed_gpu_calls':0,'manifest_count':4,'heldout_exposed_to_model':False,'production_release':False}
    try:
        manifest_raw=(root/'manifest.json').read_bytes();freeze=json.loads((root/'freeze.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
        if hashlib.sha256(manifest_raw).hexdigest()!=protocol['dataset_commitment_sha256']:raise ValueError('manifest commitment mismatch')
        manifest=json.loads(manifest_raw);workloads=manifest['workloads']
        if len(workloads)!=4:raise ValueError('manifest count mismatch')
        # Schema is evaluated here; values never reach the controlling model.
        parsed=[]
        for row in workloads:
            shape=tuple(row['shape_mnk']);seeds=row.get('seeds',manifest.get('seeds'))
            if seeds is None and 'seed' in row:seeds=[row['seed']]
            if not isinstance(seeds,list) or not seeds or len(shape)!=3:raise ValueError('unsupported private manifest schema')
            parsed.append((shape,seeds))
        prefix=root/'prefix';env={k:v for k,v in os.environ.items() if k in ['HOME','PATH','LANG','LD_LIBRARY_PATH']};env.update(PYTHONPATH=str(prefix),OPENBLAS_NUM_THREADS='1')
        with (private/'install.stdout').open('w') as out,(private/'install.stderr').open('w') as err:
            proc=subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(prefix),str(next(root.glob('*.whl')))],env=env,stdout=out,stderr=err,timeout=120)
        if proc.returncode:raise RuntimeError('private candidate installation failed')
        sys.path.insert(0,str(prefix));import torch,numpy as np
        # Reuse the frozen runner byte-for-byte under an independent evaluator namespace.
        # Only its worker observer is different; it adds stream/guard/GPU-input hashes.
        support=root/'observer';support.mkdir();(support/'__init__.py').write_text('')
        for name in ['runner.py','bounded_io.py','protocol.py','output.py']:shutil.copyfile(prefix/'kernel_agent'/name,support/name)
        shutil.copyfile(root/'sealed_worker.py',support/'pipe_worker.py')
        spec=importlib.util.spec_from_file_location('_sealed_observer',support/'__init__.py',submodule_search_locations=[str(support)]);package=importlib.util.module_from_spec(spec);sys.modules[spec.name]=package;spec.loader.exec_module(package)
        from _sealed_observer.runner import DockerRunner
        torch.set_num_threads(1)
        def inputs(shape,seed):
            m,n,k=shape;g=torch.Generator().manual_seed(seed)
            return [((torch.randn(dims,generator=g)*.25).clamp_(-4,4)).half().numpy() for dims in [(m,k),(k,n)]]
        for index,(shape,seeds) in enumerate(parsed):
            for si,seed in enumerate(seeds):
                x,y=inputs(shape,seed);reference=(torch.from_numpy(x).double()@torch.from_numpy(y).double()).float().numpy()
                snapshots=[v.tobytes() for v in [x,y]]
                for role,digest in freeze['deployment_artifacts'].items():
                    binary=prefix/'kernel_agent/data/artifacts'/(digest+'.so');job=private/('job-'+str(index)+'-'+str(si)+'-'+role)
                    summary['executed_gpu_calls']+=1;(private/'execution-started.json').write_text(json.dumps(summary))
                    actual,record=DockerRunner(binary,digest).run(x,y,job,case='cublas')
                    meta=record.get('metadata',{});expected_hashes=[hashlib.sha256(v).hexdigest() for v in snapshots]
                    good=actual is not None and all(v.tobytes()==b for v,b in zip([x,y],snapshots)) and bool(np.isfinite(actual).all()) and bool(np.all(np.abs(actual.astype(np.float64)-reference)<=3e-5+1e-4*np.abs(reference))) and meta.get('input_hashes')==expected_hashes and meta.get('guards_ok') is True and meta.get('producer_consumer_stream') is True and meta.get('output_contract') is True and meta.get('uid')==65534 and meta.get('binary_sha256')==digest
                    error=float(np.abs(actual.astype(np.float64)-reference).max()) if actual is not None else None
                    rows.append({'job':job.name,'role':role,'correct':good,'max_abs_error':error,'record':record});(private/'checks.json').write_text(json.dumps(rows,indent=2)+'\n')
                    if not record.get('cleanup',{}).get('removed'):raise RuntimeError('unconfirmed cleanup; subsequent jobs stopped')
        summary.update(status='PASS' if all(r['correct'] for r in rows) else 'FAIL',checked_outputs=len(rows),correct_outputs=sum(r['correct'] for r in rows),max_abs_error=max((r['max_abs_error'] for r in rows if r['max_abs_error'] is not None),default=None),candidate_id=freeze['candidate_id'],freeze_id=freeze['freeze_id'],protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),manifest_commitment=hashlib.sha256(manifest_raw).hexdigest(),new_model_calls=0,performance_sampling=False)
    except Exception:
        (private/'failure.log').write_text(traceback.format_exc());summary['error']='trusted_evaluator_failed; details remain private'
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
