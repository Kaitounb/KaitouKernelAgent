"""Installable entry for catalogued reviewed native operators, no free source execution."""
import argparse,hashlib,json,os,sys,uuid
from pathlib import Path
import numpy as np
from kernel_agent.registry import read_regular,atomic_json
from .array_io import load_array,encode_array
from .runtime_contract import specifications,validate_arrays
from .runner import OperatorRunner


def approved_runner(bundle,channel="development"):
    if channel not in ['development','qualified']:raise ValueError('unknown operator release channel')
    manifest=json.loads(read_regular(Path(bundle)/'aot-manifest.json',1024*1024))
    if not isinstance(manifest,dict):raise ValueError('AOT manifest must be an object')
    bid=manifest.get('bundle_id');catalog=json.loads(read_regular(Path(__file__).with_name('catalog.json'),65536))
    if bid not in catalog['bundles']:raise ValueError('bundle is not in the reviewed operator catalog')
    meta=catalog['bundles'][bid]
    if any(manifest.get(key)!=meta[key] for key in ['capability','torch','cuda','production_release']):raise ValueError('catalog environment differs')
    if channel=='qualified':
        from .qualification import approve
        certificate=approve(bid,manifest);meta={**meta,'production_release':True,'scope':certificate['scope'],'qualification':certificate}
    return OperatorRunner(bundle,bid),meta


def execute(bundle,request_path,inputs,output,work,inject_failure=None,*,channel="development"):
    runner,meta=approved_runner(bundle,channel);request=json.loads(read_regular(request_path,65536))
    if request.get('fault') is not None:raise ValueError('request fault must be null; use explicit recovery exercise flag')
    if request.get('bundle_id')!=runner.bundle_id:raise ValueError('request/catalog bundle differs')
    if channel=='qualified':
        from .qualification import check_request
        check_request(meta['qualification'],request)
    specs,_=specifications(request);arrays={name:load_array(Path(inputs)/name,*spec) for name,spec in specs.items()};validate_arrays(request,arrays)
    output=Path(output);work=Path(work)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():raise ValueError('output must be a new directory under an existing parent')
    work.mkdir(mode=0o700,exist_ok=False);attempts=[];actual=None
    for attempt in range(2):
        current={**request,'fault':inject_failure if attempt==0 else None};actual,record=runner.run(current,arrays,work/('attempt-'+str(attempt)));attempts.append(record)
        if actual is not None:break
        if not record.get('cleanup',{}).get('removed'):break
    result={'status':'PASS' if actual is not None else 'FAIL','deployment':'qualified-runtime' if channel=='qualified' else 'development-only','bundle_id':runner.bundle_id,'operation':request['operation'],'fallback_used':len(attempts)>1,'attempts':attempts,'production_release':channel=='qualified','promotion_performed':False}
    # Publish a complete output directory once, never a partial group of NPYs.
    if actual is not None:
        staging=output.with_name('.'+output.name+'-'+uuid.uuid4().hex);staging.mkdir(mode=0o700)
        try:
            for name,value in actual.items():
                with (staging/name).open('xb') as file:file.write(encode_array(value));file.flush();os.fsync(file.fileno())
            atomic_json(staging/'result.json',{k:v for k,v in result.items() if k!='attempts'})
            if output.exists():raise ValueError('output directory already exists')
            os.rename(staging,output)
        finally:
            if staging.exists():
                for path in staging.iterdir():path.unlink()
                staging.rmdir()
    atomic_json(work/'execution.json',result)
    if actual is None:raise RuntimeError('operator failed after bounded attempts; see '+str(work/'execution.json'))
    return {k:v for k,v in result.items() if k!='attempts'}


def main():
    p=argparse.ArgumentParser(description='Reviewed fixed SM86 AOT operators; evidence-bound qualified or development channel')
    sub=p.add_subparsers(dest='command',required=True)
    inspect=sub.add_parser('inspect');inspect.add_argument('--bundle',type=Path,required=True)
    run=sub.add_parser('run');run.add_argument('--bundle',type=Path,required=True);run.add_argument('--request',type=Path,required=True);run.add_argument('--inputs',type=Path,required=True);run.add_argument('--output',type=Path,required=True);run.add_argument('--work',type=Path,required=True);run.add_argument('--inject-failure',choices=['crash','timeout'],help='Explicit finite recovery exercise')
    for parser in [inspect,run]:parser.add_argument('--channel',choices=['development','qualified'],default='development')
    a=p.parse_args()
    try:
        if a.command=='inspect':
            runner,meta=approved_runner(a.bundle,a.channel);result={'status':'qualified-runtime' if a.channel=='qualified' else 'development-only','bundle_id':runner.bundle_id,**{k:v for k,v in meta.items() if k!='qualification'},'files_verified':len(runner.manifest['files'])}
        else:result=execute(a.bundle,a.request,a.inputs,a.output,a.work,a.inject_failure,channel=a.channel)
        print(json.dumps(result))
    except (ValueError,OSError,RuntimeError,KeyError,TypeError) as error:
        print(json.dumps({'status':'ERROR','error':str(error)[:1024],'production_release':False}),file=sys.stderr);raise SystemExit(1)


if __name__=='__main__':main()
