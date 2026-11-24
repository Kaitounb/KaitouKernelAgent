"""One-shot orchestration around the validated provider/guard/build/check adapters.
No release activation; a durable call reservation prevents retry after an interrupted model call.
"""
import fcntl,hashlib,json,os,stat,time,uuid
from pathlib import Path
from .identity import identity,canonical,strict_json
from .patches import apply_reviewable_patch
from .prepare import prepare
from .resources import ASSETS,verify_bundle

TERMINAL={'REJECT','CHECKS_COMPLETE','NO_PROMOTION','SCREENING_PASS','FAILED','STOPPED_UNCERTAIN_CALL','STOPPED_NEEDS_CLEANUP'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);temp=path.with_name('.'+path.name+'-'+uuid.uuid4().hex)
    try:
        with temp.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:temp.unlink(missing_ok=True)


def read_response(path):
    path=Path(path);fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as f:
        info=os.fstat(f.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0<info.st_size<=16384:raise ValueError('recorded response must be a regular JSON file <=16KiB')
        data=f.read(16385)
    return data


def status(run_dir):
    root=Path(run_dir)
    if root.is_symlink() or (root/'run.json').is_symlink():raise ValueError('symlink run/status unsupported')
    return json.loads((root/'run.json').read_text())


def execute(run_dir,*,response=None,live=False,max_calls=0,recount=False,review_sha=None,deps=None,checks_only=False,resume=False,provider_factory=None,evaluator=None):
    bundle_sha=verify_bundle();root=Path(run_dir).absolute()
    if root.is_symlink():raise ValueError('symlink run directory unsupported')
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    lock=os.open(root/'.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('optimization run is already active') from None
        raw=read_response(response) if response is not None else None
        source_kind='recorded-response' if raw is not None else 'deepseek'
        config={'source':source_kind,'response_sha256':hashlib.sha256(raw).hexdigest() if raw is not None else None,'checks_only':bool(checks_only),'recount':bool(recount),'maximum_calls':max_calls}
        def save(stage,**extra):
            state.update(status=stage,**extra);write(root/'run.json',state)
            with (root/'trajectory.jsonl').open('a') as out:
                out.write(json.dumps({'time':time.time(),'status':stage,'calls_reserved':state['calls_reserved']})+'\n');out.flush();os.fsync(out.fileno())
            return state
        if (root/'run.json').exists():
            state=status(root)
            if state['bundle_sha256']!=bundle_sha:raise ValueError('installed bundle changed during run')
            if raw is not None and state['config']['response_sha256']!=config['response_sha256']:raise ValueError('run already bound to another response')
            if state['status'] in TERMINAL:return state
            if not resume:raise ValueError('existing pending run requires --resume; model budget is not reset')
            if state['status']=='MODEL_RESERVED':return save('STOPPED_UNCERTAIN_CALL',reason='prior model call outcome unknown; no retry',verdict='REJECT')
            if state['status']=='EVALUATING':return save('STOPPED_NEEDS_CLEANUP',reason='prior evaluation interrupted; inspect retained owned-container records before a separate attempt',verdict='REJECT')
            if state['status']!='NEEDS_SOURCE_REVIEW':raise ValueError('run is not safely resumable')
            if state['config']['checks_only']!=bool(checks_only):raise ValueError('resume cannot change measurement scope')
        else:
            if resume:raise ValueError('cannot resume a missing run')
            if raw is None and (not live or max_calls!=1):raise ValueError('live proposal requires --deepseek --max-calls 1')
            if raw is not None and (live or max_calls!=0):raise ValueError('recorded response consumes zero new calls; omit live budget')
            if raw is None and not os.getenv('DEEPSEEK_API_KEY'):raise ValueError('DEEPSEEK_API_KEY missing')
            prepare(root/'prepared');state={'schema_version':1,'config':config,'bundle_sha256':bundle_sha,'parent_id':identity(4)['id'],'calls_reserved':0,'new_external_model_calls':0,'heldout_executed':False,'production_release':False,'performance_claim':False,'runtime_default':'cublas','release_activated':False,'evidence_kind':'CPU_MOCK' if evaluator or provider_factory else 'INSTALLED_ADAPTER'}
            save('PREPARED')
            if raw is None:
                save('MODEL_RESERVED',calls_reserved=1,new_external_model_calls=0 if provider_factory else None,external_call_outcome='CPU_MOCK' if provider_factory else 'UNCONFIRMED')
                try:
                    from .propose import call_once
                    call_once(root/'prepared',root/'model',provider_factory=provider_factory)
                except Exception:
                    # Preserve a completed malformed patch response for explicit recovery; never retry API.
                    if not (root/'model/response.json').is_file():return save('FAILED',reason='model failed; sanitized model/usage.json retained; no retry',verdict='REJECT')
                raw=(root/'model/response.json').read_bytes()
                state.update(new_external_model_calls=0 if provider_factory else 1,external_call_outcome='CPU_MOCK' if provider_factory else 'RESPONSE_STORED')
            else:
                (root/'response.json').write_bytes(raw)
                try:historical=strict_json(raw.decode()).get('usage')
                except (ValueError,UnicodeError,AttributeError):historical=None
                write(root/'usage.json',{'real_external_model_call':False,'calls_attempted':0,'source':'recorded response; original usage is historical','historical_usage':historical,'credentials_read':False})
            try:
                body=strict_json(raw.decode());choice=body['choices'][0]
                if body.get('model')!='deepseek-flash' or choice['finish_reason']!='stop':raise ValueError('unsupported model or incomplete response')
                content=choice['message']['content'];base=(ASSETS/'kernels/gemm/wmma_sm86.cu').read_bytes()
                try:patched=apply_reviewable_patch(content,base)
                except ValueError as error:
                    write(root/'patch-validation.json',{'strict_status':'REJECT','error':str(error)[:1000],'explicit_recount_requested':bool(recount)})
                    if not recount:raise
                    patched=apply_reviewable_patch(content,base,recount=True)
                source=root/'source';source.mkdir()
                for name in ['wmma_sm86.cu','cublas_sm86.cu','gemm_contract.cuh']:(source/name).write_bytes(patched['updated_source'] if name=='wmma_sm86.cu' else (ASSETS/'kernels/gemm'/name).read_bytes())
                patched.pop('updated_source');parent=identity(4);candidate={k:v for k,v in parent.items() if k!='id'};candidate['files']=candidate['files'].copy();candidate['files']['kernels/gemm/wmma_sm86.cu']=digest(source/'wmma_sm86.cu');candidate['id']=hashlib.sha256(canonical(candidate)).hexdigest()
                write(root/'proposal.json',patched);write(root/'source-binding.json',{'parent_id':parent['id'],'candidate':candidate,'raw_response_sha256':hashlib.sha256(raw).hexdigest(),'source_origin':source_kind,'reviewed':False})
                save('NEEDS_SOURCE_REVIEW',candidate_id=candidate['id'],source_sha256=patched['source_sha256'],source_files={p.name:digest(p) for p in source.iterdir()})
            except (IndexError,KeyError,TypeError,ValueError,UnicodeError) as error:
                return save('REJECT',reason='invalid_proposal: '+str(error)[:1000],verdict='REJECT')
        try:
            binding=json.loads((root/'source-binding.json').read_text());source=root/'source'
            if any(digest(source/name)!=value for name,value in state['source_files'].items()):raise ValueError('reviewed source changed')
            known=json.loads((ASSETS/'known-rejections.json').read_text()).get(state['candidate_id'])
            if known and not checks_only:
                write(root/'decision.json',{'verdict':'REJECT','reason':'known_candidate_aa_unstable','known_evidence':known,'incumbent_after':state['parent_id'],'new_gpu_measurements':0})
                return save('REJECT',verdict='REJECT',reason='known_candidate_aa_unstable; no resampling',known_decision_reused=True)
            if review_sha is None:return state
            if review_sha!=state['source_sha256']:raise ValueError('source review SHA mismatch')
            if deps is None:raise ValueError('--deps with existing public build dependencies required')
            binding['reviewed']=True;write(root/'source-binding.json',binding)
            write(root/'review.json',{'source_sha256':review_sha,'source_files':state['source_files'],'kind':'explicit trusted-controller source review binding; not a security proof','checks_only':checks_only})
            if evaluator or provider_factory:state['evidence_kind']='CPU_MOCK'
            save('EVALUATING')
            if evaluator is None:
                from .pipeline import run
                evaluator=run
            result=evaluator(ASSETS,source,Path(deps),root/'evaluation',binding,checks_only=checks_only)
            write(root/'evaluation-summary.json',result)
            if checks_only and result['verdict']=='KEEP':raise ValueError('checks-only cannot produce a screening KEEP')
            if 'decision' in result:write(root/'decision.json',result['decision'])
            verdict=result['verdict'];stage='CHECKS_COMPLETE' if verdict=='CORRECTNESS_ONLY' else ('SCREENING_PASS' if verdict=='KEEP' else ('NO_PROMOTION' if verdict=='NO_PROMOTION_PENDING_AB' else 'REJECT'))
            return save(stage,verdict=verdict,reason=result['reason'],qa_checks=result.get('qa_checks'),aa_status=result.get('aa_status'))
        except Exception as error:
            secret=os.getenv('DEEPSEEK_API_KEY','');message=str(error)
            if secret:message=message.replace(secret,'[REDACTED]')
            return save('FAILED',reason=message[:1000],error_type=type(error).__name__,verdict='REJECT')
    finally:
        os.close(lock)
