"""Install and exercise reviewed cuBLAS deployment versions on the declared Docker/SM86 target."""
import argparse,json,os,sys,uuid
from pathlib import Path
from .registry import Registry


def main():
    p=argparse.ArgumentParser(description='Reviewed fixed GEMM deployment with evidence-bound qualified or development channel')
    sub=p.add_subparsers(dest='command',required=True)
    for command in ['init','status','activate','rollback','run']:
        parser=sub.add_parser(command);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--channel',choices=['development','qualified'],default='development',help='Explicit reviewed GEMM release channel')
        if command=='activate':parser.add_argument('--version',required=True)
        if command=='run':
            parser.add_argument('--a',type=Path,required=True);parser.add_argument('--b',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
            parser.add_argument('--inject-failure',choices=['crash','timeout'],help='Reviewed development recovery exercise only')
    optimize=sub.add_parser('optimize',help='One bounded offline source proposal/check; no release activation')
    optimize.add_argument('--run-dir',type=Path,required=True)
    source=optimize.add_mutually_exclusive_group();source.add_argument('--response',type=Path,help='Existing DeepSeek response, zero new API calls');source.add_argument('--deepseek',action='store_true')
    optimize.add_argument('--max-calls',type=int,choices=[0,1],default=0)
    optimize.add_argument('--recount',action='store_true',help='Explicit Git hunk-count recovery; original rejection retained')
    optimize.add_argument('--review-sha',help='SHA256 of source reviewed by the trusted controller')
    optimize.add_argument('--deps',type=Path,help='Existing public build dependencies, mounted read-only')
    optimize.add_argument('--checks-only',action='store_true',help='Maintenance correctness/sanitizer check; never timing or promotion')
    optimize.add_argument('--resume',action='store_true',help='Continue only pending source review; never retry uncertain API/GPU work')
    inspect=sub.add_parser('optimize-status');inspect.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args()
    try:
        if args.command in ['optimize','optimize-status']:
            from .optimization.command import execute,status
            if args.command=='optimize-status':result=status(args.run_dir)
            else:result=execute(args.run_dir,response=args.response,live=args.deepseek,max_calls=args.max_calls,recount=args.recount,review_sha=args.review_sha,deps=args.deps,checks_only=args.checks_only,resume=args.resume)
            print(json.dumps(result))
            if result['status']=='NEEDS_SOURCE_REVIEW':raise SystemExit(3)
            if result['status'] in ['FAILED','STOPPED_UNCERTAIN_CALL','STOPPED_NEEDS_CLEANUP']:raise SystemExit(1)
            return
        from .gemm_qualification import registry as runtime_registry
        registry=runtime_registry(args.root,channel=args.channel)
        if args.command=='init':result=registry.initialize()
        elif args.command=='status':result=registry.status()
        elif args.command=='activate':result=registry.activate(args.version)
        elif args.command=='rollback':result=registry.rollback()
        else:
            from .deployment import execute,load_input
            import numpy as np
            a=load_input(args.a);b=load_input(args.b)
            # Validate dimensions before starting Docker or attempting fallback.
            if a.shape[1]!=b.shape[0]:raise ValueError('input K mismatch')
            actual,result=execute(args.root,a,b,inject_failure=args.inject_failure,channel=args.channel,registry_factory=runtime_registry)
            temp=args.output.with_name('.'+args.output.name+'-'+uuid.uuid4().hex)
            try:
                with temp.open('xb') as file:np.save(file,actual,allow_pickle=False);file.flush();os.fsync(file.fileno())
                os.replace(temp,args.output)
            finally:temp.unlink(missing_ok=True)
            result['output']=str(args.output)
        print(json.dumps(result))
    except (OSError,ValueError,RuntimeError) as error:
        print(json.dumps({'status':'ERROR','error':str(error)[:1024],'production_release':False}),file=sys.stderr);raise SystemExit(1)


if __name__=='__main__':main()
