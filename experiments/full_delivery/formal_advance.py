"""Dispatch ready branches in separate processes, respecting their frozen state.

Credential environment is process-local even with concurrent independent searches.
Pending proposals are never resent; this module never performs a GPU evaluation.
"""
import argparse,json,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from experiments.full_delivery.formal_study import verify,branch
from experiments.full_delivery.formal_results import resolve_pending


def advance(root,workers=4):
    p=verify(root);resolved=resolve_pending(root);tasks=[]
    for op,ids in p['targets'].items():
        for wid in ids:
            for rep in range(3):
                for method in ['B','C']:
                    b=branch(root,op,wid,rep,method)
                    if (b/'closed.json').exists() or any(x.is_dir() and x.name.isdecimal() and not (x/'decision.json').exists() for x in b.iterdir()):continue
                    tasks.append((op,wid,rep,method))
    def run(task):
        code="from pathlib import Path\nfrom experiments.full_delivery.formal_study import propose\nroot=Path("+repr(str(root))+ ")\nfor _ in range(13):\n r=propose(root,*"+repr(task)+")\n if r['status'] not in ['RESOLVED_EXACT_CACHE','REJECT_INVALID_PROPOSAL','MODEL_CALL_FAILED_CONSUMED']:\n  print(r,flush=True);break\n"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        return {'branch':task,'exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    print(json.dumps({'ready_branches':len(tasks),'resolved_from_cache':len(resolved)}),flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(run,t) for t in tasks]):print(json.dumps(f.result()),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();advance(a.root.resolve())
