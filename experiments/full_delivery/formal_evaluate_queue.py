"""One-shot sequential launcher for explicit trusted reviews and reservations."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from experiments.full_delivery.formal_study import ROOT,verify,sha
from kernel_agent.registry import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('queue',type=Path);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();verify(root);queue=a.queue.resolve();launches=root/'evaluation-dispatches';launches.mkdir(exist_ok=True)
    for row in json.loads(queue.read_text()):
        folder=Path(row['folder']);evidence=Path(row['evidence']);review=folder/'source-review.json';proposal=json.loads((folder/'proposal.json').read_text());assert proposal['descriptor']['id']==row['id'] and proposal['source_change']['source_sha256']==row['source']==sha(folder/'wmma_sm86.cu');reservation=json.loads((root/'evaluation-reservations'/(row['id']+'.json')).read_text());assert ROOT/reservation['evidence']==evidence and not evidence.exists();assert json.loads(review.read_text())['source_sha256']==row['source'];log=launches/row['id'];log.mkdir(exist_ok=False);atomic_json(log/'reservation.json',{'queue_sha256':sha(queue),'source_sha256':row['source'],'automatic_retry':False});print('START',row['id'],flush=True);start=time.monotonic()
        with (log/'stdout').open('x') as out,(log/'stderr').open('x') as err:r=subprocess.run([sys.executable,'-m','experiments.full_delivery.run_feedback_evaluation','--proposal-dir',str(folder/'eval-input'),'--review-file',str(review),'--output',str(evidence)],cwd=ROOT,stdout=out,stderr=err)
        atomic_json(log/'receipt.json',{'exit':r.returncode,'wall_seconds':time.monotonic()-start,'automatic_retry':False});print('END',row['id'],r.returncode,flush=True)


if __name__=='__main__':main()
