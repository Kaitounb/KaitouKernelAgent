"""CPU-only outcome ingestion, binding expected source directly to frozen proposal."""
import argparse,json
from pathlib import Path
from experiments.full_delivery.formal_study import verify
from experiments.full_delivery.audit_reviewed_source import audit
from experiments.full_delivery.formal_results import ingest_gemm
from experiments.full_delivery.formal_failures import build_failure


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--folder',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();root=a.root.resolve();folder=a.folder.resolve();evidence=a.evidence.resolve();verify(root);proposal=json.loads((folder/'proposal.json').read_text())
    if (evidence/'remote/run/evaluation/result.json').exists():
        verification=audit(evidence,proposal['source_change']['source_sha256']);outcome=ingest_gemm(root,folder,evidence)
    else:
        verification=build_failure(root,folder,evidence);outcome=json.loads((folder/'decision.json').read_text())['outcome']
    print(json.dumps({'status':verification['status'],'candidate_id':outcome['candidate_id'],'verdict':outcome['verdict'],'reason':outcome['reason'],'qa_checks':verification.get('independently_checked_qa_outputs',0),'raw_segments':verification.get('raw_segments',0)}))


if __name__=='__main__':main()
