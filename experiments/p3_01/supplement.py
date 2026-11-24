"""Targeted log-channel fix. Reuse exact built binary and re-audit archived QA on host only."""
import argparse,json
from pathlib import Path
import torch
from kernel_agent.runner import digest
from experiments.p3_01.cases import make_cases
from experiments.p3_01.evaluate import expected_before,validate_qa
from experiments.p3_01.source_runner import SourceRunner
from experiments.p3_01.source_pipeline import finish


def main():
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);old=a.previous.resolve();project=old/'project';prior=json.loads((old/'run/result.json').read_text())
    if prior['qa_checks']!=160 or prior['aa_status']!='NOT_RUN_SANITIZER_NOT_ACCEPTED' or prior['aa_sessions']:raise ValueError('not an eligible log-only supplement')
    binary=old/'run/build/artifacts/nka_gemm_v1.so'
    if digest(binary)!=prior['binary_sha256']:raise ValueError('original source binary mismatch')
    a.work.mkdir();workloads,contract,cases=make_cases(project);ledger={};checks=[];records=[];sanitizers=[];monitors=[];unstable=[]
    for group in [w['id'] for w in workloads]+['boundaries']:
        selected=[case for case in cases if case['group']==group];job=old/'run'/('qa-'+group)
        report=json.loads((job/'report.json').read_text());record=json.loads((job/'record.json').read_text())
        if report['binary_sha256']!=prior['binary_sha256'] or record['status']!='WITNESSES_READY' or not record['protected_unchanged'] or not record['cleanup']['removed']:raise ValueError('original QA binding invalid')
        checks+=validate_qa(report,record,selected,expected_before(selected),job,ledger)
        records.append({'group':group,'count':len(selected),'worker_status':'WITNESSES_READY','origin':str(job),'rechecked_cpu_only':True})
    (a.work/'oracle-ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    (a.work/'supplement-binding.json').write_text(json.dumps({'previous':str(old),'previous_result_sha256':digest(old/'run/result.json'),'binary_sha256':prior['binary_sha256'],'qa_reused':160,'qa_new_gpu_calls':0,'reason':'compute-sanitizer --log-file /dev/stderr; unchanged permissions/source/binary/thresholds; original AA never started'},indent=2)+'\n')
    def progress():
        (a.work/'progress.json').write_text(json.dumps({'checks':checks,'qa_groups':records,'sanitizers':sanitizers,'unstable':unstable},indent=2)+'\n')
    progress()
    finish(a.work,prior['candidate'],{'status':prior['build_status'],'binary_sha256':prior['binary_sha256'],'source_identity':prior['source_identity']},SourceRunner(binary,prior['binary_sha256']),workloads,contract,checks,records,ledger,sanitizers,monitors,unstable,progress)


if __name__=='__main__':main()
