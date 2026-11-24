"""CPU audit of frozen release and private raw evidence; emit aggregate facts only."""
import hashlib,json,zipfile
from pathlib import Path
from tools.verify_gemm_release_freeze import ROOT,verify


def main():
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    frozen=verify(ROOT/'releases/gemm-sm86-rc1');summary=json.loads((ROOT/'evidence/20261122-gemm-independent-acceptance/summary.json').read_text());private=ROOT/'.private/gemm-sm86-rc1-evidence';raw=private/'raw'
    assert summary['status']=='PASS' and summary['correct_outputs']==summary['checked_outputs']==summary['executed_gpu_calls']==16
    assert sha(private/'archive.zip')==summary['private_raw_evidence_sha256']
    count=0
    with zipfile.ZipFile(private/'archive.zip') as archive:
        execution=json.loads(archive.read('execution.json'));assert execution['exit']==0
        for name,meta in execution['manifest'].items():
            value=archive.read('blobs/'+meta['sha256']);assert len(value)==meta['bytes'] and hashlib.sha256(value).hexdigest()==meta['sha256']==sha(raw/name);count+=1
    seal=json.loads((ROOT/'releases/gemm-sm86-rc1/protocol.freeze.json').read_text())
    for path,digest in seal['trusted_evaluator_sha256'].items():assert sha(ROOT/path)==digest
    assert sha(raw/'manifest.json')==summary['manifest_commitment']
    assert sha(raw/'protocol.json')==summary['protocol_sha256']==frozen['protocol_sha256']
    assert sha(raw/'freeze.json')==seal['candidate_freeze_sha256']
    checks=json.loads((raw/'private/checks.json').read_text());assert len(checks)==16
    for row in checks:
        assert row['correct'] and row['record']['cleanup']['removed']
        meta=row['record']['metadata'];assert all(meta[key] is True for key in ['guards_ok','producer_consumer_stream','output_contract']);assert meta['uid']==65534
    recovery=json.loads((ROOT/'evidence/20261122-gemm-rc1-public-recovery/result.json').read_text());assert recovery['exit']==0 and recovery['result']['status']=='PASS'
    assert len(recovery['result']['records'])==6 and sum(len(x['execution']['attempts']) for x in recovery['result']['records'])==8
    result={'status':'PASS','kind':'CPU hash/metadata audit of one-shot GPU evidence, no new GPU calls','freeze_id':frozen['freeze_id'],'private_files_verified':count,'independent_outputs':16,'public_recovery_scenarios':6,'public_recovery_attempts':8,'private_raw_evidence_sha256':summary['private_raw_evidence_sha256'],'heldout_exposed_to_model':False,'performance_claim':False,'production_release':False}
    (ROOT/'evidence/20261122-gemm-independent-acceptance/verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
