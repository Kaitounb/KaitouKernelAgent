"""Create private independent workload values without exposing them to model/tool output."""
import hashlib,json,secrets
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    contract=ROOT/'workloads/mla_contract_v1.json';development=ROOT/'workloads/mla_development_v1.json';private=ROOT/'.private/mla_acceptance_v1.json'
    if private.exists():raise FileExistsError('private MLA manifest already frozen; never replace')
    choose=lambda low,high:low+secrets.randbelow(high-low+1)
    workloads=[{'id':'mla-accept-01','family':'long-latent','heads':128,'lengths':[choose(6145,7167)]}, {'id':'mla-accept-02','family':'intermediate-heads','heads':64,'lengths':[choose(769,1023) for _ in range(2)]}, {'id':'mla-accept-03','family':'ragged-independent','heads':32,'lengths':[choose(157,389) for _ in range(3)]}, {'id':'mla-accept-04','family':'batch-independent','heads':16,'lengths':[choose(257,383) for _ in range(8)]}]
    value={'contract_id':'mla-f16-r512-rope64-v1','split':'independent-acceptance','seeds':[secrets.randbits(62) for _ in range(2)],'workloads':workloads}
    raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode();private.write_bytes(raw);private.chmod(0o600)
    commitment={'contract_id':value['contract_id'],'count':4,'families':[x['family'] for x in workloads],'sha256':hashlib.sha256(raw).hexdigest(),'exact_shapes_and_seeds':'private evaluator only; not read by model','executed':False}
    (ROOT/'workloads/mla_acceptance_commitment_v1.json').write_text(json.dumps(commitment,indent=2)+'\n')
    freeze={'status':'WORKLOADS_FROZEN_NOT_GPU_ACCEPTED','public_workloads':8,'independent_workloads':4,'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [contract,development,ROOT/'workloads/mla_acceptance_commitment_v1.json']},'private_manifest_sha256':commitment['sha256'],'independent_executed':False,'heldout_exposed_to_model':False,'performance_claim':False}
    output=ROOT/'evidence/20261122-mla-workload-freeze';output.mkdir(exist_ok=False);(output/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n');print(json.dumps(freeze))


if __name__=='__main__':main()
