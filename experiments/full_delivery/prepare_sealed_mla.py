"""Freeze a reviewable MLA runtime acceptance without reading private values."""
import hashlib,json
from pathlib import Path
from kernel_agent.registry import atomic_json
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    root=ROOT/'releases/operators-sm86-rc1';root.mkdir(exist_ok=False);study=ROOT/'evidence/20261123-formal-study-v1';closed=list((study/'searches/mla').rglob('closed.json'));assert len(closed)==27 and all(json.loads(p.read_text())['best']=='initial_parent' for p in closed)
    commitment=json.loads((ROOT/'workloads/mla_acceptance_commitment_v1.json').read_text());wheel=ROOT/'artifacts/wheels/nvidia_kernel_agent-0.1.0.dev6-py3-none-any.whl';assert sha(wheel)=='d5f4bbf83dc050271af66baf35738f3800d02c2592b6fec101457122edc4fb38';manifest=ROOT/'evidence/20261123-sdk-aot-runtime-gpu/remote/bundle/aot-manifest.json'
    # Reuse the preserved original AOT manifest, discovering only public evidence.
    if not manifest.exists():
        import zipfile
        archive=next((ROOT/'artifacts/operator-runtime').glob('sm86-cfae*.zip'))
        with zipfile.ZipFile(archive) as z:
            names=[n for n in z.namelist() if n.endswith('aot-manifest.json')];assert len(names)==1;raw=z.read(names[0])
    else:raw=manifest.read_bytes()
    meta=json.loads(raw);assert meta['bundle_id']=='cfae38ce5c0fac56545207fe93588c1193646886fc5952f113fab487ef0be6eb';(root/'aot-manifest.json').write_bytes(raw)
    freeze={'scope':'Frozen trusted MLA runtime default; negative formal searches retain page16/split1. Correctness acceptance is separate from performance and production activation.','wheel_filename':wheel.name,'wheel_sha256':sha(wheel),'bundle_id':meta['bundle_id'],'aot_manifest_sha256':hashlib.sha256(raw).hexdigest(),'remote_bundle':'/tmp/nka-sdk-aot-licensed-l9fahjmi/bundle','config':{'page_size':16,'splits':1},'completed_mla_searches':27,'formal_protocol_sha256':sha(study/'protocol.json'),'runtime_activation':False,'production_release':False};atomic_json(root/'freeze.json',freeze)
    protocol={'dataset_commitment_sha256':commitment['sha256'],'manifest_count':4,'seeds_per_workload':2,'expected_gpu_invocations':8,'expected_output_tensors':16,'reference':'CPU FP64 references.mla_core outside unchanged dev6 bounded worker','tolerance':{'atol':.002,'rtol':.002},'page_permutation':True,'offset_guards':True,'producer_consumer_stream':True,'candidate_access':'per-invocation input and request only; no full manifest, reference, credentials or trusted evidence mount','one_shot':True,'automatic_retry':False,'failure_policy':'preserve private evidence; no tuning or resampling of this acceptance set','performance_sampling':False,'production_release':False};atomic_json(root/'protocol.json',protocol)
    names=['experiments/full_delivery/sealed_mla_host.py','experiments/full_delivery/run_sealed_mla.py','experiments/full_delivery/audit_sealed_mla.py'];atomic_json(root/'protocol.freeze.json',{'candidate_freeze_sha256':sha(root/'freeze.json'),'protocol_sha256':sha(root/'protocol.json'),'trusted_evaluator_sha256':{n:sha(ROOT/n) for n in names}});print(json.dumps({'status':'FROZEN_NOT_EXECUTED','manifest_count':4,'private_values_read':False,'freeze_sha256':sha(root/'freeze.json'),'protocol_sha256':sha(root/'protocol.json'),'production_release':False}))


if __name__=='__main__':main()
