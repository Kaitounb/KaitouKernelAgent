"""Post-freeze public-input recovery drill against the unchanged installed RC wheel."""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
from kernel_agent.deployment import execute
from kernel_agent.registry import Registry


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--work',type=Path,required=True);args=parser.parse_args()
    args.work.mkdir(exist_ok=False);store=args.work/'store';registry=Registry(store);registry.initialize()
    rng=np.random.default_rng(771);a=(rng.standard_normal((17,23))*.25).astype(np.float16);b=(rng.standard_normal((23,19))*.25).astype(np.float16)
    reference=(a.astype(np.float64)@b.astype(np.float64)).astype(np.float32);records=[]
    def check(label,fault=None,fallback=False):
        result,record=execute(store,a,b,inject_failure=fault)
        assert np.all(np.abs(result-reference)<=3e-5+1e-4*np.abs(reference))
        assert record['status']=='PASS' and record['fallback_used']==fallback
        assert all(row.get('cleanup',{}).get('removed') for row in record['attempts'])
        if fault:assert len(record['attempts'])==2 and record['attempts'][0]['status']=='REJECT'
        records.append({'label':label,'execution':record,'output_sha256':hashlib.sha256(result.tobytes()).hexdigest()})
    check('default')
    for fault in ['crash','timeout']:check(fault,fault,True)
    active=store/'versions/sm86-cublas-container/nka_gemm_v1.so';saved=active.read_bytes();active.chmod(0o644);active.write_bytes(b'corrupt public recovery fixture')
    check('corrupt-active',fallback=True);active.write_bytes(saved);active.chmod(0o444)
    registry.activate('sm86-cublas-original');check('explicit-old-version');pointer=registry.rollback()['pointer'];assert pointer['active']=='sm86-cublas-container';check('rollback-current')
    result={'status':'PASS','public_shape_mnk':[17,19,23],'records':records,'pointer_after_rollback':pointer,'new_model_calls':0,'performance_sampling':False,'production_release':False}
    (args.work/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','public_scenarios':len(records),'attempts':sum(len(r['execution']['attempts']) for r in records),'rollback':pointer,'production_release':False}))


if __name__=='__main__':main()
