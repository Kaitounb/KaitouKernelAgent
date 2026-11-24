"""Read private values only inside this CPU auditor; emit aggregate proof, never shapes/seeds."""
import hashlib,io,json
from pathlib import Path
import numpy as np
from kernel_agent.protocol import decode_frame
from kernel_agent.sm89.runner import validate_boundary
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    private=ROOT/'.private/sm89-rc2-independent-evidence';raw=private/'remote';public=ROOT/'evidence/20261123-sm89-independent';summary=json.loads((public/'summary.json').read_text());freeze=json.loads((public/'freeze.json').read_text());execution=json.loads((private/'execution.json').read_text())
    assert sha(private/'archive.zip')==summary['private_archive_sha256']
    for n,m in execution['manifest'].items():assert (raw/n).stat().st_size==m['bytes'] and sha(raw/n)==m['sha256']
    assert sha(raw/'manifest.json')==summary['manifest_sha256']==freeze['freeze']['manifest_sha256']
    assert sha(raw/'protocol.json')==summary['protocol_sha256']==freeze['freeze']['protocol_sha256']
    assert sha(ROOT/'experiments/full_delivery/sm89_independent_host.py')==freeze['freeze']['host_sha256']
    wheel=ROOT/'artifacts/wheels/nvidia_kernel_agent-0.1.0rc2-py3-none-any.whl';assert sha(wheel)==summary['wheel_sha256']
    manifest=json.loads((raw/'manifest.json').read_text());checks=json.loads((raw/'checks.json').read_text());protocol=json.loads((raw/'protocol.json').read_text());assert len(manifest['workloads'])==4 and len(checks)==16 and summary['status']=='PASS' and summary['executed_gpu_calls']==summary['correct_outputs']==16
    expected={'sm89-cublas-current':'72d466d30e6b777b06d8e4216bdd3f4aef1306127926ac6f0976923d3406009d','sm89-cublas-baseline':'4572c346c20a70630a2521eac8c4a83163f26d931a7d07bbcd549e28220a43de'};count=0;maximum=0.
    for i,item in enumerate(manifest['workloads']):
        assert len(item['seeds'])==2
        for j in range(2):
            case=raw/('case-'+str(i)+'-'+str(j));a=np.load(case/'a.npy',allow_pickle=False);b=np.load(case/'b.npy',allow_pickle=False);reference=(a.astype(np.float64)@b.astype(np.float64)).astype(np.float32);saved=np.load(case/'reference.npy',allow_pickle=False)
            m,n,k=item['shape_mnk'];assert a.shape==(m,k) and b.shape==(k,n) and a.dtype==b.dtype==np.float16 and np.array_equal(saved,reference)
            for role,digest in expected.items():
                job=case/role;record=json.loads((job/'record.json').read_text());assert record['status']=='OUTPUT_VALID' and record['exit']==0 and record['reason'] is None and record['cleanup']['removed'] and record['protected_files_unchanged']
                assert all(sha(job/n)==h for n,h in record['protected_hashes'].items());assert sha(job/'artifact/nka_gemm_v1.so')==digest
                assert np.array_equal(np.load(job/'inputs/a.npy',allow_pickle=False),a) and np.array_equal(np.load(job/'inputs/b.npy',allow_pickle=False),b)
                meta,payload=decode_frame(job/'capture/stdout.bin',m*n*4+16384);validate_boundary(meta);assert meta['binary_sha256']==digest and meta==record['metadata'];actual=np.load(io.BytesIO(payload),allow_pickle=False);assert payload==(job/'result.npy').read_bytes() and actual.dtype==np.float32
                error=np.abs(actual.astype(np.float64)-reference.astype(np.float64));assert np.isfinite(actual).all() and np.all(error<=protocol['atol']+protocol['rtol']*np.abs(reference));maximum=max(maximum,float(error.max()));count+=1
    assert count==16 and maximum==summary['max_abs_error']
    value={'status':'AUDIT_PASS','scope':'Fresh server-local SM89 installed API; private whole-file/frame/NPY/independently recomputed CPU FP64 and actual UID/network/readonly boundary replay','private_files_verified':len(execution['manifest']),'checked_outputs':count,'max_abs_error':maximum,'manifest_sha256':summary['manifest_sha256'],'protocol_sha256':summary['protocol_sha256'],'wheel_sha256':summary['wheel_sha256'],'private_archive_sha256':summary['private_archive_sha256'],'old_private_manifest_read':False,'exact_values_exposed_to_model':False,'new_model_calls':0,'new_event_samples':0,'production_release':False};(public/'verification.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))


if __name__=='__main__':main()
