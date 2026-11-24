"""Audit private sealed MLA evidence, emitting counts/hashes only."""
import hashlib,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import numpy as np
    from kernel_agent.protocol import decode_frame
    from kernel_agent.optimization.source_packets import unpack
    from kernel_agent.operators.array_io import decode_array
    from kernel_agent.operators.runtime_contract import specifications
    private=ROOT/'.private/operators-sm86-rc1-evidence';raw=private/'raw';release=ROOT/'releases/operators-sm86-rc1';summary=json.loads((ROOT/'evidence/20261123-mla-independent-acceptance/summary.json').read_text());assert summary['status']=='PASS' and summary['executed_gpu_calls']==summary['correct_invocations']==8 and summary['checked_output_tensors']==16 and sha(private/'archive.zip')==summary['private_raw_evidence_sha256'];assert sha(raw/'manifest.json')==summary['manifest_commitment']==json.loads((release/'protocol.json').read_text())['dataset_commitment_sha256']
    with zipfile.ZipFile(private/'archive.zip') as z:execution=json.loads(z.read('execution.json'))
    assert all((raw/n).stat().st_size==m['bytes'] and sha(raw/n)==m['sha256'] for n,m in execution['manifest'].items());checks=json.loads((raw/'private/checks.json').read_text());assert len(checks)==8;outputs_checked=0
    for row in checks:
        job=raw/'private'/row['job'];record=json.loads((job/'record.json').read_text());request=json.loads((job/'inputs/request.json').read_text());assert row['correct'] and record==row['record'];command=record['command'];assert command[command.index('--user')+1]=='65534:65534' and '--network=none' in command and '--cap-drop=ALL' in command and '--read-only' in command and '--security-opt=no-new-privileges' in command;mounts=[command[i+1] for i,x in enumerate(command) if x=='--mount'];assert len(mounts)==3 and all(m.endswith(',readonly') for m in mounts) and not any(any(s in m for s in ['manifest.json','reference-','.private','api-key']) for m in mounts);assert record['cleanup']['removed'] and record['protected_unchanged'] and all(sha(job/n)==h for n,h in record['protected_hashes'].items());meta=record['metadata'];assert meta['uid']==65534 and meta['reference_available'] is False and all(meta[k] for k in ['guards_ok','producer_consumer_stream','aot_prebuilt']);arrays={p.name:np.load(p,allow_pickle=False) for p in (job/'inputs').glob('*.npy')};assert meta['input_hashes']==[hashlib.sha256(arrays[n].tobytes()).hexdigest() for n in sorted(arrays)] and meta['gpu_input_hashes']==[hashlib.sha256(arrays[n].tobytes()).hexdigest() for n in sorted(arrays) if arrays[n].dtype.kind=='f'];_,specs=specifications(request);limits={n:int(np.prod(shape))*np.dtype(dtype).itemsize+16384 for n,(shape,dtype) in specs.items()};payload_limit=sum(limits.values())+1024*1024+(len(limits)+1)*512;frame,payload=decode_frame(job/'capture/stdout.bin',payload_limit);assert frame==meta;report,files=unpack(payload,limits)
        for name,spec in specs.items():
            assert files[name]==(job/name).read_bytes();actual=decode_array(files[name],*spec);target=np.load(job/('reference-'+name),allow_pickle=False);finite=np.isfinite(target);delta=np.abs(actual.astype(np.float64)[finite]-target[finite]);assert np.array_equal(np.isneginf(actual),np.isneginf(target)) and np.all(delta<=.002+.002*np.abs(target[finite]));outputs_checked+=1
    result={'status':'PASS','scope':'CPU private archive/SHA/NKA1/NPY/FP64-oracle/boundary replay; not new GPU execution','private_files_verified':len(execution['manifest']),'gpu_invocations':8,'independently_checked_output_tensors':outputs_checked,'private_archive_sha256':sha(private/'archive.zip'),'heldout_exposed_to_model':False,'new_model_calls':0,'performance_sampling':False,'production_release':False};(ROOT/'evidence/20261123-mla-independent-acceptance/verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
