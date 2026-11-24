"""CPU replay of low-privilege NCU packets, lifecycle and trusted-oracle witnesses."""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
from kernel_agent.protocol import decode_frame
from kernel_agent.optimization.source_packets import unpack


def audit(folder):
    folder=Path(folder);archive=folder/'archive';raw=(archive/'raw.zip').read_bytes();collection=json.loads((archive/'collection.json').read_text());assert hashlib.sha256(raw).hexdigest()==collection['archive_sha256'];root=archive/'remote'
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        manifest=json.loads(z.read('manifest.json'))
        for n,m in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==m['sha256'] and (root/n).stat().st_size==m['bytes']
    result=json.loads((root/'result.json').read_text());assert result['status']=='PASS';project=Path(__file__).resolve().parents[2];ledgers=json.loads((project/'evidence/20261123-mini-gemm-pointer-source/remote/run/evaluation/oracle-ledgers.json').read_text());limits={'execution.json':1024*1024,'metrics.csv':2*1024*1024,'report.ncu-rep':12*1024*1024,'validation.json':65536};observed=0
    for row in result['rows']:
        work=root/row['label'];record=json.loads((work/'record.json').read_text());cmd=record['command'];assert all(x in cmd for x in ['--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=32','--memory=4g','--memory-swap=4g','--cpus=2']);assert cmd[cmd.index('--user')+1]=='65534:65534' and cmd.count('--mount')==3 and all('readonly' in cmd[i+1] for i,x in enumerate(cmd) if x=='--mount');assert record['exit']==0 and record['reason'] is None and record['cleanup']['removed'] and record['protected_unchanged'];meta,packet=decode_frame(work/'capture/stdout.bin',sum(limits.values())+2*1024*1024);report,files=unpack(packet,{n:limits[n] for n in meta['artifact_names']});assert meta==record['metadata'] and meta['uid']==65534 and meta['cpu_reference_available'] is False and report=={'format':'bounded-ncu-v1','files':sorted(files)};assert all((work/n).read_bytes()==data for n,data in files.items());assert all(hashlib.sha256((work/n).read_bytes()).hexdigest()==h for n,h in record['protected_hashes'].items())
        if row['label']=='no-target-rejected':assert record['status']=='UNAVAILABLE' and meta['status']=='PROFILE_FAILED' and row['expected_rejection'];continue
        parsed=json.loads((work/'parsed.json').read_text());validation=json.loads((work/'validation.json').read_text());execution=json.loads((work/'execution.json').read_text());assert len(parsed['launches'])==3 and all(len(x['metrics'])==19 and 'gemm_wmma<4>' in x['kernel'] and all(v['status']=='observed' for v in x['metrics'].values()) for x in parsed['launches']);assert parsed['csv_sha256']==hashlib.sha256(files['metrics.csv']).hexdigest() and execution['exit']==0 and execution['cleanup']['removed'];role='incumbent' if row['label']=='baseline' else 'candidate';matches=[x for x in ledgers[role].values() if x['case']['shape_mnk']==[512,512,512] and x['case']['seed']==171 and x['case']['distribution']=='random' and not x['case']['offset_guard']];assert len(matches)==1;oracle=matches[0];assert validation['input_hashes']==oracle['input_hashes'] and validation['output_hashes']==[oracle['output_sha256']]*3 and validation['binary_sha256']==meta['binary_sha256'];observed+=3
    value={'status':'PASS','scope':'real UID65534 fixed-image NCU baseline / missing-target refusal / reviewed-source recovery, independently replayed by CPU','native_profiled_launches':observed,'metric_fields_per_profile':19,'raw_files_verified':len(manifest),'successful_profiles':2,'expected_profile_rejections':1,'host_oracle_gpu_hashes_pass':True,'new_model_calls':0,'event_sampling':False,'production_release':False};(folder/'verification.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value));return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();audit(a.folder)
