"""CPU integrity and provenance audit of accepted C initialization profiles."""
import argparse,hashlib,json
from pathlib import Path


def audit(gemm_folder,mla_folder,output):
    accepted=[];project=Path(__file__).resolve().parents[2];oracles=json.loads((project/'evidence/20261123-mini-gemm-pointer-source/remote/run/evaluation/oracle-ledgers.json').read_text())['incumbent']
    for op,folder in [('gemm',Path(gemm_folder)),('mla',Path(mla_folder))]:
        root=folder/'archive/remote';result=json.loads((root/'result.json').read_text())
        for row in result['rows']:
            if row['op']!=op:continue
            assert row['status']=='PASS' and row['launches']==3 and row['metric_fields']==19;work=root/'profiles'/op/row['workload']/str(row['repetition']);profile=json.loads((work/'profile.json').read_text());parsed=json.loads((work/'parsed.json').read_text());assert len(parsed['launches'])==3 and len(profile['facts'])==19;assert profile['profile_wall_seconds']==row['profile_wall_seconds'];csv=work/'worker/metrics.csv' if op=='gemm' else work/'native.csv';assert hashlib.sha256(csv.read_bytes()).hexdigest()==parsed['csv_sha256']==profile['provenance'][0]['csv_sha256'];assert profile['provenance'][0]['binary_sha256']==row['binary_sha256'];validation=json.loads((work/('worker/validation.json' if op=='gemm' else 'validation.json')).read_text());assert validation['status']=='PASS'
            if op=='gemm':
                record=json.loads((work/'worker/record.json').read_text());assert record['metadata']['uid']==65534 and record['cleanup']['removed'] and record['protected_unchanged'];shape=json.loads((work/'worker/inputs/request.json').read_text())['shape_mnk'];matches=[x for x in oracles.values() if x['case']['shape_mnk']==shape and x['case']['seed']==171 and x['case']['distribution']=='random' and not x['case']['offset_guard']];assert len(matches)==1 and validation['input_hashes']==matches[0]['input_hashes'] and validation['output_hashes']==[matches[0]['output_sha256']]*3
            else:assert validation['inputs_and_guards_unchanged'] and validation['page_permutation_seed']==811 and validation['offset_elements']==8 and list(validation['native_sha256'].values())==['cdc4c3ba11ef61b7e2778f93af247bc0ac53027daa8035c55b541e6ea3fd601e']
            accepted.append({'op':op,'workload':row['workload'],'repetition':row['repetition'],'profile_wall_seconds':row['profile_wall_seconds'],'profile_sha256':hashlib.sha256((work/'profile.json').read_bytes()).hexdigest()})
    assert len(accepted)==18 and len({(x['op'],x['workload'],x['repetition']) for x in accepted})==18;value={'status':'PASS','accepted_profiles':18,'native_launches':54,'metric_fields_each':19,'profile_wall_seconds':sum(x['profile_wall_seconds'] for x in accepted),'rows':accepted,'rejected_mla_dynamic_jit_profiles_retained':9,'heldout_read':False,'new_model_calls':0,'event_sampling':False,'production_release':False};Path(output).write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({k:value[k] for k in ['status','accepted_profiles','native_launches','profile_wall_seconds','new_model_calls']}));return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gemm',type=Path,required=True);p.add_argument('--mla',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.gemm,a.mla,a.output)
