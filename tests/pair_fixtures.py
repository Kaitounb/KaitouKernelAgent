"""Pure CPU simulated ledgers/reports, deliberately not real CUDA evidence."""
import copy,hashlib,json
from kernel_agent.optimization.identity import ROOT,identity,canonical,sha
from kernel_agent.optimization.cases import make_cases,case_key
from kernel_agent.optimization.pair_contract import worker_hashes,pair_id
from kernel_agent.optimization.resources import verify_bundle


def fixture(speedups=(1.06,1.06,1.06)):
    parent=identity(4);candidate=copy.deepcopy(parent);candidate.pop('id');candidate['files']['kernels/gemm/wmma_sm86.cu']='a'*64;candidate['id']=hashlib.sha256(canonical(candidate)).hexdigest()
    descriptor={'schema_version':1,'parent':parent,'candidate':candidate,'binary_sha256':{'incumbent':json.loads((ROOT/'workloads/gemm_measurement_v1.json').read_text())['initial_template']['binary_sha256'],'candidate':'b'*64},'candidate_source_sha256':{name:candidate['files']['kernels/gemm/'+name] for name in ['wmma_sm86.cu','cublas_sm86.cu','gemm_contract.cuh']},'protocol_sha256':sha(ROOT/'workloads/gemm_measurement_v1.json'),'bundle_sha256':verify_bundle(),'worker_sha256':worker_hashes(),'environment':{'capability':[8,6],'torch':'2.14.0+cu130','cuda':'13.0'}}
    workloads,contract,cases=make_cases(ROOT);ledgers={role:{} for role in ['incumbent','candidate']}
    h=lambda text:hashlib.sha256(text.encode()).hexdigest()
    for role in ledgers:
        for case in cases:
            key=case_key(case);ledgers[role][key]={'case':case,'binary_sha256':descriptor['binary_sha256'][role],'output_sha256':h(role+key),'input_hashes':[h('a'+key),h('b'+key)],'max_abs_error':0.0}
    def check(case,role):
        import struct
        value=ledgers[role][case_key(case)];return {**case,'input_hashes':value['input_hashes'],'output_sha256':value['output_sha256'],'guard_hashes':[hashlib.sha256(struct.pack('<e',7.0)*34).hexdigest()]*2 if case['offset_guard'] else [],'fresh_output':True,'producer_consumer_stream':True}
    aa=[];ab=[]
    for mode,collection in [('aa',aa),('ab',ab)]:
        for session in range(3):
            hashes=descriptor['binary_sha256'];pid=pair_id(descriptor)
            record={'status':'WITNESSES_READY','exit':0,'reason':None,'protected_unchanged':True,'cleanup':{'removed':True,'after':{'exit':1,'stderr':'Error: No such container: CPU-fixture'}},'source_binary_sha256':hashes['candidate'],'incumbent_sha256':hashes['incumbent'],'pair_sha256':pid,'protected_hashes':{'artifact/'+name:val for name,val in descriptor['worker_sha256'].items()}}
            record['protected_hashes'].update({'artifact/'+role+'/nka_gemm_v1.so':digest for role,digest in hashes.items()});record['protected_hashes']['inputs/request.json']='f'*64
            report={'status':'PASS','mode':mode,'session':session,'graph_launches':32,'replays':12,'binary_sha256':hashes['candidate'],'candidate_binary_sha256':hashes['candidate'],'incumbent_binary_sha256':hashes['incumbent'],'pair_sha256':pid,'capability':[8,6],'torch':'2.14.0+cu130','cuda':'13.0','cpu_reference_available':False,'module_identity_distinct':True,'function_identity_distinct':True,'module_paths':{r:'/artifact/'+r+'/nka_gemm_v1.so' for r in hashes},'module_names':{r:'_nka_'+r+'.nka_gemm_v1' for r in hashes}}
            report['backends']={'incumbent':'template','candidate':'template'}
            for phase in ['checks_before','checks_after']:report[phase]=[check(c,'candidate') for c in contract] if mode=='aa' else {r:[check(c,r) for c in contract] for r in hashes}
            rows=[]
            for w in workloads:
                for count in [1,16]:
                    keys=[case_key({'shape_mnk':w['shape_mnk'],'seed':701+i,'distribution':'random','offset_guard':False}) for i in range(count)];segments=[]
                    for pair in range(9):
                        order=['incumbent','candidate'] if (pair+session)%2==0 else ['candidate','incumbent']
                        segments += [{'pair':pair,'label':role,'order':i,'elapsed_ms':10.0 if role=='incumbent' or mode=='aa' else 10/speedups[session]} for i,role in enumerate(order)]
                    rows.append({'workload':w['id'],'shape_mnk':w['shape_mnk'],'input_count':count,'capture_order':['incumbent','candidate'] if session%2==0 else ['candidate','incumbent'],'segments':segments,'input_hashes':[ledgers['candidate'][k]['input_hashes'] for k in keys],'output_hashes':{label:[ledgers['candidate' if mode=='aa' else label][keys[i%count]]['output_sha256'] for i in range(32)] for label in hashes}})
            report['candidate_aa' if mode=='aa' else 'measurements']=rows;collection.append({'report':report,'record':record})
    return aa,ab,ledgers,descriptor,contract,workloads
