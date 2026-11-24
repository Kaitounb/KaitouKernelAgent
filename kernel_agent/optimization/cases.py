"""Trusted public development cases and ledger keys; never reads heldout data."""
import json
from pathlib import Path
from .identity import BOUNDARIES


def make_cases(project):
    development=json.loads((Path(project)/'workloads/gemm_development_v1.json').read_text())['workloads'];cases=[]
    def add(group,shape,seed,dist='random',offset=False,kind='contract'):
        cases.append({'id':'q'+str(len(cases)).zfill(3),'group':group,'kind':kind,'shape_mnk':list(shape),'seed':seed,'distribution':dist,'offset_guard':offset})
    for w in development:
        for seed in [171,172]:add(w['id'],w['shape_mnk'],seed)
    for shape,dist,offset in [*BOUNDARIES,((16,32,16),'random',False),((32,48,48),'random',False)]:
        for seed in [171,172]:add('boundaries',shape,seed,dist,offset)
    contract=cases.copy()
    for w in development:
        for seed in range(701,717):add(w['id'],w['shape_mnk'],seed,kind='graph')
    return development,contract,cases


def case_key(case):
    return json.dumps({k:case[k] for k in ['shape_mnk','seed','distribution','offset_guard']},sort_keys=True)
