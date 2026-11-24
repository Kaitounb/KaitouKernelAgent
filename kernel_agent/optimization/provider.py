"""Pinned original KernelAgent provider loader, moved for installation."""
import importlib,json,sys,types
from .identity import ROOT,sha


def provider_class():
    folder=ROOT/'provenance/upstream-reviews/20261121-p1-03/KernelAgent/utils/providers'
    records=json.loads((folder.parents[1]/'source-record.json').read_text())
    for row in records:
        if sha(folder.parents[1]/row['path'])!=row['sha256']:
            raise RuntimeError('upstream provider file drift')
    package=types.ModuleType('nka_kernelagent_providers');package.__path__=[str(folder)]
    sys.modules[package.__name__]=package
    upstream=importlib.import_module(package.__name__+'.openai_base')
    class DeepSeekProvider(upstream.OpenAICompatibleProvider):
        @property
        def name(self):return 'deepseek'
        def _build_api_params(self,model_name,messages,**kwargs):
            params=super()._build_api_params(model_name,messages,**kwargs)
            params.update(response_format={'type':'json_object'},extra_body={'thinking':{'type':'disabled'}})
            return params
    return DeepSeekProvider

