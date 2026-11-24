"""Export validated SDK binaries using FlashInfer's existing AOT layout; no recompilation."""
import argparse,hashlib,importlib.metadata,json,shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--softmax',type=Path,required=True);p.add_argument('--torch-license',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);a.output.chmod(0o755)
    shutil.copytree(a.prefix/'flashinfer',a.output/'flashinfer',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for path in a.prefix.glob('flashinfer_python-*.dist-info'):shutil.copytree(path,a.output/path.name)
    import einops
    module=Path(einops.__file__).parent;shutil.copytree(module,a.output/'einops',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    distribution=importlib.metadata.distribution('einops');metadata=Path(distribution._path);shutil.copytree(metadata,a.output/metadata.name)
    binaries=[]
    for source in sorted(a.cache.glob('*/cached_ops/*/*.so')):
        # The validated cache has no arbitrary names or source-supplied path traversal.
        name=source.parent.name
        if source.name!=name+'.so':raise ValueError('unexpected SDK binary layout')
        target=a.output/'flashinfer/data/aot'/name/source.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);binaries.append(str(target.relative_to(a.output)))
    if not binaries:raise ValueError('no validated native SDK binaries found')
    softmax=a.output/'softmax'/a.softmax.name;softmax.parent.mkdir();shutil.copyfile(a.softmax,softmax)
    notices=a.output/'THIRD_PARTY_NOTICES.txt';notices.write_text('FlashInfer 0.2.8: Apache-2.0, original notices in flashinfer_python-0.2.8.dist-info/licenses/.\nPyTorch v2.7.1 PersistentSoftmax.cuh: BSD-3-Clause, original license in licenses/PYTORCH_LICENSE.\nEinops 0.8.1: MIT, original notices retained in its dist-info.\nCUDA/Torch runtime libraries are supplied by the pinned existing image, not copied into this bundle.\n')
    (a.output/'licenses').mkdir();shutil.copyfile(a.torch_license,a.output/'licenses/PYTORCH_LICENSE')
    files={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.rglob('*') if f.is_file() and not f.is_symlink()}
    body={'format':'flashinfer-native-aot-v1','scope':'reviewed SM86 developer runtime bundle; qualification pending','torch':'2.14.0+cu130','cuda':'13.0','capability':[8,6],'sdk_version':'0.2.8','sdk_revision':'3f8317c67ac129facb8525b2fb581be42f634459','sdk_compiler_adapter':'jit/core.py C++20 flags only','einops_version':importlib.metadata.version('einops'),'aot_binaries':binaries,'softmax_binary':str(softmax.relative_to(a.output)),'files':files,'production_release':False}
    manifest={**body,'bundle_id':hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()};(a.output/'aot-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'status':'EXPORTED_NOT_RUNTIME_ACCEPTED','bundle_id':manifest['bundle_id'],'aot_binaries':len(binaries),'einops_version':body['einops_version'],'files':len(files)}))


if __name__=='__main__':main()
