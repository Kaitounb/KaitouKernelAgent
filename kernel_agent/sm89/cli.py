"""Installed entrypoint for allowlisted SM89 SDK versions; no source execution commands."""
import argparse,json,os,sys,uuid
from pathlib import Path
from kernel_agent.deployment import execute,load_input
from .catalog import registry
from .runner import NativeRunner


def main():
    p=argparse.ArgumentParser(description='Reviewed SM89 cuBLAS runtime preview; target production acceptance pending');sub=p.add_subparsers(dest='command',required=True)
    for command in ['init','status','activate','rollback','run']:
        parser=sub.add_parser(command);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--channel',choices=['development','qualified'],default='development')
        if command=='activate':parser.add_argument('--version',required=True)
        if command=='run':
            for name in ['a','b','output','worker-runtime']:parser.add_argument('--'+name,type=Path,required=True)
            parser.add_argument('--inject-failure',choices=['crash','timeout'])
    a=p.parse_args()
    try:
        store=registry(a.root,channel=a.channel)
        if a.command=='init':result=store.initialize()
        elif a.command=='status':result=store.status()
        elif a.command=='activate':result=store.activate(a.version)
        elif a.command=='rollback':result=store.rollback()
        else:
            import numpy as np
            if a.channel=='qualified':
                from kernel_agent.registry import read_regular
                runtime=json.loads(read_regular(a.worker_runtime,65536));accepted=store.catalog['qualified_environment']
                if any(runtime.get(k)!=accepted[v] for k,v in [('gpu_identity','gpu_identity'),('manifest_sha256','public_runtime_manifest_sha256'),('python_sha256','python_sha256')]):raise ValueError('SM89 qualified runtime binding differs')
            x=load_input(a.a);y=load_input(a.b)
            actual,result=execute(a.root,x,y,inject_failure=a.inject_failure,channel=a.channel,registry_factory=registry,runner_factory=lambda binary,digest:NativeRunner(binary,digest,runtime=a.worker_runtime))
            temporary=a.output.with_name('.'+a.output.name+'-'+uuid.uuid4().hex)
            try:
                with temporary.open('xb') as f:np.save(f,actual,allow_pickle=False);f.flush();os.fsync(f.fileno())
                os.replace(temporary,a.output)
            finally:temporary.unlink(missing_ok=True)
            result.update(output=str(a.output),target='sm89-reviewed-native',source_execution_enabled=False)
        print(json.dumps(result))
    except (OSError,ValueError,RuntimeError,KeyError) as error:
        print(json.dumps({'status':'ERROR','error':str(error)[:1024],'production_release':False}),file=sys.stderr);raise SystemExit(1)


if __name__=='__main__':main()
