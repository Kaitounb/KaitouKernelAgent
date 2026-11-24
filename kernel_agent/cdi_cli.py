"""Installed RTX3060 nativeCDI entrypoint for reviewed fixed cuBLAS only."""
import argparse,json,os,sys,uuid
from pathlib import Path
from kernel_agent.deployment import execute,load_input
from .cdi_qualification import registry
from .cdi_runner import DockerRunner


def main():
    p=argparse.ArgumentParser(description='Evidence-bound RTX3060 reviewed cuBLAS runtime');sub=p.add_subparsers(dest='command',required=True)
    for command in ['init','status','activate','rollback','run']:
        parser=sub.add_parser(command);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--channel',choices=['development','qualified'],default='development')
        if command=='activate':parser.add_argument('--version',required=True)
        if command=='run':
            for name in ['a','b','output']:parser.add_argument('--'+name,type=Path,required=True)
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
            x=load_input(a.a);y=load_input(a.b)
            actual,result=execute(a.root,x,y,inject_failure=a.inject_failure,channel=a.channel,registry_factory=registry,runner_factory=DockerRunner)
            temporary=a.output.with_name('.'+a.output.name+'-'+uuid.uuid4().hex)
            try:
                with temporary.open('xb') as f:np.save(f,actual,allow_pickle=False);f.flush();os.fsync(f.fileno())
                os.replace(temporary,a.output)
            finally:temporary.unlink(missing_ok=True)
            result.update(output=str(a.output),target='rtx3060-reviewed-cdi',source_execution_enabled=False)
        print(json.dumps(result))
    except (OSError,ValueError,RuntimeError,KeyError) as error:
        print(json.dumps({'status':'ERROR','error':str(error)[:1024],'production_release':False}),file=sys.stderr);raise SystemExit(1)


if __name__=='__main__':main()
