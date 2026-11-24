"""Existing atomic package-pointer operation, restricted to the two accepted installation prefixes."""
import argparse,fcntl,json,os,sys,uuid
from pathlib import Path

def switch(root,package):
 loc=json.loads((root/'location.json').read_text());target=loc['prefix'] if package=='release' else loc.get('previous_prefix');assert package in ['release','preview']
 if not target or not (Path(target)/'kernel_agent/cdi_cli.py').is_file():raise ValueError('accepted package prefix is unavailable')
 temporary=root/('.application-next-'+uuid.uuid4().hex)
 try:temporary.symlink_to(Path(target));os.replace(temporary,root/'current')
 finally:temporary.unlink(missing_ok=True)
 return {'status':'PACKAGE_ACTIVATED','package':package,'current_prefix':str((root/'current').resolve()),'development_channel_required':package=='preview'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--deployment-root',type=Path,required=True);p.add_argument('package',choices=['release','preview']);a=p.parse_args();lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);print(json.dumps(switch(a.deployment_root.resolve(),a.package)))
if __name__=='__main__':
 try:main()
 except (OSError,ValueError,AssertionError) as error:print(json.dumps({'status':'ERROR','error':str(error)[:512]}),file=sys.stderr);raise SystemExit(1)
