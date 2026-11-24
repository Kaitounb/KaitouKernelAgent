"""Read-only project state/runtime packaging on the old authorized server."""
import argparse,fcntl,json,os,platform,subprocess,sys,time
from pathlib import Path
from backup_archive import snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;roots=json.loads((root/'roots.json').read_text());assert all(str(p).startswith('/tmp/nka-') for p in roots);base=Path('/tmp/nka-full-server-dqnrbkzs');lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);metadata=root/'metadata';metadata.mkdir();records=[]
 for command in [[str(base/'tools-v2/bin/uv'),'pip','freeze','--python',str(base/'venv/bin/python')],['/usr/local/cuda-12.6/bin/nvcc','--version'],['/usr/local/cuda-12.6/bin/ncu','--version'],['ldd','--version'],['uname','-a']]:
  r=subprocess.run(command,capture_output=True,text=True,timeout=60);records.append({'command':command,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
 (metadata/'environment.json').write_text(json.dumps({'platform':platform.platform(),'os_release':Path('/etc/os-release').read_text(),'venv_python_realpath':os.path.realpath(base/'venv/bin/python'),'records':records,'system_cuda_not_in_runtime_archive':True,'system_driver_not_archived':True},indent=2)+'\n')
 packages=[base/n for n in ['venv','python','tools-v2','uv-wheel']];exclude=[base/'uv-cache',*packages];critical=snapshot([(Path(p),str(Path(p).relative_to('/'))) for p in roots]+[(metadata,'metadata')],root/'critical',exclude);(root/'progress.json').write_text(json.dumps({'status':'CRITICAL_PACKED','critical':critical},indent=2)+'\n')
 runtime=snapshot([(p,str(p.relative_to('/'))) for p in packages],root/'runtime');result={'status':'PACKED','root':str(root),'critical':critical,'runtime':runtime,'excluded_uv_cache_duplicate_or_temporary':True,'originals_modified':False,'new_gpu_calls':0,'new_model_calls':0};(root/'progress.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
