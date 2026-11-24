"""Apply one backed-up, authorized HAGS setting. Never restarts Windows."""
import argparse,base64,hashlib,json,uuid
from pathlib import Path
from tools.ssh_3070 import ROOT,execute_python

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);script=(ROOT/'experiments/full_delivery/windows_hags_setting.ps1').read_text();backup='C:\\Windows\\Temp\\nka-hags-'+uuid.uuid4().hex+'\\backup.json';command='& {\n'+script+'\n} -Action Enable -BackupPath '+"'"+backup+"'";encoded=base64.b64encode(command.encode('utf-16-le')).decode();plan={'action':'Back up original single HwSchMode value and set DWORD2; no Windows reboot','script_sha256':hashlib.sha256(script.encode()).hexdigest(),'backup_path':backup,'actual_hags_enabled_claimed':False,'new_gpu_measurement':False};(a.output/'plan.json').write_text(json.dumps(plan,indent=2));(a.output/'setting.ps1').write_text(script)
 payload='ENCODED='+repr(encoded)+'\nBACKUP='+repr(backup)+'\n'+r'''
import fcntl,json,subprocess
from pathlib import Path,PureWindowsPath
lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
r=subprocess.run(['docker','ps','--filter','name=nka-','--format','{{.ID}}'],capture_output=True,text=True,timeout=20);assert r.returncode==0 and not r.stdout.strip()
r=subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',ENCODED],capture_output=True,text=True,timeout=60)
if r.returncode:print(json.dumps({'status':'FAILED','exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}))
else:
 result=json.loads(r.stdout);assert result['Status']=='PASS' and result['After']['Value']==2 and not result['RebootPerformed'];path=Path('/mnt/c').joinpath(*PureWindowsPath(BACKUP).parts[1:]);original=json.loads(path.read_text(encoding='utf-8-sig'));print(json.dumps({'status':'CONFIGURED_PENDING_REBOOT','result':result,'original_backup':original,'reboot_performed':False,'qualification_passed':False,'production_release':False}))
'''
 (a.output/'driver.py').write_text(payload);r=execute_python(payload,120);(a.output/'transport.stderr').write_text(r.stderr)
 if r.returncode:(a.output/'transport.stdout').write_text(r.stdout);raise RuntimeError('Setting transport failed; inspect existing state; do not replay')
 v=json.loads(r.stdout);(a.output/'result.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
