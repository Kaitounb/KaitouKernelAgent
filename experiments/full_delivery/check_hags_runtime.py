"""Read the actual post-boot Windows HAGS mode; registry alone is insufficient."""
import argparse,base64,hashlib,json,uuid
from pathlib import Path
from tools.ssh_3070 import ROOT,execute_python

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--require-enabled',action='store_true');a=p.parse_args();assert not a.output.exists();identity=uuid.uuid4().hex
 payload='IDENTITY='+repr(identity)+'\n'+r'''
import base64,fcntl,hashlib,json,re,subprocess,time
from pathlib import Path
lock=open('/tmp/nka-reviewed-gpu-execution.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
name='nka-hags-runtime-'+IDENTITY+'.txt';windows='C:/Windows/Temp/'+name;target=Path('/mnt/c/Windows/Temp')/name
r=subprocess.run(['/mnt/c/Windows/System32/dxdiag.exe','/whql:off','/t',windows],capture_output=True,timeout=90);assert r.returncode==0
for _ in range(60):
 if target.exists() and target.stat().st_size>0:break
 time.sleep(1)
raw=target.read_bytes();text=raw.decode('utf-16' if raw.startswith(b'\xff\xfe') else 'utf-8',errors='replace');cards=[];current=None
for line in text.splitlines():
 s=line.strip()
 if s.startswith('Card name:'):
  current={'name':s.split(':',1)[1].strip()};cards.append(current)
 elif s.startswith('Driver Model:') and current is not None:current['driver_model']=s.split(':',1)[1].strip()
 elif s.startswith('Hardware Scheduling:') and current is not None:
  m=re.search(r'Supported:(True|False) Enabled:(True|False)',s)
  if m:current.update(hags_supported=m[1]=='True',hags_enabled=m[2]=='True')
selected=[c for c in cards if c['name']=='NVIDIA GeForce RTX 3070'];assert len(selected)==1 and 'hags_enabled' in selected[0]
script="[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @{BootUtc=(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')} | ConvertTo-Json -Compress"
p=subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',script],capture_output=True,text=True,timeout=30);assert p.returncode==0;boot=json.loads(p.stdout)
g=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=uuid,driver_version,name','--format=csv,noheader'],capture_output=True,text=True,timeout=15);assert g.returncode==0
print(json.dumps({'actual_adapter':selected[0],'windows_boot_utc':boot['BootUtc'],'gpu_identity':g.stdout.strip(),'raw_sha256':hashlib.sha256(raw).hexdigest(),'raw_base64':base64.b64encode(raw).decode(),'remote_raw_path':windows}))
'''
 r=execute_python(payload,210)
 if r.returncode:raise RuntimeError('Read-only HAGS runtime check unavailable; do not launch qualification')
 value=json.loads(r.stdout);raw=base64.b64decode(value.pop('raw_base64'));assert hashlib.sha256(raw).hexdigest()==value['raw_sha256'];private=ROOT/'.private/admission-repair-hardware'/identity;private.mkdir(mode=0o700,parents=True,exist_ok=False);f=private/'dxdiag-original.txt';f.write_bytes(raw);f.chmod(0o600)
 value.update(status='HAGS_ENABLED' if value['actual_adapter']['hags_enabled'] else 'HAGS_DISABLED_PENDING_REBOOT_OR_POLICY',scope='Actual trusted DxDiag runtime state; no qualification or private acceptance execution',private_raw_preserved=True,new_model_calls=0,production_release=False);a.output.mkdir(parents=True,exist_ok=False);(a.output/'runtime.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
 if a.require_enabled and (not value['actual_adapter']['hags_supported'] or not value['actual_adapter']['hags_enabled']):raise SystemExit(2)
if __name__=='__main__':main()
