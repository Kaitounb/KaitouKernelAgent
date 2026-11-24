"""Read-only evidence recovery via a binary SFTP transfer; never reruns candidate/API/GPU."""
import argparse,hashlib,json,os,re,shutil,subprocess,zipfile
from pathlib import Path
from tools.ssh_3070 import execute_python,ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--remote',required=True);p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
    if not re.fullmatch(r'/tmp/nka-p3-01-source-[a-z0-9_]+',a.remote) or a.evidence.exists():p.error('expected an owned source run and new output')
    a.evidence.mkdir();shutil.copytree(a.original/'sources',a.evidence/'sources')
    for name in ['source-hashes.json','source-review.json']:shutil.copyfile(a.original/name,a.evidence/name)
    payload='SOURCE='+repr(a.remote)+'\n'+r'''
import hashlib,json,stat,uuid,zipfile
from pathlib import Path
root=Path(SOURCE);result=json.loads((root/'run/result.json').read_text())
assert result['qa_checks']==160 and result['aa_status']=='NOT_RUN_SANITIZER_NOT_ACCEPTED'
win=Path('/mnt/c/Users/Administrator/AppData/Local/Temp');assert win.is_dir()
target=win/('nka-p3-01-'+uuid.uuid4().hex+'.zip');manifest={};blobs={}
for item in root.rglob('*'):
 rel=item.relative_to(root)
 if any(x in ['project','deps','artifact','__pycache__'] for x in rel.parts):continue
 info=item.lstat()
 if not stat.S_ISREG(info.st_mode):continue
 assert info.st_size<=128*1024*1024
 digest=hashlib.sha256(item.read_bytes()).hexdigest();manifest[str(rel)]={'sha256':digest,'bytes':info.st_size};blobs.setdefault(digest,item)
record={'remote_work':str(root),'exit':0,'manifest':manifest,'collection':'read-only recovery; original finished result retained; no new pipeline execution'}
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as out:
 out.writestr('execution.json',json.dumps(record))
 for digest,item in blobs.items():out.write(item,'blobs/'+digest)
print(json.dumps({'windows_path':'C:/Users/Administrator/AppData/Local/Temp/'+target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size}))
'''
    (a.evidence/'collect-remote.py').write_text(payload);r=execute_python(payload,timeout=180)
    (a.evidence/'collect-stderr.log').write_text(r.stderr)
    if r.returncode or not r.stdout.strip():raise RuntimeError('read-only collection failed; no measurements rerun')
    record=json.loads(r.stdout);(a.evidence/'transfer-binding.json').write_text(json.dumps(record,indent=2)+'\n')
    path=record['windows_path']
    if not re.fullmatch(r'C:/Users/Administrator/AppData/Local/Temp/nka-p3-01-[0-9a-f]{32}\.zip',path):raise ValueError('unexpected download path')
    lines=(ROOT/'.private/3070-connection.txt').read_text().splitlines();line=next(x for x in lines if 'Windows' in x and '密码' in x)
    account=re.search(r'账号\s*([^，,\s]+)',line).group(1);password=re.search(r'密码[：:]\s*(\S+)',line).group(1)
    env={k:v for k,v in os.environ.items() if 'API_KEY' not in k.upper() and 'AUTH_TOKEN' not in k.upper()};env['SSHPASS']=password
    target=a.evidence/'transport.zip'
    cmd=['/opt/homebrew/bin/sshpass','-e','scp','-q','-P','21022','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',account+'@127.0.0.1:'+path,str(target)]
    transfer=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=1800);(a.evidence/'scp-stderr.log').write_text(transfer.stderr)
    if transfer.returncode:raise RuntimeError('binary evidence transfer failed; raw original remains remote')
    if target.stat().st_size!=record['bytes'] or hashlib.sha256(target.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('download archive checksum mismatch')
    with zipfile.ZipFile(target) as archive:
        execution=json.loads(archive.read('execution.json'));remote=a.evidence/'remote';remote.mkdir();blobs=a.evidence/'blobs';blobs.mkdir()
        for rel,meta in execution['manifest'].items():
            p=Path(rel)
            if p.is_absolute() or '..' in p.parts:raise ValueError('invalid evidence path')
            digest=meta['sha256'];blob=blobs/digest
            if not blob.exists():
                data=archive.read('blobs/'+digest)
                if len(data)!=meta['bytes'] or hashlib.sha256(data).hexdigest()!=digest:raise ValueError('archive member changed')
                blob.write_bytes(data);blob.chmod(0o444)
            output=remote/p;output.parent.mkdir(parents=True,exist_ok=True);os.link(blob,output)
    (a.evidence/'execution.json').write_text(json.dumps(execution,indent=2)+'\n');print(json.dumps({'status':'COLLECTED_NO_REEXECUTION','archive_bytes':record['bytes'],'files':len(execution['manifest'])}))


if __name__=='__main__':main()
