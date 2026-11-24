"""Authorized SSH transport factored from the existing P0/P1 experiment scripts."""
import os,re,socket,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def execute_python(payload,timeout=1200):
    lines=(ROOT/'.private/3070-connection.txt').read_text().splitlines()
    windows=next(x for x in lines if 'Windows' in x and '密码' in x)
    account=lambda x:re.search(r'账号\s*([^，,\s]+)',x).group(1)
    password=lambda x:re.search(r'密码[：:]\s*(\S+)',x).group(1)
    env=os.environ.copy()
    for name in list(env):
        if 'API_KEY' in name.upper() or 'AUTH_TOKEN' in name.upper():env.pop(name)
    try:
        with socket.create_connection(('127.0.0.1',21022),timeout=1):pass
    except OSError:
        cloud=next(x for x in lines if '腾讯云' in x and '密码' in x and x!=windows)
        host=re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b',cloud).group(0)
        port=re.search(r'端口\s*(\d+)',cloud).group(1);frp=re.search(r'本地转发端口\s*(\d+)',windows).group(1)
        env['SSHPASS']=password(cloud)
        cmd=['/opt/homebrew/bin/sshpass','-e','ssh','-M','-S','/tmp/nvidia-kernel-agent-p0-04-3070.sock','-fNT','-L',f'127.0.0.1:21022:127.0.0.1:{frp}',
             '-p',port,'-o','StrictHostKeyChecking=yes','-o','ExitOnForwardFailure=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3',account(cloud)+'@'+host]
        result=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=30)
        if result.returncode:raise ConnectionError('Authorized tunnel unavailable: '+result.stderr)
    env['SSHPASS']=password(windows)
    cmd=['/opt/homebrew/bin/sshpass','-e','ssh','-p','21022','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',account(windows)+'@127.0.0.1',
         'wsl -d Ubuntu -u root -- /home/dministrator/cuda-lab/.venv/bin/python -']
    return subprocess.run(cmd,input=payload,env=env,text=True,capture_output=True,timeout=timeout)
