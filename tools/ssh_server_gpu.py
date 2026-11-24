"""Authorized server transport. Credential stays in private file and SSHPASS child environment."""
import json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def execute_python(payload,timeout=120,*,connection_path=None,control_path=None):
    connection=json.loads((Path(connection_path) if connection_path is not None else ROOT/'.private/server-gpu-connection.json').read_text());env={k:v for k,v in os.environ.items() if 'API_KEY' not in k.upper() and 'AUTH_TOKEN' not in k.upper()};env['SSHPASS']=connection['password']
    cmd=['/opt/homebrew/bin/sshpass','-e','ssh','-T','-p',str(connection['port']),'-o','StrictHostKeyChecking=accept-new','-o','ConnectTimeout=15','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3',connection['user']+'@'+connection['host'],'python3 -']
    if control_path is not None:cmd[3:3]=['-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath='+str(control_path)]
    return subprocess.run(cmd,input=payload,env=env,capture_output=True,text=True,timeout=timeout)
