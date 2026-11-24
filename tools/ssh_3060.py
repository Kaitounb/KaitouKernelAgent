"""Dedicated authorized3060 transport with a private reusable SSH control socket."""
from .ssh_server_gpu import ROOT,execute_python as _execute

def execute_python(payload,timeout=120):
 return _execute(payload,timeout,connection_path=ROOT/'.private/server-gpu-3060-next-connection.json',control_path=ROOT/'.private/ssh-3060-next.sock')
