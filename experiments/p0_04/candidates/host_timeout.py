"""Finite GPU work, then a host wait with one child in the worker process group."""
import json
import signal
import subprocess
import sys
import time
import torch
from baseline import launch as correct


def launch(x, y, n):
    correct(x, y, n)
    torch.cuda.synchronize()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])

    def terminate(signum, frame):
        child.wait(timeout=1)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    print(json.dumps({"fault": "host_timeout", "gpu_synchronized": True,
                      "child_pid": child.pid}), flush=True)
    time.sleep(3600)
