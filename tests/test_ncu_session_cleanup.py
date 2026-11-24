"""Real Linux process-tree regression for compiler children using another process group."""
import os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from experiments.p1_04.ncu_adapter import stop_profile_session


@unittest.skipUnless(sys.platform.startswith('linux'),'requires Linux /proc sessions')
class SessionCleanupTests(unittest.TestCase):
    def test_child_in_different_group_is_removed_without_touching_controller(self):
        with tempfile.TemporaryDirectory() as temp:
            marker=Path(temp)/'child.pid'
            code="import os,subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],preexec_fn=os.setpgrp);Path(sys.argv[1]).write_text(str(p.pid));time.sleep(20)"
            proc=subprocess.Popen([sys.executable,'-c',code,str(marker)],start_new_session=True)
            try:
                deadline=time.monotonic()+3
                while not marker.exists() and time.monotonic()<deadline:time.sleep(.02)
                self.assertTrue(marker.exists());child=int(marker.read_text());self.assertNotEqual(os.getpgid(child),proc.pid);self.assertEqual(os.getsid(child),proc.pid)
                result=stop_profile_session(proc.pid);self.assertTrue(result['removed']);self.assertEqual(result['remaining_running'],[]);proc.wait(timeout=3)
                self.assertNotEqual(os.getpgrp(),proc.pid)
            finally:
                # The same owned session is safe to drain even after an assertion.
                stop_profile_session(proc.pid)
                if proc.poll() is None:proc.kill();proc.wait(timeout=3)


if __name__=='__main__':unittest.main()
