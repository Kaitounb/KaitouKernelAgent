"""Real subprocess/byte-limit checks; no GPU, Docker, NumPy, model or network."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from kernel_agent.bounded_io import capture,confirmed_removed
from kernel_agent.protocol import HEADER,MAGIC,META_LIMIT,decode_frame,encode_frame


class BoundedIOTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def invoke(self,script,**kwargs):
        self.calls=[]
        def cleanup(reason):self.calls.append(reason);return {'called':True}
        return capture([sys.executable,'-c',script],self.root/'capture',timeout=kwargs.get('timeout',3),stdout_limit=kwargs.get('stdout_limit',128),stderr_limit=128,cleanup=cleanup)
    def test_removal_confirmation_is_specific(self):
        for message in ['error: no such object: name','Error: No such container: name']:
            self.assertTrue(confirmed_removed({'exit':1,'stderr':message}))
        for result in [{'exit':None,'stderr':'timeout'},{'exit':1,'stderr':'Cannot connect to Docker daemon'},{'exit':0,'stderr':'No such object'}]:
            self.assertFalse(confirmed_removed(result))
    def test_log_creation_failure_never_starts_worker(self):
        directory=self.root/'capture';directory.mkdir();(directory/'stdout.bin').mkdir()
        with patch('kernel_agent.bounded_io.subprocess.Popen') as spawn:
            with self.assertRaises(OSError):self.invoke('print("never")')
            spawn.assert_not_called()
        self.assertEqual(self.calls,[])
    def test_log_write_error_still_cleans_up(self):
        real_open=Path.open
        class BrokenWriter:
            def __init__(self,file):self.file=file
            def write(self,data):raise OSError('injected disk full')
            def close(self):self.file.close()
        def opened(path,*args,**kwargs):
            file=real_open(path,*args,**kwargs)
            return BrokenWriter(file) if path.name=='stdout.bin' else file
        with patch('kernel_agent.bounded_io.Path.open',new=opened):result=self.invoke('print("output")')
        self.assertEqual(result['reason'],'host_io_error');self.assertEqual(self.calls,['host_io_error']);self.assertTrue(result['cleanup']['called'])
    def test_cleanup_exception_remains_unconfirmed(self):
        def failed(reason):raise OSError('injected control failure')
        result=capture([sys.executable,'-c','print("ok")'],self.root/'capture',timeout=3,stdout_limit=128,stderr_limit=128,cleanup=failed)
        self.assertFalse(result['cleanup']['removed']);self.assertEqual(result['cleanup']['error_type'],'OSError')
    def test_small_output_success(self):
        r=self.invoke('print("ok")');self.assertEqual(r['exit'],0);self.assertIsNone(r['reason']);self.assertEqual(self.calls,[None])
    def test_stdout_flood_capped(self):
        r=self.invoke('import os; os.write(1,b"x"*65536)');self.assertEqual(r['reason'],'stdout_limit');self.assertEqual((self.root/'capture/stdout.bin').stat().st_size,128);self.assertEqual(len(self.calls),1)
    def test_stderr_flood_capped(self):
        r=self.invoke('import os; os.write(2,b"x"*65536)');self.assertEqual(r['reason'],'stderr_limit');self.assertEqual(r['stored_bytes']['stderr'],128)
    def test_timeout_cleanup(self):
        r=self.invoke('import time; time.sleep(10)',timeout=.1);self.assertEqual(r['reason'],'timeout');self.assertEqual(self.calls,['timeout'])
    def test_exit_failure_preserved(self):self.assertEqual(self.invoke('raise SystemExit(17)')['exit'],17)
    def frame(self,data):p=self.root/'frame';p.write_bytes(data);return p
    def test_frame_round_trip(self):
        self.assertEqual(decode_frame(self.frame(encode_frame({'uid':65534},b'abc')),10),({'uid':65534},b'abc'))
    def test_large_header_rejected(self):
        with self.assertRaises(ValueError):decode_frame(self.frame(HEADER.pack(MAGIC,META_LIMIT+1,0)),10)
    def test_truncated_payload(self):
        with self.assertRaises(ValueError):decode_frame(self.frame(encode_frame({},b'abc')[:-1]),10)
    def test_trailing_data_rejected(self):
        with self.assertRaises(ValueError):decode_frame(self.frame(encode_frame({})+b'x'),10)
    def test_nested_duplicate_nonfinite_metadata(self):
        for data in [b'{"a":{}}',b'{"a":1,"a":2}',b'{"a":NaN}',b'[]']:
            with self.subTest(data=data),self.assertRaises(ValueError):decode_frame(self.frame(HEADER.pack(MAGIC,len(data),0)+data),10)
    def test_invalid_utf8(self):
        with self.assertRaises(ValueError):decode_frame(self.frame(HEADER.pack(MAGIC,1,0)+b'\xff'),10)
    def test_payload_limit(self):
        with self.assertRaises(ValueError):decode_frame(self.frame(encode_frame({},b'abc')),2)


if __name__=='__main__':unittest.main()
