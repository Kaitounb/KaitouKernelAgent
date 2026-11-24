"""CPU-only hostile output-file checks; uses NumPy already present on the target host."""
import io,os,tempfile,unittest
from pathlib import Path
import numpy as np
try:
    from experiments.p3_02.supervisor import load_output
except ModuleNotFoundError:
    from supervisor import load_output


class OutputDecoderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'output.npy';self.shape=(2,3)
    def tearDown(self):self.temp.cleanup()
    def rejected(self):
        with self.assertRaises((ValueError,OSError,EOFError)):load_output(self.path,self.shape)
    def test_valid_float32(self):
        expected=np.arange(6,dtype=np.float32).reshape(self.shape);np.save(self.path,expected)
        np.testing.assert_array_equal(load_output(self.path,self.shape),expected)
    def test_symlink_rejected(self):
        target=self.path.with_name('other');target.write_text('untrusted');self.path.symlink_to(target);self.rejected()
    def test_fifo_rejected_without_blocking(self):os.mkfifo(self.path);self.rejected()
    def test_wrong_shape(self):np.save(self.path,np.zeros((3,2),np.float32));self.rejected()
    def test_wrong_dtype(self):np.save(self.path,np.zeros(self.shape,np.float64));self.rejected()
    def test_object_payload_not_unpickled(self):np.save(self.path,np.full(self.shape,object(),dtype=object));self.rejected()
    def test_forged_huge_shape(self):
        with self.path.open('wb') as out:np.lib.format.write_array_header_1_0(out,{'descr':'<f4','fortran_order':False,'shape':(10**12,10**12)})
        self.rejected()
    def test_truncated_payload(self):
        np.save(self.path,np.zeros(self.shape,np.float32));self.path.write_bytes(self.path.read_bytes()[:-4]);self.rejected()
    def test_oversized_sparse_file(self):
        with self.path.open('wb') as out:out.truncate(20000)
        self.rejected()
    def test_fortran_rejected(self):np.save(self.path,np.asfortranarray(np.zeros(self.shape,np.float32)));self.rejected()


if __name__=='__main__':unittest.main()
