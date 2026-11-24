"""NPY checks extracted unchanged from the verified P3-02 supervisor.load_output."""
import os,stat
import numpy as np


def load_output(path,shape):
    # Never follow a candidate-created symlink or let a forged header allocate unbounded host memory.
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as file:
        info=os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size>int(np.prod(shape))*4+16384:raise ValueError('output type/size invalid')
        version=np.lib.format.read_magic(file)
        if version==(1,0):observed,fortran,dtype=np.lib.format.read_array_header_1_0(file,max_header_size=10000)
        elif version==(2,0):observed,fortran,dtype=np.lib.format.read_array_header_2_0(file,max_header_size=10000)
        else:raise ValueError('unsupported numpy format')
        if observed!=shape or fortran or dtype!=np.dtype('float32') or dtype.hasobject:raise ValueError('output shape/dtype/order invalid')
        if info.st_size-file.tell()!=int(np.prod(shape))*4:raise ValueError('output payload size mismatch')
        file.seek(0);return np.load(file,allow_pickle=False,max_header_size=10000)
