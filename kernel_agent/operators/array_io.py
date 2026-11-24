"""Typed NPY admission adapted from the verified runtime output reader, no pickle."""
import io,os,stat
from pathlib import Path
import numpy as np


def decode_array(raw,shape,dtype):
    expected=np.dtype(dtype);shape=tuple(shape);size=int(np.prod(shape))*expected.itemsize
    if len(raw)>size+16384:raise ValueError('NPY exceeds expected array size')
    file=io.BytesIO(raw);version=np.lib.format.read_magic(file)
    if version==(1,0):observed,fortran,actual=np.lib.format.read_array_header_1_0(file,max_header_size=10000)
    elif version==(2,0):observed,fortran,actual=np.lib.format.read_array_header_2_0(file,max_header_size=10000)
    else:raise ValueError('unsupported NPY version')
    if observed!=shape or fortran or actual!=expected or actual.hasobject or len(raw)-file.tell()!=size:raise ValueError('NPY shape/dtype/order/payload mismatch')
    file.seek(0);return np.load(file,allow_pickle=False,max_header_size=10000)


def load_array(path,shape,dtype):
    limit=int(np.prod(shape))*np.dtype(dtype).itemsize+16384;fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as file:
        info=os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size>limit:raise ValueError('NPY input type/size invalid')
        raw=file.read(limit+1)
    return decode_array(raw,shape,dtype)


def encode_array(array):
    raw=io.BytesIO();np.save(raw,array,allow_pickle=False);return raw.getvalue()
