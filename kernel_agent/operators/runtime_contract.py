"""Bounded request and tensor specs for reviewed native operator artifacts only."""
import numpy as np


def _integer(value,low,high):
    if type(value) is not int or not low<=value<=high:raise ValueError('operator dimension outside contract')
    return value


def specifications(request):
    if not isinstance(request,dict) or request.get('operation') not in ['softmax','attention','mla']:raise ValueError('unknown reviewed operator')
    op=request['operation'];common={'operation','bundle_id','fault'}
    if request.get('fault') not in [None,'crash','timeout','constant']:raise ValueError('unknown reviewed failure fixture')
    if type(request.get('bundle_id')) is not str or len(request['bundle_id'])!=64:raise ValueError('bundle binding required')
    if op=='softmax':
        if set(request)!=common|{'rows','width','dtype'} or request['dtype'] not in ['float16','float32']:raise ValueError('softmax request schema mismatch')
        rows=_integer(request['rows'],1,65536);width=_integer(request['width'],1,16384);inputs={'x.npy':((rows,width),request['dtype'])};outputs={'output.npy':((rows,width),'float32')}
    elif op=='attention':
        if set(request)!=common|{'batch','sq','sk','heads','dim','causal','mask'} or type(request['causal']) is not bool or type(request['mask']) is not bool:raise ValueError('attention request schema mismatch')
        b=_integer(request['batch'],1,16);sq=_integer(request['sq'],1,8192);sk=_integer(request['sk'],sq,8192);h=_integer(request['heads'],1,128);d=request['dim']
        if type(d) is not int or d not in [64,128]:raise ValueError('unsupported attention dimension')
        inputs={'q.npy':((b,sq,h,d),'float16'),'k.npy':((b,sk,h,d),'float16'),'v.npy':((b,sk,h,d),'float16')}
        if request['mask']:inputs['mask.npy']=((sq,sk),'bool')
        outputs={'output.npy':((b,sq,h,d),'float16')}
    else:
        if set(request)!=common|{'lengths','heads','page_size','splits','num_pages'}:raise ValueError('MLA request schema mismatch')
        lengths=request['lengths']
        if not isinstance(lengths,list) or not 1<=len(lengths)<=16:raise ValueError('MLA batch outside contract')
        for n in lengths:_integer(n,0,8192)
        h=request['heads'];page=request['page_size'];splits=request['splits']
        if type(h) is not int or h not in [16,32,64,128] or type(page) is not int or page not in [1,16,64] or type(splits) is not int or splits not in [1,2,4]:raise ValueError('invalid controlled MLA configuration')
        pages=sum((n+page-1)//page for n in lengths)
        if type(request['num_pages']) is not int or request['num_pages']!=pages:raise ValueError('MLA physical page count mismatch')
        b=len(lengths);inputs={'q_abs.npy':((b,h,512),'float16'),'q_rope.npy':((b,h,64),'float16'),'cache.npy':((pages,page,512),'float16'),'rope_cache.npy':((pages,page,64),'float16'),'indptr.npy':((b+1,),'int32'),'indices.npy':((pages,),'int32')};outputs={'z.npy':((b,h,512),'float16'),'lse.npy':((b,h),'float32')}
    total=sum(int(np.prod(shape))*np.dtype(dtype).itemsize for shape,dtype in inputs.values())
    if total>192*1024*1024:raise ValueError('operator input exceeds declared 192MiB admission limit')
    return inputs,outputs


def validate_arrays(request,arrays):
    inputs,outputs=specifications(request)
    if set(arrays)!=set(inputs):raise ValueError('operator input names differ')
    for name,(shape,dtype) in inputs.items():
        value=arrays[name]
        if value.shape!=shape or value.dtype!=np.dtype(dtype) or not value.flags.c_contiguous:raise ValueError('operator array layout differs')
        if value.dtype.kind=='f':
            limit=10000 if request['operation']=='softmax' else 4
            if not np.isfinite(value).all() or np.any(np.abs(value)>limit):raise ValueError('operator finite-value contract violated')
    if request['operation']=='mla':
        ptr=arrays['indptr.npy'].tolist();indices=arrays['indices.npy'].tolist();page=request['page_size'];lengths=request['lengths'];expected=[0]
        for n in lengths:expected.append(expected[-1]+(n+page-1)//page)
        if ptr!=expected or sorted(indices)!=list(range(request['num_pages'])):raise ValueError('invalid/nonpermutation MLA page table')
    return outputs
