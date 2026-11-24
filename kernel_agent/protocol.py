"""Bounded byte framing. No pickle, object hooks, candidate paths or recursive metadata."""
import json,struct
from pathlib import Path

MAGIC=b'NKA1';HEADER=struct.Struct('!4sIQ');META_LIMIT=8192


def _pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate metadata key')
        result[key]=value
    return result


def _constant(value):raise ValueError('nonfinite metadata constant')


def validate_metadata(meta):
    if not isinstance(meta,dict) or len(meta)>64:raise ValueError('metadata must be a small flat object')
    for key,value in meta.items():
        if not isinstance(key,str) or len(key)>128:raise ValueError('invalid metadata key')
        if value is None or type(value) is bool:continue
        if type(value) is int and -(2**63)<=value<2**63:continue
        if type(value) is str and len(value)<=2048:continue
        if type(value) is list and len(value)<=32 and all(type(x) is str and len(x)<=128 for x in value):continue
        raise ValueError('metadata type/size invalid')
    return meta


def encode_frame(meta,payload=b''):
    validate_metadata(meta)
    encoded=json.dumps(meta,separators=(',',':'),allow_nan=False).encode()
    if len(encoded)>META_LIMIT:raise ValueError('metadata too large')
    return HEADER.pack(MAGIC,len(encoded),len(payload))+encoded+payload


def decode_frame(path,max_payload):
    # This file is created by the trusted bounded pipe reader, never mounted in the worker.
    path=Path(path)
    if path.stat().st_size>HEADER.size+META_LIMIT+max_payload:raise ValueError('frame too large')
    with path.open('rb') as file:
        head=file.read(HEADER.size)
        if len(head)!=HEADER.size:raise ValueError('truncated frame header')
        magic,meta_bytes,payload_bytes=HEADER.unpack(head)
        if magic!=MAGIC or meta_bytes>META_LIMIT or payload_bytes>max_payload:raise ValueError('invalid frame header/limits')
        if path.stat().st_size!=HEADER.size+meta_bytes+payload_bytes:raise ValueError('frame length mismatch')
        try:meta=json.loads(file.read(meta_bytes),object_pairs_hook=_pairs,parse_constant=_constant)
        except (UnicodeError,RecursionError) as error:raise ValueError('invalid metadata encoding/depth') from error
        validate_metadata(meta)
        return meta,file.read(payload_bytes)
