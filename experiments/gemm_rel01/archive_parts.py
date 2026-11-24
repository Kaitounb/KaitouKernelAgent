"""Lossless frame/ZIP_STORED member segmentation for content-addressed evidence transport.
No extraction/execution; reconstruction must match each original whole-file hash.
"""
import io,struct,zipfile
from kernel_agent.protocol import HEADER,MAGIC


def parts(data):
    try:
        magic,meta,payload=HEADER.unpack(data[:HEADER.size])
        offset=HEADER.size+meta
        if magic!=MAGIC or offset+payload!=len(data):return [data]
        raw=data[offset:];cuts=[]
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for entry in archive.infolist():
                if entry.compress_type!=zipfile.ZIP_STORED or entry.flag_bits&1:return [data]
                local=entry.header_offset
                if raw[local:local+4]!=b'PK\x03\x04':return [data]
                name,extra=struct.unpack_from('<HH',raw,local+26);start=offset+local+30+name+extra;end=start+entry.file_size
                if not offset<=start<end<=len(data):return [data]
                cuts.append((start,end))
        result=[];cursor=0
        for start,end in sorted(cuts):
            if start<cursor:return [data]
            if start>cursor:result.append(data[cursor:start])
            result.append(data[start:end]);cursor=end
        if cursor<len(data):result.append(data[cursor:])
        return result or [data]
    except (ValueError,struct.error,zipfile.BadZipFile,EOFError):return [data]
