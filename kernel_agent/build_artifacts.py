"""Closed-name, bounded build-result decoding. Never loads native code on the host."""
import io,stat,struct,zipfile

LIMITS={'nka_gemm_v1.so':8*1024*1024,'build.ninja':128*1024,'main.cpp':256*1024,'cuda.cu':256*1024}
ARCHIVE_LIMIT=9*1024*1024


def check_elf(data):
    # Basic platform tag only. Correctness/trust requires a separate isolated run.
    if len(data)<64 or data[:6]!=b'\x7fELF\x02\x01' or struct.unpack_from('<HH',data,16)!=(3,62):raise ValueError('expected x86-64 little-endian ELF shared library')


def decode_artifacts(payload):
    if len(payload)>ARCHIVE_LIMIT:raise ValueError('artifact archive exceeds limit')
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries=archive.infolist()
            if len(entries)!=len(LIMITS) or {e.filename for e in entries}!=set(LIMITS):raise ValueError('artifact names/count invalid')
            result={}
            for info in entries:
                mode=stat.S_IFMT(info.external_attr>>16)
                if info.filename in result or mode not in [0,stat.S_IFREG] or info.is_dir():raise ValueError('artifact entry type/duplicate invalid')
                if info.flag_bits&1 or info.compress_type!=zipfile.ZIP_STORED or info.file_size!=info.compress_size:raise ValueError('only unencrypted stored artifacts accepted')
                if not 0<info.file_size<=LIMITS[info.filename]:raise ValueError('artifact entry size invalid')
                data=archive.read(info)
                if len(data)!=info.file_size:raise ValueError('artifact truncated')
                result[info.filename]=data
    except (zipfile.BadZipFile,RuntimeError,NotImplementedError,EOFError) as error:raise ValueError('invalid artifact archive') from error
    check_elf(result['nka_gemm_v1.so'])
    return result
