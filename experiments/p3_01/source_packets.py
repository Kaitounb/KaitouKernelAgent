"""Bounded standard ZIP_STORED results; expected files/lengths come only from trusted cases."""
import io,json,stat,zipfile

REPORT_LIMIT=1024*1024


def pack(report,files):
    raw=io.BytesIO()
    with zipfile.ZipFile(raw,'w',compression=zipfile.ZIP_STORED) as archive:
        archive.writestr('report.json',json.dumps(report,allow_nan=False))
        for name,data in files.items():archive.writestr(name,data)
    return raw.getvalue()


def unpack(payload,limits):
    wanted={'report.json':REPORT_LIMIT,**limits}
    if len(payload)>sum(wanted.values())+len(wanted)*512:raise ValueError('source packet too large')
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            info=archive.infolist()
            if len(info)!=len(wanted) or {i.filename for i in info}!=set(wanted):raise ValueError('source packet names/count mismatch')
            files={}
            for entry in info:
                if entry.filename in files or stat.S_IFMT(entry.external_attr>>16) not in [0,stat.S_IFREG] or entry.is_dir() or entry.flag_bits&1 or entry.compress_type!=zipfile.ZIP_STORED or entry.file_size!=entry.compress_size or not 0<entry.file_size<=wanted[entry.filename]:raise ValueError('source packet entry invalid')
                files[entry.filename]=archive.read(entry)
            try:report=json.loads(files.pop('report.json'))
            except (UnicodeError,RecursionError) as error:raise ValueError('invalid source report') from error
            if not isinstance(report,dict):raise ValueError('source report must be an object')
            return report,files
    except (zipfile.BadZipFile,RuntimeError,EOFError) as error:raise ValueError('invalid source packet') from error
