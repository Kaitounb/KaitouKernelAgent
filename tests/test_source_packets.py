"""CPU-only hostile result-envelope tests; no numerical or GPU acceptance."""
import io,json,stat,unittest,warnings,zipfile
from experiments.p3_01.source_packets import pack,unpack


def packet(entries,compression=zipfile.ZIP_STORED):
    stream=io.BytesIO()
    with warnings.catch_warnings(),zipfile.ZipFile(stream,'w',compression=compression) as archive:
        warnings.simplefilter('ignore',UserWarning)
        for name,data in entries:archive.writestr(name,data)
    return stream.getvalue()


class SourcePacketTests(unittest.TestCase):
    def test_roundtrip(self):
        self.assertEqual(unpack(pack({'status':'PASS'},{'q0.npy':b'npy'}),{'q0.npy':8}),({'status':'PASS'},{'q0.npy':b'npy'}))

    def test_duplicate_or_unexpected_names(self):
        for entries in [[('report.json','{}'),('report.json','{}')],[('report.json','{}'),('../q0.npy','x')]]:
            with self.subTest(entries=entries),self.assertRaises(ValueError):unpack(packet(entries),{'q0.npy':8})

    def test_compression_rejected(self):
        with self.assertRaises(ValueError):unpack(packet([('report.json','{}'),('q0.npy','x')],zipfile.ZIP_DEFLATED),{'q0.npy':8})

    def test_nonobject_and_malformed_report_rejected(self):
        for raw in ['[]','null','"text"','{',b'\xff']:
            with self.subTest(raw=raw),self.assertRaises(ValueError):unpack(packet([('report.json',raw)]),{})

    def test_oversize_or_empty_entry_rejected(self):
        for raw in [b'',b'123456789']:
            with self.subTest(raw=raw),self.assertRaises(ValueError):unpack(pack({}, {'q0.npy':raw}),{'q0.npy':8})

    def test_symlink_and_truncation_rejected(self):
        name=zipfile.ZipInfo('q0.npy');name.create_system=3;name.external_attr=(stat.S_IFLNK|0o777)<<16
        for raw in [packet([('report.json','{}'),(name,'target')]),pack({}, {'q0.npy':b'x'})[:-12]]:
            with self.subTest(raw=len(raw)),self.assertRaises(ValueError):unpack(raw,{'q0.npy':8})


if __name__=='__main__':unittest.main()
