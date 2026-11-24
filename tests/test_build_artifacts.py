"""Host-side artifact parsing tests. No NumPy, native import, Docker, compiler or GPU."""
import io,stat,struct,unittest,zipfile
from kernel_agent.build_artifacts import LIMITS,ARCHIVE_LIMIT,check_elf,decode_artifacts


def elf():
    data=bytearray(64);data[:6]=b'\x7fELF\x02\x01';struct.pack_into('<HH',data,16,3,62);return bytes(data)


def archive(entries=None,compression=zipfile.ZIP_STORED,symlink=False):
    raw=io.BytesIO();entries=entries if entries is not None else [(name,elf() if name.endswith('.so') else b'build trace') for name in LIMITS]
    with zipfile.ZipFile(raw,'w',compression=compression) as file:
        for name,data in entries:
            info=zipfile.ZipInfo(name);info.compress_type=compression;info.create_system=3
            info.external_attr=((stat.S_IFLNK if symlink and name.endswith('.so') else stat.S_IFREG)|0o644)<<16
            file.writestr(info,data)
    return raw.getvalue()


class BuildArtifactTests(unittest.TestCase):
    def test_fixed_regular_files(self):self.assertEqual(decode_artifacts(archive())['nka_gemm_v1.so'],elf())
    def test_wrong_magic_rejected(self):
        with self.assertRaises(ValueError):check_elf(b'not ELF')
    def test_wrong_architecture_rejected(self):
        data=bytearray(elf());struct.pack_into('<H',data,18,183)
        with self.assertRaises(ValueError):check_elf(data)
    def test_symlink_rejected(self):
        with self.assertRaises(ValueError):decode_artifacts(archive(symlink=True))
    def test_traversal_rejected(self):
        with self.assertRaises(ValueError):decode_artifacts(archive([('../nka_gemm_v1.so',elf())]))
    def test_missing_artifact_rejected(self):
        with self.assertRaises(ValueError):decode_artifacts(archive([('nka_gemm_v1.so',elf())]))
    def test_duplicate_rejected(self):
        entries=[(name,elf() if name.endswith('.so') else b'trace') for name in LIMITS];entries[-1]=entries[0]
        with self.assertWarns(UserWarning):payload=archive(entries)
        with self.assertRaises(ValueError):decode_artifacts(payload)
    def test_compressed_entries_rejected(self):
        with self.assertRaises(ValueError):decode_artifacts(archive(compression=zipfile.ZIP_DEFLATED))
    def test_oversized_entry_rejected(self):
        entries=[(name,elf() if name.endswith('.so') else b'trace') for name in LIMITS];entries[1]=('build.ninja',b'x'*(LIMITS['build.ninja']+1))
        with self.assertRaises(ValueError):decode_artifacts(archive(entries))
    def test_oversized_envelope_rejected(self):
        with self.assertRaises(ValueError):decode_artifacts(b'x'*(ARCHIVE_LIMIT+1))
    def test_corrupt_zip_rejected(self):
        with self.assertRaises(ValueError):decode_artifacts(archive()[:32])


if __name__=='__main__':unittest.main()
