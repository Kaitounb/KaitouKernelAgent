"""CPU-only lossless transport regression; no candidate execution."""
import io,unittest,zipfile
from kernel_agent.protocol import encode_frame
from experiments.gemm_rel01.archive_parts import parts


class EvidencePartTests(unittest.TestCase):
    def packet(self,compression=zipfile.ZIP_STORED):
        out=io.BytesIO()
        with zipfile.ZipFile(out,'w',compression=compression) as archive:
            archive.writestr('q0.npy',b'one-output'*1024);archive.writestr('report.json','{"status":"PASS"}')
        return encode_frame({'kind':'source-eval-v1'},out.getvalue())
    def test_members_are_reusable_exact_bytes(self):
        raw=self.packet();segments=parts(raw);self.assertEqual(b''.join(segments),raw);self.assertIn(b'one-output'*1024,segments);self.assertGreater(len(segments),2)
    def test_compressed_packets_remain_whole(self):
        raw=self.packet(zipfile.ZIP_DEFLATED);self.assertEqual(parts(raw),[raw])
    def test_invalid_or_truncated_frames_remain_lossless(self):
        raw=self.packet()
        for data in [b'',b'log text',raw[:-8],b'wrong'+raw,encode_frame({},b'not zip')]:
            with self.subTest(size=len(data)):self.assertEqual(b''.join(parts(data)),data)


if __name__=='__main__':unittest.main()
