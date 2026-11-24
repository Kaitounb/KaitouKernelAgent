"""Negative checks only touch a temporary synthetic project, never the references."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "provenance_checker", Path(__file__).resolve().parents[1]/"tools/check_provenance.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/"references").mkdir()
        (self.root/"provenance").mkdir()
        (self.root/"references/example.txt").write_bytes(b"original snapshot")
        refs = {"references/example.txt":checker.digest(b"original snapshot")}
        self.lock = {"schema_version":1,"reference_files":refs,
            "reference_tree_sha256":checker.tree_digest(refs),"historical_source_files":{},
            "license_files":{},"historical_manifests":{}}
        self.write_lock()

    def write_lock(self):
        (self.root/"provenance/reference-lock.json").write_text(json.dumps(self.lock))

    def tearDown(self):
        self.temp.cleanup()

    def test_original_passes(self):
        self.assertEqual(checker.verify(self.root)["locked_files"],1)

    def test_changed_reference_rejected(self):
        (self.root/"references/example.txt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"locked file changed"):
            checker.verify(self.root)

    def test_added_reference_rejected(self):
        (self.root/"references/unlisted.txt").write_bytes(b"new")
        with self.assertRaisesRegex(ValueError,"inventory changed"):
            checker.verify(self.root)

    def test_missing_reference_rejected(self):
        (self.root/"references/example.txt").unlink()
        with self.assertRaisesRegex(ValueError,"inventory changed"):
            checker.verify(self.root)

    def test_inconsistent_tree_identity_rejected(self):
        self.lock["reference_tree_sha256"] = "0"*64
        self.write_lock()
        with self.assertRaisesRegex(ValueError,"identity is inconsistent"):
            checker.verify(self.root)

    def test_unsafe_paths_rejected_before_read(self):
        for path in ["../outside", "/tmp/outside", ".private/credentials.txt"]:
            with self.assertRaisesRegex(ValueError,"invalid locked path"):
                checker.checked_path(self.root,path)


if __name__ == "__main__":
    unittest.main()
