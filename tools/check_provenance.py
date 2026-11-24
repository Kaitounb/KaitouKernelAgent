"""Offline hash/inventory check. Never imports or runs reference implementations."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tree_digest(files):
    return digest(json.dumps(files, sort_keys=True, separators=(",", ":")).encode())


def checked_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.parts[0] == ".private":
        raise ValueError("invalid locked path")
    target = root/path
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("locked path escapes project")
    return target


def verify(root):
    lock = json.loads((root/"provenance/reference-lock.json").read_text())
    if lock["schema_version"] != 1:
        raise ValueError("unsupported lock schema")
    references = lock["reference_files"]
    if tree_digest(references) != lock["reference_tree_sha256"]:
        raise ValueError("reference tree identity is inconsistent")
    actual = {str(p.relative_to(root)) for p in (root/"references").rglob("*") if p.is_file()}
    if actual != set(references):
        raise ValueError("reference inventory changed")
    groups = [references, lock["historical_source_files"], lock["license_files"], lock["historical_manifests"]]
    count = 0
    for group in groups:
        for relative, sha in group.items():
            path = checked_path(root, relative)
            if digest(path.read_bytes()) != sha:
                raise ValueError("locked file changed: " + relative)
            count += 1
    evidence_files = 0
    for relative in lock["historical_manifests"]:
        manifest = checked_path(root, relative)
        for name, sha in json.loads(manifest.read_text()).items():
            target = manifest.parent/name
            if not target.resolve().is_relative_to(manifest.parent.resolve()):
                raise ValueError("historical evidence path escapes its run")
            if digest(target.read_bytes()) != sha:
                raise ValueError("historical evidence changed: " + str(target.relative_to(root)))
            evidence_files += 1
    return {"status":"PROVENANCE_PASS", "evidence_type":"STATIC_HASH_AUDIT",
            "reference_files":len(references), "locked_files":count,
            "historical_evidence_files":evidence_files,
            "reference_tree_sha256":lock["reference_tree_sha256"],
            "gpu_run":False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = verify(args.root.resolve())
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
