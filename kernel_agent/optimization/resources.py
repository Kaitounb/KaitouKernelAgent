"""Read-only, explicitly pinned public build and provider resources bundled in the wheel."""
import hashlib,json
from pathlib import Path

PACKAGE=Path(__file__).resolve().parent
ASSETS=PACKAGE/'assets'


def verify_bundle():
    manifest=json.loads((PACKAGE/'bundle.json').read_text())
    for rel,digest in manifest['sha256'].items():
        path=PACKAGE/rel
        if path.is_symlink() or not path.resolve().is_relative_to(PACKAGE) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('installed optimization resource changed: '+rel)
    return hashlib.sha256((PACKAGE/'bundle.json').read_bytes()).hexdigest()
