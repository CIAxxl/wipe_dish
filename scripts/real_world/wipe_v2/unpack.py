"""Verify, assemble and safely unpack the delivered Wipe V2 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

def digest(path: Path) -> str:
    """Hash a file with bounded memory."""
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    """Preserve transport files and verify all extracted members."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory with the downloaded V2 files")
    root = parser.parse_args().directory.resolve()
    volumes = json.loads((root / 'transport_volumes.json').read_text())
    archive = root / volumes['archive']
    if not archive.exists():
        partial = archive.with_suffix('.assembling')
        with partial.open('wb') as output:
            for entry in volumes['volumes']:
                source = root / entry['path']
                assert source.stat().st_size == entry['bytes']
                assert digest(source) == entry['sha256']
                with source.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        output.write(chunk)
        assert partial.stat().st_size == volumes['archive_bytes']
        assert digest(partial) == volumes['archive_sha256']
        partial.replace(archive)
    assert digest(archive) == volumes['archive_sha256']
    expected = {entry['path']: entry for entry in json.loads((root / 'archive_members.json').read_text())}
    data = root / 'wipe_dish_v2_pca_v1'
    if not data.exists():
        with tarfile.open(archive) as source:
            members = source.getmembers()
            for member in members:
                target = (root / member.name).resolve()
                assert target.is_relative_to(data.resolve()), member.name
                assert member.isfile() or member.isdir(), member.name
            assert {member.name for member in members if member.isfile()} == set(expected)
            source.extractall(root, members=members)
    for name, entry in expected.items():
        path = root / name
        assert path.stat().st_size == entry['bytes'], name
        assert digest(path) == entry['sha256'], name
    report = {'passed': True, 'archive_sha256': volumes['archive_sha256'],
              'verified_members': len(expected), 'data': str(data)}
    (root / 'extraction_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
