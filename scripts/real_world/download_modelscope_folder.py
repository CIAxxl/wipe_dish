#!/usr/bin/env python3
"""Download one ModelScope dataset folder with size and SHA-256 verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


CHUNK_SIZE = 1024 * 1024


class SafeRedirect(HTTPRedirectHandler):
    """Strip the session cookie when ModelScope redirects to another host."""

    def redirect_request(self, request, fp, code, message, headers, new_url):
        redirected = super().redirect_request(request, fp, code, message, headers, new_url)
        if redirected and urlparse(new_url).hostname != urlparse(request.full_url).hostname:
            for key in list(redirected.headers):
                if key.lower() == "cookie":
                    del redirected.headers[key]
        return redirected


def open_url(url: str, token: str, offset: int = 0):
    """Open an authenticated URL, optionally resuming from a byte offset."""
    headers = {"Cookie": f"m_session_id={token}"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return build_opener(SafeRedirect()).open(Request(url, headers=headers), timeout=180)


def digest_file(path: Path) -> str:
    """Return the SHA-256 of a local file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    entry: dict,
    destination_root: Path,
    base_url: str,
    token: str,
    destination_relative: PurePosixPath | None = None,
) -> None:
    """Resume a file download and verify it against the repository manifest."""
    relative = PurePosixPath(entry["Path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe repository path: {relative}")
    output_relative = destination_relative or relative
    if output_relative.is_absolute() or ".." in output_relative.parts:
        raise ValueError(f"Unsafe destination path: {output_relative}")
    destination = destination_root.joinpath(*output_relative.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = int(entry["Size"])
    expected_sha = entry["Sha256"].lower()
    if destination.exists() and destination.stat().st_size == size:
        if digest_file(destination) == expected_sha:
            print(f"VERIFIED existing {relative} {size}", flush=True)
            return
    partial = destination.with_name(destination.name + ".part")
    url = base_url + quote(str(relative), safe="/")
    for attempt in range(1, 6):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            partial.unlink()
            offset = 0
        try:
            with open_url(url, token, offset) as response:
                if offset and response.status != 206:
                    offset = 0
                mode = "ab" if offset else "wb"
                with partial.open(mode) as stream:
                    while chunk := response.read(CHUNK_SIZE):
                        stream.write(chunk)
            if partial.stat().st_size != size:
                raise ValueError(f"Incomplete file: {partial.stat().st_size}/{size}")
            if digest_file(partial) != expected_sha:
                partial.unlink()
                raise ValueError("SHA-256 mismatch")
            partial.replace(destination)
            print(f"VERIFIED downloaded {relative} {size}", flush=True)
            return
        except Exception as error:
            print(f"RETRY {relative} attempt={attempt} reason={type(error).__name__}", flush=True)
            if attempt == 5:
                raise RuntimeError(f"Failed to download {relative}") from error
            time.sleep(min(2**attempt, 16))


def main() -> int:
    """Download the requested folder and report a verified file count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    token = sys.stdin.readline().strip()
    if not token:
        raise ValueError("ModelScope token is required on stdin")
    prefix = args.folder.strip("/") + "/"
    api_url = (
        f"https://www.modelscope.cn/api/v1/datasets/{args.dataset}/repo/tree"
        "?Revision=master&Root=&Recursive=true"
    )
    with open_url(api_url, token) as response:
        manifest = json.load(response)
    if manifest.get("Code") != 200:
        raise RuntimeError(f"ModelScope manifest failed: {manifest.get('Message')}")
    entries = [
        entry for entry in manifest["Data"]["Files"]
        if entry["Type"] == "blob" and entry["Path"].startswith(prefix)
    ]
    if not entries:
        raise RuntimeError(f"No files found under {prefix}")
    print(
        f"MANIFEST folder={prefix} files={len(entries)} "
        f"bytes={sum(int(entry['Size']) for entry in entries)}",
        flush=True,
    )
    base_url = f"https://www.modelscope.cn/datasets/{args.dataset}/resolve/master/"
    for entry in entries:
        download(entry, args.output, base_url, token)
    print(f"RESULT verified={len(entries)} failed=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
