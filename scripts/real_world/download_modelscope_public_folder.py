#!/usr/bin/env python3
"""Download and verify every file in a public ModelScope dataset folder."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
from pathlib import Path, PurePosixPath
from urllib.parse import quote


def main() -> int:
    """Download a complete folder at a fixed repository revision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--strip-folder", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    helper_path = Path(__file__).with_name("download_modelscope_folder.py")
    spec = importlib.util.spec_from_file_location("modelscope_downloader", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {helper_path}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

    api_url = (
        f"https://www.modelscope.cn/api/v1/datasets/{args.dataset}/repo/tree"
        f"?Revision={quote(args.revision)}&Root={quote(args.folder.strip('/'))}&Recursive=true"
        "&PageSize=1000&PageNumber=1"
    )
    with helper.open_url(api_url, "") as response:
        manifest = json.load(response)
    if manifest.get("Code") != 200:
        raise RuntimeError(f"ModelScope listing failed: {manifest.get('Message')}")
    files = manifest["Data"]["Files"]
    if len(files) != manifest["Data"]["TotalCount"]:
        raise RuntimeError("ModelScope listing is incomplete")
    prefix = args.folder.strip("/") + "/"
    entries = [
        entry for entry in files
        if entry["Type"] == "blob" and entry["Path"].startswith(prefix)
    ]
    if not entries:
        raise RuntimeError(f"No files under {prefix}")
    revision = manifest["Data"]["LatestCommitter"]["ShortId"]
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    (args.evidence_dir / "modelscope_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    total_bytes = sum(int(entry["Size"]) for entry in entries)
    print(
        f"MANIFEST files={len(entries)} bytes={total_bytes} revision={revision}",
        flush=True,
    )
    base_url = f"https://www.modelscope.cn/datasets/{args.dataset}/resolve/{revision}/"
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(
            lambda entry: helper.download(
                entry, args.output, base_url, "",
                PurePosixPath(entry["Path"]).relative_to(args.folder.strip("/"))
                if args.strip_folder else None,
            ),
            sorted(entries, key=lambda entry: int(entry["Size"])),
        ))
    result = {
        "dataset": args.dataset,
        "folder": prefix,
        "revision": revision,
        "verified": len(entries),
        "failed": 0,
        "bytes": total_bytes,
        "output": str(args.output if args.strip_folder else args.output / args.folder.strip("/")),
    }
    (args.evidence_dir / "download_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print("RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
