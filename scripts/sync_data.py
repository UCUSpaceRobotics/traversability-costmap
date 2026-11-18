#!/usr/bin/env python3
"""Download rosbags of the traversability dataset from the Hugging Face Hub, or upload them to it.

Bags are selected from data/manifest.csv. The first argument is required:
  download  copy the selected bags from the Hub into data/raw/
  upload    send the selected bags from data/raw/ to the Hub (one commit)
  delete    delete the local .mcap files of the selected bags (the git-ignored part); metadata.yaml,
            session.yaml and the Hub are not touched. Bags with split=exclude are included.
            Needs a filter (--bag / --session / --split / --ride-type) or --all; asks for confirmation (--yes skips it)
The Hugging Face dataset repo has the same layout as data/: raw/<session>/<bag>/.
Only .mcap files are downloaded: metadata.yaml and session.yaml are tracked in git
Files that are already downloaded are skipped.

Upload sends, for the selected bags, the whole bag folder (.mcap + metadata.yaml)
and, for their sessions, session.yaml and zed2i_record.yaml. Bags with split=exclude are included.
Needs a write token.

Usage examples (run from anywhere inside the repo):
    python scripts/sync_data.py download                                         # all train + val bags
    python scripts/sync_data.py download --split val                             # only validation bags
    python scripts/sync_data.py download --session 2026-10-17_ucu-cinema
    python scripts/sync_data.py download --ride-type stonepath --ride-type flat
    python scripts/sync_data.py download --bag 20261017_162915_v0p5
    python scripts/sync_data.py download --revision v0.1                          # exact dataset version
    python scripts/sync_data.py download --dry-run                                # show which files would be downloaded/skipped

    python scripts/sync_data.py delete --bag 20261017_162915_v0p5 --bag 20261017_163716_v0p5
    python scripts/sync_data.py delete --session 2026-10-17_ucu-cinema --dry-run   # list what would be deleted
    python scripts/sync_data.py delete --all --yes                                # every local bag, including exclude

    python scripts/sync_data.py upload --session 2026-11-15_stryiskiy-park --session 2026-11-15_stryiskiy-park-raw
    python scripts/sync_data.py upload --bag 20261017_162915_v0p5 --commit-message "add bag"
    python scripts/sync_data.py upload --session 2026-11-15_stryiskiy-park --dry-run   # show which files would be new/changed/unchanged

Requires: pip install huggingface_hub, then `hf auth login` (a read token is enough for download,
a write token for ucu-space-robotics is needed for upload).
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

# --- Dataset settings ---------------------------------------------------------
HF_REPO_ID = "ucu-space-robotics/traversability-costmap"
DEFAULT_REVISION = "main"

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "data" / "manifest.csv"
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["download", "upload", "delete"], help="download from the Hub, upload to it, or delete local .mcap files")
    p.add_argument("--split", action="append", help="train / val (repeatable)")
    p.add_argument("--session", action="append", help="session folder name (repeatable)")
    p.add_argument("--ride-type", action="append", help="ride type, e.g. stonepath (repeatable)")
    p.add_argument("--bag", action="append", help="exact bag name (repeatable)")
    p.add_argument("--commit-message", help="commit message for upload")
    p.add_argument("--revision", default=DEFAULT_REVISION, help=f"dataset tag/branch/commit (default: {DEFAULT_REVISION})")
    p.add_argument("--include-excluded", action="store_true", help="also download bags with split=exclude")
    p.add_argument("--all", action="store_true", help="delete: select every bag in the manifest (including exclude)")
    p.add_argument("--yes", action="store_true", help="delete: do not ask for confirmation")
    p.add_argument("--no-verify", action="store_true", help="skip sha256 check")
    p.add_argument("--dry-run", action="store_true", help="compare with the Hub and print what would be downloaded/uploaded, without doing it")
    return p.parse_args()


def select_bags(args: argparse.Namespace) -> list[dict]:
    if not MANIFEST.exists():
        sys.exit(f"Manifest not found: {MANIFEST}")
    with MANIFEST.open(newline="") as f:
        rows = list(csv.DictReader(f))

    def keep(r: dict) -> bool:
        if r["split"] == "exclude" and not args.include_excluded and not args.bag and args.mode == "download":
            return False
        return (
            (not args.split or r["split"] in args.split)
            and (not args.session or r["session"] in args.session)
            and (not args.ride_type or r["ride_type"] in args.ride_type)
            and (not args.bag or r["bag"] in args.bag)
        )

    selected = [r for r in rows if keep(r)]
    if args.bag:
        missing = set(args.bag) - {r["bag"] for r in selected}
        if missing:
            sys.exit(f"Bags not in manifest: {', '.join(sorted(missing))}")
    return selected


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(rows: list[dict]) -> bool:
    """Check bags that have a sha256 in the manifest (single-file bags)."""
    ok = True
    for r in rows:
        expected = r.get("sha256", "").strip()
        if not expected:
            continue
        mcaps = sorted((RAW_DIR / r["session"] / r["bag"]).glob("*.mcap"))
        if len(mcaps) != 1:
            print(f"  ? {r['bag']}: expected 1 .mcap file, found {len(mcaps)}; skipping check")
            continue
        if sha256_of(mcaps[0]) != expected:
            print(f"  ✗ {r['bag']}: sha256 mismatch, delete the folder and download again")
            ok = False
        else:
            print(f"  ✓ {r['bag']}")
    return ok


def remote_files(prefix_dirs: list[str]) -> dict[str, tuple[int, str | None, str | None]] | None:
    """Files on the Hub under raw/<dir> as {path relative to raw/: (size, sha256 or None, git blob id)}.

    Returns None if the Hub cannot be reached (then the dry run only lists local files)."""
    api = HfApi()
    out = {}
    for d in prefix_dirs:
        try:
            for item in api.list_repo_tree(HF_REPO_ID, path_in_repo=f"raw/{d}", recursive=True, repo_type="dataset"):
                if hasattr(item, "size"):  # files only, not folders
                    sha = item.lfs.sha256 if getattr(item, "lfs", None) else None
                    out[item.path[len("raw/"):]] = (item.size, sha, item.blob_id)
        except Exception as e:
            if "404" in str(e) or "NotFound" in type(e).__name__:
                continue  # session not on the Hub yet: everything in it is new
            print(f"  (could not read the Hub: {type(e).__name__}: {e})")
            return None
    return out


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fmt_mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def dry_run_upload(files: list[str], sessions: list[str]) -> None:
    remote = remote_files(sessions)
    new = changed = same = 0
    new_bytes = changed_bytes = 0
    for rel in files:
        local = RAW_DIR / rel
        size = local.stat().st_size
        if remote is None:
            print(f"  ? {rel} ({fmt_mb(size)})")
            continue
        r = remote.get(rel)
        if r is None:
            status, new, new_bytes = "NEW      ", new + 1, new_bytes + size
        else:
            r_size, r_sha, r_blob = r
            if r_size != size:
                is_same = False
            elif r_sha is not None:  # large file: same size is treated as the same file (hashing GBs is slow)
                is_same = True
            else:
                is_same = git_blob_sha1(local) == r_blob
            if is_same:
                status, same = "unchanged", same + 1
            else:
                status, changed, changed_bytes = "CHANGED  ", changed + 1, changed_bytes + size
        print(f"  {status} {rel} ({fmt_mb(size)})")
    if remote is not None:
        print(f"\nWould upload: {new} new ({fmt_mb(new_bytes)}), {changed} changed ({fmt_mb(changed_bytes)}); "
              f"{same} unchanged files are skipped. Nothing is deleted on the Hub.")
    print("Dry run: nothing was uploaded.")


def dry_run_download(rows: list[dict]) -> None:
    sessions = sorted({r["session"] for r in rows})
    remote = remote_files(sessions)
    todo = skip = 0
    todo_bytes = 0
    for r in rows:
        prefix = f"{r['session']}/{r['bag']}/"
        local_dir = RAW_DIR / r["session"] / r["bag"]
        if remote is None:
            print(f"  ? {prefix}")
            continue
        mcaps = {k: v for k, v in remote.items() if k.startswith(prefix) and k.endswith(".mcap")}
        if not mcaps:
            print(f"  MISSING ON HUB {prefix}")
            continue
        for rel, (r_size, _, _) in sorted(mcaps.items()):
            local = RAW_DIR / rel
            if local.exists() and local.stat().st_size == r_size:
                status, skip = "present  ", skip + 1
            else:
                status, todo, todo_bytes = ("UPDATE  " if local.exists() else "DOWNLOAD"), todo + 1, todo_bytes + r_size
            print(f"  {status} {rel} ({fmt_mb(r_size)})")
    if remote is not None:
        print(f"\nWould download: {todo} files ({fmt_mb(todo_bytes)}); {skip} already present are skipped. "
              f"YAML files are never downloaded.")
    print("Dry run: nothing was downloaded.")


def upload(rows: list[dict], args: argparse.Namespace) -> None:
    """Upload the selected bags and the YAML files of their sessions in one commit."""
    patterns = []
    for r in rows:
        if not (RAW_DIR / r["session"] / r["bag"]).is_dir():
            sys.exit(f"Bag folder not found locally: {r['session']}/{r['bag']}")
        patterns.append(f"{r['session']}/{r['bag']}/*")
    sessions = sorted({r["session"] for r in rows})
    for session in sessions:
        patterns += [f"{session}/session.yaml", f"{session}/zed2i_record.yaml"]

    if args.dry_run:
        files = sorted({p.relative_to(RAW_DIR).as_posix() for pat in patterns for p in RAW_DIR.glob(pat) if p.is_file()})
        dry_run_upload(files, sessions)
        return

    message = args.commit_message or f"add bags: {', '.join(sessions)}"
    commit = HfApi().upload_folder(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        folder_path=RAW_DIR,
        path_in_repo="raw",
        allow_patterns=patterns,
        commit_message=message,
    )
    print(f"Uploaded: {commit}")


def delete_local(rows: list[dict], args: argparse.Namespace) -> None:
    """Delete the local .mcap files of the selected bags."""
    files = sorted(f for r in rows for f in (RAW_DIR / r["session"] / r["bag"]).glob("*.mcap"))
    if not files:
        print("No local .mcap files for the selected bags.")
        return
    total = sum(f.stat().st_size for f in files)
    for f in files:
        print(f"  {f.relative_to(RAW_DIR).as_posix()} ({fmt_mb(f.stat().st_size)})")
    print(f"\n{len(files)} files, {fmt_mb(total)}")
    if args.dry_run:
        print("Dry run: nothing was deleted.")
        return
    if not args.yes and input("Delete these files? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return
    for f in files:
        f.unlink()
    print("Deleted.")


def main() -> None:
    args = parse_args()
    if args.mode == "delete" and not (args.all or args.bag or args.session or args.split or args.ride_type):
        sys.exit("delete needs a filter (--bag / --session / --split / --ride-type) or --all.")
    rows = select_bags(args)
    if not rows:
        sys.exit("No bags match the given filters.")

    if args.mode == "delete":
        print(f"Local bags selected for deletion: {len(rows)}")
        delete_local(rows, args)
        return

    total_mb = sum(float(r.get("size_mb") or 0) for r in rows)
    print(f"Dataset {HF_REPO_ID} @ {args.revision} ({'upload' if (args.mode == "upload") else 'download'}): {len(rows)} bags, ~{total_mb:.0f} MB")
    for r in rows:
        print(f"  - [{r['split']:>5}] {r['session']}/{r['bag']}")
    if (args.mode == "upload"):
        upload(rows, args)
        return
    if args.dry_run:
        dry_run_download(rows)
        return

    patterns = [f"raw/{r['session']}/{r['bag']}/*.mcap" for r in rows]

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=DATA_DIR,
        allow_patterns=patterns,
    )
    print(f"Downloaded to {RAW_DIR}")

    if not args.no_verify:
        print("Verifying checksums:")
        if not verify(rows):
            sys.exit(1)


if __name__ == "__main__":
    main()