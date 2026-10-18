#!/usr/bin/env python3
"""Download rosbags of the traversability dataset from the Hugging Face Hub.

Bags are selected from data/manifest.csv and downloaded into data/raw/,
keeping the layout <session>/<bag>/ used in the Hugging Face dataset repo.
Files that are already downloaded are skipped.

Usage examples (run from anywhere inside the repo):
    python scripts/download_data.py                                         # all train + val bags
    python scripts/download_data.py --split val                             # only validation bags
    python scripts/download_data.py --session 2026-10-17_ucu-cinema
    python scripts/download_data.py --ride-type stonepath --ride-type flat
    python scripts/download_data.py --bag flat_05ms_20261017_162915
    python scripts/download_data.py --revision v0.1                          # exact dataset version
    python scripts/download_data.py --dry-run                                # only list what would be downloaded

Requires: pip install huggingface_hub, then `hf auth login` (a read token is enough).
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# --- Dataset settings ---------------------------------------------------------
HF_REPO_ID = "erc-samples/traversability-costmap"
DEFAULT_REVISION = "main"

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "data" / "manifest.csv"
RAW_DIR = REPO_ROOT / "data" / "raw"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", action="append", help="train / val (repeatable)")
    p.add_argument("--session", action="append", help="session folder name (repeatable)")
    p.add_argument("--ride-type", action="append", help="ride type, e.g. stonepath (repeatable)")
    p.add_argument("--bag", action="append", help="exact bag name (repeatable)")
    p.add_argument("--revision", default=DEFAULT_REVISION, help=f"dataset tag/branch/commit (default: {DEFAULT_REVISION})")
    p.add_argument("--include-excluded", action="store_true", help="also download bags with split=exclude")
    p.add_argument("--no-verify", action="store_true", help="skip sha256 check")
    p.add_argument("--dry-run", action="store_true", help="only print the selected bags")
    return p.parse_args()


def select_bags(args: argparse.Namespace) -> list[dict]:
    if not MANIFEST.exists():
        sys.exit(f"Manifest not found: {MANIFEST}")
    with MANIFEST.open(newline="") as f:
        rows = list(csv.DictReader(f))

    def keep(r: dict) -> bool:
        if r["split"] == "exclude" and not args.include_excluded and not args.bag:
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


def main() -> None:
    args = parse_args()
    rows = select_bags(args)
    if not rows:
        sys.exit("No bags match the given filters.")

    total_mb = sum(float(r.get("size_mb") or 0) for r in rows)
    print(f"Dataset {HF_REPO_ID} @ {args.revision}: {len(rows)} bags, ~{total_mb:.0f} MB")
    for r in rows:
        print(f"  - [{r['split']:>5}] {r['session']}/{r['bag']}")
    if args.dry_run:
        return

    patterns = [f"{r['session']}/{r['bag']}/*" for r in rows]
    patterns += sorted({f"{r['session']}/session.yaml" for r in rows})

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=RAW_DIR,
        allow_patterns=patterns,
    )
    print(f"Downloaded to {RAW_DIR}")

    if not args.no_verify:
        print("Verifying checksums:")
        if not verify(rows):
            sys.exit(1)


if __name__ == "__main__":
    main()