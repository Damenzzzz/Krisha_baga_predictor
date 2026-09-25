"""Resume detail enrichment, then dedup, reconcile Qdrant and smoke-test.

Runs finalization even when details stops on a block/CAPTCHA, so partial progress
is reflected in the vector payload. Returns the details exit code after checks.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--city")
    ap.add_argument("--smoke-city", default="almaty_oblast")
    args = ap.parse_args()
    details = [sys.executable, "-m", "krisha", "details", "--refresh-missing"]
    if args.limit is not None:
        details += ["--limit", str(args.limit)]
    if args.city:
        details += ["--city", args.city]
    code = subprocess.run(details, check=False).returncode
    for command in (
        ["krisha", "dedup"],
        ["scripts.resync_qdrant_payload"],
        ["scripts.smoke_similar", "--city", args.smoke_city],
    ):
        result = subprocess.run([sys.executable, "-m", *command], check=False)
        if result.returncode:
            return result.returncode
    return code


if __name__ == "__main__":
    raise SystemExit(main())
