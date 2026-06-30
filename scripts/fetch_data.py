"""eval:fetch-data — provision WANDS query.csv/product.csv/label.csv into dataset/wands/.

Pure infra (README "Dataset"). Clones the upstream WANDS repo into a temp dir
and copies the three TSV files into dataset/wands/. Skips files that already
exist unless WANDS_FORCE=1. Does NOT touch results/ or any harness code.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

WANDS_REPO = os.environ.get("WANDS_REPO_URL", "https://github.com/wayfair/WANDS.git")
FILES = ("query.csv", "product.csv", "label.csv")


def main() -> int:
    dest = Path(os.environ.get("WANDS_DEST", "dataset/wands")).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    force = os.environ.get("WANDS_FORCE") == "1"

    if not force and all((dest / f).exists() for f in FILES):
        print(f"WANDS files already present in {dest}; set WANDS_FORCE=1 to refetch")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        clone_dir = Path(tmp) / "WANDS"
        print(f"cloning {WANDS_REPO} -> {clone_dir}")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", WANDS_REPO, str(clone_dir)],
                check=True,
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f"failed to clone WANDS: {exc}", file=sys.stderr)
            return 1

        src = clone_dir / "dataset"
        missing = [f for f in FILES if not (src / f).exists()]
        if missing:
            print(f"upstream is missing expected files: {missing}", file=sys.stderr)
            return 1
        for f in FILES:
            shutil.copy2(src / f, dest / f)
            print(f"copied {f} -> {dest / f}")

    print(f"WANDS provisioned into {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
