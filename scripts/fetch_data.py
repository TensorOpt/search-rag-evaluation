"""eval:fetch-data — provision WANDS query.csv/product.csv/label.csv into dataset/wands/.

Pure infra (README "Dataset"). Clones the upstream WANDS repo into a temp dir
and copies the three TSV files into dataset/wands/. Skips files that already
exist unless WANDS_FORCE=1. Does NOT touch results/ or any harness code.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from benchmark.logging_setup import get_logger, setup_logging

log = get_logger(__name__)

WANDS_REPO = os.environ.get("WANDS_REPO_URL", "https://github.com/wayfair/WANDS.git")
FILES = ("query.csv", "product.csv", "label.csv")


def main() -> int:
    setup_logging()
    dest = Path(os.environ.get("WANDS_DEST", "dataset/wands")).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    force = os.environ.get("WANDS_FORCE") == "1"

    if not force and all((dest / f).exists() for f in FILES):
        log.info("WANDS files already present in %s; set WANDS_FORCE=1 to refetch", dest)
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        clone_dir = Path(tmp) / "WANDS"
        log.info("cloning %s -> %s", WANDS_REPO, clone_dir)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", WANDS_REPO, str(clone_dir)],
                check=True,
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            log.error("failed to clone WANDS: %s", exc)
            return 1

        src = clone_dir / "dataset"
        missing = [f for f in FILES if not (src / f).exists()]
        if missing:
            log.error("upstream is missing expected files: %s", missing)
            return 1
        for f in FILES:
            shutil.copy2(src / f, dest / f)
            log.info("copied %s -> %s", f, dest / f)

    log.info("WANDS provisioned into %s", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
