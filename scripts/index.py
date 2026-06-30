"""eval:index — placeholder entry point, wired in Phase 11.

Non-destructive: imports cleanly, prints a notice, exits 0. The real index
build (register embedding endpoints -> ensure_index -> bulk_index, §3.5) is
delivered in Phase 11 via benchmark.runner / ElasticsearchIndexer.
"""

from __future__ import annotations


def main() -> int:
    print("eval:index is a placeholder; wired in Phase 11")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
