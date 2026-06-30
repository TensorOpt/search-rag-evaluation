"""eval:wait-for-es — poll ES cluster health until at least yellow.

Pure infra (README "Wait for cluster health"). Polls
${ES_URL}/_cluster/health?wait_for_status=yellow until ready or timeout.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request


def main() -> int:
    es_url = os.environ.get("ES_URL", "http://localhost:9200").rstrip("/")
    deadline = time.monotonic() + float(os.environ.get("ES_WAIT_TIMEOUT", "120"))
    url = f"{es_url}/_cluster/health?wait_for_status=yellow&timeout=10s"

    while True:
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310 (local ES)
                body = json.loads(resp.read().decode("utf-8"))
            status = body.get("status")
            if status in ("yellow", "green"):
                print(f"ES cluster healthy: status={status}")
                return 0
            print(f"ES status={status}, waiting...")
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            print(f"ES not reachable yet ({exc}); retrying...")

        if time.monotonic() >= deadline:
            print(f"timed out waiting for ES at {es_url}", file=sys.stderr)
            return 1
        time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
