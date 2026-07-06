"""eval:wait-for-es — poll ES cluster health until at least yellow.

Pure infra (README "Wait for cluster health"). Polls
${ES_URL}/_cluster/health?wait_for_status=yellow until ready or timeout.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from benchmark.common.logging_setup import get_logger, setup_logging

log = get_logger(__name__)


def main() -> int:
    setup_logging()
    es_url = os.environ.get("ES_URL", "http://localhost:9200").rstrip("/")
    deadline = time.monotonic() + float(os.environ.get("ES_WAIT_TIMEOUT", "120"))
    url = f"{es_url}/_cluster/health?wait_for_status=yellow&timeout=10s"

    while True:
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310 (local ES)
                body = json.loads(resp.read().decode("utf-8"))
            status = body.get("status")
            if status in ("yellow", "green"):
                log.info("ES cluster healthy: status=%s", status)
                return 0
            log.info("ES status=%s, waiting...", status)
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            log.warning("ES not reachable yet (%s); retrying...", exc)

        if time.monotonic() >= deadline:
            log.error("timed out waiting for ES at %s", es_url)
            return 1
        time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
