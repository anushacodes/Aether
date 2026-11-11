"""Check the running container pilot through real HTTP, with no configured model calls."""

from __future__ import annotations

import argparse
import json
import re
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def check(api_url: str, frontend_url: str) -> dict[str, object]:
    """Wait for the worker's three-section pilot, then check browser-facing boundaries."""
    api_url = api_url.rstrip("/")
    frontend_url = frontend_url.rstrip("/")
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{api_url}/health/ready", timeout=5) as response:
                ready = json.load(response)
            assert ready["documents"] == 3, ready
            break
        except URLError:
            time.sleep(1)
    else:
        raise TimeoutError("The container worker did not publish the pilot within 90 seconds")
    question = {"query": "versioning", "mode": "bm25"}
    request = Request(
        f"{api_url}/retrieval",
        data=json.dumps(question).encode(),
        headers={"Content-Type": "application/json", "Origin": frontend_url},
    )
    with urlopen(request, timeout=10) as response:
        assert response.headers["Access-Control-Allow-Origin"] == frontend_url
        hits = json.load(response)["hits"]
    assert len(hits) == 3 and all(hit["chunk"]["original_text"] for hit in hits)
    request = Request(
        f"{api_url}/search/stream",
        data=json.dumps(question).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=15) as response:
        stream = response.read().decode()
    assert "event: sources" in stream and "model_not_configured" in stream
    assert "event: done" not in stream
    with urlopen(frontend_url, timeout=10) as response:
        html = response.read().decode()
    assert "Inspect the sources" in html
    scripts = re.findall(r'<script[^>]+src="([^"]+)"', html)
    assert scripts, "Production HTML must reference built JavaScript assets"
    for path in scripts:
        with urlopen(f"{frontend_url}{path}", timeout=10) as response:
            assert response.status == 200
    return {
        "passed": [
            "worker-published pilot is ready",
            "real source lookup and configured CORS",
            "unconfigured model preserves streamed sources without false success",
            "production frontend and JavaScript assets are served",
        ],
        "documents": len(hits),
        "real_aws_requests": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:18000")
    parser.add_argument("--frontend-url", default="http://localhost:13000")
    args = parser.parse_args()
    print(json.dumps(check(args.api_url, args.frontend_url), indent=2))
