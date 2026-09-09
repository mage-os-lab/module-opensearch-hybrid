from __future__ import annotations

import os

from poc.os_client import OpenSearchClient
from poc.smoke import EXPECTED_VERSION


def main() -> None:
    url = os.environ.get("OPENSEARCH_HYBRID_OS_URL", "http://127.0.0.1:9201")
    with OpenSearchClient(url) as client:
        version = client.wait_until_ready(expected_version=EXPECTED_VERSION)
    print(f"OpenSearch {version} is ready at {url}")


if __name__ == "__main__":
    main()
