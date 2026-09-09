from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, cast

import httpx

from poc.manifest import write_json
from poc.radial_similarity_capture import HttpClient, capture_radial_similarity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture exact-vs-radial MageOS product similarity evidence"
    )
    parser.add_argument("--url", required=True)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--username")
    parser.add_argument("--password-env", default="OPENSEARCH_PASSWORD")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    specification = json.loads(args.specification.read_text())
    if not isinstance(specification, dict):
        parser.error("specification must be a JSON object")
    auth = None
    if args.username is not None:
        password = os.environ.get(args.password_env)
        if password is None:
            parser.error(f"{args.password_env} is not set")
        auth = (args.username, password)
    with httpx.Client(base_url=args.url.rstrip("/"), timeout=30.0, auth=auth) as client:
        capture = capture_radial_similarity(
            cast(HttpClient, client),
            cast(dict[str, Any], specification),
            args.url,
        )
    write_json(args.output, capture)
    print(
        f"wrote {args.output}; split={capture['split']}; "
        f"cases={len(specification['cases'])}; thresholds={len(capture['thresholds'])}"
    )


if __name__ == "__main__":
    main()
