from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from poc.manifest import write_json
from poc.radial_similarity_calibration import evaluate_capture


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate captured exact-vs-radial MageOS product similarity evidence"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    capture = json.loads(args.input.read_text())
    if not isinstance(capture, dict):
        parser.error("input must be a JSON object")
    report = evaluate_capture(cast(dict[str, Any], capture))
    write_json(args.output, report)
    print(
        f"wrote {args.output}; eligible_candidates={len(report['eligible_candidates'])}; "
        "human_selection_required=true"
    )
    if report["eligible_candidates"] == []:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
