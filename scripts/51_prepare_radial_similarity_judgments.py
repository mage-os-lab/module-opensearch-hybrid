from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from poc.manifest import write_json
from poc.radial_similarity_judgments import (
    bind_completed_judgments,
    prepare_judgment_template,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare or bind score-blinded merchant radial judgments"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="create a merchant judgment template")
    prepare.add_argument("--capture", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)

    bind = subparsers.add_parser("bind", help="bind completed judgments to a capture")
    bind.add_argument("--capture", type=Path, required=True)
    bind.add_argument("--judgments", type=Path, required=True)
    bind.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    capture = _load_object(args.capture)
    if args.command == "prepare":
        output = prepare_judgment_template(capture)
    else:
        output = bind_completed_judgments(capture, _load_object(args.judgments))
    write_json(args.output, output)
    case_count = (
        len(output["cases"])
        if args.command == "prepare"
        else len(output["merchant_judgments"]["identity"]["cases"])
    )
    print(f"wrote {args.output}; command={args.command}; cases={case_count}")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], value)


if __name__ == "__main__":
    main()
