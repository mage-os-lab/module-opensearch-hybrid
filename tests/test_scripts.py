from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_numbered_python_scripts_have_executable_main_guards() -> None:
    missing = [
        script.name
        for script in sorted((ROOT / "scripts").glob("[0-9]*.py"))
        if 'if __name__ == "__main__":' not in script.read_text()
    ]

    assert missing == []
