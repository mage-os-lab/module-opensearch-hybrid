from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SERVER_ORIGIN = "http://ml-commons-package-server:8080"


def _load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _service_block(compose_path: Path) -> str:
    marker = "\n  ml-commons-package-server:\n"
    text = compose_path.read_text()
    assert marker in text
    remainder = text.split(marker, maxsplit=1)[1]
    match = re.search(r"\n  [a-zA-Z0-9_-]+:\n", remainder)
    return remainder if match is None else remainder[: match.start()]


def _make_dry_run(target: str) -> str:
    completed = subprocess.run(
        ["make", "-n", target],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_package_binding_uses_exact_compose_internal_url() -> None:
    package_script = _load_script("26_package_ml_commons_dense.py")

    binding = package_script.package_server_binding("model.name.zip")

    assert binding == {
        "service": "ml-commons-package-server",
        "port": 8080,
        "network_scope": "compose_internal",
        "url": f"{PACKAGE_SERVER_ORIGIN}/model.name.zip",
        "trusted_url_regex": (
            r"^http://ml\-commons\-package\-server:8080/model\.name\.zip$"
        ),
    }
    assert re.fullmatch(
        str(binding["trusted_url_regex"]), str(binding["url"])
    )
    assert not re.fullmatch(
        str(binding["trusted_url_regex"]),
        f"{PACKAGE_SERVER_ORIGIN}/different.zip",
    )


def test_smoke_rejects_unbound_or_mismatched_package_server_metadata() -> None:
    smoke_script = _load_script("27_smoke_ml_commons_dense.py")
    url = f"{PACKAGE_SERVER_ORIGIN}/model.zip"
    package = {
        "schema_version": 2,
        "opensearch_target": "3.8.0",
        "eligible_for_decision": False,
        "registration_body": {"url": url},
        "package_server": {
            "service": "ml-commons-package-server",
            "port": 8080,
            "network_scope": "compose_internal",
            "url": url,
            "trusted_url_regex": (
                r"^http://ml\-commons\-package\-server:8080/model\.zip$"
            ),
        },
    }

    assert smoke_script.validate_package_server_binding(package, "3.8.0") == (
        r"^http://ml\-commons\-package\-server:8080/model\.zip$"
    )

    mismatched = {
        **package,
        "registration_body": {"url": "http://127.0.0.1:18080/model.zip"},
    }
    with pytest.raises(ValueError, match="registration URL"):
        smoke_script.validate_package_server_binding(mismatched, "3.8.0")

    with pytest.raises(ValueError, match="schema version"):
        smoke_script.validate_package_server_binding(
            {**package, "schema_version": 1}, "3.8.0"
        )


@pytest.mark.parametrize(
    "compose_name",
    ["docker-compose.yml", "compose.opensearch-2.19.yml"],
)
def test_compose_package_server_is_private_and_read_only(compose_name: str) -> None:
    block = _service_block(ROOT / compose_name)

    assert 'profiles: ["ml-commons"]' in block
    assert "python:3.13.7-alpine3.22@sha256:" in block
    assert "./data/cache/models/ml-commons" in block
    assert "target: /srv/packages" in block
    assert "read_only: true" in block
    assert "ports:" not in block
    assert "expose:" in block
    assert '"8080"' in block
    assert 'no-new-privileges:true' in block


def test_versioned_make_targets_bind_matching_manifest_and_stack() -> None:
    package_38 = _make_dry_run("ml-commons-package-3.8")
    package_219 = _make_dry_run("ml-commons-package-2.19")
    smoke_38 = _make_dry_run("ml-commons-smoke-3.8")
    smoke_219 = _make_dry_run("ml-commons-smoke-2.19")

    assert "--opensearch-version 3.8.0" in package_38
    assert "--opensearch-version 2.19.6" in package_219
    assert "--profile ml-commons" in smoke_38
    assert "docker-compose.yml" in smoke_38
    assert "arctic-package.json" in smoke_38
    assert "http://127.0.0.1:9201" in smoke_38
    assert "--profile ml-commons" in smoke_219
    assert "compose.opensearch-2.19.yml" in smoke_219
    assert "arctic-package-opensearch-2.x.json" in smoke_219
    assert "http://127.0.0.1:9219" in smoke_219
