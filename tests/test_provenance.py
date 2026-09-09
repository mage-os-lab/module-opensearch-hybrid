from __future__ import annotations

import copy
import hashlib
import importlib
import json
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from poc.config import ConfigError
from poc.dense import run_wands_dense_benchmark
from poc.os_client import OpenSearchClient
from poc.provenance import (
    ProvenanceError,
    collect_code_revision,
    collect_live_benchmark_environment,
    collect_manifest_provenance,
    load_benchmark_profile,
    registered_opensearch_url,
    require_registered_opensearch_client,
    source_tree_sha256,
    verify_container_environment,
    verify_container_resources,
    verify_decision_provenance,
    verify_live_benchmark_environment,
    verify_opensearch_environment,
)

ROOT = Path(__file__).resolve().parents[1]
PERSIST_BENCHMARK_ENVIRONMENT = cast(
    Callable[..., bool],
    importlib.import_module(
        "scripts.00_verify_benchmark_profile"
    ).persist_benchmark_environment,
)


def _fixture_profile_text() -> str:
    return "\n".join(
        (
            "schema_version = 2",
            'profile_id = "fixture"',
            "requires_live_verification = true",
            "cpu_limit = 8",
            "memory_limit_bytes = 17179869184",
            'compose_project = "fixture-project"',
            'compose_service = "opensearch"',
            'compose_files = ["compose.yml"]',
            'container_image = "opensearchproject/opensearch:3.8.0"',
            'container_image_id = "sha256:' + "b" * 64 + '"',
            'opensearch_url = "http://127.0.0.1:9201"',
            'opensearch_version = "3.8.0"',
            'plugin_version = "3.8.0.0"',
            'plugin_components = ["opensearch-knn", "opensearch-neural-search"]',
            'latency_required_architecture = "x86_64"',
            "",
        )
    )


def _fixture_live_facts() -> dict[str, Any]:
    container_id = "a" * 64
    empty_contract_hash = hashlib.sha256(b"{}").hexdigest()
    return {
        "profile_id": "fixture",
        "compose_project": "fixture-project",
        "compose_service": "opensearch",
        "compose_files": ["compose.yml"],
        "container_id": container_id,
        "image": "opensearchproject/opensearch:3.8.0",
        "image_id": "sha256:" + "b" * 64,
        "cpu_limit": 8,
        "memory_limit_bytes": 16 * 1024**3,
        "architecture": "arm64",
        "latency_architecture_eligible": False,
        "limits_verified": True,
        "runtime": {
            "repository_digest": (
                "opensearchproject/opensearch@sha256:" + "b" * 64
            ),
            "platform_manifest_digest": "sha256:" + "c" * 64,
            "platform_os": "linux",
            "contract": {},
            "contract_sha256": empty_contract_hash,
        },
        "opensearch": {
            "url": "http://127.0.0.1:9201",
            "version": "3.8.0",
            "node_name": container_id[:12],
            "plugins": [
                {"component": "opensearch-knn", "version": "3.8.0.0"},
                {
                    "component": "opensearch-neural-search",
                    "version": "3.8.0.0",
                },
            ],
        },
    }


def _fixture_environment(verified_at: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "verified_at": verified_at,
        **_fixture_live_facts(),
    }


def _fixture_rendered_config() -> dict[str, Any]:
    return {
        "name": "fixture-project",
        "networks": {"default": {"name": "fixture-project_default"}},
        "services": {
            "opensearch": {
                "cpus": 8,
                "command": None,
                "entrypoint": None,
                "environment": {"FIXTURE_SETTING": "true"},
                "healthcheck": {
                    "test": ["CMD", "true"],
                    "timeout": "6s",
                    "interval": "5s",
                    "retries": 60,
                    "start_period": "30s",
                },
                "image": "opensearchproject/opensearch:3.8.0",
                "mem_limit": str(16 * 1024**3),
                "networks": {"default": None},
                "ports": [
                    {
                        "host_ip": "127.0.0.1",
                        "target": 9200,
                        "published": "9201",
                        "protocol": "tcp",
                    }
                ],
                "restart": "unless-stopped",
                "ulimits": {
                    "memlock": {"soft": -1, "hard": -1},
                    "nofile": {"soft": 65536, "hard": 65536},
                },
                "volumes": [
                    {
                        "type": "volume",
                        "source": "fixture-data",
                        "target": "/usr/share/opensearch/data",
                    }
                ],
            }
        },
        "volumes": {"fixture-data": {"name": "fixture-project_fixture-data"}},
    }


def _fixture_image_inspect() -> dict[str, Any]:
    image_id = "sha256:" + "b" * 64
    return {
        "Id": image_id,
        "Architecture": "arm64",
        "Os": "linux",
        "RepoDigests": [f"opensearchproject/opensearch@{image_id}"],
        "Config": {
            "Env": ["PATH=/usr/bin"],
            "Entrypoint": ["./entrypoint.sh"],
            "Cmd": ["opensearch"],
            "User": "1000",
            "WorkingDir": "/usr/share/opensearch",
        },
    }


def _fixture_container_inspect(tmp_path: Path) -> dict[str, Any]:
    del tmp_path
    ports = {
        "9200/tcp": [{"HostIp": "127.0.0.1", "HostPort": "9201"}]
    }
    return {
        "Id": "a" * 64,
        "Image": "sha256:" + "b" * 64,
        "State": {"Running": True, "Health": {"Status": "healthy"}},
        "Config": {
            "Image": "opensearchproject/opensearch:3.8.0",
            "Env": ["FIXTURE_SETTING=true", "PATH=/usr/bin"],
            "Entrypoint": ["./entrypoint.sh"],
            "Cmd": ["opensearch"],
            "User": "1000",
            "WorkingDir": "/usr/share/opensearch",
            "Healthcheck": {
                "Test": ["CMD", "true"],
                "Timeout": 6_000_000_000,
                "Interval": 5_000_000_000,
                "Retries": 60,
                "StartPeriod": 30_000_000_000,
            },
            "Labels": {},
        },
        "HostConfig": {
            "NanoCpus": 8_000_000_000,
            "Memory": 16 * 1024**3,
            "MemoryReservation": 0,
            "MemorySwap": 32 * 1024**3,
            "CpuPeriod": 0,
            "CpuQuota": 0,
            "CpuShares": 0,
            "CpusetCpus": "",
            "PidsLimit": None,
            "ShmSize": 64 * 1024**2,
            "PortBindings": ports,
            "Ulimits": [
                {"Name": "memlock", "Soft": -1, "Hard": -1},
                {"Name": "nofile", "Soft": 65536, "Hard": 65536},
            ],
            "NetworkMode": "fixture-project_default",
            "RestartPolicy": {"Name": "unless-stopped", "MaximumRetryCount": 0},
            "Privileged": False,
            "PublishAllPorts": False,
            "ReadonlyRootfs": False,
            "CapAdd": None,
            "CapDrop": None,
            "SecurityOpt": None,
            "CgroupnsMode": "private",
            "IpcMode": "private",
            "PidMode": "",
            "UsernsMode": "",
        },
        "Mounts": [
            {
                "Type": "volume",
                "Name": "fixture-project_fixture-data",
                "Destination": "/usr/share/opensearch/data",
                "RW": True,
                "Propagation": "",
            }
        ],
        "NetworkSettings": {
            "Ports": ports,
            "Networks": {"fixture-project_default": {}},
        },
        "ImageManifestDescriptor": {
            "digest": "sha256:" + "c" * 64,
            "platform": {"architecture": "arm64", "os": "linux"},
        },
    }


def _write_environment(path: Path, facts: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "verified_at": "2026-08-22T12:00:00+00:00",
                **facts,
            }
        )
        + "\n"
    )


def test_source_tree_hash_changes_for_code_but_ignores_generated_evidence(
    tmp_path: Path,
) -> None:
    (tmp_path / "poc").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "poc/example.py").write_text("VALUE = 1\n")
    (tmp_path / "results/result.json").write_text("{}\n")
    (tmp_path / "README.md").write_text("descriptive documentation\n")
    (tmp_path / "REPORT.md").write_text("generated findings\n")
    (tmp_path / ".gitignore").write_text("results/\n")
    (tmp_path / "EXPERIMENT-AMENDMENTS.md").write_text("protocol amendment\n")
    first = source_tree_sha256(tmp_path)

    (tmp_path / "results/result.json").write_text('{"changed": true}\n')
    assert source_tree_sha256(tmp_path) == first

    (tmp_path / "README.md").write_text("updated descriptive documentation\n")
    (tmp_path / "REPORT.md").write_text("updated generated findings\n")
    (tmp_path / ".gitignore").write_text("results/\ndata/cache/\n")
    assert source_tree_sha256(tmp_path) == first

    (tmp_path / "EXPERIMENT-AMENDMENTS.md").write_text("changed protocol amendment\n")
    assert source_tree_sha256(tmp_path) != first
    (tmp_path / "EXPERIMENT-AMENDMENTS.md").write_text("protocol amendment\n")
    assert source_tree_sha256(tmp_path) == first

    (tmp_path / "poc/example.py").write_text("VALUE = 2\n")
    assert source_tree_sha256(tmp_path) != first


def test_code_revision_reports_commit_and_source_dirtiness(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "poc").mkdir()
    source = tmp_path / "poc/example.py"
    source.write_text("VALUE = 1\n")
    subprocess.run(["git", "add", "poc/example.py"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )

    clean = collect_code_revision(tmp_path)
    assert clean.git_commit is not None
    assert clean.source_dirty is False

    source.write_text("VALUE = 2\n")
    assert collect_code_revision(tmp_path).source_dirty is True


def test_benchmark_profile_registers_the_resource_ceiling() -> None:
    profile = load_benchmark_profile(ROOT / "config/benchmark.toml")

    assert profile.cpu_limit == 8
    assert profile.memory_limit_bytes == 16 * 1024**3
    assert profile.latency_required_architecture == "x86_64"
    assert profile.requires_live_verification is True
    assert profile.compose_project == "opensearch-hybrid-os38"
    assert profile.compose_service == "opensearch"
    assert profile.compose_files == ("docker-compose.yml", "compose.bench.yml")
    assert profile.container_image == "opensearchproject/opensearch:3.8.0"
    assert profile.container_image_id == (
        "sha256:bcc1797519726ceb6d651d4a3e60b7c30da91793914a8dfe75fd441d4f641509"
    )
    assert profile.opensearch_url == "http://127.0.0.1:9201"
    assert profile.opensearch_version == "3.8.0"
    assert "opensearch-knn" in profile.plugin_components
    assert "opensearch-neural-search" in profile.plugin_components

    transfer = load_benchmark_profile(ROOT / "config/benchmark-2.19.toml")
    assert transfer.profile_id == "opensearch-2.19-local-8cpu-16g"
    assert transfer.compose_project == "opensearch-hybrid-os219"
    assert transfer.compose_files == ("compose.opensearch-2.19.yml",)
    assert transfer.opensearch_url == "http://127.0.0.1:9219"
    assert transfer.opensearch_version == "2.19.6"


def test_registered_opensearch_url_fails_closed_on_a_different_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_path = tmp_path / "benchmark.toml"
    profile_path.write_text(_fixture_profile_text())

    monkeypatch.delenv("OPENSEARCH_HYBRID_OS_URL", raising=False)
    assert registered_opensearch_url(
        profile_path,
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    ) == "http://127.0.0.1:9201"

    monkeypatch.setenv("OPENSEARCH_HYBRID_OS_URL", "http://127.0.0.1:9201/")
    assert registered_opensearch_url(
        profile_path,
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    ) == "http://127.0.0.1:9201"

    monkeypatch.setenv("OPENSEARCH_HYBRID_OS_URL", "http://127.0.0.1:9219")
    with pytest.raises(ProvenanceError, match="registered benchmark profile"):
        registered_opensearch_url(
            profile_path,
            environment_variable="OPENSEARCH_HYBRID_OS_URL",
        )

    monkeypatch.setenv("OPENSEARCH_HYBRID_OS_URL", "not a URL")
    with pytest.raises(ProvenanceError, match="invalid"):
        registered_opensearch_url(
            profile_path,
            environment_variable="OPENSEARCH_HYBRID_OS_URL",
        )


def test_registered_wands_and_trec_endpoints_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENSEARCH_HYBRID_OS_URL", raising=False)
    monkeypatch.delenv("OPENSEARCH_HYBRID_TREC_OS_URL", raising=False)

    assert registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    ) == "http://127.0.0.1:9201"
    assert registered_opensearch_url(
        ROOT / "config/benchmark-2.19.toml",
        environment_variable="OPENSEARCH_HYBRID_TREC_OS_URL",
    ) == "http://127.0.0.1:9219"

    monkeypatch.setenv("OPENSEARCH_HYBRID_TREC_OS_URL", "http://127.0.0.1:9201")
    with pytest.raises(ProvenanceError, match="registered benchmark profile"):
        registered_opensearch_url(
            ROOT / "config/benchmark-2.19.toml",
            environment_variable="OPENSEARCH_HYBRID_TREC_OS_URL",
        )


def test_registered_client_guard_rejects_the_other_profile_endpoint() -> None:
    with OpenSearchClient("http://127.0.0.1:9201/") as client:
        require_registered_opensearch_client(
            client,
            ROOT / "config/benchmark.toml",
        )

    with (
        OpenSearchClient("http://127.0.0.1:9219") as client,
        pytest.raises(ProvenanceError, match="client endpoint"),
    ):
        require_registered_opensearch_client(
            client,
            ROOT / "config/benchmark.toml",
        )


def test_core_decision_entry_point_rejects_the_other_profile_endpoint() -> None:
    with (
        OpenSearchClient("http://127.0.0.1:9219") as client,
        pytest.raises(ProvenanceError, match="client endpoint"),
    ):
        run_wands_dense_benchmark(
            client,
            root=ROOT,
            index="not-reached",
            models=(),
        )


def test_decision_script_clients_use_registered_profile_endpoints() -> None:
    diagnostic_scripts = {
        "00_verify.py",
        "01_verify.py",
        "27_smoke_ml_commons_dense.py",
        "32_smoke_trec_sparse.py",
    }
    for path in sorted((ROOT / "scripts").glob("*.py")):
        source = path.read_text()
        if "OpenSearchClient(" not in source or path.name in diagnostic_scripts:
            continue
        assert (
            "registered_opensearch_url(" in source
            or "trec_opensearch_url(" in source
        ), f"{path.name} constructs a decision client without a registered endpoint"
        assert "os.environ.get(\"OPENSEARCH_HYBRID_OS_URL\"" not in source
        assert "os.environ.get(\"OPENSEARCH_HYBRID_WANDS_OS_URL\"" not in source
        assert "os.environ.get(\"OPENSEARCH_HYBRID_TREC_OS_URL\"" not in source


def test_container_resource_verifier_requires_exact_registered_limits() -> None:
    profile = load_benchmark_profile(ROOT / "config/benchmark.toml")
    inspect = {
        "Id": "a" * 64,
        "Image": profile.container_image_id,
        "Config": {"Image": "opensearchproject/opensearch:3.8.0"},
        "HostConfig": {
            "NanoCpus": 8_000_000_000,
            "Memory": 16 * 1024**3,
        },
    }

    facts = verify_container_resources(inspect, profile, architecture="aarch64")
    assert facts["limits_verified"] is True
    assert facts["architecture"] == "arm64"
    assert facts["image_id"] == profile.container_image_id

    wrong_image = copy.deepcopy(inspect)
    wrong_image["Image"] = "sha256:" + "c" * 64
    with pytest.raises(ProvenanceError, match="unregistered image ID"):
        verify_container_resources(wrong_image, profile, architecture="aarch64")

    inspect["HostConfig"]["NanoCpus"] = 0  # type: ignore[index]
    try:
        verify_container_resources(inspect, profile, architecture="aarch64")
    except ProvenanceError as exc:
        assert "CPU limit" in str(exc)
    else:
        raise AssertionError("accepted a container without the registered CPU limit")


def test_container_environment_requires_registered_image_project_and_compose_files(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "benchmark.toml"
    profile_path.write_text(_fixture_profile_text())
    profile = load_benchmark_profile(profile_path)
    compose_path = tmp_path / "compose.yml"
    compose_path.write_text("services: {}\n")
    inspect = _fixture_container_inspect(tmp_path)
    config = inspect["Config"]
    assert isinstance(config, dict)
    config["Labels"] = {
        "com.docker.compose.project": "fixture-project",
        "com.docker.compose.service": "opensearch",
        "com.docker.compose.project.working_dir": str(tmp_path),
        "com.docker.compose.project.config_files": str(compose_path),
    }
    rendered_config = _fixture_rendered_config()
    image_inspect = _fixture_image_inspect()

    facts = verify_container_environment(
        inspect,
        profile,
        root=tmp_path,
        architecture="aarch64",
        rendered_config=rendered_config,
        image_inspect=image_inspect,
    )
    assert facts["profile_id"] == "fixture"
    assert facts["container_id"] == "a" * 64
    assert facts["limits_verified"] is True
    assert facts["runtime"]["repository_digest"] == (
        "opensearchproject/opensearch@sha256:" + "b" * 64
    )
    assert facts["runtime"]["contract"]["mounts"] == [
        {
            "type": "volume",
            "source": "fixture-data",
            "target": "/usr/share/opensearch/data",
            "read_only": False,
            "propagation": "",
        }
    ]
    assert len(facts["runtime"]["contract_sha256"]) == 64

    forged = copy.deepcopy(inspect)
    forged["Config"]["Labels"]["com.docker.compose.project"] = "other"
    with pytest.raises(ProvenanceError, match="Compose project"):
        verify_container_environment(
            forged,
            profile,
            root=tmp_path,
            architecture="aarch64",
            rendered_config=rendered_config,
            image_inspect=image_inspect,
        )

    stale = copy.deepcopy(inspect)
    stale["Config"]["Image"] = "opensearchproject/opensearch:latest"
    with pytest.raises(ProvenanceError, match="container image"):
        verify_container_environment(
            stale,
            profile,
            root=tmp_path,
            architecture="aarch64",
            rendered_config=rendered_config,
            image_inspect=image_inspect,
        )

    privileged = copy.deepcopy(inspect)
    privileged["HostConfig"]["Privileged"] = True
    with pytest.raises(ProvenanceError, match="runtime"):
        verify_container_environment(
            privileged,
            profile,
            root=tmp_path,
            architecture="aarch64",
            rendered_config=rendered_config,
            image_inspect=image_inspect,
        )


@pytest.mark.parametrize(
    "case",
    (
        "environment",
        "entrypoint",
        "command",
        "ports",
        "ulimits",
        "network",
        "resources",
        "mounts",
        "read_only",
        "platform_manifest",
        "repository_digest",
        "rendered_environment",
    ),
)
def test_container_runtime_rejects_decision_relevant_docker_drift(
    tmp_path: Path,
    case: str,
) -> None:
    profile_path = tmp_path / "benchmark.toml"
    profile_path.write_text(_fixture_profile_text())
    profile = load_benchmark_profile(profile_path)
    compose_path = tmp_path / "compose.yml"
    compose_path.write_text("services: {}\n")
    inspect = _fixture_container_inspect(tmp_path)
    config = inspect["Config"]
    assert isinstance(config, dict)
    config["Labels"] = {
        "com.docker.compose.project": "fixture-project",
        "com.docker.compose.service": "opensearch",
        "com.docker.compose.project.working_dir": str(tmp_path),
        "com.docker.compose.project.config_files": str(compose_path),
    }
    rendered_config = _fixture_rendered_config()
    image_inspect = _fixture_image_inspect()

    if case == "environment":
        config["Env"] = ["FIXTURE_SETTING=false", "PATH=/usr/bin"]
    elif case == "entrypoint":
        config["Entrypoint"] = ["/bin/false"]
    elif case == "command":
        config["Cmd"] = ["not-opensearch"]
    elif case == "ports":
        host = inspect["HostConfig"]
        assert isinstance(host, dict)
        host["PortBindings"] = {
            "9200/tcp": [{"HostIp": "0.0.0.0", "HostPort": "9201"}]
        }
    elif case == "ulimits":
        host = inspect["HostConfig"]
        assert isinstance(host, dict)
        host["Ulimits"] = [{"Name": "nofile", "Soft": 1024, "Hard": 1024}]
    elif case == "network":
        host = inspect["HostConfig"]
        assert isinstance(host, dict)
        host["NetworkMode"] = "bridge"
    elif case == "resources":
        host = inspect["HostConfig"]
        assert isinstance(host, dict)
        host["MemorySwap"] = 0
    elif case == "mounts":
        mounts = inspect["Mounts"]
        assert isinstance(mounts, list)
        mount = mounts[0]
        assert isinstance(mount, dict)
        mount["RW"] = False
    elif case == "read_only":
        host = inspect["HostConfig"]
        assert isinstance(host, dict)
        host["ReadonlyRootfs"] = True
    elif case == "platform_manifest":
        descriptor = inspect["ImageManifestDescriptor"]
        assert isinstance(descriptor, dict)
        descriptor["digest"] = "not-a-digest"
    elif case == "repository_digest":
        image_inspect["RepoDigests"] = []
    elif case == "rendered_environment":
        services = rendered_config["services"]
        assert isinstance(services, dict)
        service = services["opensearch"]
        assert isinstance(service, dict)
        service["environment"] = {"FIXTURE_SETTING": "other"}
    else:
        raise AssertionError(f"unhandled fixture case {case}")

    with pytest.raises(ProvenanceError):
        verify_container_environment(
            inspect,
            profile,
            root=tmp_path,
            architecture="aarch64",
            rendered_config=rendered_config,
            image_inspect=image_inspect,
        )


def test_opensearch_environment_requires_registered_version_node_and_plugins(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "benchmark.toml"
    profile_path.write_text(_fixture_profile_text())
    profile = load_benchmark_profile(profile_path)
    container_id = "a" * 64
    root_response = {
        "name": container_id[:12],
        "version": {"distribution": "opensearch", "number": "3.8.0"},
    }
    plugins = [
        {"component": "opensearch-neural-search", "version": "3.8.0.0"},
        {"component": "opensearch-knn", "version": "3.8.0.0"},
    ]

    assert (
        verify_opensearch_environment(
            profile,
            root_response=root_response,
            plugins=plugins,
            container_id=container_id,
        )
        == _fixture_live_facts()["opensearch"]
    )

    with pytest.raises(ProvenanceError, match="plugin inventory"):
        verify_opensearch_environment(
            profile,
            root_response=root_response,
            plugins=plugins[:-1],
            container_id=container_id,
        )

    drifted_root = copy.deepcopy(root_response)
    drifted_root["name"] = "c" * 12
    with pytest.raises(ProvenanceError, match="node identity"):
        verify_opensearch_environment(
            profile,
            root_response=drifted_root,
            plugins=plugins,
            container_id=container_id,
        )


def test_decision_provenance_requires_a_clean_bound_commit_and_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "--allow-empty",
            "-qm",
            "empty fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    empty_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (tmp_path / "config").mkdir()
    (tmp_path / "poc").mkdir()
    (tmp_path / "results/environment").mkdir(parents=True)
    profile_path = tmp_path / "config/benchmark.toml"
    profile_path.write_text(_fixture_profile_text())
    (tmp_path / "poc/example.py").write_text("VALUE = 1\n")
    environment_path = tmp_path / "results/environment/benchmark-profile.json"
    live_facts = _fixture_live_facts()
    _write_environment(environment_path, live_facts)
    subprocess.run(
        ["git", "add", "config/benchmark.toml", "poc/example.py"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    provenance = collect_manifest_provenance(
        tmp_path,
        profile_path=profile_path,
        environment_path=environment_path,
    )
    monkeypatch.setattr(
        "poc.provenance._collect_live_environment_facts",
        lambda *, root, profile: copy.deepcopy(live_facts),
    )

    assert verify_decision_provenance(
        provenance,
        root=tmp_path,
        profile_path=profile_path,
        environment_path=environment_path,
        require_current_code_revision=True,
    )

    for key, value in (
        ("git_commit", None),
        ("source_dirty", True),
        ("source_tree_sha256", "0" * 64),
    ):
        tampered = copy.deepcopy(provenance)
        tampered["code_revision"][key] = value
        assert not verify_decision_provenance(
            tampered,
            root=tmp_path,
            profile_path=profile_path,
            environment_path=environment_path,
        )

    tampered_profile = copy.deepcopy(provenance)
    tampered_profile["benchmark_profile"]["cpu_limit"] = 4
    assert not verify_decision_provenance(
        tampered_profile,
        root=tmp_path,
        profile_path=profile_path,
        environment_path=environment_path,
    )

    tampered_environment = copy.deepcopy(provenance)
    tampered_environment["benchmark_environment"]["sha256"] = "f" * 64
    assert not verify_decision_provenance(
        tampered_environment,
        root=tmp_path,
        profile_path=profile_path,
        environment_path=environment_path,
    )

    empty_revision = copy.deepcopy(provenance)
    empty_revision["code_revision"] = {
        "git_commit": empty_commit,
        "source_tree_sha256": hashlib.sha256().hexdigest(),
        "source_dirty": False,
    }
    assert not verify_decision_provenance(
        empty_revision,
        root=tmp_path,
        profile_path=profile_path,
        environment_path=environment_path,
    )

    forged_facts = copy.deepcopy(live_facts)
    forged_facts["container_id"] = "c" * 64
    forged_opensearch = forged_facts["opensearch"]
    assert isinstance(forged_opensearch, dict)
    forged_opensearch["node_name"] = "c" * 12
    _write_environment(environment_path, forged_facts)
    forged_provenance = collect_manifest_provenance(
        tmp_path,
        profile_path=profile_path,
        environment_path=environment_path,
    )
    assert not verify_decision_provenance(
        forged_provenance,
        root=tmp_path,
        profile_path=profile_path,
        environment_path=environment_path,
    )

    _write_environment(environment_path, live_facts)
    provenance = collect_manifest_provenance(
        tmp_path,
        profile_path=profile_path,
        environment_path=environment_path,
    )
    stale_live_facts = copy.deepcopy(live_facts)
    stale_live_facts["container_id"] = "d" * 64
    stale_opensearch = stale_live_facts["opensearch"]
    assert isinstance(stale_opensearch, dict)
    stale_opensearch["node_name"] = "d" * 12
    monkeypatch.setattr(
        "poc.provenance._collect_live_environment_facts",
        lambda *, root, profile: stale_live_facts,
    )
    assert not verify_decision_provenance(
        provenance,
        root=tmp_path,
        profile_path=profile_path,
        environment_path=environment_path,
    )


def test_current_revision_allows_result_only_commits_but_rejects_source_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "config").mkdir()
    (tmp_path / "poc").mkdir()
    (tmp_path / "results/environment").mkdir(parents=True)
    profile_path = tmp_path / "config/benchmark.toml"
    profile_path.write_text(_fixture_profile_text())
    source_path = tmp_path / "poc/example.py"
    source_path.write_text("VALUE = 1\n")
    environment_path = tmp_path / "results/environment/benchmark-profile.json"
    live_facts = _fixture_live_facts()
    _write_environment(environment_path, live_facts)
    subprocess.run(
        ["git", "add", "config/benchmark.toml", "poc/example.py"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "source fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    provenance = collect_manifest_provenance(
        tmp_path,
        profile_path=profile_path,
        environment_path=environment_path,
    )
    monkeypatch.setattr(
        "poc.provenance._collect_live_environment_facts",
        lambda *, root, profile: copy.deepcopy(live_facts),
    )

    result_path = tmp_path / "results/seal.json"
    result_path.write_text('{"sealed": true}\n')
    subprocess.run(["git", "add", "results/seal.json"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "evidence seal",
        ],
        cwd=tmp_path,
        check=True,
    )
    assert verify_decision_provenance(
        provenance,
        root=tmp_path,
        profile_path=profile_path,
        environment_path=environment_path,
        require_current_code_revision=True,
    )

    source_path.write_text("VALUE = 2\n")
    assert not verify_decision_provenance(
        provenance,
        root=tmp_path,
        profile_path=profile_path,
        environment_path=environment_path,
        require_current_code_revision=True,
    )

    def unavailable(*, root: Path, profile: object) -> dict[str, Any]:
        raise ProvenanceError("Docker is unavailable")

    monkeypatch.setattr(
        "poc.provenance._collect_live_environment_facts",
        unavailable,
    )
    assert not verify_decision_provenance(
        provenance,
        root=tmp_path,
        profile_path=profile_path,
        environment_path=environment_path,
    )


def test_schema_one_environment_fails_before_any_live_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "config").mkdir()
    (tmp_path / "poc").mkdir()
    (tmp_path / "results/environment").mkdir(parents=True)
    profile_path = tmp_path / "config/benchmark.toml"
    profile_path.write_text(_fixture_profile_text())
    (tmp_path / "poc/example.py").write_text("VALUE = 1\n")
    environment_path = tmp_path / "results/environment/benchmark-profile.json"
    environment_path.write_text(json.dumps({"schema_version": 1}) + "\n")
    subprocess.run(
        ["git", "add", "config/benchmark.toml", "poc/example.py"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    provenance = collect_manifest_provenance(
        tmp_path,
        profile_path=profile_path,
        environment_path=environment_path,
    )

    live_probe_calls = 0

    def unexpected_probe(*, root: Path, profile: object) -> dict[str, Any]:
        nonlocal live_probe_calls
        live_probe_calls += 1
        return _fixture_live_facts()

    monkeypatch.setattr(
        "poc.provenance._collect_live_environment_facts",
        unexpected_probe,
    )
    assert not verify_decision_provenance(
        provenance,
        root=tmp_path,
        profile_path=profile_path,
        environment_path=environment_path,
    )
    assert live_probe_calls == 0


def test_live_environment_capture_has_schema_two_and_cannot_disable_live_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_path = tmp_path / "benchmark.toml"
    profile_path.write_text(_fixture_profile_text())
    profile = load_benchmark_profile(profile_path)
    live_facts = _fixture_live_facts()
    monkeypatch.setattr(
        "poc.provenance._collect_live_environment_facts",
        lambda *, root, profile: live_facts,
    )
    verified_at = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)

    environment = collect_live_benchmark_environment(
        tmp_path,
        profile,
        verified_at=verified_at,
    )
    assert environment == {
        "schema_version": 2,
        "verified_at": verified_at.isoformat(),
        **live_facts,
    }
    assert verify_live_benchmark_environment(
        environment,
        root=tmp_path,
        profile=profile,
    )

    disabled_profile = profile_path.read_text().replace(
        "requires_live_verification = true",
        "requires_live_verification = false",
    )
    profile_path.write_text(disabled_profile)
    with pytest.raises(ConfigError, match="live verification"):
        load_benchmark_profile(profile_path)


def test_environment_capture_preserves_bytes_when_only_timestamp_changes(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "benchmark.toml"
    profile_path.write_text(_fixture_profile_text())
    profile = load_benchmark_profile(profile_path)
    output_path = tmp_path / "benchmark-profile.json"
    existing = _fixture_environment("2026-08-22T12:00:00+00:00")
    existing_bytes = json.dumps(existing, separators=(",", ":")).encode()
    output_path.write_bytes(existing_bytes)
    candidate = _fixture_environment("2026-08-22T12:05:00+00:00")

    assert not PERSIST_BENCHMARK_ENVIRONMENT(
        output_path,
        candidate,
        profile=profile,
    )
    assert output_path.read_bytes() == existing_bytes


@pytest.mark.parametrize("invalid_existing", (False, True))
def test_environment_capture_rewrites_changed_or_invalid_artifacts(
    tmp_path: Path,
    invalid_existing: bool,
) -> None:
    profile_path = tmp_path / "benchmark.toml"
    profile_path.write_text(_fixture_profile_text())
    profile = load_benchmark_profile(profile_path)
    output_path = tmp_path / "benchmark-profile.json"
    candidate = _fixture_environment("2026-08-22T12:05:00+00:00")
    if invalid_existing:
        output_path.write_text("{invalid")
    else:
        existing = _fixture_environment("2026-08-22T12:00:00+00:00")
        existing["container_id"] = "d" * 64
        opensearch = cast(dict[str, Any], existing["opensearch"])
        opensearch["node_name"] = "d" * 12
        output_path.write_text(json.dumps(existing))

    assert PERSIST_BENCHMARK_ENVIRONMENT(
        output_path,
        candidate,
        profile=profile,
    )
    assert json.loads(output_path.read_text()) == candidate
