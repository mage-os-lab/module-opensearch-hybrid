from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import re
import subprocess
import tarfile
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from poc.config import ConfigError
from poc.os_client import OpenSearchClient

_SOURCE_DIRECTORIES = ("config", "poc", "scripts", "tests")
_SOURCE_FILES = (
    ".python-version",
    "EXPERIMENT-AMENDMENTS.md",
    "Makefile",
    "compose.bench.yml",
    "compose.opensearch-2.19.yml",
    "docker-compose.yml",
    "pyproject.toml",
    "uv.lock",
)
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
_PLUGIN_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+")
_DURATION = re.compile(r"(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<unit>ns|us|ms|s|m|h)")
_ENVIRONMENT_KEYS = {
    "architecture",
    "compose_files",
    "compose_project",
    "compose_service",
    "container_id",
    "cpu_limit",
    "image",
    "image_id",
    "latency_architecture_eligible",
    "limits_verified",
    "memory_limit_bytes",
    "opensearch",
    "profile_id",
    "runtime",
    "schema_version",
    "verified_at",
}


@dataclass(frozen=True, slots=True)
class CodeRevision:
    git_commit: str | None
    source_tree_sha256: str
    source_dirty: bool


@dataclass(frozen=True, slots=True)
class BenchmarkProfile:
    profile_id: str
    requires_live_verification: bool
    cpu_limit: int
    memory_limit_bytes: int
    compose_project: str
    compose_service: str
    compose_files: tuple[str, ...]
    container_image: str
    container_image_id: str
    opensearch_url: str
    opensearch_version: str
    plugin_version: str
    plugin_components: tuple[str, ...]
    latency_required_architecture: str


class ProvenanceError(RuntimeError):
    """Raised when the live benchmark environment violates the registered profile."""


def source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files: list[Path] = []
    for name in _SOURCE_FILES:
        path = root / name
        if path.is_file():
            files.append(path)
    for directory in _SOURCE_DIRECTORIES:
        path = root / directory
        if not path.is_dir():
            continue
        files.extend(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file()
            and "__pycache__" not in candidate.parts
            and candidate.suffix not in {".pyc", ".pyo"}
        )
    for path in sorted(set(files), key=lambda candidate: candidate.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def source_tree_sha256_at_revision(root: Path, revision: str) -> str | None:
    if _GIT_COMMIT.fullmatch(revision) is None:
        return None
    listing = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", revision],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if listing.returncode != 0:
        return None
    names = listing.stdout.splitlines()
    if not any(name.startswith("config/") for name in names) or not any(
        name.startswith("poc/") for name in names
    ):
        return None
    selected_roots = [
        directory
        for directory in _SOURCE_DIRECTORIES
        if any(name.startswith(f"{directory}/") for name in names)
    ]
    selected_roots.extend(name for name in _SOURCE_FILES if name in names)
    if not selected_roots:
        return None
    archive = subprocess.run(
        ["git", "archive", "--format=tar", revision, "--", *selected_roots],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if archive.returncode != 0:
        return None
    files: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as handle:
            for member in handle.getmembers():
                if not member.isfile():
                    continue
                path = Path(member.name)
                if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                    continue
                extracted = handle.extractfile(member)
                if extracted is None:
                    return None
                files[path.as_posix()] = extracted.read()
    except tarfile.TarError:
        return None
    digest = hashlib.sha256()
    for relative, payload in sorted(files.items()):
        encoded = relative.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def collect_code_revision(root: Path) -> CodeRevision:
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    git_commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    status_result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    source_dirty = status_result.returncode != 0 or any(
        _is_source_status_line(line) for line in status_result.stdout.splitlines() if line.strip()
    )
    return CodeRevision(
        git_commit=git_commit,
        source_tree_sha256=source_tree_sha256(root),
        source_dirty=source_dirty,
    )


def load_benchmark_profile(path: Path) -> BenchmarkProfile:
    with path.open("rb") as handle:
        values = tomllib.load(handle)
    if values.get("schema_version") != 2:
        raise ConfigError("unsupported benchmark profile schema")
    profile_id = values.get("profile_id")
    requires_live_verification = values.get("requires_live_verification")
    cpu_limit = values.get("cpu_limit")
    memory_limit_bytes = values.get("memory_limit_bytes")
    compose_project = values.get("compose_project")
    compose_service = values.get("compose_service")
    compose_files = values.get("compose_files")
    container_image = values.get("container_image")
    container_image_id = values.get("container_image_id")
    opensearch_url = values.get("opensearch_url")
    opensearch_version = values.get("opensearch_version")
    plugin_version = values.get("plugin_version")
    plugin_components = values.get("plugin_components")
    architecture = values.get("latency_required_architecture")
    if not isinstance(profile_id, str) or not profile_id:
        raise ConfigError("benchmark profile ID must be a non-empty string")
    if requires_live_verification is not True:
        raise ConfigError("benchmark profile must require live verification")
    if not isinstance(cpu_limit, int) or isinstance(cpu_limit, bool) or cpu_limit <= 0:
        raise ConfigError("benchmark CPU limit must be a positive integer")
    if (
        not isinstance(memory_limit_bytes, int)
        or isinstance(memory_limit_bytes, bool)
        or memory_limit_bytes <= 0
    ):
        raise ConfigError("benchmark memory limit must be a positive integer")
    if not isinstance(compose_project, str) or not compose_project:
        raise ConfigError("benchmark Compose project must be a non-empty string")
    if not isinstance(compose_service, str) or not compose_service:
        raise ConfigError("benchmark Compose service must be a non-empty string")
    if (
        not isinstance(compose_files, list)
        or not compose_files
        or not all(isinstance(value, str) and value for value in compose_files)
    ):
        raise ConfigError("benchmark compose files must be a non-empty string list")
    if len(compose_files) != len(set(compose_files)) or any(
        Path(value).is_absolute() or ".." in Path(value).parts for value in compose_files
    ):
        raise ConfigError("benchmark Compose files must be unique safe relative paths")
    if not isinstance(container_image, str) or not container_image:
        raise ConfigError("benchmark container image must be a non-empty string")
    if (
        not isinstance(container_image_id, str)
        or _IMAGE_ID.fullmatch(container_image_id) is None
    ):
        raise ConfigError("benchmark container image ID must be an exact sha256 digest")
    if not isinstance(opensearch_url, str) or not _is_local_opensearch_url(opensearch_url):
        raise ConfigError("benchmark OpenSearch URL must be a loopback HTTP endpoint")
    if not isinstance(opensearch_version, str) or _VERSION.fullmatch(opensearch_version) is None:
        raise ConfigError("benchmark OpenSearch version must be an exact release")
    if not isinstance(plugin_version, str) or _PLUGIN_VERSION.fullmatch(plugin_version) is None:
        raise ConfigError("benchmark plugin version must be an exact release")
    if (
        not isinstance(plugin_components, list)
        or not plugin_components
        or not all(isinstance(value, str) and value for value in plugin_components)
        or plugin_components != sorted(plugin_components)
        or len(plugin_components) != len(set(plugin_components))
    ):
        raise ConfigError(
            "benchmark plugin components must be a sorted unique non-empty string list"
        )
    if architecture not in {"x86_64", "arm64"}:
        raise ConfigError("benchmark latency architecture must be x86_64 or arm64")
    return BenchmarkProfile(
        profile_id=profile_id,
        requires_live_verification=True,
        cpu_limit=cpu_limit,
        memory_limit_bytes=memory_limit_bytes,
        compose_project=compose_project,
        compose_service=compose_service,
        compose_files=tuple(compose_files),
        container_image=container_image,
        container_image_id=container_image_id,
        opensearch_url=opensearch_url,
        opensearch_version=opensearch_version,
        plugin_version=plugin_version,
        plugin_components=tuple(plugin_components),
        latency_required_architecture=architecture,
    )


def registered_opensearch_url(
    profile_path: Path,
    *,
    environment_variable: str,
) -> str:
    """Return the profile endpoint and reject a conflicting process override."""

    profile = load_benchmark_profile(profile_path)
    registered = profile.opensearch_url.rstrip("/")
    override = os.environ.get(environment_variable)
    if override is None:
        return registered
    if not _is_local_opensearch_url(override):
        raise ProvenanceError(
            f"{environment_variable} contains an invalid OpenSearch endpoint"
        )
    normalized_override = override.rstrip("/")
    if normalized_override != registered:
        raise ProvenanceError(
            f"{environment_variable} does not match the registered benchmark profile: "
            f"expected {registered!r}, found {normalized_override!r}"
        )
    return registered


def registered_opensearch_client(
    profile_path: Path,
    *,
    environment_variable: str,
    timeout: float,
) -> OpenSearchClient:
    """Construct a client whose endpoint is bound to one registered profile."""

    return OpenSearchClient(
        registered_opensearch_url(
            profile_path,
            environment_variable=environment_variable,
        ),
        timeout=timeout,
    )


def require_registered_opensearch_client(
    client: object,
    profile_path: Path,
) -> None:
    """Reject a production client that is bound to a different profile endpoint."""

    if not isinstance(client, OpenSearchClient):
        return
    profile = load_benchmark_profile(profile_path)
    registered = profile.opensearch_url.rstrip("/")
    if client.base_url != registered:
        raise ProvenanceError(
            "OpenSearch client endpoint does not match the registered benchmark profile: "
            f"expected {registered!r}, found {client.base_url!r}"
        )


def verify_container_resources(
    inspect: Mapping[str, Any],
    profile: BenchmarkProfile,
    *,
    architecture: str,
) -> dict[str, Any]:
    host_config = inspect.get("HostConfig")
    container_config = inspect.get("Config")
    if not isinstance(host_config, Mapping) or not isinstance(container_config, Mapping):
        raise ProvenanceError("Docker inspection is missing container resource metadata")

    expected_nanocpus = profile.cpu_limit * 1_000_000_000
    actual_nanocpus = host_config.get("NanoCpus")
    if actual_nanocpus != expected_nanocpus:
        raise ProvenanceError(
            "container CPU limit does not match the registered profile: "
            f"expected {profile.cpu_limit}, found {actual_nanocpus!r} NanoCPUs"
        )

    actual_memory = host_config.get("Memory")
    if actual_memory != profile.memory_limit_bytes:
        raise ProvenanceError(
            "container memory limit does not match the registered profile: "
            f"expected {profile.memory_limit_bytes}, found {actual_memory!r}"
        )

    normalized_architecture = _normalize_architecture(architecture)
    container_id = inspect.get("Id")
    image = container_config.get("Image")
    image_id = inspect.get("Image")
    if (
        not isinstance(container_id, str)
        or _CONTAINER_ID.fullmatch(container_id) is None
        or not isinstance(image, str)
        or not image
        or image_id != profile.container_image_id
    ):
        raise ProvenanceError(
            "Docker inspection is missing container identity or uses an unregistered image ID"
        )
    return {
        "container_id": container_id,
        "image": image,
        "image_id": image_id,
        "cpu_limit": profile.cpu_limit,
        "memory_limit_bytes": profile.memory_limit_bytes,
        "architecture": normalized_architecture,
        "latency_architecture_eligible": (
            normalized_architecture == profile.latency_required_architecture
        ),
        "limits_verified": True,
    }


def verify_container_environment(
    inspect: Mapping[str, Any],
    profile: BenchmarkProfile,
    *,
    root: Path,
    architecture: str,
    rendered_config: Mapping[str, Any] | None = None,
    image_inspect: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = inspect.get("State")
    container_config = inspect.get("Config")
    if (
        not isinstance(state, Mapping)
        or state.get("Running") is not True
        or not isinstance(container_config, Mapping)
    ):
        raise ProvenanceError("Docker inspection does not describe a running container")
    if container_config.get("Image") != profile.container_image:
        raise ProvenanceError(
            "live container image does not match the registered profile: "
            f"expected {profile.container_image!r}, found {container_config.get('Image')!r}"
        )
    labels = container_config.get("Labels")
    if not isinstance(labels, Mapping):
        raise ProvenanceError("Docker inspection is missing Compose labels")
    if labels.get("com.docker.compose.project") != profile.compose_project:
        raise ProvenanceError("live Compose project does not match the registered profile")
    if labels.get("com.docker.compose.service") != profile.compose_service:
        raise ProvenanceError("live Compose service does not match the registered profile")

    expected_working_directory = root.resolve()
    working_directory = labels.get("com.docker.compose.project.working_dir")
    if (
        not isinstance(working_directory, str)
        or Path(working_directory).resolve() != expected_working_directory
    ):
        raise ProvenanceError("live Compose working directory does not match the repository")
    compose_label = labels.get("com.docker.compose.project.config_files")
    if not isinstance(compose_label, str) or not compose_label:
        raise ProvenanceError("Docker inspection is missing live Compose files")
    live_compose_paths = tuple(
        _resolve_compose_path(value, working_directory)
        for value in compose_label.split(",")
        if value.strip()
    )
    expected_compose_paths = tuple(
        (root / relative).resolve() for relative in profile.compose_files
    )
    if live_compose_paths != expected_compose_paths:
        raise ProvenanceError(
            "live Compose files do not match the registered profile: "
            f"expected {expected_compose_paths!r}, found {live_compose_paths!r}"
        )
    if not all(path.is_file() for path in expected_compose_paths):
        raise ProvenanceError("a registered Compose file is missing from the repository")

    rendered = rendered_config or _load_rendered_compose_config(root, profile)
    registered_image = image_inspect or _load_image_inspection(profile)
    runtime = _verify_rendered_container_runtime(
        inspect,
        registered_image,
        rendered,
        profile,
        root=root,
        architecture=architecture,
    )

    return {
        "profile_id": profile.profile_id,
        "compose_project": profile.compose_project,
        "compose_service": profile.compose_service,
        "compose_files": list(profile.compose_files),
        **verify_container_resources(inspect, profile, architecture=architecture),
        "runtime": runtime,
    }


def _load_rendered_compose_config(
    root: Path,
    profile: BenchmarkProfile,
) -> Mapping[str, Any]:
    command = ["docker", "compose", "--project-name", profile.compose_project]
    for relative in profile.compose_files:
        command.extend(("--file", str((root / relative).resolve())))
    command.extend(("config", "--format", "json"))
    try:
        value = json.loads(_run_command(*command))
    except json.JSONDecodeError as exc:
        raise ProvenanceError("rendered Docker Compose configuration is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise ProvenanceError("rendered Docker Compose configuration schema differs")
    return value


def _load_image_inspection(profile: BenchmarkProfile) -> Mapping[str, Any]:
    try:
        value = json.loads(
            _run_command("docker", "inspect", "--type=image", profile.container_image_id)
        )
    except json.JSONDecodeError as exc:
        raise ProvenanceError("Docker image inspection returned invalid JSON") from exc
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(value[0], Mapping)
    ):
        raise ProvenanceError("Docker image inspection did not return exactly one image")
    return value[0]


def _verify_rendered_container_runtime(
    inspect: Mapping[str, Any],
    image_inspect: Mapping[str, Any],
    rendered_config: Mapping[str, Any],
    profile: BenchmarkProfile,
    *,
    root: Path,
    architecture: str,
) -> dict[str, Any]:
    container_config = _mapping_value(inspect, "Config", "container runtime Config")
    host_config = _mapping_value(inspect, "HostConfig", "container runtime HostConfig")
    network_settings = _mapping_value(
        inspect,
        "NetworkSettings",
        "container runtime NetworkSettings",
    )
    image_config = _mapping_value(image_inspect, "Config", "registered image Config")
    services = _mapping_value(rendered_config, "services", "rendered Compose services")
    service = services.get(profile.compose_service)
    if not isinstance(service, Mapping):
        raise ProvenanceError("registered service is missing from rendered Compose configuration")
    if rendered_config.get("name") != profile.compose_project:
        raise ProvenanceError("rendered Compose project does not match the registered profile")
    if service.get("image") != profile.container_image:
        raise ProvenanceError("rendered Compose image does not match the registered profile")

    normalized_architecture = _normalize_architecture(architecture)
    if (
        image_inspect.get("Id") != profile.container_image_id
        or image_inspect.get("Os") != "linux"
        or _normalize_architecture(str(image_inspect.get("Architecture", "")))
        != normalized_architecture
    ):
        raise ProvenanceError(
            "registered Docker image identity or platform differs from the live runtime"
        )
    repository = _container_image_repository(profile.container_image)
    repository_digest = f"{repository}@{profile.container_image_id}"
    repo_digests = image_inspect.get("RepoDigests")
    if (
        not isinstance(repo_digests, list)
        or repository_digest not in repo_digests
        or any(not isinstance(value, str) for value in repo_digests)
    ):
        raise ProvenanceError("registered Docker image repository digest is missing")
    descriptor = _mapping_value(
        inspect,
        "ImageManifestDescriptor",
        "live image manifest descriptor",
    )
    platform_value = _mapping_value(
        descriptor,
        "platform",
        "live image manifest platform",
    )
    platform_digest = descriptor.get("digest")
    if (
        not isinstance(platform_digest, str)
        or _IMAGE_ID.fullmatch(platform_digest) is None
        or platform_value.get("os") != "linux"
        or _normalize_architecture(str(platform_value.get("architecture", "")))
        != normalized_architecture
    ):
        raise ProvenanceError("live image manifest digest or platform differs")

    expected_environment = _parse_compose_environment(service.get("environment"))
    merged_environment = _parse_container_environment(image_config.get("Env"))
    merged_environment.update(expected_environment)
    live_environment = _parse_container_environment(container_config.get("Env"))
    if live_environment != merged_environment:
        raise ProvenanceError("container runtime environment differs from rendered Compose")

    entrypoint = _expected_process_vector(
        service.get("entrypoint"),
        image_config.get("Entrypoint"),
        name="entrypoint",
    )
    command = _expected_process_vector(
        service.get("command"),
        image_config.get("Cmd"),
        name="command",
    )
    user = _expected_process_string(service.get("user"), image_config.get("User"))
    working_directory = _expected_process_string(
        service.get("working_dir"),
        image_config.get("WorkingDir"),
    )
    if (
        _string_vector(container_config.get("Entrypoint"), "live entrypoint") != entrypoint
        or _string_vector(container_config.get("Cmd"), "live command") != command
        or container_config.get("User", "") != user
        or container_config.get("WorkingDir", "") != working_directory
    ):
        raise ProvenanceError("container runtime process differs from rendered Compose")

    ports = _normalize_rendered_ports(service.get("ports"))
    live_ports = _normalize_live_ports(host_config.get("PortBindings"))
    realized_ports = _normalize_live_ports(network_settings.get("Ports"))
    if live_ports != ports or realized_ports != ports:
        raise ProvenanceError("container runtime port bindings differ from rendered Compose")

    ulimits = _normalize_rendered_ulimits(service.get("ulimits"))
    if _normalize_live_ulimits(host_config.get("Ulimits")) != ulimits:
        raise ProvenanceError("container runtime ulimits differ from rendered Compose")

    mounts = _normalize_rendered_mounts(
        service.get("volumes"),
        rendered_config,
        profile,
        root=root,
    )
    live_mounts = _normalize_live_mounts(inspect.get("Mounts"), root=root)
    if live_mounts != mounts:
        raise ProvenanceError("container runtime mounts differ from rendered Compose")
    portable_mounts = [
        {key: value for key, value in mount.items() if key != "runtime_source"}
        for mount in mounts
    ]

    networks = _normalize_rendered_networks(service, rendered_config, profile)
    if (
        _normalize_live_networks(network_settings.get("Networks")) != networks
        or host_config.get("NetworkMode") not in networks
    ):
        raise ProvenanceError("container runtime networks differ from rendered Compose")

    security = {
        "cap_add": _optional_string_list(service.get("cap_add"), "Compose cap_add"),
        "cap_drop": _optional_string_list(service.get("cap_drop"), "Compose cap_drop"),
        "cgroupns_mode": str(service.get("cgroup", "private")),
        "ipc_mode": str(service.get("ipc", "private")),
        "pid_mode": str(service.get("pid", "")),
        "privileged": bool(service.get("privileged", False)),
        "publish_all_ports": bool(service.get("publish_all_ports", False)),
        "read_only": bool(service.get("read_only", False)),
        "security_opt": _optional_string_list(
            service.get("security_opt"),
            "Compose security_opt",
        ),
        "userns_mode": str(service.get("userns_mode", "")),
    }
    live_security = {
        "cap_add": _optional_string_list(host_config.get("CapAdd"), "live CapAdd"),
        "cap_drop": _optional_string_list(host_config.get("CapDrop"), "live CapDrop"),
        "cgroupns_mode": str(host_config.get("CgroupnsMode", "")),
        "ipc_mode": str(host_config.get("IpcMode", "")),
        "pid_mode": str(host_config.get("PidMode", "")),
        "privileged": host_config.get("Privileged"),
        "publish_all_ports": host_config.get("PublishAllPorts"),
        "read_only": host_config.get("ReadonlyRootfs"),
        "security_opt": _optional_string_list(
            host_config.get("SecurityOpt"),
            "live SecurityOpt",
        ),
        "userns_mode": str(host_config.get("UsernsMode", "")),
    }
    if live_security != security:
        raise ProvenanceError("container runtime security settings differ from rendered Compose")

    resources = _normalize_rendered_resources(service, profile)
    live_resources = {
        "cpu_limit_nanocpus": host_config.get("NanoCpus"),
        "cpu_period": host_config.get("CpuPeriod"),
        "cpu_quota": host_config.get("CpuQuota"),
        "cpu_shares": host_config.get("CpuShares"),
        "cpuset_cpus": host_config.get("CpusetCpus"),
        "memory_limit_bytes": host_config.get("Memory"),
        "memory_reservation_bytes": host_config.get("MemoryReservation"),
        "memory_swap_bytes": host_config.get("MemorySwap"),
        "pids_limit": host_config.get("PidsLimit"),
        "shm_size_bytes": host_config.get("ShmSize"),
    }
    if live_resources != resources:
        raise ProvenanceError("container runtime resource settings differ from rendered Compose")

    restart_policy = _normalize_rendered_restart_policy(service.get("restart"))
    if _normalize_live_restart_policy(host_config.get("RestartPolicy")) != restart_policy:
        raise ProvenanceError("container runtime restart policy differs from rendered Compose")

    healthcheck = _normalize_rendered_healthcheck(service.get("healthcheck"))
    if _normalize_live_healthcheck(container_config.get("Healthcheck")) != healthcheck:
        raise ProvenanceError("container runtime healthcheck differs from rendered Compose")
    state = _mapping_value(inspect, "State", "container runtime State")
    state_health = _mapping_value(state, "Health", "container runtime health state")
    if state_health.get("Status") != "healthy":
        raise ProvenanceError("container runtime is not healthy")

    contract: dict[str, Any] = {
        "environment_sha256": _canonical_json_sha256(merged_environment),
        "entrypoint": entrypoint,
        "command": command,
        "user": user,
        "working_directory": working_directory,
        "ports": ports,
        "ulimits": ulimits,
        "mounts": portable_mounts,
        "networks": networks,
        "security": security,
        "resources": resources,
        "restart_policy": restart_policy,
        "healthcheck": healthcheck,
    }
    return {
        "repository_digest": repository_digest,
        "platform_manifest_digest": platform_digest,
        "platform_os": "linux",
        "contract": contract,
        "contract_sha256": _canonical_json_sha256(contract),
    }


def _mapping_value(
    value: Mapping[str, Any],
    key: str,
    name: str,
) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ProvenanceError(f"{name} is missing")
    return result


def _container_image_repository(image: str) -> str:
    name = image.rsplit("/", 1)[-1]
    return image.rsplit(":", 1)[0] if ":" in name else image


def _parse_compose_environment(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ProvenanceError("rendered Compose environment schema differs")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or not isinstance(item, str):
            raise ProvenanceError("rendered Compose environment schema differs")
        result[key] = item
    return result


def _parse_container_environment(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ProvenanceError("container runtime environment schema differs")
    result: dict[str, str] = {}
    for item in value:
        assert isinstance(item, str)
        key, separator, content = item.partition("=")
        if not separator or not key or key in result:
            raise ProvenanceError("container runtime environment schema differs")
        result[key] = content
    return result


def _expected_process_vector(
    compose_value: object,
    image_value: object,
    *,
    name: str,
) -> list[str]:
    return _string_vector(
        image_value if compose_value is None else compose_value,
        f"rendered {name}",
    )


def _string_vector(value: object, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ProvenanceError(f"{name} schema differs")
    return list(value)


def _expected_process_string(compose_value: object, image_value: object) -> str:
    value = image_value if compose_value is None else compose_value
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ProvenanceError("rendered process string schema differs")
    return value


def _normalize_rendered_ports(value: object) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ProvenanceError("rendered Compose port schema differs")
    ports: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ProvenanceError("rendered Compose port schema differs")
        target = item.get("target")
        published = item.get("published")
        protocol = item.get("protocol", "tcp")
        host_ip = item.get("host_ip", "0.0.0.0")
        if (
            not isinstance(target, int)
            or isinstance(target, bool)
            or not isinstance(published, str)
            or not published.isdigit()
            or not isinstance(protocol, str)
            or not isinstance(host_ip, str)
        ):
            raise ProvenanceError("rendered Compose port schema differs")
        ports.append(
            {
                "container_port": target,
                "host_ip": host_ip,
                "host_port": int(published),
                "protocol": protocol,
            }
        )
    return sorted(
        ports,
        key=lambda item: (
            item["container_port"],
            item["protocol"],
            item["host_ip"],
            item["host_port"],
        ),
    )


def _normalize_live_ports(value: object) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, Mapping):
        raise ProvenanceError("container runtime port binding schema differs")
    ports: list[dict[str, Any]] = []
    for key, bindings in value.items():
        if not isinstance(key, str) or "/" not in key or not isinstance(bindings, list):
            raise ProvenanceError("container runtime port binding schema differs")
        port_text, protocol = key.split("/", 1)
        if not port_text.isdigit():
            raise ProvenanceError("container runtime port binding schema differs")
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise ProvenanceError("container runtime port binding schema differs")
            host_ip = binding.get("HostIp")
            host_port = binding.get("HostPort")
            if (
                not isinstance(host_ip, str)
                or not isinstance(host_port, str)
                or not host_port.isdigit()
            ):
                raise ProvenanceError("container runtime port binding schema differs")
            ports.append(
                {
                    "container_port": int(port_text),
                    "host_ip": host_ip,
                    "host_port": int(host_port),
                    "protocol": protocol,
                }
            )
    return sorted(
        ports,
        key=lambda item: (
            item["container_port"],
            item["protocol"],
            item["host_ip"],
            item["host_port"],
        ),
    )


def _normalize_rendered_ulimits(value: object) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, Mapping):
        raise ProvenanceError("rendered Compose ulimit schema differs")
    limits: list[dict[str, Any]] = []
    for name, item in value.items():
        if not isinstance(name, str) or not name or not isinstance(item, Mapping):
            raise ProvenanceError("rendered Compose ulimit schema differs")
        soft = item.get("soft")
        hard = item.get("hard")
        if (
            not isinstance(soft, int)
            or isinstance(soft, bool)
            or not isinstance(hard, int)
            or isinstance(hard, bool)
        ):
            raise ProvenanceError("rendered Compose ulimit schema differs")
        limits.append({"name": name, "soft": soft, "hard": hard})
    return sorted(limits, key=lambda item: item["name"])


def _normalize_live_ulimits(value: object) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ProvenanceError("container runtime ulimit schema differs")
    limits: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ProvenanceError("container runtime ulimit schema differs")
        name = item.get("Name")
        soft = item.get("Soft")
        hard = item.get("Hard")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(soft, int)
            or isinstance(soft, bool)
            or not isinstance(hard, int)
            or isinstance(hard, bool)
        ):
            raise ProvenanceError("container runtime ulimit schema differs")
        limits.append({"name": name, "soft": soft, "hard": hard})
    return sorted(limits, key=lambda item: item["name"])


def _normalize_rendered_mounts(
    value: object,
    rendered_config: Mapping[str, Any],
    profile: BenchmarkProfile,
    *,
    root: Path,
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ProvenanceError("rendered Compose mount schema differs")
    top_level_volumes = rendered_config.get("volumes", {})
    if not isinstance(top_level_volumes, Mapping):
        raise ProvenanceError("rendered Compose volume registry schema differs")
    mounts: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ProvenanceError("rendered Compose mount schema differs")
        mount_type = item.get("type")
        source = item.get("source")
        target = item.get("target")
        read_only = item.get("read_only", False)
        if (
            mount_type not in {"bind", "volume"}
            or not isinstance(source, str)
            or not source
            or not isinstance(target, str)
            or not target.startswith("/")
            or not isinstance(read_only, bool)
        ):
            raise ProvenanceError("rendered Compose mount schema differs")
        if mount_type == "volume":
            volume = top_level_volumes.get(source)
            if not isinstance(volume, Mapping):
                raise ProvenanceError("rendered Compose named volume is not registered")
            actual_source = volume.get("name", f"{profile.compose_project}_{source}")
            if not isinstance(actual_source, str) or not actual_source:
                raise ProvenanceError("rendered Compose named volume schema differs")
            portable_source = source
            propagation = ""
        else:
            expected_source = _normalize_host_path(Path(source).resolve())
            actual_source = expected_source
            portable_source = _portable_mount_source(expected_source, root)
            bind = item.get("bind", {})
            if not isinstance(bind, Mapping):
                raise ProvenanceError("rendered Compose bind mount schema differs")
            propagation_value = bind.get("propagation", "rprivate")
            if not isinstance(propagation_value, str):
                raise ProvenanceError("rendered Compose bind mount propagation differs")
            propagation = propagation_value
        mounts.append(
            {
                "type": mount_type,
                "source": portable_source,
                "runtime_source": actual_source,
                "target": target,
                "read_only": read_only,
                "propagation": propagation,
            }
        )
    return sorted(mounts, key=lambda item: (item["target"], item["type"]))


def _normalize_live_mounts(value: object, *, root: Path) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ProvenanceError("container runtime mount schema differs")
    mounts: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ProvenanceError("container runtime mount schema differs")
        mount_type = item.get("Type")
        target = item.get("Destination")
        read_write = item.get("RW")
        propagation = item.get("Propagation", "")
        if (
            mount_type not in {"bind", "volume"}
            or not isinstance(target, str)
            or not target.startswith("/")
            or not isinstance(read_write, bool)
            or not isinstance(propagation, str)
        ):
            raise ProvenanceError("container runtime mount schema differs")
        if mount_type == "volume":
            runtime_source = item.get("Name")
            if not isinstance(runtime_source, str) or not runtime_source:
                raise ProvenanceError("container runtime named volume schema differs")
            prefix, separator, portable_source = runtime_source.partition("_")
            if not separator or not prefix or not portable_source:
                raise ProvenanceError("container runtime named volume schema differs")
        else:
            source = item.get("Source")
            if not isinstance(source, str) or not source:
                raise ProvenanceError("container runtime bind mount schema differs")
            runtime_source = _normalize_host_path(Path(source))
            portable_source = _portable_mount_source(runtime_source, root)
        mounts.append(
            {
                "type": mount_type,
                "source": portable_source,
                "runtime_source": runtime_source,
                "target": target,
                "read_only": not read_write,
                "propagation": propagation,
            }
        )
    return sorted(mounts, key=lambda item: (item["target"], item["type"]))


def _normalize_host_path(path: Path) -> str:
    text = path.as_posix()
    prefixes = ("/host_mnt", "/run/desktop/mnt/host")
    for prefix in prefixes:
        if text.startswith(f"{prefix}/"):
            return text[len(prefix) :]
    return text


def _portable_mount_source(source: str, root: Path) -> str:
    source_path = Path(source)
    try:
        return source_path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return source


def _normalize_rendered_networks(
    service: Mapping[str, Any],
    rendered_config: Mapping[str, Any],
    profile: BenchmarkProfile,
) -> list[str]:
    service_networks = service.get("networks", {"default": None})
    network_registry = rendered_config.get("networks", {})
    if not isinstance(service_networks, Mapping) or not isinstance(
        network_registry,
        Mapping,
    ):
        raise ProvenanceError("rendered Compose network schema differs")
    names: list[str] = []
    for key in service_networks:
        if not isinstance(key, str) or not key:
            raise ProvenanceError("rendered Compose network schema differs")
        registered = network_registry.get(key)
        if not isinstance(registered, Mapping):
            raise ProvenanceError("rendered Compose network is not registered")
        name = registered.get("name", f"{profile.compose_project}_{key}")
        if not isinstance(name, str) or not name:
            raise ProvenanceError("rendered Compose network name differs")
        names.append(name)
    return sorted(names)


def _normalize_live_networks(value: object) -> list[str]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not key for key in value
    ):
        raise ProvenanceError("container runtime network schema differs")
    return sorted(value)


def _optional_string_list(value: object, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ProvenanceError(f"{name} schema differs")
    return sorted(value)


def _normalize_rendered_resources(
    service: Mapping[str, Any],
    profile: BenchmarkProfile,
) -> dict[str, int | str | None]:
    cpus = service.get("cpus")
    memory = service.get("mem_limit")
    if not isinstance(cpus, int | float) or isinstance(cpus, bool):
        raise ProvenanceError("rendered Compose CPU limit schema differs")
    if not isinstance(memory, int | str) or isinstance(memory, bool):
        raise ProvenanceError("rendered Compose memory limit schema differs")
    try:
        memory_bytes = int(memory)
    except ValueError as exc:
        raise ProvenanceError("rendered Compose memory limit schema differs") from exc
    if cpus != profile.cpu_limit or memory_bytes != profile.memory_limit_bytes:
        raise ProvenanceError("rendered Compose resources differ from the registered profile")
    pids_limit = service.get("pids_limit")
    if pids_limit is not None and (
        not isinstance(pids_limit, int) or isinstance(pids_limit, bool)
    ):
        raise ProvenanceError("rendered Compose pids limit schema differs")
    return {
        "cpu_limit_nanocpus": int(float(cpus) * 1_000_000_000),
        "cpu_period": int(service.get("cpu_period", 0)),
        "cpu_quota": int(service.get("cpu_quota", 0)),
        "cpu_shares": int(service.get("cpu_shares", 0)),
        "cpuset_cpus": str(service.get("cpuset", "")),
        "memory_limit_bytes": memory_bytes,
        "memory_reservation_bytes": int(service.get("mem_reservation", 0)),
        "memory_swap_bytes": int(service.get("memswap_limit", memory_bytes * 2)),
        "pids_limit": pids_limit,
        "shm_size_bytes": int(service.get("shm_size", 64 * 1024**2)),
    }


def _normalize_rendered_restart_policy(value: object) -> dict[str, int | str]:
    if value is None or value == "no":
        return {"name": "no", "maximum_retry_count": 0}
    if not isinstance(value, str):
        raise ProvenanceError("rendered Compose restart policy schema differs")
    name, separator, count = value.partition(":")
    if separator and (name != "on-failure" or not count.isdigit()):
        raise ProvenanceError("rendered Compose restart policy schema differs")
    return {
        "name": name,
        "maximum_retry_count": int(count) if separator else 0,
    }


def _normalize_live_restart_policy(value: object) -> dict[str, int | str]:
    if not isinstance(value, Mapping):
        raise ProvenanceError("container runtime restart policy schema differs")
    name = value.get("Name")
    count = value.get("MaximumRetryCount")
    if (
        not isinstance(name, str)
        or not isinstance(count, int)
        or isinstance(count, bool)
    ):
        raise ProvenanceError("container runtime restart policy schema differs")
    return {"name": name, "maximum_retry_count": count}


def _normalize_rendered_healthcheck(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProvenanceError("rendered Compose healthcheck schema differs")
    test = _string_vector(value.get("test"), "rendered healthcheck test")
    retries = value.get("retries")
    if not isinstance(retries, int) or isinstance(retries, bool):
        raise ProvenanceError("rendered Compose healthcheck retries schema differs")
    return {
        "test": test,
        "interval_nanoseconds": _duration_nanoseconds(value.get("interval")),
        "timeout_nanoseconds": _duration_nanoseconds(value.get("timeout")),
        "start_period_nanoseconds": _duration_nanoseconds(value.get("start_period")),
        "retries": retries,
    }


def _normalize_live_healthcheck(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProvenanceError("container runtime healthcheck schema differs")
    test = _string_vector(value.get("Test"), "live healthcheck test")
    result = {
        "test": test,
        "interval_nanoseconds": value.get("Interval"),
        "timeout_nanoseconds": value.get("Timeout"),
        "start_period_nanoseconds": value.get("StartPeriod"),
        "retries": value.get("Retries"),
    }
    if any(
        not isinstance(item, int) or isinstance(item, bool)
        for key, item in result.items()
        if key != "test"
    ):
        raise ProvenanceError("container runtime healthcheck schema differs")
    return result


def _duration_nanoseconds(value: object) -> int:
    if not isinstance(value, str):
        raise ProvenanceError("rendered Compose duration schema differs")
    match = _DURATION.fullmatch(value)
    if match is None:
        raise ProvenanceError("rendered Compose duration schema differs")
    multipliers = {
        "ns": 1,
        "us": 1_000,
        "ms": 1_000_000,
        "s": 1_000_000_000,
        "m": 60_000_000_000,
        "h": 3_600_000_000_000,
    }
    return int(float(match.group("value")) * multipliers[match.group("unit")])


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def normalize_plugin_inventory(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ProvenanceError("OpenSearch plugin inventory is missing")
    inventory: list[tuple[str, str]] = []
    for plugin in value:
        if not isinstance(plugin, Mapping) or set(plugin) != {"component", "version"}:
            raise ProvenanceError("OpenSearch plugin inventory schema differs")
        component = plugin.get("component")
        version = plugin.get("version")
        if (
            not isinstance(component, str)
            or not component
            or not isinstance(version, str)
            or not version
        ):
            raise ProvenanceError("OpenSearch plugin inventory schema differs")
        inventory.append((component, version))
    if len(inventory) != len(set(inventory)):
        raise ProvenanceError("OpenSearch plugin inventory contains duplicates")
    return [
        {"component": component, "version": version} for component, version in sorted(inventory)
    ]


def verify_opensearch_environment(
    profile: BenchmarkProfile,
    *,
    root_response: Mapping[str, Any],
    plugins: object,
    container_id: str,
) -> dict[str, Any]:
    version = root_response.get("version")
    node_name = root_response.get("name")
    if (
        not isinstance(version, Mapping)
        or version.get("distribution") != "opensearch"
        or version.get("number") != profile.opensearch_version
    ):
        raise ProvenanceError("live OpenSearch version does not match the registered profile")
    if not isinstance(node_name, str) or not node_name or not container_id.startswith(node_name):
        raise ProvenanceError("live OpenSearch node identity differs from the container")
    normalized_plugins = normalize_plugin_inventory(plugins)
    expected_plugins = _expected_plugin_inventory(profile)
    if normalized_plugins != expected_plugins:
        raise ProvenanceError(
            "live OpenSearch plugin inventory does not match the registered profile"
        )
    return {
        "url": profile.opensearch_url,
        "version": profile.opensearch_version,
        "node_name": node_name,
        "plugins": normalized_plugins,
    }


def collect_live_benchmark_environment(
    root: Path,
    profile: BenchmarkProfile,
    *,
    verified_at: datetime | None = None,
) -> dict[str, Any]:
    timestamp = verified_at or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ProvenanceError("benchmark verification timestamp must include a timezone")
    return {
        "schema_version": 2,
        "verified_at": timestamp.isoformat(),
        **_collect_live_environment_facts(root=root, profile=profile),
    }


def verify_live_benchmark_environment(
    environment: Mapping[str, Any],
    *,
    root: Path,
    profile: BenchmarkProfile,
) -> bool:
    try:
        recorded_facts = _validate_environment_snapshot(environment, profile)
        live_facts = _collect_live_environment_facts(root=root, profile=profile)
    except Exception:
        return False
    return recorded_facts == live_facts


def benchmark_environment_snapshots_match(
    existing: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    profile: BenchmarkProfile,
) -> bool:
    """Compare validated live facts while ignoring only their capture timestamps."""

    try:
        existing_facts = _validate_environment_snapshot(existing, profile)
        candidate_facts = _validate_environment_snapshot(candidate, profile)
    except ProvenanceError:
        return False
    return existing_facts == candidate_facts


def _collect_live_environment_facts(
    *,
    root: Path,
    profile: BenchmarkProfile,
) -> dict[str, Any]:
    container_id = _find_live_container(profile)
    try:
        inspect_value = json.loads(_run_command("docker", "inspect", container_id))
    except json.JSONDecodeError as exc:
        raise ProvenanceError("Docker inspection returned invalid JSON") from exc
    if (
        not isinstance(inspect_value, list)
        or len(inspect_value) != 1
        or not isinstance(inspect_value[0], Mapping)
    ):
        raise ProvenanceError("Docker inspection did not return exactly one container")
    inspect = inspect_value[0]
    architecture = _run_command("docker", "exec", container_id, "uname", "-m")
    container_facts = verify_container_environment(
        inspect,
        profile,
        root=root,
        architecture=architecture,
    )
    try:
        with OpenSearchClient(profile.opensearch_url, timeout=5.0) as client:
            root_response = client.request("GET", "/")
            plugins = client.request("GET", "/_cat/plugins?format=json&h=component,version")
    except Exception as exc:
        raise ProvenanceError("registered OpenSearch endpoint could not be verified live") from exc
    if not isinstance(root_response, Mapping):
        raise ProvenanceError("live OpenSearch root response schema differs")
    opensearch_facts = verify_opensearch_environment(
        profile,
        root_response=root_response,
        plugins=plugins,
        container_id=str(container_facts["container_id"]),
    )
    return {**container_facts, "opensearch": opensearch_facts}


def _find_live_container(profile: BenchmarkProfile) -> str:
    output = _run_command(
        "docker",
        "ps",
        "--filter",
        f"label=com.docker.compose.project={profile.compose_project}",
        "--filter",
        f"label=com.docker.compose.service={profile.compose_service}",
        "--format",
        "{{.ID}}",
    )
    container_ids = [value for value in output.splitlines() if value]
    if len(container_ids) != 1:
        raise ProvenanceError(
            f"expected exactly one live {profile.compose_project} "
            f"{profile.compose_service} container, found {len(container_ids)}"
        )
    return container_ids[0]


def _run_command(*command: str) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ProvenanceError(f"could not run {command[0]}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise ProvenanceError(
            f"{' '.join(command[:2])} failed with status {result.returncode}: {detail}"
        )
    return result.stdout.strip()


def _validate_environment_snapshot(
    environment: Mapping[str, Any],
    profile: BenchmarkProfile,
) -> dict[str, Any]:
    if set(environment) != _ENVIRONMENT_KEYS or environment.get("schema_version") != 2:
        raise ProvenanceError("benchmark environment schema differs")
    verified_at = environment.get("verified_at")
    if not isinstance(verified_at, str):
        raise ProvenanceError("benchmark environment timestamp is missing")
    try:
        timestamp = datetime.fromisoformat(verified_at)
    except ValueError as exc:
        raise ProvenanceError("benchmark environment timestamp is invalid") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ProvenanceError("benchmark environment timestamp must include a timezone")
    if (
        environment.get("profile_id") != profile.profile_id
        or environment.get("compose_project") != profile.compose_project
        or environment.get("compose_service") != profile.compose_service
        or environment.get("compose_files") != list(profile.compose_files)
        or environment.get("image") != profile.container_image
        or environment.get("image_id") != profile.container_image_id
        or environment.get("cpu_limit") != profile.cpu_limit
        or environment.get("memory_limit_bytes") != profile.memory_limit_bytes
        or environment.get("limits_verified") is not True
    ):
        raise ProvenanceError("benchmark environment differs from the registered profile")
    container_id = environment.get("container_id")
    image_id = environment.get("image_id")
    architecture = environment.get("architecture")
    if (
        not isinstance(container_id, str)
        or _CONTAINER_ID.fullmatch(container_id) is None
        or not isinstance(image_id, str)
        or _IMAGE_ID.fullmatch(image_id) is None
        or not isinstance(architecture, str)
        or architecture not in {"arm64", "x86_64"}
        or environment.get("latency_architecture_eligible")
        is not (architecture == profile.latency_required_architecture)
    ):
        raise ProvenanceError("benchmark environment container facts are invalid")
    runtime = environment.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "contract",
        "contract_sha256",
        "platform_manifest_digest",
        "platform_os",
        "repository_digest",
    }:
        raise ProvenanceError("benchmark environment container runtime schema differs")
    contract = runtime.get("contract")
    contract_hash = runtime.get("contract_sha256")
    platform_digest = runtime.get("platform_manifest_digest")
    if (
        not isinstance(contract, Mapping)
        or not isinstance(contract_hash, str)
        or _SHA256.fullmatch(contract_hash) is None
        or contract_hash != _canonical_json_sha256(contract)
        or runtime.get("repository_digest")
        != f"{_container_image_repository(profile.container_image)}@{profile.container_image_id}"
        or not isinstance(platform_digest, str)
        or _IMAGE_ID.fullmatch(platform_digest) is None
        or runtime.get("platform_os") != "linux"
    ):
        raise ProvenanceError("benchmark environment container runtime facts are invalid")
    opensearch = environment.get("opensearch")
    if not isinstance(opensearch, Mapping) or set(opensearch) != {
        "node_name",
        "plugins",
        "url",
        "version",
    }:
        raise ProvenanceError("benchmark environment OpenSearch schema differs")
    node_name = opensearch.get("node_name")
    plugins = normalize_plugin_inventory(opensearch.get("plugins"))
    if (
        opensearch.get("url") != profile.opensearch_url
        or opensearch.get("version") != profile.opensearch_version
        or not isinstance(node_name, str)
        or not node_name
        or not container_id.startswith(node_name)
        or opensearch.get("plugins") != plugins
        or plugins != _expected_plugin_inventory(profile)
    ):
        raise ProvenanceError("benchmark environment OpenSearch facts are invalid")
    return {
        key: value
        for key, value in environment.items()
        if key not in {"schema_version", "verified_at"}
    }


def _expected_plugin_inventory(profile: BenchmarkProfile) -> list[dict[str, str]]:
    return [
        {"component": component, "version": profile.plugin_version}
        for component in profile.plugin_components
    ]


def _resolve_compose_path(value: str, working_directory: str) -> Path:
    path = Path(value.strip())
    if not path.is_absolute():
        path = Path(working_directory) / path
    return path.resolve()


def collect_manifest_provenance(
    root: Path,
    *,
    profile_path: Path | None = None,
    environment_path: Path | None = None,
) -> dict[str, Any]:
    profile_path = profile_path or root / "config/benchmark.toml"
    profile = load_benchmark_profile(profile_path)
    environment_path = environment_path or root / "results/environment/benchmark-profile.json"
    environment: dict[str, object] = {"status": "not_verified"}
    if environment_path.is_file():
        environment = {
            "status": "verified_artifact_present",
            "path": str(environment_path.relative_to(root)),
            "sha256": _file_sha256(environment_path),
        }
    return {
        "code_revision": asdict(collect_code_revision(root)),
        "benchmark_profile": _benchmark_profile_payload(profile),
        "benchmark_profile_sha256": _file_sha256(profile_path),
        "benchmark_environment": environment,
        "generator_host": {
            "architecture": platform.machine(),
            "system": platform.system(),
        },
    }


def verify_decision_provenance(
    provenance: Mapping[str, Any],
    *,
    root: Path,
    profile_path: Path,
    environment_path: Path,
    require_current_code_revision: bool = False,
) -> bool:
    try:
        profile = load_benchmark_profile(profile_path)
        profile_relative = profile_path.relative_to(root).as_posix()
        environment_relative = environment_path.relative_to(root).as_posix()
        profile_hash = _file_sha256(profile_path)
        environment_hash = _file_sha256(environment_path)
    except (ConfigError, OSError, ValueError):
        return False
    recorded_profile = provenance.get("benchmark_profile")
    if not isinstance(recorded_profile, Mapping):
        return False
    expected_profile = _benchmark_profile_payload(profile)
    normalized_profile = dict(recorded_profile)
    for key in ("compose_files", "plugin_components"):
        values = normalized_profile.get(key)
        if isinstance(values, tuple):
            normalized_profile[key] = list(values)
    if (
        normalized_profile != expected_profile
        or provenance.get("benchmark_profile_sha256") != profile_hash
    ):
        return False
    expected_environment = {
        "status": "verified_artifact_present",
        "path": environment_relative,
        "sha256": environment_hash,
    }
    if provenance.get("benchmark_environment") != expected_environment:
        return False
    revision = provenance.get("code_revision")
    if not isinstance(revision, Mapping) or set(revision) != {
        "git_commit",
        "source_tree_sha256",
        "source_dirty",
    }:
        return False
    commit = revision.get("git_commit")
    source_hash = revision.get("source_tree_sha256")
    if (
        not isinstance(commit, str)
        or _GIT_COMMIT.fullmatch(commit) is None
        or not isinstance(source_hash, str)
        or _SHA256.fullmatch(source_hash) is None
        or revision.get("source_dirty") is not False
        or source_tree_sha256_at_revision(root, commit) != source_hash
        or _revision_file_sha256(root, commit, profile_relative) != profile_hash
    ):
        return False
    if require_current_code_revision:
        current_revision = collect_code_revision(root)
        if (
            current_revision.source_dirty
            or current_revision.source_tree_sha256 != source_hash
        ):
            return False
    host = provenance.get("generator_host")
    if (
        not isinstance(host, Mapping)
        or set(host) != {"architecture", "system"}
        or not isinstance(host.get("architecture"), str)
        or not host.get("architecture")
        or not isinstance(host.get("system"), str)
        or not host.get("system")
    ):
        return False
    try:
        with environment_path.open(encoding="utf-8") as handle:
            environment = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(environment, Mapping) and verify_live_benchmark_environment(
        environment,
        root=root,
        profile=profile,
    )


def _revision_file_sha256(root: Path, revision: str, relative_path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _is_source_status_line(line: str) -> bool:
    raw_path = line[3:].strip().strip('"')
    if " -> " in raw_path:
        raw_path = raw_path.split(" -> ", 1)[1]
    path = Path(raw_path)
    return path.name in _SOURCE_FILES or (bool(path.parts) and path.parts[0] in _SOURCE_DIRECTORIES)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_architecture(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "aarch64": "arm64",
        "arm64": "arm64",
        "amd64": "x86_64",
        "x86_64": "x86_64",
    }
    return aliases.get(normalized, normalized)


def _benchmark_profile_payload(profile: BenchmarkProfile) -> dict[str, Any]:
    payload = asdict(profile)
    payload["compose_files"] = list(profile.compose_files)
    payload["plugin_components"] = list(profile.plugin_components)
    return payload


def _is_local_opensearch_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "http"
        and parsed.hostname == "127.0.0.1"
        and port is not None
        and parsed.username is None
        and parsed.password is None
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
    )
