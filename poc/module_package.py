from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

PACKAGE_NAME = "mage-os/module-opensearch-hybrid"
PACKAGE_TYPE = "magento2-module"
MANIFEST_SCHEMA_VERSION = 1
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

_EXCLUDED_DIRECTORIES = frozenset({".git", ".phpunit.cache", "__pycache__", "vendor"})
_EXCLUDED_FILENAMES = frozenset({".DS_Store"})
# The canonical repository includes the encoder and Python tooling beside the module.
_EXCLUDED_ROOTS = frozenset(
    {
        ".github",
        ".venv",
        ".uv-cache",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "artifacts",
        "config",
        "data",
        "dist",
        "poc",
        "results",
        "runs",
        "scripts",
        "services",
        "tests",
        ".worktrees",
    }
)
_EXCLUDED_ROOT_FILES = frozenset(
    {
        ".env",
        ".gitignore",
        ".python-version",
        "Makefile",
        "pyproject.toml",
        "uv.lock",
        "composer.lock",
        "docker-compose.yml",
        "compose.bench.yml",
        "compose.opensearch-2.19.yml",
    }
)
_EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo"})
_NON_RUNTIME_ROOTS = frozenset({"Test", "dev", "docs"})
_NON_RUNTIME_FILES = frozenset({"README.md", "phpunit.xml.dist"})
_RUNTIME_TEXT_SUFFIXES = frozenset({".css", ".html", ".js", ".json", ".php", ".phtml", ".xml"})
_FORBIDDEN_RUNTIME_REFERENCES = (
    "../",
    "/rocket-search/",  # Reject dependencies on the original harness checkout too.
    "/opensearch-hybrid/",
    "/module-opensearch-hybrid/",
    "poc/",
    "scripts/",
    "results/",
    "runs/",
)


@dataclass(frozen=True)
class PayloadFile:
    path: str
    size: int
    sha256: str
    mode: str

    def as_manifest(self) -> dict[str, str | int]:
        return {
            "path": self.path,
            "bytes": self.size,
            "sha256": self.sha256,
            "mode": self.mode,
        }


def build_module_package(
    module_root: Path,
    archive_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    root = _validated_root(module_root)
    _validate_output_path(root, archive_path, "archive")
    _validate_output_path(root, manifest_path, "manifest")
    if archive_path.resolve(strict=False) == manifest_path.resolve(strict=False):
        raise ValueError("archive and manifest paths must be different")
    if archive_path.exists():
        raise FileExistsError(f"archive already exists: {archive_path}")
    if manifest_path.exists():
        raise FileExistsError(f"manifest already exists: {manifest_path}")

    payload = collect_payload(root)
    validate_module_boundary(root, payload)
    manifest_files = [item.as_manifest() for item in payload]
    payload_bytes = sum(item.size for item in payload)
    payload_sha256 = _canonical_sha256(manifest_files)

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    archive_temp = _temporary_path(archive_path)
    manifest_temp = _temporary_path(manifest_path)
    try:
        _write_archive(root, payload, archive_temp)
        archive_bytes = archive_temp.read_bytes()
        manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "package": {
                "name": PACKAGE_NAME,
                "type": PACKAGE_TYPE,
            },
            "payload": {
                "root": ".",
                "file_count": len(payload),
                "bytes": payload_bytes,
                "sha256": payload_sha256,
                "files": manifest_files,
            },
            "archive": {
                "format": "zip",
                "compression": "stored",
                "bytes": len(archive_bytes),
                "sha256": hashlib.sha256(archive_bytes).hexdigest(),
            },
        }
        manifest_temp.write_bytes(_canonical_json(manifest) + b"\n")
        verify_archive(archive_temp, manifest)
        os.replace(archive_temp, archive_path)
        os.replace(manifest_temp, manifest_path)
    finally:
        archive_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)
    return manifest


def verify_module_payload(module_root: Path, manifest: dict[str, Any]) -> None:
    root = _validated_root(module_root)
    _validate_manifest_identity(manifest)
    payload = collect_payload(root)
    validate_module_boundary(root, payload)
    actual_files = [item.as_manifest() for item in payload]
    expected_payload = _manifest_payload(manifest)
    if actual_files != expected_payload.get("files"):
        raise ValueError("module payload files do not match the manifest")
    if len(payload) != expected_payload.get("file_count"):
        raise ValueError("module payload file count does not match the manifest")
    if sum(item.size for item in payload) != expected_payload.get("bytes"):
        raise ValueError("module payload byte count does not match the manifest")
    if _canonical_sha256(actual_files) != expected_payload.get("sha256"):
        raise ValueError("module payload digest does not match the manifest")


def verify_archive(archive_path: Path, manifest: dict[str, Any]) -> None:
    _validate_manifest_identity(manifest)
    expected_payload = _manifest_payload(manifest)
    expected_files = cast(list[dict[str, Any]], expected_payload.get("files"))
    expected_by_path = {str(item["path"]): item for item in expected_files}
    archive_bytes = archive_path.read_bytes()
    archive_manifest = manifest.get("archive")
    if not isinstance(archive_manifest, dict):
        raise ValueError("manifest archive must be an object")
    if archive_manifest.get("format") != "zip" or archive_manifest.get("compression") != "stored":
        raise ValueError("manifest archive contract is unsupported")
    if len(archive_bytes) != archive_manifest.get("bytes"):
        raise ValueError("archive byte count does not match the manifest")
    if hashlib.sha256(archive_bytes).hexdigest() != archive_manifest.get("sha256"):
        raise ValueError("archive digest does not match the manifest")

    with zipfile.ZipFile(archive_path, "r") as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)):
            raise ValueError("archive contains duplicate paths")
        if names != sorted(expected_by_path):
            raise ValueError("archive paths do not match the payload manifest")
        for member in members:
            _validate_archive_member(member)
            expected = expected_by_path[member.filename]
            content = archive.read(member)
            if len(content) != expected["bytes"]:
                raise ValueError(f"archive byte count mismatch for {member.filename}")
            if hashlib.sha256(content).hexdigest() != expected["sha256"]:
                raise ValueError(f"archive digest mismatch for {member.filename}")
            mode = stat.S_IMODE(member.external_attr >> 16)
            if f"{mode:04o}" != expected["mode"]:
                raise ValueError(f"archive mode mismatch for {member.filename}")


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("package manifest must be a JSON object")
    return cast(dict[str, Any], value)


def collect_payload(module_root: Path) -> tuple[PayloadFile, ...]:
    root = _validated_root(module_root)
    payload: list[PayloadFile] = []
    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        retained_directories: list[str] = []
        for name in sorted(directory_names):
            candidate = current_path / name
            relative = candidate.relative_to(root)
            _reject_symlink(candidate, relative)
            if not _is_excluded(relative):
                retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in sorted(file_names):
            candidate = current_path / name
            relative = candidate.relative_to(root)
            _reject_symlink(candidate, relative)
            if _is_excluded(relative):
                continue
            metadata = candidate.stat(follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"package payload is not a regular file: {relative.as_posix()}")
            content = candidate.read_bytes()
            mode = 0o755 if metadata.st_mode & stat.S_IXUSR else 0o644
            payload.append(
                PayloadFile(
                    path=relative.as_posix(),
                    size=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                    mode=f"{mode:04o}",
                )
            )
    return tuple(sorted(payload, key=lambda item: item.path))


def validate_module_boundary(module_root: Path, payload: tuple[PayloadFile, ...]) -> None:
    by_path = {item.path: item for item in payload}
    composer_entry = by_path.get("composer.json")
    if composer_entry is None:
        raise ValueError("module payload is missing composer.json")
    composer = json.loads((module_root / composer_entry.path).read_text(encoding="utf-8"))
    if not isinstance(composer, dict):
        raise ValueError("composer.json must be a JSON object")
    if composer.get("name") != PACKAGE_NAME:
        raise ValueError(f"composer package name must be {PACKAGE_NAME}")
    if composer.get("type") != PACKAGE_TYPE:
        raise ValueError(f"composer package type must be {PACKAGE_TYPE}")

    for item in payload:
        relative = PurePosixPath(item.path)
        if relative.parts[0] in _NON_RUNTIME_ROOTS or item.path in _NON_RUNTIME_FILES:
            continue
        if relative.suffix.lower() not in _RUNTIME_TEXT_SUFFIXES:
            continue
        content = (module_root / item.path).read_text(encoding="utf-8")
        for forbidden in _FORBIDDEN_RUNTIME_REFERENCES:
            if forbidden in content:
                raise ValueError(
                    f"runtime file references the parent evidence repository: "
                    f"{item.path} ({forbidden})"
                )


def _write_archive(
    module_root: Path,
    payload: tuple[PayloadFile, ...],
    destination: Path,
) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for item in payload:
            info = zipfile.ZipInfo(item.path, date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            mode = int(item.mode, 8)
            info.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(info, (module_root / item.path).read_bytes())


def _validate_archive_member(member: zipfile.ZipInfo) -> None:
    path = PurePosixPath(member.filename)
    if member.is_dir() or path.is_absolute() or ".." in path.parts or "\\" in member.filename:
        raise ValueError(f"unsafe archive path: {member.filename}")
    if member.date_time != ZIP_TIMESTAMP:
        raise ValueError(f"archive timestamp is not deterministic: {member.filename}")
    if member.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"archive member is not stored: {member.filename}")
    if not stat.S_ISREG(member.external_attr >> 16):
        raise ValueError(f"archive member is not a regular file: {member.filename}")


def _validated_root(module_root: Path) -> Path:
    if module_root.is_symlink():
        raise ValueError("module root must not be a symbolic link")
    root = module_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("module root must be a directory")
    return root


def _validate_output_path(module_root: Path, output: Path, label: str) -> None:
    resolved = output.resolve(strict=False)
    if resolved == module_root or (
        resolved.is_relative_to(module_root) and not resolved.is_relative_to(module_root / "dist")
    ):
        raise ValueError(f"{label} path must be outside the module root or inside dist/")


def _temporary_path(destination: Path) -> Path:
    descriptor, value = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(value)


def _reject_symlink(candidate: Path, relative: Path) -> None:
    if candidate.is_symlink():
        raise ValueError(f"package payload contains a symbolic link: {relative.as_posix()}")


def _is_excluded(relative: Path) -> bool:
    return (
        relative.parts[0] in _EXCLUDED_ROOTS
        or (len(relative.parts) == 1 and relative.name in _EXCLUDED_ROOT_FILES)
        or any(part in _EXCLUDED_DIRECTORIES for part in relative.parts)
        or relative.name in _EXCLUDED_FILENAMES
        or relative.suffix in _EXCLUDED_SUFFIXES
    )


def _validate_manifest_identity(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported package manifest schema")
    package = manifest.get("package")
    if not isinstance(package, dict):
        raise ValueError("manifest package must be an object")
    if package.get("name") != PACKAGE_NAME or package.get("type") != PACKAGE_TYPE:
        raise ValueError("manifest package identity is invalid")


def _manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = manifest.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise ValueError("manifest payload is invalid")
    return cast(dict[str, Any], payload)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()
