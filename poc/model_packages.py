from __future__ import annotations

import hashlib
import os
import re
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from poc.datasets import DatasetIntegrityError, FileFacts, file_facts

_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class PinnedZipPackage:
    url: str
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class PinnedZipMember:
    name: str
    destination: Path
    sha256: str | None = None
    bytes: int | None = None


def fetch_and_extract_pinned_zip(
    package: PinnedZipPackage,
    *,
    package_path: Path,
    members: tuple[PinnedZipMember, ...],
    repair: bool = False,
) -> dict[str, str]:
    """Fetch a pinned ZIP and atomically materialize only registered members."""
    _validate_registration(package, package_path=package_path, members=members)
    package_status = _fetch_package(
        package,
        destination=package_path,
        repair=repair,
    )
    archive_facts, archive_members = _inspect_registered_members(
        package_path,
        members,
    )

    statuses: dict[str, str] = {"package": package_status}
    pending: list[PinnedZipMember] = []
    for member in members:
        expected = archive_facts[member.name]
        destination = member.destination
        if destination.is_symlink():
            raise DatasetIntegrityError(
                f"refusing symbolic-link model destination {destination}"
            )
        if destination.exists():
            if not destination.is_file():
                raise DatasetIntegrityError(
                    f"model destination is not a regular file: {destination}"
                )
            if file_facts(destination) == expected:
                statuses[member.name] = "cached"
                continue
            if not repair:
                raise DatasetIntegrityError(
                    f"refusing to replace invalid existing file {destination}; "
                    "rerun with --repair"
                )
            statuses[member.name] = "repaired"
        else:
            statuses[member.name] = "extracted"
        pending.append(member)

    _extract_registered_members(
        package_path,
        pending,
        archive_members=archive_members,
        archive_facts=archive_facts,
    )
    return statuses


def _validate_registration(
    package: PinnedZipPackage,
    *,
    package_path: Path,
    members: tuple[PinnedZipMember, ...],
) -> None:
    if not package.url.startswith("https://artifacts.opensearch.org/"):
        raise ValueError("model package must use the official OpenSearch HTTPS origin")
    if _SHA256.fullmatch(package.sha256) is None:
        raise ValueError("model package SHA-256 must be 64 lowercase hexadecimal characters")
    if (
        not isinstance(package.bytes, int)
        or isinstance(package.bytes, bool)
        or package.bytes <= 0
    ):
        raise ValueError("model package bytes must be a positive integer")
    if not members:
        raise ValueError("at least one model package member must be registered")
    if package_path.is_symlink():
        raise DatasetIntegrityError(f"refusing symbolic-link package path {package_path}")

    names: set[str] = set()
    destinations: set[Path] = set()
    for member in members:
        if (
            not member.name
            or member.name in {".", ".."}
            or "/" in member.name
            or "\\" in member.name
            or Path(member.name).name != member.name
        ):
            raise ValueError("registered ZIP member must be an exact basename")
        if member.destination.name != member.name:
            raise ValueError("registered ZIP member destination must preserve its basename")
        if member.name in names:
            raise ValueError(f"duplicate registered ZIP member {member.name}")
        if member.destination in destinations:
            raise ValueError(f"duplicate model member destination {member.destination}")
        if member.destination == package_path:
            raise ValueError("model member destination must differ from the package path")
        names.add(member.name)
        destinations.add(member.destination)

        has_sha256 = member.sha256 is not None
        has_bytes = member.bytes is not None
        if has_sha256 != has_bytes:
            raise ValueError("member SHA-256 and bytes must be registered together")
        if has_sha256 and (
            not isinstance(member.sha256, str)
            or _SHA256.fullmatch(member.sha256) is None
            or not isinstance(member.bytes, int)
            or isinstance(member.bytes, bool)
            or member.bytes <= 0
        ):
            raise ValueError("registered model member identity is invalid")


def _fetch_package(
    package: PinnedZipPackage,
    *,
    destination: Path,
    repair: bool,
) -> str:
    existing = destination.exists()
    if existing:
        if not destination.is_file():
            raise DatasetIntegrityError(
                f"model package path is not a regular file: {destination}"
            )
        if file_facts(destination) == FileFacts(
            sha256=package.sha256,
            bytes=package.bytes,
        ):
            return "cached"
        if not repair:
            raise DatasetIntegrityError(
                f"refusing to replace invalid existing file {destination}; "
                "rerun with --repair"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    request = urllib.request.Request(
        package.url,
        headers={"User-Agent": "opensearch-hybrid-benchmark/0.1"},
    )
    try:
        with _open_url(request) as response, temporary.open("wb") as output:
            _copy_stream(response, output)
            output.flush()
            os.fsync(output.fileno())
        facts = file_facts(temporary)
        if facts != FileFacts(sha256=package.sha256, bytes=package.bytes):
            raise DatasetIntegrityError(
                "downloaded model package does not match registered bytes: "
                f"sha256={facts.sha256}, bytes={facts.bytes}"
            )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return "repaired" if existing else "downloaded"


def _inspect_registered_members(
    package_path: Path,
    members: tuple[PinnedZipMember, ...],
) -> tuple[dict[str, FileFacts], dict[str, zipfile.ZipInfo]]:
    facts: dict[str, FileFacts] = {}
    archive_members: dict[str, zipfile.ZipInfo] = {}
    try:
        with zipfile.ZipFile(package_path) as archive:
            for member in members:
                matches = [
                    info for info in archive.infolist() if info.filename == member.name
                ]
                if len(matches) != 1 or matches[0].is_dir():
                    raise DatasetIntegrityError(
                        f"pinned model package must contain one registered ZIP member "
                        f"named {member.name}"
                    )
                info = matches[0]
                with archive.open(info, "r") as source:
                    current = _stream_facts(source)
                if current.bytes != info.file_size:
                    raise DatasetIntegrityError(
                        f"registered ZIP member {member.name} has inconsistent size"
                    )
                if member.sha256 is not None:
                    if member.bytes is None:
                        raise ValueError("registered model member bytes are missing")
                    if current != FileFacts(
                        sha256=member.sha256,
                        bytes=member.bytes,
                    ):
                        raise DatasetIntegrityError(
                            f"registered ZIP member {member.name} does not match pinned bytes"
                        )
                facts[member.name] = current
                archive_members[member.name] = info
    except DatasetIntegrityError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise DatasetIntegrityError(
            f"pinned model package is not a readable ZIP archive: {package_path}"
        ) from error
    return facts, archive_members


def _extract_registered_members(
    package_path: Path,
    members: list[PinnedZipMember],
    *,
    archive_members: dict[str, zipfile.ZipInfo],
    archive_facts: dict[str, FileFacts],
) -> None:
    if not members:
        return
    prepared: list[tuple[Path, Path]] = []
    try:
        with zipfile.ZipFile(package_path) as archive:
            for member in members:
                member.destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = _temporary_path(member.destination)
                prepared.append((temporary, member.destination))
                with (
                    archive.open(archive_members[member.name], "r") as source,
                    temporary.open("wb") as output,
                ):
                    _copy_stream(source, output)
                    output.flush()
                    os.fsync(output.fileno())
                if file_facts(temporary) != archive_facts[member.name]:
                    raise DatasetIntegrityError(
                        f"extracted ZIP member {member.name} differs from pinned package"
                    )
        for temporary, destination in prepared:
            temporary.replace(destination)
    except DatasetIntegrityError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise DatasetIntegrityError(
            f"failed to extract registered members from {package_path}"
        ) from error
    finally:
        for temporary, _ in prepared:
            temporary.unlink(missing_ok=True)


def _temporary_path(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".part",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def _stream_facts(source: IO[bytes]) -> FileFacts:
    digest = hashlib.sha256()
    byte_count = 0
    while chunk := source.read(_CHUNK_BYTES):
        digest.update(chunk)
        byte_count += len(chunk)
    return FileFacts(sha256=digest.hexdigest(), bytes=byte_count)


def _copy_stream(source: IO[bytes], destination: IO[bytes]) -> None:
    while chunk := source.read(_CHUNK_BYTES):
        destination.write(chunk)


def _open_url(request: urllib.request.Request) -> Any:
    return urllib.request.urlopen(request, timeout=120)
