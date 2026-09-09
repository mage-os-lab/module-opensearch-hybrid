from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pytest

import poc.model_packages as model_packages
from poc.datasets import DatasetIntegrityError
from poc.model_packages import (
    PinnedZipMember,
    PinnedZipPackage,
    fetch_and_extract_pinned_zip,
)


def test_fetch_extracts_only_registered_members_and_reuses_valid_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_bytes = _package_bytes(
        {
            "model.pt": b"registered model",
            "tokenizer.json": b'{"registered":true}',
            "../outside.txt": b"must not escape",
            "unused.txt": b"must not be extracted",
        }
    )
    requests: list[str] = []

    def open_url(request: urllib.request.Request) -> io.BytesIO:
        requests.append(request.full_url)
        return io.BytesIO(package_bytes)

    monkeypatch.setattr(model_packages, "_open_url", open_url)
    package_path = tmp_path / "cache" / "model.zip"
    model_path = tmp_path / "models" / "model.pt"
    tokenizer_path = tmp_path / "models" / "tokenizer.json"
    package = _package_spec(package_bytes)
    members = (
        _member("model.pt", model_path, b"registered model"),
        _member("tokenizer.json", tokenizer_path, b'{"registered":true}'),
    )

    assert fetch_and_extract_pinned_zip(
        package,
        package_path=package_path,
        members=members,
    ) == {
        "package": "downloaded",
        "model.pt": "extracted",
        "tokenizer.json": "extracted",
    }
    assert model_path.read_bytes() == b"registered model"
    assert tokenizer_path.read_bytes() == b'{"registered":true}'
    assert not (tmp_path / "outside.txt").exists()
    assert not (tmp_path / "models" / "unused.txt").exists()

    assert fetch_and_extract_pinned_zip(
        package,
        package_path=package_path,
        members=members,
    ) == {
        "package": "cached",
        "model.pt": "cached",
        "tokenizer.json": "cached",
    }
    assert requests == [package.url]


def test_fetch_refuses_invalid_existing_files_until_repair_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_bytes = _package_bytes({"model.pt": b"registered model"})
    monkeypatch.setattr(
        model_packages,
        "_open_url",
        lambda request: io.BytesIO(package_bytes),
    )
    package = _package_spec(package_bytes)
    package_path = tmp_path / "model.zip"
    destination = tmp_path / "model.pt"
    member = _member("model.pt", destination, b"registered model")

    package_path.write_bytes(b"corrupt package")
    with pytest.raises(DatasetIntegrityError, match="rerun with --repair"):
        fetch_and_extract_pinned_zip(
            package,
            package_path=package_path,
            members=(member,),
        )
    assert package_path.read_bytes() == b"corrupt package"

    assert fetch_and_extract_pinned_zip(
        package,
        package_path=package_path,
        members=(member,),
        repair=True,
    ) == {"package": "repaired", "model.pt": "extracted"}

    destination.write_bytes(b"corrupt extracted model")
    with pytest.raises(DatasetIntegrityError, match="rerun with --repair"):
        fetch_and_extract_pinned_zip(
            package,
            package_path=package_path,
            members=(member,),
        )
    assert destination.read_bytes() == b"corrupt extracted model"

    assert fetch_and_extract_pinned_zip(
        package,
        package_path=package_path,
        members=(member,),
        repair=True,
    ) == {"package": "cached", "model.pt": "repaired"}
    assert destination.read_bytes() == b"registered model"


def test_fetch_validates_members_before_writing_any_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_bytes = _package_bytes({"model.pt": b"registered model"})
    monkeypatch.setattr(
        model_packages,
        "_open_url",
        lambda request: io.BytesIO(package_bytes),
    )
    missing_path = tmp_path / "model.pt"

    with pytest.raises(DatasetIntegrityError, match="registered ZIP member"):
        fetch_and_extract_pinned_zip(
            _package_spec(package_bytes),
            package_path=tmp_path / "model.zip",
            members=(
                _member("model.pt", missing_path, b"registered model"),
                PinnedZipMember(
                    name="tokenizer.json",
                    destination=tmp_path / "tokenizer.json",
                ),
            ),
        )

    assert not missing_path.exists()
    assert not (tmp_path / "tokenizer.json").exists()


def test_fetch_rejects_an_archive_member_that_differs_from_its_registered_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_bytes = _package_bytes({"model.pt": b"unexpected model"})
    monkeypatch.setattr(
        model_packages,
        "_open_url",
        lambda request: io.BytesIO(package_bytes),
    )
    destination = tmp_path / "model.pt"

    with pytest.raises(DatasetIntegrityError, match="does not match pinned bytes"):
        fetch_and_extract_pinned_zip(
            _package_spec(package_bytes),
            package_path=tmp_path / "model.zip",
            members=(_member("model.pt", destination, b"registered model"),),
        )

    assert not destination.exists()


@pytest.mark.parametrize("member_name", ["../model.pt", "nested/model.pt", "..", ""])
def test_fetch_rejects_non_basename_member_registrations(
    tmp_path: Path,
    member_name: str,
) -> None:
    package_bytes = _package_bytes({"model.pt": b"registered model"})

    with pytest.raises(ValueError, match="basename"):
        fetch_and_extract_pinned_zip(
            _package_spec(package_bytes),
            package_path=tmp_path / "model.zip",
            members=(
                PinnedZipMember(
                    name=member_name,
                    destination=tmp_path / "model.pt",
                ),
            ),
        )


def _package_bytes(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


def _package_spec(content: bytes) -> PinnedZipPackage:
    return PinnedZipPackage(
        url="https://artifacts.opensearch.org/models/fixture.zip",
        sha256=hashlib.sha256(content).hexdigest(),
        bytes=len(content),
    )


def _member(name: str, destination: Path, content: bytes) -> PinnedZipMember:
    return PinnedZipMember(
        name=name,
        destination=destination,
        sha256=hashlib.sha256(content).hexdigest(),
        bytes=len(content),
    )
