from __future__ import annotations

import hashlib
import re
import tomllib
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from poc.config import ConfigError
from poc.manifest import write_json

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


class DatasetIntegrityError(RuntimeError):
    """Raised when downloaded or prepared dataset bytes violate their contract."""


@dataclass(frozen=True, slots=True)
class DatasetFileSpec:
    name: str
    filename: str
    url: str
    sha256: str
    bytes: int

    @classmethod
    def from_mapping(cls, name: str, values: dict[str, Any]) -> DatasetFileSpec:
        filename = _required_string(name, "filename", values.get("filename"))
        if Path(filename).name != filename:
            raise ConfigError(f"dataset file {name!r} filename must not contain directories")
        url = _required_string(name, "url", values.get("url"))
        if not url.startswith("https://"):
            raise ConfigError(f"dataset file {name!r} must use an HTTPS URL")
        sha256 = _required_string(name, "sha256", values.get("sha256"))
        if _SHA256.fullmatch(sha256) is None:
            raise ConfigError(f"dataset file {name!r} has an invalid SHA-256")
        byte_count = values.get("bytes")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count <= 0:
            raise ConfigError(f"dataset file {name!r} bytes must be a positive integer")
        return cls(name=name, filename=filename, url=url, sha256=sha256, bytes=byte_count)


@dataclass(frozen=True, slots=True)
class WandsDatasetSpec:
    source: str
    revision: str
    expected_products: int
    expected_queries: int
    expected_judgments: int
    expected_unique_pairs: int
    expected_duplicate_pairs: int
    expected_conflicting_pairs: int
    license: str
    primary_gain_mapping: str
    duplicate_policy: str
    split_seed: str
    files: dict[str, DatasetFileSpec]


@dataclass(frozen=True, slots=True)
class TrecProductSearchDatasetSpec:
    source: str
    revision: str
    expected_products: int
    expected_queries: int
    expected_judged_queries: int
    expected_judgments: int
    expected_unique_judged_products: int
    license: str
    primary_gain_mapping: str
    query_authority: str
    files: dict[str, DatasetFileSpec]


@dataclass(frozen=True, slots=True)
class FileFacts:
    sha256: str
    bytes: int


def load_wands_config(path: Path | str = Path("config/datasets.toml")) -> WandsDatasetSpec:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != 1:
        raise ConfigError(f"unsupported dataset registry schema in {config_path}")
    datasets = raw.get("datasets")
    if not isinstance(datasets, dict) or not isinstance(datasets.get("wands"), dict):
        raise ConfigError(f"dataset registry {config_path} contains no WANDS entry")
    values = datasets["wands"]

    revision = _required_string("wands", "revision", values.get("revision"))
    if _REVISION.fullmatch(revision) is None:
        raise ConfigError("WANDS revision must be an immutable 40-character commit SHA")
    raw_files = values.get("files")
    if not isinstance(raw_files, dict) or set(raw_files) != {"product", "query", "label"}:
        raise ConfigError("WANDS must configure product, query, and label files")
    files = {
        name: DatasetFileSpec.from_mapping(name, file_values)
        for name, file_values in raw_files.items()
        if isinstance(name, str) and isinstance(file_values, dict)
    }
    if len(files) != 3:
        raise ConfigError("WANDS file entries must be tables")
    if any(revision not in file.url for file in files.values()):
        raise ConfigError("every WANDS file URL must contain the pinned revision")

    return WandsDatasetSpec(
        source=_required_string("wands", "source", values.get("source")),
        revision=revision,
        expected_products=_required_positive_int(values, "expected_products"),
        expected_queries=_required_positive_int(values, "expected_queries"),
        expected_judgments=_required_positive_int(values, "expected_judgments"),
        expected_unique_pairs=_required_positive_int(values, "expected_unique_pairs"),
        expected_duplicate_pairs=_required_positive_int(values, "expected_duplicate_pairs"),
        expected_conflicting_pairs=_required_positive_int(values, "expected_conflicting_pairs"),
        license=_required_string("wands", "license", values.get("license")),
        primary_gain_mapping=_required_string(
            "wands", "primary_gain_mapping", values.get("primary_gain_mapping")
        ),
        duplicate_policy=_required_string(
            "wands", "duplicate_policy", values.get("duplicate_policy")
        ),
        split_seed=_required_string("wands", "split_seed", values.get("split_seed")),
        files=files,
    )


def load_trec_product_search_config(
    path: Path | str = Path("config/datasets.toml"),
) -> TrecProductSearchDatasetSpec:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != 1:
        raise ConfigError(f"unsupported dataset registry schema in {config_path}")
    datasets = raw.get("datasets")
    if not isinstance(datasets, dict) or not isinstance(
        datasets.get("trec_product_search_2024"), dict
    ):
        raise ConfigError(
            f"dataset registry {config_path} contains no TREC Product Search 2024 entry"
        )
    values = datasets["trec_product_search_2024"]
    revision = _required_string(
        "trec_product_search_2024", "revision", values.get("revision")
    )
    if _REVISION.fullmatch(revision) is None:
        raise ConfigError("TREC Product Search revision must be an immutable commit SHA")
    raw_files = values.get("files")
    if not isinstance(raw_files, dict) or set(raw_files) != {
        "corpus",
        "queries",
        "qrels",
    }:
        raise ConfigError("TREC Product Search must configure corpus, queries, and qrels")
    files = {
        name: DatasetFileSpec.from_mapping(name, file_values)
        for name, file_values in raw_files.items()
        if isinstance(name, str) and isinstance(file_values, dict)
    }
    if len(files) != 3:
        raise ConfigError("TREC Product Search file entries must be tables")
    if revision not in files["corpus"].url:
        raise ConfigError("TREC Product Search corpus URL must contain the pinned revision")
    section = "trec_product_search_2024"
    return TrecProductSearchDatasetSpec(
        source=_required_string(section, "source", values.get("source")),
        revision=revision,
        expected_products=_required_positive_int(
            values, "expected_products", section=section
        ),
        expected_queries=_required_positive_int(
            values, "expected_queries", section=section
        ),
        expected_judged_queries=_required_positive_int(
            values, "expected_judged_queries", section=section
        ),
        expected_judgments=_required_positive_int(
            values, "expected_judgments", section=section
        ),
        expected_unique_judged_products=_required_positive_int(
            values, "expected_unique_judged_products", section=section
        ),
        license=_required_string(section, "license", values.get("license")),
        primary_gain_mapping=_required_string(
            section, "primary_gain_mapping", values.get("primary_gain_mapping")
        ),
        query_authority=_required_string(
            section, "query_authority", values.get("query_authority")
        ),
        files=files,
    )


def file_facts(path: Path) -> FileFacts:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    return FileFacts(sha256=digest.hexdigest(), bytes=byte_count)


def verify_dataset_file(path: Path, spec: DatasetFileSpec) -> FileFacts:
    if not path.is_file():
        raise DatasetIntegrityError(f"missing dataset file: {path}")
    facts = file_facts(path)
    if facts.sha256 != spec.sha256 or facts.bytes != spec.bytes:
        raise DatasetIntegrityError(
            f"dataset file {path} does not match pinned bytes: "
            f"sha256={facts.sha256}, bytes={facts.bytes}"
        )
    return facts


def download_verified_file(destination: Path, spec: DatasetFileSpec, *, repair: bool) -> str:
    if destination.exists():
        try:
            verify_dataset_file(destination, spec)
            return "cached"
        except DatasetIntegrityError:
            if not repair:
                raise DatasetIntegrityError(
                    f"refusing to replace invalid existing file {destination}; rerun with --repair"
                ) from None

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    request = urllib.request.Request(
        spec.url,
        headers={"User-Agent": "opensearch-hybrid-benchmark/0.1"},
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        verify_dataset_file(temporary, spec)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return "downloaded"


def fetch_wands(
    spec: WandsDatasetSpec, destination: Path, *, repair: bool = False
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for name in ("product", "query", "label"):
        file_spec = spec.files[name]
        statuses[name] = download_verified_file(
            destination / file_spec.filename,
            file_spec,
            repair=repair,
        )
    facts = verify_wands_raw(spec, destination)
    write_json(
        destination / "source-manifest.json",
        {
            "schema_version": 1,
            "dataset": "WANDS",
            "source": spec.source,
            "revision": spec.revision,
            "license": spec.license,
            "files": {name: asdict(file_facts_) for name, file_facts_ in facts.items()},
        },
    )
    return statuses


def verify_wands_raw(spec: WandsDatasetSpec, source: Path) -> dict[str, FileFacts]:
    return {
        name: verify_dataset_file(source / file_spec.filename, file_spec)
        for name, file_spec in spec.files.items()
    }


def fetch_trec_product_search(
    spec: TrecProductSearchDatasetSpec,
    destination: Path,
    *,
    repair: bool = False,
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for name in ("corpus", "queries", "qrels"):
        file_spec = spec.files[name]
        statuses[name] = download_verified_file(
            destination / file_spec.filename,
            file_spec,
            repair=repair,
        )
    facts = verify_trec_product_search_raw(spec, destination)
    write_json(
        destination / "source-manifest.json",
        {
            "schema_version": 1,
            "dataset": "TREC Product Search 2024",
            "source": spec.source,
            "revision": spec.revision,
            "license": spec.license,
            "query_authority": spec.query_authority,
            "files": {name: asdict(file_facts_) for name, file_facts_ in facts.items()},
        },
    )
    return statuses


def verify_trec_product_search_raw(
    spec: TrecProductSearchDatasetSpec,
    source: Path,
) -> dict[str, FileFacts]:
    return {
        name: verify_dataset_file(source / file_spec.filename, file_spec)
        for name, file_spec in spec.files.items()
    }


def _required_string(section: str, key: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{section} {key} must be a non-empty string")
    return value


def _required_positive_int(
    values: dict[str, Any], key: str, *, section: str = "wands"
) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{section} {key} must be a positive integer")
    return value
