from __future__ import annotations

import gzip
import json
import zipfile
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, TypedDict, cast

import pytest

from poc.datasets import (
    DatasetFileSpec,
    DatasetIntegrityError,
    TrecProductSearchDatasetSpec,
    file_facts,
)
from poc.trec_sparse import TrecSparseSpec
from poc.trec_sparse_indexing import build_trec_sparse_index, verify_trec_sparse_index
from poc.trec_sparse_validation import (
    SampledSparseRecord,
    deterministic_sample_positions,
    scan_trec_sparse_stream,
    validate_recovered_trec_sparse,
    verify_trec_sparse_validation_manifest,
)


class ValidationFixture(TypedDict):
    root: Path
    dataset_spec: TrecProductSearchDatasetSpec
    sparse_spec: TrecSparseSpec
    corpus_path: Path
    embeddings_path: Path
    completion_manifest_path: Path
    package_path: Path
    model_path: Path
    tokenizer_path: Path
    validation_manifest_path: Path


@pytest.fixture(autouse=True)
def _validation_runtime_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "poc.trec_sparse_validation.verify_decision_provenance",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "poc.trec_sparse_validation._implementation_fingerprints",
        lambda root: {
            "poc/trec_sparse_validation.py": {
                "path": "poc/trec_sparse_validation.py",
                "sha256": "1" * 64,
                "bytes": 1,
            }
        },
    )


def test_validation_positions_are_deterministic_and_span_the_full_stream() -> None:
    assert deterministic_sample_positions(11, 5) == (0, 2, 5, 7, 10)
    assert deterministic_sample_positions(3, 3) == (0, 1, 2)

    with pytest.raises(ValueError, match="cannot exceed"):
        deterministic_sample_positions(2, 3)


def test_full_scan_binds_product_order_and_positive_finite_weights(
    tmp_path: Path,
) -> None:
    corpus, embeddings, _ = _write_sparse_fixture(tmp_path)

    scan = scan_trec_sparse_stream(
        corpus,
        embeddings,
        expected_records=3,
        sample_positions=(0, 2),
    )

    assert scan.records == 3
    assert scan.corpus_records == 3
    assert [sample.product_id for sample in scan.samples] == ["doc-1", "doc-3"]
    assert scan.embeddings_sha256 == file_facts(embeddings).sha256

    invalid = _canonical_line("doc-1", {"one": 1.0})
    invalid += _canonical_line("doc-2", {"bad": float("nan")})
    invalid += _canonical_line("doc-3", {"three": 3.0})
    embeddings.write_bytes(invalid)
    with pytest.raises(DatasetIntegrityError, match="finite positive"):
        scan_trec_sparse_stream(
            corpus,
            embeddings,
            expected_records=3,
            sample_positions=(0, 2),
        )


def test_recovered_validation_requires_exact_sample_reencoding_and_binds_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_validation_fixture(tmp_path)
    monkeypatch.setattr(
        "poc.trec_sparse_validation.collect_manifest_provenance",
        lambda *args, **kwargs: _clean_provenance(),
    )

    def reproduce(
        samples: Sequence[SampledSparseRecord],
        **kwargs: object,
    ) -> tuple[tuple[bytes, ...], dict[str, object]]:
        return tuple(sample.canonical_line for sample in samples), _sample_runtime()

    monkeypatch.setattr(
        "poc.trec_sparse_validation.reencode_trec_sparse_sample", reproduce
    )

    result = validate_recovered_trec_sparse(
        **fixture,
        sample_size=3,
    )

    assert result["all_checks_passed"] is True
    assert result["quality_evidence_eligible_for_decision"] is False
    assert result["quality_ineligibility_reason"] == (
        "legacy schema v1 generation provenance is unknown"
    )
    assert result["latency_evidence_eligible_for_decision"] is False
    assert result["generation_provenance_status"] == "unknown_pre_commit_process"
    assert result["completion_manifest_provenance_interpretation"] == (
        "completion_time_only_not_generator_attribution"
    )
    sample = cast(dict[str, Any], result["sample_reencoding"])
    completion = cast(dict[str, Any], result["completion_manifest"])
    assert sample["compared_records"] == 3
    assert completion["path"] == (
        "results/trec-product-search/neural-sparse/precompute-full-manifest.json"
    )
    verified = verify_trec_sparse_validation_manifest(
        root=fixture["root"],
        dataset_spec=fixture["dataset_spec"],
        sparse_spec=fixture["sparse_spec"],
        corpus_path=fixture["corpus_path"],
        embeddings_path=fixture["embeddings_path"],
        completion_manifest_path=fixture["completion_manifest_path"],
        package_path=fixture["package_path"],
        model_path=fixture["model_path"],
        tokenizer_path=fixture["tokenizer_path"],
        validation_manifest_path=fixture["validation_manifest_path"],
        expected_sample_size=3,
    )
    assert verified == result

    fixture["embeddings_path"].write_bytes(
        fixture["embeddings_path"].read_bytes() + b"\n"
    )
    with pytest.raises(DatasetIntegrityError, match="embeddings .*differs"):
        verify_trec_sparse_validation_manifest(
            root=fixture["root"],
            dataset_spec=fixture["dataset_spec"],
            sparse_spec=fixture["sparse_spec"],
            corpus_path=fixture["corpus_path"],
            embeddings_path=fixture["embeddings_path"],
            completion_manifest_path=fixture["completion_manifest_path"],
            package_path=fixture["package_path"],
            model_path=fixture["model_path"],
            tokenizer_path=fixture["tokenizer_path"],
            validation_manifest_path=fixture["validation_manifest_path"],
            expected_sample_size=3,
        )


def test_recovered_validation_rejects_a_sample_value_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_validation_fixture(tmp_path)
    monkeypatch.setattr(
        "poc.trec_sparse_validation.collect_manifest_provenance",
        lambda *args, **kwargs: _clean_provenance(),
    )

    def mismatch(
        samples: Sequence[SampledSparseRecord],
        **kwargs: object,
    ) -> tuple[tuple[bytes, ...], dict[str, object]]:
        lines = [sample.canonical_line for sample in samples]
        lines[1] = _canonical_line(samples[1].product_id, {"changed": 9.0})
        return tuple(lines), _sample_runtime()

    monkeypatch.setattr(
        "poc.trec_sparse_validation.reencode_trec_sparse_sample", mismatch
    )

    with pytest.raises(DatasetIntegrityError, match="sample re-encoding differs"):
        validate_recovered_trec_sparse(
            **fixture,
            sample_size=3,
        )

    assert not fixture["validation_manifest_path"].exists()


def test_validation_verifier_requires_completion_time_only_provenance_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_validation_fixture(tmp_path)
    monkeypatch.setattr(
        "poc.trec_sparse_validation.collect_manifest_provenance",
        lambda *args, **kwargs: _clean_provenance(),
    )

    def reproduce(
        samples: Sequence[SampledSparseRecord],
        **kwargs: object,
    ) -> tuple[tuple[bytes, ...], dict[str, object]]:
        return tuple(sample.canonical_line for sample in samples), _sample_runtime()

    monkeypatch.setattr(
        "poc.trec_sparse_validation.reencode_trec_sparse_sample", reproduce
    )
    result = validate_recovered_trec_sparse(**fixture, sample_size=3)
    result["generation_provenance_status"] = "claimed_generator_commit"
    fixture["validation_manifest_path"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(DatasetIntegrityError, match="provenance interpretation"):
        verify_trec_sparse_validation_manifest(
            root=fixture["root"],
            dataset_spec=fixture["dataset_spec"],
            sparse_spec=fixture["sparse_spec"],
            corpus_path=fixture["corpus_path"],
            embeddings_path=fixture["embeddings_path"],
            completion_manifest_path=fixture["completion_manifest_path"],
            package_path=fixture["package_path"],
            model_path=fixture["model_path"],
            tokenizer_path=fixture["tokenizer_path"],
            validation_manifest_path=fixture["validation_manifest_path"],
            expected_sample_size=3,
        )


def test_validation_verifier_rejects_stale_implementation_fingerprints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_validation_fixture(tmp_path)
    monkeypatch.setattr(
        "poc.trec_sparse_validation.collect_manifest_provenance",
        lambda *args, **kwargs: _clean_provenance(),
    )
    fingerprints = {
        "poc/trec_sparse_validation.py": {
            "path": "poc/trec_sparse_validation.py",
            "sha256": "1" * 64,
            "bytes": 1,
        }
    }
    monkeypatch.setattr(
        "poc.trec_sparse_validation._implementation_fingerprints",
        lambda root: fingerprints,
        raising=False,
    )

    def reproduce(
        samples: Sequence[SampledSparseRecord],
        **kwargs: object,
    ) -> tuple[tuple[bytes, ...], dict[str, object]]:
        return tuple(sample.canonical_line for sample in samples), _sample_runtime()

    monkeypatch.setattr(
        "poc.trec_sparse_validation.reencode_trec_sparse_sample", reproduce
    )
    validate_recovered_trec_sparse(**fixture, sample_size=3)
    fingerprints["poc/trec_sparse_validation.py"] = {
        "path": "poc/trec_sparse_validation.py",
        "sha256": "2" * 64,
        "bytes": 1,
    }

    with pytest.raises(DatasetIntegrityError, match="implementation fingerprints"):
        verify_trec_sparse_validation_manifest(
            root=fixture["root"],
            dataset_spec=fixture["dataset_spec"],
            sparse_spec=fixture["sparse_spec"],
            corpus_path=fixture["corpus_path"],
            embeddings_path=fixture["embeddings_path"],
            completion_manifest_path=fixture["completion_manifest_path"],
            package_path=fixture["package_path"],
            model_path=fixture["model_path"],
            tokenizer_path=fixture["tokenizer_path"],
            validation_manifest_path=fixture["validation_manifest_path"],
            expected_sample_size=3,
        )


def test_validation_verifier_replays_the_official_model_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_validation_fixture(tmp_path)
    monkeypatch.setattr(
        "poc.trec_sparse_validation.collect_manifest_provenance",
        lambda *args, **kwargs: _clean_provenance(),
    )

    def reproduce(
        samples: Sequence[SampledSparseRecord],
        **kwargs: object,
    ) -> tuple[tuple[bytes, ...], dict[str, object]]:
        return tuple(sample.canonical_line for sample in samples), _sample_runtime()

    monkeypatch.setattr(
        "poc.trec_sparse_validation.reencode_trec_sparse_sample", reproduce
    )
    validate_recovered_trec_sparse(**fixture, sample_size=3)

    def forged_replay(
        samples: Sequence[SampledSparseRecord],
        **kwargs: object,
    ) -> tuple[tuple[bytes, ...], dict[str, object]]:
        lines = [sample.canonical_line for sample in samples]
        lines[-1] = _canonical_line(samples[-1].product_id, {"forged": 4.0})
        return tuple(lines), _sample_runtime()

    monkeypatch.setattr(
        "poc.trec_sparse_validation.reencode_trec_sparse_sample", forged_replay
    )
    with pytest.raises(DatasetIntegrityError, match="does not replay"):
        verify_trec_sparse_validation_manifest(
            root=fixture["root"],
            dataset_spec=fixture["dataset_spec"],
            sparse_spec=fixture["sparse_spec"],
            corpus_path=fixture["corpus_path"],
            embeddings_path=fixture["embeddings_path"],
            completion_manifest_path=fixture["completion_manifest_path"],
            package_path=fixture["package_path"],
            model_path=fixture["model_path"],
            tokenizer_path=fixture["tokenizer_path"],
            validation_manifest_path=fixture["validation_manifest_path"],
            expected_sample_size=3,
        )


def test_index_manifest_binds_validation_hash_and_quality_eligibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_validation_fixture(tmp_path)
    clean = _clean_provenance()
    monkeypatch.setattr(
        "poc.trec_sparse_validation.collect_manifest_provenance",
        lambda *args, **kwargs: clean,
    )
    monkeypatch.setattr(
        "poc.trec_sparse_indexing.collect_manifest_provenance",
        lambda *args, **kwargs: clean,
    )
    monkeypatch.setattr(
        "poc.trec_sparse_indexing.verify_decision_provenance",
        lambda *args, **kwargs: True,
    )

    def reproduce(
        samples: Sequence[SampledSparseRecord],
        **kwargs: object,
    ) -> tuple[tuple[bytes, ...], dict[str, object]]:
        return tuple(sample.canonical_line for sample in samples), _sample_runtime()

    monkeypatch.setattr(
        "poc.trec_sparse_validation.reencode_trec_sparse_sample", reproduce
    )

    validate_recovered_trec_sparse(
        **fixture,
        sample_size=3,
    )
    monkeypatch.setattr(
        "poc.trec_sparse_indexing.verify_trec_sparse_validation_manifest",
        lambda **kwargs: verify_trec_sparse_validation_manifest(
            **kwargs, expected_sample_size=3
        ),
    )
    manifest_path = tmp_path / "results/trec-product-search/neural-sparse/index-manifest.json"
    client = FakeOpenSearchClient(document_count=3)

    manifest = build_trec_sparse_index(
        cast(Any, client),
        root=fixture["root"],
        dataset_spec=fixture["dataset_spec"],
        sparse_spec=fixture["sparse_spec"],
        corpus_path=fixture["corpus_path"],
        embeddings_path=fixture["embeddings_path"],
        precompute_manifest_path=fixture["completion_manifest_path"],
        validation_manifest_path=fixture["validation_manifest_path"],
        package_path=fixture["package_path"],
        model_path=fixture["model_path"],
        tokenizer_path=fixture["tokenizer_path"],
        manifest_path=manifest_path,
    )

    validation_facts = file_facts(fixture["validation_manifest_path"])
    validation = cast(dict[str, Any], manifest["validation_manifest"])
    assert manifest["schema_version"] == 2
    assert manifest["quality_evidence_eligible_for_decision"] is False
    assert manifest["latency_evidence_eligible_for_decision"] is False
    assert validation["sha256"] == validation_facts.sha256
    assert validation["bytes"] == validation_facts.bytes
    assert client.indexed_document_ids == ["doc-1", "doc-2", "doc-3"]
    assert cast(dict[str, Any], manifest["index_content"])["documents"] == 3

    client.indexed_documents["doc-2"][
        fixture["sparse_spec"].embedding_field
    ] = {"altered": 9.0}
    with pytest.raises(DatasetIntegrityError, match="source differs for document doc-2"):
        verify_trec_sparse_index(
            cast(Any, client),
            root=fixture["root"],
            dataset_spec=fixture["dataset_spec"],
            sparse_spec=fixture["sparse_spec"],
            corpus_path=fixture["corpus_path"],
            embeddings_path=fixture["embeddings_path"],
            precompute_manifest_path=fixture["completion_manifest_path"],
            validation_manifest_path=fixture["validation_manifest_path"],
            package_path=fixture["package_path"],
            model_path=fixture["model_path"],
            tokenizer_path=fixture["tokenizer_path"],
            manifest_path=manifest_path,
        )


def test_schema_v2_clean_generation_can_be_quality_eligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_validation_fixture(tmp_path)
    _upgrade_completion_to_schema_v2(fixture)
    monkeypatch.setattr(
        "poc.trec_sparse_validation.collect_manifest_provenance",
        lambda *args, **kwargs: _clean_provenance(),
    )

    def reproduce(
        samples: Sequence[SampledSparseRecord],
        **kwargs: object,
    ) -> tuple[tuple[bytes, ...], dict[str, object]]:
        return tuple(sample.canonical_line for sample in samples), _sample_runtime()

    monkeypatch.setattr(
        "poc.trec_sparse_validation.reencode_trec_sparse_sample", reproduce
    )

    result = validate_recovered_trec_sparse(**fixture, sample_size=3)

    assert result["generation_provenance_status"] == (
        "verified_clean_schema_v2_start"
    )
    assert result["generation_provenance_eligible_for_decision"] is True
    assert result["completion_quality_evidence_eligible_for_decision"] is True
    assert result["quality_evidence_eligible_for_decision"] is True
    assert result["quality_ineligibility_reason"] is None


def test_schema_v2_completion_rejects_a_rehashed_false_checkpoint_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_validation_fixture(tmp_path)
    _upgrade_completion_to_schema_v2(fixture)
    monkeypatch.setattr(
        "poc.trec_sparse_validation.collect_manifest_provenance",
        lambda *args, **kwargs: _clean_provenance(),
    )
    checkpoint_path = fixture["embeddings_path"].with_name(
        f".{fixture['embeddings_path'].name}.checkpoint.json"
    )
    checkpoint = cast(dict[str, Any], json.loads(checkpoint_path.read_text()))
    prefix = cast(dict[str, Any], checkpoint["checkpoint_prefix"])
    prefix["stream_sha256"] = "f" * 64
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n"
    )
    completion = cast(
        dict[str, Any], json.loads(fixture["completion_manifest_path"].read_text())
    )
    completion["checkpoint_binding"] = _artifact(fixture["root"], checkpoint_path)
    fixture["completion_manifest_path"].write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(DatasetIntegrityError, match="checkpoint content binding"):
        validate_recovered_trec_sparse(**fixture, sample_size=3)


def test_schema_v2_completion_cannot_overstate_ineligible_start_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_validation_fixture(tmp_path)
    _upgrade_completion_to_schema_v2(fixture)
    monkeypatch.setattr(
        "poc.trec_sparse_validation.collect_manifest_provenance",
        lambda *args, **kwargs: _clean_provenance(),
    )
    monkeypatch.setattr(
        "poc.trec_sparse_validation.verify_decision_provenance",
        lambda *args, **kwargs: kwargs.get("require_current_code_revision") is True,
    )

    with pytest.raises(DatasetIntegrityError, match="overstates provenance"):
        validate_recovered_trec_sparse(**fixture, sample_size=3)


def test_index_quality_requires_current_start_captured_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_validation_fixture(tmp_path)
    _upgrade_completion_to_schema_v2(fixture)
    clean = _clean_provenance()
    monkeypatch.setattr(
        "poc.trec_sparse_validation.collect_manifest_provenance",
        lambda *args, **kwargs: clean,
    )
    monkeypatch.setattr(
        "poc.trec_sparse_indexing.collect_manifest_provenance",
        lambda *args, **kwargs: clean,
    )
    provenance_requirements: list[bool] = []

    def eligible_index_provenance(*args: object, **kwargs: object) -> bool:
        provenance_requirements.append(
            bool(kwargs.get("require_current_code_revision"))
        )
        return True

    monkeypatch.setattr(
        "poc.trec_sparse_indexing.verify_decision_provenance",
        eligible_index_provenance,
    )

    def reproduce(
        samples: Sequence[SampledSparseRecord],
        **kwargs: object,
    ) -> tuple[tuple[bytes, ...], dict[str, object]]:
        return tuple(sample.canonical_line for sample in samples), _sample_runtime()

    monkeypatch.setattr(
        "poc.trec_sparse_validation.reencode_trec_sparse_sample", reproduce
    )
    validate_recovered_trec_sparse(**fixture, sample_size=3)
    monkeypatch.setattr(
        "poc.trec_sparse_indexing.verify_trec_sparse_validation_manifest",
        lambda **kwargs: verify_trec_sparse_validation_manifest(
            **kwargs, expected_sample_size=3
        ),
    )
    manifest_path = tmp_path / "results/trec-product-search/neural-sparse/index-manifest.json"
    client = FakeOpenSearchClient(document_count=3)
    manifest = build_trec_sparse_index(
        cast(Any, client),
        root=fixture["root"],
        dataset_spec=fixture["dataset_spec"],
        sparse_spec=fixture["sparse_spec"],
        corpus_path=fixture["corpus_path"],
        embeddings_path=fixture["embeddings_path"],
        precompute_manifest_path=fixture["completion_manifest_path"],
        validation_manifest_path=fixture["validation_manifest_path"],
        package_path=fixture["package_path"],
        model_path=fixture["model_path"],
        tokenizer_path=fixture["tokenizer_path"],
        manifest_path=manifest_path,
    )
    assert manifest["indexing_provenance_eligible_for_decision"] is True
    assert manifest["schema_version"] == 2
    assert manifest["quality_evidence_eligible_for_decision"] is True
    assert provenance_requirements == [True]

    provenance_requirements.clear()

    def ineligible_index_provenance(
        *_args: object, **kwargs: object
    ) -> bool:
        provenance_requirements.append(
            bool(kwargs.get("require_current_code_revision"))
        )
        return False

    monkeypatch.setattr(
        "poc.trec_sparse_indexing.verify_decision_provenance",
        ineligible_index_provenance,
    )
    with pytest.raises(DatasetIntegrityError, match="index evidence eligibility"):
        verify_trec_sparse_index(
            cast(Any, client),
            root=fixture["root"],
            dataset_spec=fixture["dataset_spec"],
            sparse_spec=fixture["sparse_spec"],
            corpus_path=fixture["corpus_path"],
            embeddings_path=fixture["embeddings_path"],
            precompute_manifest_path=fixture["completion_manifest_path"],
            validation_manifest_path=fixture["validation_manifest_path"],
            package_path=fixture["package_path"],
            model_path=fixture["model_path"],
            tokenizer_path=fixture["tokenizer_path"],
            manifest_path=manifest_path,
        )
    assert provenance_requirements == [False]

    diagnostic_manifest = build_trec_sparse_index(
        cast(Any, FakeOpenSearchClient(document_count=3)),
        root=fixture["root"],
        dataset_spec=fixture["dataset_spec"],
        sparse_spec=fixture["sparse_spec"],
        corpus_path=fixture["corpus_path"],
        embeddings_path=fixture["embeddings_path"],
        precompute_manifest_path=fixture["completion_manifest_path"],
        validation_manifest_path=fixture["validation_manifest_path"],
        package_path=fixture["package_path"],
        model_path=fixture["model_path"],
        tokenizer_path=fixture["tokenizer_path"],
        manifest_path=tmp_path / "results/diagnostic-index-manifest.json",
    )
    assert diagnostic_manifest["indexing_provenance_eligible_for_decision"] is False
    assert diagnostic_manifest["quality_evidence_eligible_for_decision"] is False


def _write_validation_fixture(tmp_path: Path) -> ValidationFixture:
    root = tmp_path
    corpus, embeddings, records = _write_sparse_fixture(root)
    package = root / "data/cache/models/neural-sparse/doc-v2-distill.zip"
    model = (
        root
        / "data/cache/models/neural-sparse/doc-v2-distill"
        / "opensearch-neural-sparse-encoding-doc-v2-distill.pt"
    )
    tokenizer = model.with_name("tokenizer.json")
    package.parent.mkdir(parents=True, exist_ok=True)
    model.parent.mkdir(parents=True, exist_ok=True)
    model_payload = b"official torchscript"
    tokenizer_payload = b"official tokenizer"
    model.write_bytes(model_payload)
    tokenizer.write_bytes(tokenizer_payload)
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(model.name, model_payload)
        archive.writestr(tokenizer.name, tokenizer_payload)
    package_facts = file_facts(package)
    corpus_facts = file_facts(corpus)
    dataset_spec = TrecProductSearchDatasetSpec(
        source="https://example.test/trec",
        revision="a" * 40,
        expected_products=records,
        expected_queries=1,
        expected_judged_queries=1,
        expected_judgments=1,
        expected_unique_judged_products=1,
        license="test",
        primary_gain_mapping="test",
        query_authority="test",
        files={
            "corpus": DatasetFileSpec(
                name="corpus",
                filename=corpus.name,
                url="https://example.test/corpus",
                sha256=corpus_facts.sha256,
                bytes=corpus_facts.bytes,
            )
        },
    )
    sparse_spec = TrecSparseSpec(
        document_model_name="official-doc-model",
        document_model_version="1.0.0",
        document_model_format="TORCH_SCRIPT",
        document_model_url="https://example.test/model.zip",
        document_model_sha256=package_facts.sha256,
        document_model_bytes=package_facts.bytes,
        query_tokenizer_name="official-query-tokenizer",
        query_tokenizer_version="1.0.1",
        query_tokenizer_format="TORCH_SCRIPT",
        query_tokenizer_content_sha256=file_facts(tokenizer).sha256,
        query_tokenizer_content_bytes=file_facts(tokenizer).bytes,
        index_name="test-index",
        text_field="sparse_text",
        embedding_field="sparse_embedding",
        shards=1,
        replicas=0,
        bulk_request_size=100,
        expected_opensearch_version="2.19.6",
        inference_batch_size=2,
        maximum_token_length=256,
        maximum_value_ratio=0.1,
        minimum_preflight_documents_per_second=150.0,
    )
    completion = root / "results/trec-product-search/neural-sparse/precompute-full-manifest.json"
    completion.parent.mkdir(parents=True, exist_ok=True)
    completion_payload = {
        "schema_version": 1,
        "dataset": "TREC Product Search 2024",
        "scope": "full_corpus",
        "records": records,
        "record_limit": None,
        "runtime": "torchscript_cpu",
        "batch_size": sparse_spec.inference_batch_size,
        "maximum_token_length": sparse_spec.maximum_token_length,
        "minimum_preflight_documents_per_second": (
            sparse_spec.minimum_preflight_documents_per_second
        ),
        "document_recipe": "title.description",
        "pruning": {
            "type": "max_ratio",
            "maximum_value_ratio": sparse_spec.maximum_value_ratio,
            "implementation": "local_precomputed_equivalent",
        },
        "source_corpus": _artifact(root, corpus),
        "document_model": {
            "name": sparse_spec.document_model_name,
            "version": sparse_spec.document_model_version,
            "format": sparse_spec.document_model_format,
            "package_url": sparse_spec.document_model_url,
            "package": _artifact(root, package),
            "torchscript": _artifact(root, model),
            "tokenizer": _artifact(root, tokenizer),
        },
        "embeddings": {
            **_artifact(root, embeddings),
            "stream_sha256": file_facts(embeddings).sha256,
        },
    }
    completion.write_text(json.dumps(completion_payload, indent=2) + "\n")
    return ValidationFixture(
        root=root,
        dataset_spec=dataset_spec,
        sparse_spec=sparse_spec,
        corpus_path=corpus,
        embeddings_path=embeddings,
        completion_manifest_path=completion,
        package_path=package,
        model_path=model,
        tokenizer_path=tokenizer,
        validation_manifest_path=(
            root
            / "results/trec-product-search/neural-sparse/precompute-full-validation-manifest.json"
        ),
    )


def _write_sparse_fixture(root: Path) -> tuple[Path, Path, int]:
    corpus = root / "data/raw/trec-product-search-2024/collection.trec.gz"
    embeddings = (
        root
        / "data/cache/neural-sparse/trec-product-search-doc-v2-distill-full.jsonl"
    )
    corpus.parent.mkdir(parents=True, exist_ok=True)
    embeddings.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(corpus, "wt", encoding="utf-8") as handle:
        handle.write("doc-1\tTitle one\tDescription one\n")
        handle.write("doc-2\tTitle two\tDescription two\n")
        handle.write("doc-3\tTitle three\tDescription three\n")
    embeddings.write_bytes(
        _canonical_line("doc-1", {"one": 1.0})
        + _canonical_line("doc-2", {"two": 2.0})
        + _canonical_line("doc-3", {"three": 3.0})
    )
    return corpus, embeddings, 3


def _upgrade_completion_to_schema_v2(fixture: ValidationFixture) -> None:
    completion = cast(
        dict[str, Any], json.loads(fixture["completion_manifest_path"].read_text())
    )
    generation_provenance = _clean_provenance()
    embeddings = file_facts(fixture["embeddings_path"])
    checkpoint = {
        "schema_version": 2,
        "artifact_type": "trec_sparse_precompute_checkpoint_binding",
        "dataset": "TREC Product Search 2024",
        "scope": "full_corpus",
        "record_limit": None,
        "output_filename": fixture["embeddings_path"].name,
        "generation_provenance": generation_provenance,
        "source_corpus": completion["source_corpus"],
        "document_model": completion["document_model"],
        "runtime": "torchscript_cpu",
        "batch_size": fixture["sparse_spec"].inference_batch_size,
        "maximum_token_length": fixture["sparse_spec"].maximum_token_length,
        "minimum_preflight_documents_per_second": (
            fixture["sparse_spec"].minimum_preflight_documents_per_second
        ),
        "pruning": completion["pruning"],
        "document_recipe": "title.description",
        "checkpoint_prefix": {
            "records": fixture["dataset_spec"].expected_products,
            "bytes": embeddings.bytes,
            "stream_sha256": embeddings.sha256,
        },
    }
    checkpoint_path = fixture["embeddings_path"].with_name(
        f".{fixture['embeddings_path'].name}.checkpoint.json"
    )
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")
    completion.update(
        {
            "schema_version": 2,
            "generation_provenance": generation_provenance,
            "benchmark_provenance": generation_provenance,
            "quality_evidence_eligible_for_decision": True,
            "resumed_records": 0,
            "checkpoint_resume_verified": False,
            "checkpoint_binding": _artifact(fixture["root"], checkpoint_path),
            "encoded_records_this_session": fixture["dataset_spec"].expected_products,
            "throughput_scope": "current_process_session",
        }
    )
    fixture["completion_manifest_path"].write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )


def _canonical_line(product_id: str, embedding: dict[str, float]) -> bytes:
    return (
        json.dumps(
            {"product_id": product_id, "embedding": embedding},
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _artifact(root: Path, path: Path) -> dict[str, object]:
    facts = file_facts(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
    }


def _clean_provenance() -> dict[str, object]:
    return {
        "code_revision": {
            "git_commit": "a" * 40,
            "source_tree_sha256": "b" * 64,
            "source_dirty": False,
        },
        "benchmark_profile": {"profile_id": "test"},
        "benchmark_profile_sha256": "c" * 64,
        "benchmark_environment": {"status": "test"},
        "generator_host": {"architecture": "test", "system": "test"},
    }


def _sample_runtime() -> dict[str, object]:
    return {
        "engine": "torchscript_cpu",
        "torch_version": "test",
        "tokenizers_version": "test",
        "numpy_version": "test",
        "inference_batch_size": 2,
        "maximum_token_length": 256,
        "maximum_value_ratio": 0.1,
    }


class FakeOpenSearchClient:
    def __init__(self, *, document_count: int) -> None:
        self.document_count = document_count
        self.indexed_document_ids: list[str] = []
        self.indexed_documents: dict[str, dict[str, Any]] = {}
        self.definition: dict[str, Any] | None = None
        self.write_block = False

    def wait_until_ready(self, *, expected_version: str) -> str:
        return expected_version

    def delete_index(self, index: str) -> None:
        return None

    def create_index(self, index: str, definition: object) -> None:
        self.definition = cast(dict[str, Any], definition)

    def bulk_index(
        self,
        index: str,
        documents: Any,
        *,
        batch_size: int,
    ) -> None:
        self.indexed_documents = {
            document_id: deepcopy(dict(document))
            for document_id, document in documents
        }
        self.indexed_document_ids = list(self.indexed_documents)

    def refresh(self, index: str) -> None:
        return None

    def force_merge(self, index: str) -> None:
        return None

    def update_index_settings(self, index: str, settings: object) -> None:
        if settings == {"index": {"blocks": {"write": True}}}:
            self.write_block = True
        return None

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: object | None = None,
        params: object | None = None,
    ) -> dict[str, object]:
        if path.endswith("/_count"):
            return {"count": self.document_count}
        if path.endswith("/_mget"):
            assert method == "POST"
            assert isinstance(json_body, dict)
            return {
                "docs": [
                    {
                        "_id": document_id,
                        "found": document_id in self.indexed_documents,
                        "_source": deepcopy(
                            self.indexed_documents.get(document_id)
                        ),
                    }
                    for document_id in cast(list[str], json_body["ids"])
                ]
            }
        if path.endswith("/_stats/docs,segments"):
            return {
                "_all": {
                    "primaries": {
                        "docs": {"count": self.document_count, "deleted": 0},
                        "segments": {"count": 1},
                    }
                }
            }
        if path.endswith("/_stats/store"):
            return {"_all": {"primaries": {"store": {"size_in_bytes": 123}}}}
        if path.endswith("/_mapping"):
            assert self.definition is not None
            return {"test-index": {"mappings": self.definition["mappings"]}}
        if path.endswith("/_settings"):
            assert self.definition is not None
            definition_settings = cast(
                dict[str, Any], self.definition["settings"]
            )
            index_settings = dict(
                cast(dict[str, Any], definition_settings["index"])
            )
            index_settings["refresh_interval"] = "1s"
            if self.write_block:
                index_settings["blocks"] = {"write": "true"}
            index_settings["analysis"] = definition_settings["analysis"]
            return {
                "test-index": {
                    "settings": {
                        "index": {
                            "uuid": "test-uuid",
                            "creation_date": "1",
                            **index_settings,
                        }
                    }
                }
            }
        return {"test-index": {"settings": {"index": {"uuid": "test-uuid"}}}}
