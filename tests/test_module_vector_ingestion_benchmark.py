from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from poc.benchmark_source_identity import verify_source_identity
from poc.module_vector_ingestion import (
    BenchmarkConfig,
    assert_decision_environment,
    build_bulk_payload,
    generate_vectors,
    summarize_comparison,
)

ROOT = Path(__file__).resolve().parents[1]


def test_numeric_and_base64_payloads_preserve_float32_vector_bytes() -> None:
    vectors = generate_vectors(document_count=4, dimension=256, seed=20260826)

    numeric = build_bulk_payload(
        index="numeric-index",
        start_document=1,
        vectors=vectors,
        wire_format="numeric",
    )
    encoded = build_bulk_payload(
        index="base64-index",
        start_document=1,
        vectors=vectors,
        wire_format="base64",
    )

    numeric_lines = [json.loads(line) for line in numeric.splitlines()]
    encoded_lines = [json.loads(line) for line in encoded.splitlines()]
    for offset in range(4):
        numeric_action = numeric_lines[offset * 2]
        encoded_action = encoded_lines[offset * 2]
        numeric_source = numeric_lines[offset * 2 + 1]
        encoded_source = encoded_lines[offset * 2 + 1]

        assert numeric_action["index"]["_id"] == encoded_action["index"]["_id"]
        assert numeric_action["index"]["version_type"] == "external_gte"
        assert encoded_action["index"]["version_type"] == "external_gte"
        assert {
            key: value for key, value in numeric_source.items() if key != "embedding"
        } == {key: value for key, value in encoded_source.items() if key != "embedding"}
        numeric_bytes = np.asarray(numeric_source["embedding"], dtype="<f4").tobytes()
        encoded_bytes = base64.b64decode(encoded_source["embedding"], validate=True)
        assert numeric_bytes == encoded_bytes
        assert numeric_bytes == vectors[offset].astype("<f4", copy=False).tobytes()

    assert len(encoded.encode()) < len(numeric.encode()) * 0.4


def test_vectors_are_reproducible_and_normalized() -> None:
    first = generate_vectors(document_count=32, dimension=256, seed=7)
    second = generate_vectors(document_count=32, dimension=256, seed=7)

    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.linalg.norm(first, axis=1) == pytest.approx(np.ones(32), abs=1e-6)


def test_benchmark_cli_requires_bound_source_and_environment_identity() -> None:
    source = (ROOT / "scripts/47_benchmark_module_vector_ingestion.py").read_text()
    benchmark = (ROOT / "poc/module_vector_ingestion.py").read_text()
    attester = (ROOT / "poc/benchmark_source_identity.py").read_text()
    makefile = (ROOT / "Makefile").read_text()
    documentation = (
        ROOT / "docs/VECTOR-INGESTION-QUALIFICATION.md"
    ).read_text()
    readme = (ROOT / "README.md").read_text()

    assert '"--decision-run"' in source
    assert '"--opensearch-image"' in source
    assert '"--topology"' in source
    assert '"--source-identity"' in source
    assert "assert_decision_environment(" in source
    assert "platform.machine()" in source
    assert "platform.system()" in source
    assert '"git", "diff", "--quiet", "HEAD", "--"' in attester
    assert "source_tree_dirty" in source
    assert "run_benchmark(" in source
    assert "write_json(" in source
    assert "module-vector-ingestion-smoke:" in makefile
    assert "module-vector-ingestion-decision:" in makefile
    assert "--documents 50000" in makefile
    assert "--trials 5" in makefile
    assert "--decision-run" in makefile
    assert "module-vector-ingestion-smoke" in documentation
    assert "module-vector-ingestion-decision" in documentation
    assert "60%" in documentation
    assert "5%" in documentation
    assert "VECTOR-INGESTION-QUALIFICATION.md" in readme
    assert "worker_process_cpu_seconds" in benchmark
    assert "worker_rss_bytes" in benchmark
    assert "worker_peak_rss_bytes" in benchmark
    assert '"retry_count": 0' in benchmark
    assert '"dead_letter_count": 0' in benchmark
    assert '"transport_mode": "direct_bulk_no_queue"' in benchmark


def test_external_source_identity_requires_exact_clean_committed_bytes(
    tmp_path: Path,
) -> None:
    source_path = Path("benchmark.py")
    source = tmp_path / source_path
    source.write_text("print('qualified')\n")
    identity = {
        "git_commit": "a" * 40,
        "source_tree_dirty": False,
        "untracked_source_paths": [],
        "file_sha256": {
            str(source_path): hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    }

    verified = verify_source_identity(tmp_path, (source_path,), identity)

    assert verified["git_commit"] == "a" * 40
    assert verified["verification_method"] == "external_manifest_file_hashes"
    source.write_text("print('changed')\n")
    with pytest.raises(ValueError, match="hash does not match"):
        verify_source_identity(tmp_path, (source_path,), identity)


@pytest.mark.parametrize(
    "identity",
    [
        {
            "git_commit": "not-a-commit",
            "source_tree_dirty": False,
            "untracked_source_paths": [],
            "file_sha256": {"benchmark.py": "a" * 64},
        },
        {
            "git_commit": "a" * 40,
            "source_tree_dirty": True,
            "untracked_source_paths": [],
            "file_sha256": {"benchmark.py": "a" * 64},
        },
    ],
)
def test_external_source_identity_rejects_uncommitted_provenance(
    tmp_path: Path,
    identity: dict[str, object],
) -> None:
    source_path = Path("benchmark.py")
    (tmp_path / source_path).write_text("print('qualified')\n")

    with pytest.raises(ValueError, match="committed"):
        verify_source_identity(tmp_path, (source_path,), identity)


def test_decision_summary_applies_registered_median_gates() -> None:
    trials = []
    for trial in range(1, 6):
        trials.extend(
            [
                {
                    "arm": "numeric",
                    "batch_size": 64,
                    "trial": trial,
                    "serialized_bytes": 5_000_000,
                    "documents_per_second": 1_000.0,
                    "bulk_latency": {"p95_ms": 100.0},
                    "failed_items": 0,
                    "document_count": 50_000,
                    "vector_count": 50_000,
                },
                {
                    "arm": "base64",
                    "batch_size": 64,
                    "trial": trial,
                    "serialized_bytes": 1_500_000,
                    "documents_per_second": 990.0,
                    "bulk_latency": {"p95_ms": 102.0},
                    "failed_items": 0,
                    "document_count": 50_000,
                    "vector_count": 50_000,
                },
            ]
        )

    summary = summarize_comparison(trials, decision_run=True)

    assert summary["payload_reduction_fraction"] == pytest.approx(0.7)
    assert summary["throughput_ratio"] == pytest.approx(0.99)
    assert summary["p95_latency_ratio"] == pytest.approx(1.02)
    assert summary["gates"] == {
        "complete_and_error_free": True,
        "payload_reduction_at_least_60_percent": True,
        "throughput_regression_at_most_5_percent": True,
        "p95_latency_regression_at_most_5_percent": True,
    }
    assert summary["eligible_for_decision"] is True


def test_decision_environment_requires_same_host_linux_amd64() -> None:
    image = "opensearchproject/opensearch:3.8.0@sha256:" + "a" * 64

    assert_decision_environment(
        decision_run=True,
        topology="same-host",
        machine="x86_64",
        system="Linux",
        opensearch_image=image,
    )
    with pytest.raises(ValueError, match="Linux x86_64"):
        assert_decision_environment(
            decision_run=True,
            topology="same-host",
            machine="arm64",
            system="Darwin",
            opensearch_image=image,
        )
    with pytest.raises(ValueError, match="same-host"):
        assert_decision_environment(
            decision_run=True,
            topology="remote",
            machine="x86_64",
            system="Linux",
            opensearch_image=image,
        )
    with pytest.raises(ValueError, match="immutable"):
        assert_decision_environment(
            decision_run=True,
            topology="same-host",
            machine="x86_64",
            system="Linux",
            opensearch_image="opensearchproject/opensearch:3.8.0",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"document_count": 0}, "document count"),
        ({"dimension": 255}, "256 dimensions"),
        ({"batch_sizes": (0,)}, "batch size"),
        ({"trials": 0}, "trial count"),
        ({"decision_run": True, "document_count": 49_999}, "50,000 documents"),
        (
            {"decision_run": True, "document_count": 50_000, "trials": 4},
            "five trials",
        ),
        (
            {
                "decision_run": True,
                "document_count": 50_000,
                "trials": 5,
                "batch_sizes": (32, 64),
            },
            "one frozen batch size",
        ),
    ],
)
def test_benchmark_configuration_rejects_unqualified_runs(
    kwargs: dict[str, object], message: str
) -> None:
    defaults: dict[str, object] = {
        "document_count": 10_000,
        "dimension": 256,
        "batch_sizes": (16, 32, 64),
        "trials": 1,
        "warmup_documents": 1_000,
        "seed": 20260826,
        "decision_run": False,
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=message):
        BenchmarkConfig(**defaults)  # type: ignore[arg-type]
