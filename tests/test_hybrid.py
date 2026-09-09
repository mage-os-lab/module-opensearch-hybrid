from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import poc.hybrid as hybrid_module
from poc.config import WANDS_INT8_MODEL_NAMES, ModelSpec, load_model_registry
from poc.datasets import DatasetIntegrityError, file_facts
from poc.dense import query_vectors_sha256
from poc.experiments import BM25Profile, load_bm25_experiments
from poc.hybrid import (
    derive_hybrid_run_eligibility,
    execute_exact_hybrid_run,
    expected_hybrid_artifact_keys,
    hybrid_pipeline_id,
    hybrid_trec_score_for_rank,
    hybrid_upstream_artifact_bindings_valid,
    select_hybrid_model,
    select_hybrid_weight,
    verify_registered_hybrid_inputs,
    verify_wands_index_after_decision_replay,
)
from poc.indexing import load_wands_index_config
from poc.query_runtime import load_query_runtime_spec
from poc.trec import RunRecord, write_run

ROOT = Path(__file__).resolve().parents[1]


def test_hybrid_weight_selection_uses_registered_order_as_tie_break() -> None:
    weights = (0.2, 0.3, 0.4)
    metrics = {
        0.2: {"ndcg@10": 0.71},
        0.3: {"ndcg@10": 0.73},
        0.4: {"ndcg@10": 0.73},
    }

    assert select_hybrid_weight(weights, metrics, metric="ndcg@10") == 0.3


def test_hybrid_model_selection_uses_development_metrics_only() -> None:
    metrics = {
        "gte": {"ndcg@10": 0.73},
        "granite": {"ndcg@10": 0.75},
    }

    assert (
        select_hybrid_model(("gte", "granite"), metrics, metric="ndcg@10")
        == "granite"
    )


def test_hybrid_trec_score_is_derived_from_final_fused_rank() -> None:
    assert hybrid_trec_score_for_rank(1) == 100.0
    assert hybrid_trec_score_for_rank(100) == 1.0


def test_hybrid_artifact_keys_are_exactly_the_registered_grid_and_test_winner() -> None:
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")

    assert expected_hybrid_artifact_keys(experiments) == {
        "dev/lw020",
        "dev/lw030",
        "dev/lw040",
        "dev/lw050",
        "dev/lw060",
        "test/selected",
    }


@pytest.mark.parametrize("upstream_eligible", (False, "true", 1, None))
def test_hybrid_held_out_eligibility_rejects_non_true_upstream_evidence(
    upstream_eligible: object,
) -> None:
    eligible, reason = derive_hybrid_run_eligibility(
        decision_scope="held_out",
        benchmark_provenance_valid=True,
        upstream_evidence_eligible=upstream_eligible,
        selected_model_eligible=True,
    )

    assert eligible is False
    assert reason == "hybrid upstream evidence is not decision eligible"


def test_hybrid_tuning_artifacts_remain_explicitly_ineligible() -> None:
    eligible, reason = derive_hybrid_run_eligibility(
        decision_scope="tuning",
        benchmark_provenance_valid=True,
        upstream_evidence_eligible=True,
        selected_model_eligible=True,
    )

    assert eligible is False
    assert reason == "development tuning result"


def test_hybrid_final_gate_uses_selected_model_not_rejected_candidates() -> None:
    assert hybrid_module.hybrid_upstream_evidence_eligible(
        index_eligible=True,
        bm25_eligible=True,
        int8_summary_eligible=True,
    ) is True

    eligible, reason = derive_hybrid_run_eligibility(
        decision_scope="held_out",
        benchmark_provenance_valid=True,
        upstream_evidence_eligible=True,
        selected_model_eligible=False,
    )

    assert eligible is False
    assert reason == "selected int8 dense model is not decision eligible"


def test_hybrid_upstream_bindings_fail_when_source_bytes_change(tmp_path: Path) -> None:
    source = tmp_path / "results/source.json"
    source.parent.mkdir(parents=True)
    source.write_text("one\n")
    facts = file_facts(source)
    evidence = {
        "source": {
            "path": "results/source.json",
            "sha256": facts.sha256,
            "bytes": facts.bytes,
        }
    }

    assert hybrid_upstream_artifact_bindings_valid(tmp_path, evidence) is True
    source.write_text("two\n")
    assert hybrid_upstream_artifact_bindings_valid(tmp_path, evidence) is False


def test_hybrid_inputs_must_equal_registered_model_runtime_experiment_and_index() -> None:
    registry = load_model_registry(ROOT / "config/models.toml")
    models = tuple(registry[name] for name in WANDS_INT8_MODEL_NAMES)
    runtime = load_query_runtime_spec(ROOT / "config/query_runtime.toml")
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")
    index = load_wands_index_config(ROOT / "config/indexes.toml").name

    verified = verify_registered_hybrid_inputs(
        root=ROOT,
        index=index,
        models=models,
        runtime=runtime,
        experiments=experiments,
    )
    assert verified["index"] == index
    assert verified["models"] == list(WANDS_INT8_MODEL_NAMES)

    with pytest.raises(DatasetIntegrityError, match="index"):
        verify_registered_hybrid_inputs(
            root=ROOT,
            index=f"{index}-unregistered",
            models=models,
            runtime=runtime,
            experiments=experiments,
        )
    with pytest.raises(DatasetIntegrityError, match="model"):
        verify_registered_hybrid_inputs(
            root=ROOT,
            index=index,
            models=tuple(reversed(models)),
            runtime=runtime,
            experiments=experiments,
        )
    with pytest.raises(DatasetIntegrityError, match="runtime"):
        verify_registered_hybrid_inputs(
            root=ROOT,
            index=index,
            models=models,
            runtime=replace(runtime, batch_size=runtime.batch_size + 1),
            experiments=experiments,
        )
    with pytest.raises(DatasetIntegrityError, match="experiment"):
        verify_registered_hybrid_inputs(
            root=ROOT,
            index=index,
            models=models,
            runtime=runtime,
            experiments=replace(experiments, primary_metric="map@10"),
        )


def test_exact_hybrid_run_uses_frozen_bm25_clause_and_registered_pipeline() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.pipelines: list[tuple[str, dict[str, object]]] = []
            self.requests: list[tuple[str, dict[str, object], str | None]] = []

        def request(self, method: str, path: str) -> dict[str, object]:
            assert method == "GET"
            pipeline_id = path.rsplit("/", 1)[-1]
            return {
                pipeline_id: next(
                    definition
                    for current_id, definition in reversed(self.pipelines)
                    if current_id == pipeline_id
                )
            }

        def put_search_pipeline(
            self, pipeline: str, definition: dict[str, object]
        ) -> None:
            self.pipelines.append((pipeline, definition))

        def search(
            self,
            index: str,
            request: dict[str, object],
            *,
            pipeline: str | None = None,
        ) -> dict[str, object]:
            self.requests.append((index, request, pipeline))
            return {
                "hits": {
                    "hits": [
                        {"_id": "20", "_score": 0.9},
                        {"_id": "10", "_score": 0.8},
                    ]
                }
            }

    model = ModelSpec.from_mapping(
        "test_model",
        {
            "hf_id": "example/test",
            "revision": "1" * 40,
            "dims": 2,
            "normalize": True,
            "max_seq_length": 512,
            "bulk_dtype": "float32",
            "use_memory_efficient_attention": False,
            "trust_remote_code": False,
            "query_prefix": "",
            "document_prefix": "",
            "query_template": "{query}",
            "document_template": "{title}",
            "license": "Apache-2.0",
            "decision_eligible": True,
            "contamination": "none_known",
        },
    )
    profile = BM25Profile.from_mapping(
        {
            "name": "frozen",
            "fields": ["title^8", "description"],
            "query_type": "best_fields",
            "operator": "or",
            "tie_breaker": 0.1,
        }
    )
    client = FakeClient()

    records = execute_exact_hybrid_run(
        client,  # type: ignore[arg-type]
        index="products",
        queries={"1": "coffee table"},
        query_vectors={"1": np.asarray([1.0, 0.0], dtype=np.float32)},
        profile=profile,
        model=model,
        lexical_weight=0.3,
        tag="test-hybrid",
    )

    expected_pipeline = hybrid_pipeline_id(model, 0.3)
    assert client.pipelines[0][0] == expected_pipeline
    assert client.requests[0][2] == expected_pipeline
    request = client.requests[0][1]
    assert request["query"]["hybrid"]["queries"][0] == profile.query("coffee table")  # type: ignore[index]
    assert [(record.document_id, record.score) for record in records] == [
        ("20", 100.0),
        ("10", 99.0),
    ]


def test_exact_hybrid_run_rejects_pipeline_mutation_during_replay() -> None:
    class MutatingClient:
        def __init__(self) -> None:
            self.pipeline_id = ""
            self.pipeline: dict[str, object] = {}

        def put_search_pipeline(
            self, pipeline: str, definition: dict[str, object]
        ) -> None:
            self.pipeline_id = pipeline
            self.pipeline = definition

        def request(self, method: str, path: str) -> dict[str, object]:
            assert method == "GET"
            assert path == f"/_search/pipeline/{self.pipeline_id}"
            return {self.pipeline_id: self.pipeline}

        def search(
            self,
            index: str,
            request: dict[str, object],
            *,
            pipeline: str | None = None,
        ) -> dict[str, object]:
            del index, request, pipeline
            self.pipeline = {"phase_results_processors": []}
            return {"hits": {"hits": [{"_id": "20", "_score": 0.9}]}}

    model = _fixture_model()
    profile = BM25Profile.from_mapping(
        {
            "name": "frozen",
            "fields": ["title"],
            "query_type": "best_fields",
            "operator": "or",
            "tie_breaker": 0.0,
        }
    )

    with pytest.raises(DatasetIntegrityError, match="search pipeline differs"):
        execute_exact_hybrid_run(
            MutatingClient(),  # type: ignore[arg-type]
            index="products",
            queries={"1": "coffee table"},
            query_vectors={"1": np.asarray([1.0, 0.0], dtype=np.float32)},
            profile=profile,
            model=model,
            lexical_weight=0.3,
            tag="test-hybrid",
        )


def test_index_replay_guard_rejects_post_scan_document_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hybrid_module,
        "load_wands_config",
        lambda path: SimpleNamespace(expected_products=1),
    )
    monkeypatch.setattr(
        hybrid_module,
        "load_wands_index_config",
        lambda path: SimpleNamespace(name="products"),
    )
    monkeypatch.setattr(
        hybrid_module,
        "verify_wands_lexical_index",
        lambda *args, **kwargs: {
            "uuid": "same-index",
            "index_content": {"sha256": "mutated-after-initial-scan"},
        },
    )

    with pytest.raises(DatasetIntegrityError, match="changed during decision replay"):
        verify_wands_index_after_decision_replay(
            object(),  # type: ignore[arg-type]
            root=tmp_path,
            index="products",
            expected_verification={
                "uuid": "same-index",
                "index_content": {"sha256": "initial-scan"},
            },
        )


def test_hybrid_live_replay_rejects_altered_nonselected_dev_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiments = load_bm25_experiments(ROOT / "config/experiments.toml")
    model = _fixture_model()
    profile = BM25Profile.from_mapping(
        {
            "name": "frozen",
            "fields": ["title"],
            "query_type": "best_fields",
            "operator": "or",
            "tie_breaker": 0.0,
        }
    )
    prepared = tmp_path / "data/prepared/wands"
    prepared.mkdir(parents=True)
    for split in ("dev", "test"):
        (prepared / f"queries.{split}.jsonl").write_text(
            json.dumps({"query_id": "1", "query": "table", "split": split})
            + "\n"
        )

    vectors = {"1": np.asarray([1.0, 0.0], dtype=np.float32)}
    vector_hash = query_vectors_sha256(vectors)
    artifacts: dict[str, dict[str, str]] = {}
    for weight in experiments.lexical_weight_grid:
        key = f"dev/lw{round(weight * 100):03d}"
        tag = f"fixture-{key.replace('/', '-')}"
        artifacts[key] = _write_replay_fixture(
            tmp_path,
            key=key,
            tag=tag,
            document_id="unexpected" if weight == experiments.lexical_weight_grid[0] else "d1",
            vector_hash=vector_hash,
        )
    artifacts["test/selected"] = _write_replay_fixture(
        tmp_path,
        key="test/selected",
        tag="fixture-test-selected",
        document_id="d1",
        vector_hash=vector_hash,
    )

    class FakeBackend:
        model_artifact_sha256 = "a" * 64

    monkeypatch.setattr(
        hybrid_module,
        "create_int8_query_backend",
        lambda **kwargs: FakeBackend(),
    )
    monkeypatch.setattr(
        hybrid_module,
        "encode_queries",
        lambda *args, **kwargs: vectors,
    )

    def execute_live(*args: object, **kwargs: Any) -> list[RunRecord]:
        return [
            RunRecord(
                query_id="1",
                document_id="d1",
                rank=1,
                score=100.0,
                tag=str(kwargs["tag"]),
            )
        ]

    monkeypatch.setattr(hybrid_module, "execute_exact_hybrid_run", execute_live)

    with pytest.raises(DatasetIntegrityError, match="dev/lw020"):
        hybrid_module._verify_hybrid_live_reruns(
            object(),  # type: ignore[arg-type]
            root=tmp_path,
            index="fixture",
            model=model,
            runtime=load_query_runtime_spec(ROOT / "config/query_runtime.toml"),
            profile=profile,
            experiments=experiments,
            summary={
                "selected_lexical_weight": 0.5,
                "artifacts": artifacts,
            },
        )


def _fixture_model() -> ModelSpec:
    return ModelSpec.from_mapping(
        "fixture",
        {
            "hf_id": "example/fixture",
            "revision": "1" * 40,
            "dims": 2,
            "normalize": True,
            "max_seq_length": 512,
            "bulk_dtype": "float32",
            "use_memory_efficient_attention": False,
            "trust_remote_code": False,
            "query_prefix": "",
            "document_prefix": "",
            "query_template": "{query}",
            "document_template": "{title}",
            "license": "Apache-2.0",
            "decision_eligible": True,
            "contamination": "none_known",
        },
    )


def _write_replay_fixture(
    root: Path,
    *,
    key: str,
    tag: str,
    document_id: str,
    vector_hash: str,
) -> dict[str, str]:
    run_path = root / f"runs/{tag}.trec"
    write_run(
        run_path,
        [
            RunRecord(
                query_id="1",
                document_id=document_id,
                rank=1,
                score=100.0,
                tag=tag,
            )
        ],
    )
    manifest_path = root / f"runs/{tag}.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "tag": tag,
                "query_vectors_sha256": vector_hash,
            }
        )
        + "\n"
    )
    return {
        "run": str(run_path.relative_to(root)),
        "manifest": str(manifest_path.relative_to(root)),
    }
