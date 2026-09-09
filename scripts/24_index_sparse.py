from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from poc.datasets import file_facts, load_wands_config
from poc.index_content import verify_live_index_content
from poc.index_evidence import lock_live_index_for_decisions
from poc.indexing import (
    verify_wands_preparation_evidence,
    wands_completion_provenance_evidence,
)
from poc.manifest import canonical_sha256, collect_index_facts, read_json
from poc.neural_sparse import (
    build_sparse_index_definition,
    build_sparse_ingest_pipeline,
    iter_precomputed_sparse_documents,
    load_neural_sparse_spec,
    verify_wands_sparse_precompute,
)
from poc.os_client import OpenSearchClient
from poc.provenance import collect_manifest_provenance, registered_opensearch_url
from poc.smoke import EXPECTED_VERSION

ROOT = Path(__file__).resolve().parents[1]
MODEL_ARTIFACT = ROOT / "results/wands/neural-sparse/model-deployment.json"
INDEX_ARTIFACT = ROOT / "results/wands/neural-sparse/index-manifest.json"


def main() -> None:
    benchmark_provenance = cast(
        dict[str, Any],
        json.loads(json.dumps(collect_manifest_provenance(ROOT))),
    )
    spec = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    dataset = load_wands_config(ROOT / "config/datasets.toml")
    preparation_evidence = verify_wands_preparation_evidence(
        root=ROOT,
        dataset_spec=dataset,
        prepared_directory=ROOT / "data/prepared/wands",
    )
    model_artifact = cast(dict[str, Any], read_json(MODEL_ARTIFACT))
    model_id = str(model_artifact["model_id"])
    pipeline = build_sparse_ingest_pipeline(spec, model_id=model_id)
    definition = build_sparse_index_definition(spec, use_default_pipeline=False)
    products_path = ROOT / "data/prepared/wands/products.jsonl"
    embeddings_path = ROOT / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl"
    precompute_manifest_path = (
        ROOT / "results/wands/neural-sparse/precompute-manifest.json"
    )
    precompute_evidence = verify_wands_sparse_precompute(
        root=ROOT,
        dataset=dataset,
        spec=spec,
        products_path=products_path,
        embeddings_path=embeddings_path,
        manifest_path=precompute_manifest_path,
    )
    url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    with OpenSearchClient(url, timeout=300) as client:
        version = client.wait_until_ready(expected_version=EXPECTED_VERSION)
        live_model = cast(
            dict[str, Any],
            client.request("GET", f"/_plugins/_ml/models/{model_id}"),
        )
        if live_model.get("model_state") != "DEPLOYED":
            raise RuntimeError("neural-sparse model is not deployed")
        client.request(
            "PUT",
            f"/_ingest/pipeline/{spec.ingest_pipeline}",
            json_body=pipeline,
        )
        client.delete_index(spec.index_name)
        client.create_index(spec.index_name, definition)
        client.bulk_index(
            spec.index_name,
            iter_precomputed_sparse_documents(
                products_path,
                embeddings_path,
                spec,
            ),
            batch_size=spec.bulk_request_size,
        )
        client.refresh(spec.index_name)
        client.force_merge(spec.index_name)
        client.update_index_settings(
            spec.index_name,
            {"index": {"refresh_interval": "1s"}},
        )
        index_write_block = lock_live_index_for_decisions(
            client,
            index=spec.index_name,
        )
        facts = collect_index_facts(client, spec.index_name)
        index_content = verify_live_index_content(
            client,
            index=spec.index_name,
            expected_documents=iter_precomputed_sparse_documents(
                products_path,
                embeddings_path,
                spec,
            ),
            expected_count=dataset.expected_products,
            float32_fields=(spec.embedding_field,),
        )
    if facts.document_count != dataset.expected_products:
        raise RuntimeError("neural-sparse index contains the wrong product count")
    provenance_eligible, completion_revision = wands_completion_provenance_evidence(
        ROOT,
        benchmark_provenance,
    )
    quality_eligible = (
        provenance_eligible
        and preparation_evidence.get("eligible_for_decision") is True
        and precompute_evidence.get("eligible_for_decision") is True
    )
    manifest = {
        "schema_version": 2,
        "dataset": "WANDS",
        "source_revision": dataset.revision,
        "opensearch_version": version,
        "index": asdict(facts),
        "index_definition_sha256": canonical_sha256(definition),
        "index_content": index_content,
        "index_write_block": index_write_block,
        "ingest_pipeline": pipeline,
        "ingest_pipeline_sha256": canonical_sha256(pipeline),
        "model_id": model_id,
        "model_artifact_sha256": file_facts(MODEL_ARTIFACT).sha256,
        "prepared_products_sha256": file_facts(products_path).sha256,
        "precomputed_embeddings_sha256": file_facts(embeddings_path).sha256,
        "precompute_manifest_sha256": file_facts(precompute_manifest_path).sha256,
        "precompute_evidence": precompute_evidence,
        "preparation_evidence": preparation_evidence,
        "query_mode": "doc_only_builtin_analyzer",
        "query_analyzer": spec.query_analyzer,
        "benchmark_provenance": benchmark_provenance,
        "completion_code_revision": completion_revision,
        "quality_evidence_eligible_for_decision": quality_eligible,
        "latency_evidence_eligible_for_decision": False,
    }
    _write_json_exact(INDEX_ARTIFACT, manifest)
    print(
        f"indexed neural-sparse WANDS: {facts.document_count} products, "
        f"{facts.segment_count} segment, model={model_id}, "
        f"quality_eligible={quality_eligible}"
    )


def _write_json_exact(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()
