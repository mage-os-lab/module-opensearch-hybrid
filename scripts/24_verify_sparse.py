from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from poc.datasets import DatasetIntegrityError, file_facts, load_wands_config
from poc.index_content import verify_live_index_content
from poc.index_evidence import INDEX_WRITE_BLOCK_EVIDENCE, verify_live_index_settings
from poc.indexing import (
    verify_wands_preparation_evidence,
    wands_recorded_provenance_valid,
)
from poc.manifest import canonical_sha256, collect_index_facts, read_json, write_json
from poc.neural_sparse import (
    build_sparse_index_definition,
    build_sparse_ingest_pipeline,
    collect_wands_sparse_tokenizer_probe,
    iter_precomputed_sparse_documents,
    load_neural_sparse_spec,
    verify_wands_sparse_precompute,
    verify_wands_sparse_tokenizer_probe,
)
from poc.os_client import OpenSearchClient
from poc.provenance import registered_opensearch_url

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    spec = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    dataset = load_wands_config(ROOT / "config/datasets.toml")
    manifest_path = ROOT / "results/wands/neural-sparse/index-manifest.json"
    manifest = cast(dict[str, Any], read_json(manifest_path))
    model_artifact_path = ROOT / "results/wands/neural-sparse/model-deployment.json"
    model_artifact = cast(dict[str, Any], read_json(model_artifact_path))
    embeddings_path = ROOT / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl"
    precompute_manifest_path = (
        ROOT / "results/wands/neural-sparse/precompute-manifest.json"
    )
    model_id = str(model_artifact["model_id"])
    definition = build_sparse_index_definition(spec, use_default_pipeline=False)
    pipeline = build_sparse_ingest_pipeline(spec, model_id=model_id)
    preparation_evidence = verify_wands_preparation_evidence(
        root=ROOT,
        dataset_spec=dataset,
        prepared_directory=ROOT / "data/prepared/wands",
    )
    precompute_evidence = verify_wands_sparse_precompute(
        root=ROOT,
        dataset=dataset,
        spec=spec,
        products_path=ROOT / "data/prepared/wands/products.jsonl",
        embeddings_path=embeddings_path,
        manifest_path=precompute_manifest_path,
    )
    url = registered_opensearch_url(
        ROOT / "config/benchmark.toml",
        environment_variable="OPENSEARCH_HYBRID_OS_URL",
    )
    with OpenSearchClient(url, timeout=120) as client:
        facts = collect_index_facts(client, spec.index_name)
        if facts.document_count != dataset.expected_products:
            raise DatasetIntegrityError("live neural-sparse index product count differs")
        mapping = cast(
            dict[str, Any], client.request("GET", f"/{spec.index_name}/_mapping")
        )
        live_mapping = cast(dict[str, Any], mapping[spec.index_name])["mappings"]
        if live_mapping != definition["mappings"]:
            raise DatasetIntegrityError("live neural-sparse mapping differs")
        verify_live_index_settings(
            client,
            index=spec.index_name,
            definition=definition,
        )
        index_content = verify_live_index_content(
            client,
            index=spec.index_name,
            expected_documents=iter_precomputed_sparse_documents(
                ROOT / "data/prepared/wands/products.jsonl",
                embeddings_path,
                spec,
            ),
            expected_count=dataset.expected_products,
            float32_fields=(spec.embedding_field,),
        )
        if manifest.get("index_content") != index_content:
            raise DatasetIntegrityError("live neural-sparse index content differs")
        live_pipeline = cast(
            dict[str, Any],
            client.request("GET", f"/_ingest/pipeline/{spec.ingest_pipeline}"),
        )[spec.ingest_pipeline]
        live_pipeline.pop("version", None)
        if live_pipeline != pipeline:
            raise DatasetIntegrityError("live neural-sparse ingest pipeline differs")
        probe = collect_wands_sparse_tokenizer_probe(
            client,
            root=ROOT,
            spec=spec,
        )
    expected_manifest = {
        "schema_version": 2,
        "dataset": "WANDS",
        "source_revision": dataset.revision,
        "index": asdict(facts),
        "index_definition_sha256": canonical_sha256(definition),
        "index_write_block": INDEX_WRITE_BLOCK_EVIDENCE,
        "ingest_pipeline_sha256": canonical_sha256(pipeline),
        "model_id": model_id,
        "model_artifact_sha256": file_facts(model_artifact_path).sha256,
        "prepared_products_sha256": file_facts(
            ROOT / "data/prepared/wands/products.jsonl"
        ).sha256,
        "precomputed_embeddings_sha256": file_facts(embeddings_path).sha256,
        "precompute_manifest_sha256": file_facts(precompute_manifest_path).sha256,
        "precompute_evidence": precompute_evidence,
        "preparation_evidence": preparation_evidence,
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise DatasetIntegrityError(f"neural-sparse manifest {key} differs")
    quality_eligible = (
        wands_recorded_provenance_valid(
            ROOT,
            provenance=manifest.get("benchmark_provenance"),
            completion_revision=manifest.get("completion_code_revision"),
        )
        and preparation_evidence.get("eligible_for_decision") is True
        and precompute_evidence.get("eligible_for_decision") is True
    )
    if (
        manifest.get("quality_evidence_eligible_for_decision") is not quality_eligible
        or manifest.get("latency_evidence_eligible_for_decision") is not False
    ):
        raise DatasetIntegrityError("neural-sparse index eligibility differs")
    inspection = {
        "schema_version": 2,
        **probe,
        "status": "passed",
        "index_manifest": {
            "path": str(manifest_path.relative_to(ROOT)),
            "sha256": file_facts(manifest_path).sha256,
            "bytes": file_facts(manifest_path).bytes,
        },
        "quality_evidence_eligible_for_decision": quality_eligible,
    }
    write_json(
        ROOT / "results/wands/neural-sparse/tokenizer-inspection.json",
        inspection,
    )
    with OpenSearchClient(url, timeout=120) as client:
        verify_wands_sparse_tokenizer_probe(
            client,
            root=ROOT,
            spec=spec,
            index_evidence={
                "artifact": inspection["index_manifest"],
                "eligible_for_decision": quality_eligible,
            },
        )
    print(
        f"verified neural-sparse index {facts.uuid}: {facts.document_count} products; "
        f"tokenizer overlap={len(cast(list[str], probe['overlap']))}; "
        f"quality_eligible={quality_eligible}"
    )


if __name__ == "__main__":
    main()
