from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from poc.datasets import file_facts, load_wands_config
from poc.indexing import iter_prepared_products
from poc.manifest import write_json
from poc.neural_sparse import (
    WANDS_SPARSE_MAXIMUM_TOKEN_LENGTH,
    WandsSparseEncoder,
    canonical_sparse_embedding_line,
    create_wands_sparse_encoder,
    load_neural_sparse_spec,
    sparse_document_text,
    verify_neural_sparse_model_artifacts,
    verify_wands_sparse_precompute,
    wands_sparse_pruning_recipe,
    wands_sparse_replay_contract,
)
from poc.provenance import collect_manifest_provenance, verify_decision_provenance

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = ROOT / "data/cache/models/neural-sparse/doc-v3-distill"
MODEL_PATH = MODEL_DIRECTORY / "opensearch-neural-sparse-encoding-doc-v3-distill.pt"
TOKENIZER_PATH = MODEL_DIRECTORY / "tokenizer.json"
PRODUCTS_PATH = ROOT / "data/prepared/wands/products.jsonl"
OUTPUT = ROOT / "data/cache/neural-sparse/wands-doc-v3-distill.jsonl"
MANIFEST = ROOT / "results/wands/neural-sparse/precompute-manifest.json"


def main() -> None:
    generation_provenance = collect_manifest_provenance(
        ROOT,
        profile_path=ROOT / "config/benchmark.toml",
        environment_path=ROOT / "results/environment/benchmark-profile.json",
    )
    spec = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    dataset = load_wands_config(ROOT / "config/datasets.toml")
    model_artifacts = verify_neural_sparse_model_artifacts(
        spec,
        root=ROOT,
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
    )
    encoder = create_wands_sparse_encoder(
        spec=spec,
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    digest = hashlib.sha256()
    records = 0
    started = time.perf_counter()
    pending: list[tuple[str, str]] = []
    with temporary.open("w", encoding="utf-8") as handle:
        for product_id, document in iter_prepared_products(PRODUCTS_PATH):
            pending.append((product_id, sparse_document_text(document)))
            if len(pending) == spec.inference_batch_size:
                records += _encode_batch(
                    pending,
                    encoder=encoder,
                    handle=handle,
                    digest=digest,
                )
                pending.clear()
        if pending:
            records += _encode_batch(
                pending,
                encoder=encoder,
                handle=handle,
                digest=digest,
            )
    temporary.replace(OUTPUT)
    elapsed = time.perf_counter() - started
    provenance_eligible = verify_decision_provenance(
        generation_provenance,
        root=ROOT,
        profile_path=ROOT / "config/benchmark.toml",
        environment_path=ROOT / "results/environment/benchmark-profile.json",
        require_current_code_revision=True,
    )
    write_json(
        MANIFEST,
        {
            "schema_version": 2,
            "created_at": datetime.now(UTC).isoformat(),
            "model": spec.model_name,
            "model_version": spec.model_version,
            **model_artifacts,
            "products_sha256": file_facts(PRODUCTS_PATH).sha256,
            "runtime": encoder.runtime,
            "mps_status": "unsupported_traced_cpu_constants_and_pytorch_shader_assertion",
            "batch_size": spec.inference_batch_size,
            "maximum_token_length": WANDS_SPARSE_MAXIMUM_TOKEN_LENGTH,
            "pruning": wands_sparse_pruning_recipe(spec),
            "deterministic_replay": wands_sparse_replay_contract(
                products_path=PRODUCTS_PATH,
                dataset=dataset,
                spec=spec,
            ),
            "document_recipe": (
                "title.brand.category.product_class.features.description"
            ),
            "records": records,
            "elapsed_seconds": elapsed,
            "embeddings": {
                "path": str(OUTPUT.relative_to(ROOT)),
                "sha256": digest.hexdigest(),
                "file_sha256": file_facts(OUTPUT).sha256,
                "bytes": file_facts(OUTPUT).bytes,
            },
            "generation_provenance": generation_provenance,
            "benchmark_provenance": generation_provenance,
            "generation_provenance_eligible_for_decision": provenance_eligible,
            "quality_evidence_eligible_for_decision": provenance_eligible,
            "quality_ineligibility_reason": (
                None
                if provenance_eligible
                else "precompute did not retain current decision-eligible start provenance"
            ),
        },
    )
    del encoder
    verify_wands_sparse_precompute(
        root=ROOT,
        dataset=dataset,
        spec=spec,
        products_path=PRODUCTS_PATH,
        embeddings_path=OUTPUT,
        manifest_path=MANIFEST,
    )
    print(f"precomputed {records} neural-sparse documents in {elapsed:.2f}s")


def _encode_batch(
    batch: list[tuple[str, str]],
    *,
    encoder: WandsSparseEncoder,
    handle: Any,
    digest: Any,
) -> int:
    encoded = encoder.encode(batch)
    if len(encoded) != len(batch):
        raise ValueError("official sparse model returned the wrong batch size")
    for (expected_product_id, _), (product_id, embedding) in zip(
        batch, encoded, strict=True
    ):
        if product_id != expected_product_id:
            raise ValueError("official sparse model returned products out of order")
        line = canonical_sparse_embedding_line(product_id, embedding)
        handle.write(line)
        digest.update(line.encode())
    return len(batch)


if __name__ == "__main__":
    main()
