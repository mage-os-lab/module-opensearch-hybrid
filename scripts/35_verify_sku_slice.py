from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from poc.datasets import file_facts
from poc.manifest import read_json
from poc.sku_slice import load_sku_queries, load_sku_slice_spec
from poc.trec import read_qrels

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    spec = load_sku_slice_spec(ROOT / "config/sku_slice.toml")
    manifest = cast(
        dict[str, Any], read_json(ROOT / "results/wands/sku-slice/preparation.json")
    )
    query_artifact = cast(dict[str, Any], manifest["queries"])
    qrels_artifact = cast(dict[str, Any], manifest["qrels"])
    query_path = ROOT / str(query_artifact["path"])
    qrels_path = ROOT / str(qrels_artifact["path"])
    queries = load_sku_queries(query_path)
    qrels = read_qrels(qrels_path)
    if len(queries) != spec.query_count or len(qrels) != spec.query_count:
        raise ValueError("synthetic SKU slice count differs")
    if query_artifact.get("sha256") != file_facts(query_path).sha256:
        raise ValueError("synthetic SKU query hash differs")
    if qrels_artifact.get("sha256") != file_facts(qrels_path).sha256:
        raise ValueError("synthetic SKU qrels hash differs")
    expected_pairs = {
        (query_id, query["expected_document_id"])
        for query_id, query in queries.items()
    }
    actual_pairs = {(qrel.query_id, qrel.document_id) for qrel in qrels}
    if actual_pairs != expected_pairs or any(qrel.relevance != 2 for qrel in qrels):
        raise ValueError("synthetic SKU qrels differ from prepared targets")
    print(f"verified {len(queries)} synthetic SKU and known-item queries")


if __name__ == "__main__":
    main()
