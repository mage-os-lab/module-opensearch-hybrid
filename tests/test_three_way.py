from __future__ import annotations

from pathlib import Path

from poc.config import load_model_registry
from poc.neural_sparse import load_neural_sparse_spec
from poc.three_way import load_three_way_spec
from poc.three_way_indexing import build_three_way_index_definition

ROOT = Path(__file__).resolve().parents[1]


def test_three_way_grid_has_equal_budget_and_normalized_distinct_weights() -> None:
    spec = load_three_way_spec(ROOT / "config/three_way.toml")

    assert spec.index_name == "opensearch-hybrid-wands-dense-sparse-v1"
    assert spec.dense_model == "arctic_embed_m_v2"
    assert len(spec.candidates) == 5
    assert len({candidate.weights for candidate in spec.candidates}) == 5
    assert all(abs(sum(candidate.weights) - 1.0) < 1e-12 for candidate in spec.candidates)


def test_three_way_index_contains_exact_dense_and_rank_features_fields() -> None:
    spec = load_three_way_spec(ROOT / "config/three_way.toml")
    sparse = load_neural_sparse_spec(ROOT / "config/neural_sparse.toml")
    model = load_model_registry(ROOT / "config/models.toml")[spec.dense_model]

    definition = build_three_way_index_definition(spec, sparse, model)
    properties = definition["mappings"]["properties"]

    assert definition["settings"]["index"]["knn"] is True
    assert properties["embedding_arctic_embed_m_v2"] == {
        "type": "knn_vector",
        "dimension": 256,
        "space_type": "cosinesimil",
        "method": {
            "name": "hnsw",
            "engine": "lucene",
            "space_type": "cosinesimil",
            "parameters": {"ef_construction": 128, "m": 16},
        },
    }
    assert properties[sparse.embedding_field] == {"type": "rank_features"}
