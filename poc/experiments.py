from __future__ import annotations

import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from poc.config import ConfigError
from poc.search import build_configured_bm25_query, build_integrator_bm25_query

_FIELD = re.compile(r"[a-z_]+(?:\.[a-z_]+)?(?:\^(?:\d+(?:\.\d+)?|\.\d+))?")


@dataclass(frozen=True, slots=True)
class BM25Profile:
    name: str
    fields: tuple[str, ...]
    query_type: str
    operator: str
    tie_breaker: float
    strategy: str = "multi_match"
    minimum_should_match: str | None = None
    phrase_fields: tuple[str, ...] = ()
    prefix_fields: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> BM25Profile:
        name = values.get("name")
        fields = values.get("fields")
        query_type = values.get("query_type")
        operator = values.get("operator")
        tie_breaker = values.get("tie_breaker")
        strategy = values.get("strategy", "multi_match")
        minimum_should_match = values.get("minimum_should_match")
        phrase_fields = values.get("phrase_fields", [])
        prefix_fields = values.get("prefix_fields", [])
        if not isinstance(name, str) or not name:
            raise ConfigError("BM25 profile name must be a non-empty string")
        if (
            not isinstance(fields, list)
            or not fields
            or not all(isinstance(field, str) and _FIELD.fullmatch(field) for field in fields)
        ):
            raise ConfigError(f"BM25 profile {name!r} has invalid fields")
        if len(set(fields)) != len(fields):
            raise ConfigError(f"BM25 profile {name!r} has duplicate fields")
        if query_type != "best_fields":
            raise ConfigError(f"BM25 profile {name!r} uses an unsupported query type")
        if operator not in {"or", "and"}:
            raise ConfigError(f"BM25 profile {name!r} has an invalid operator")
        if (
            not isinstance(tie_breaker, int | float)
            or isinstance(tie_breaker, bool)
            or not 0.0 <= float(tie_breaker) <= 1.0
        ):
            raise ConfigError(f"BM25 profile {name!r} has an invalid tie breaker")
        if strategy not in {"multi_match", "competent_integrator"}:
            raise ConfigError(f"BM25 profile {name!r} has an invalid strategy")
        for label, configured_fields in (
            ("phrase", phrase_fields),
            ("prefix", prefix_fields),
        ):
            if not isinstance(configured_fields, list) or not all(
                isinstance(field, str) and _FIELD.fullmatch(field)
                for field in configured_fields
            ):
                raise ConfigError(f"BM25 profile {name!r} has invalid {label} fields")
        if strategy == "competent_integrator" and (
            not isinstance(minimum_should_match, str)
            or not minimum_should_match
            or not phrase_fields
            or not prefix_fields
        ):
            raise ConfigError(
                f"BM25 profile {name!r} has an incomplete integrator query strategy"
            )
        if strategy == "multi_match" and (
            minimum_should_match is not None or phrase_fields or prefix_fields
        ):
            raise ConfigError(f"BM25 profile {name!r} mixes query strategies")
        return cls(
            name=name,
            fields=tuple(fields),
            query_type=query_type,
            operator=operator,
            tie_breaker=float(tie_breaker),
            strategy=strategy,
            minimum_should_match=minimum_should_match,
            phrase_fields=tuple(phrase_fields),
            prefix_fields=tuple(prefix_fields),
        )

    def query(self, text: str) -> dict[str, Any]:
        if self.strategy == "competent_integrator":
            if self.minimum_should_match is None:
                raise ConfigError("integrator profile has no minimum-should-match rule")
            return build_integrator_bm25_query(
                text,
                fields=self.fields,
                operator=self.operator,
                tie_breaker=self.tie_breaker,
                minimum_should_match=self.minimum_should_match,
                phrase_fields=self.phrase_fields,
                prefix_fields=self.prefix_fields,
            )
        return build_configured_bm25_query(
            text,
            fields=self.fields,
            query_type=self.query_type,
            operator=self.operator,
            tie_breaker=self.tie_breaker,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BM25Experiments:
    naive: BM25Profile
    magento_emulation: BM25Profile
    competent_integrator: BM25Profile
    tuning_candidates: tuple[BM25Profile, ...]
    hybrid_weight_count: int
    lexical_weight_grid: tuple[float, ...]
    rrf_rank_constants: tuple[int, ...]
    fusion_normalization: str
    fusion_combination: str
    primary_metric: str
    minimum_paired_delta: float
    alpha: float
    multiple_testing_correction: str
    maximum_trec_orphan_rate: float
    minimum_judged_at_10: float
    minimum_ann_recall_at_100: float
    ann_ef_search_grid: tuple[int, ...]


def load_bm25_experiments(
    path: Path | str = Path("config/experiments.toml"),
) -> BM25Experiments:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != 1:
        raise ConfigError(f"unsupported experiment schema in {config_path}")
    bm25 = raw.get("bm25")
    fusion = raw.get("fusion")
    decision = raw.get("decision")
    ann = raw.get("ann")
    if (
        not isinstance(bm25, dict)
        or not isinstance(fusion, dict)
        or not isinstance(decision, dict)
        or not isinstance(ann, dict)
    ):
        raise ConfigError("experiment config must contain BM25 and fusion sections")
    raw_naive = bm25.get("naive")
    raw_magento = bm25.get("magento_emulation")
    raw_competent = bm25.get("competent_integrator")
    raw_candidates = bm25.get("tuning_candidates")
    weight_grid = fusion.get("lexical_weight_grid")
    rrf_rank_constants = fusion.get("rrf_rank_constants")
    normalization = fusion.get("normalization")
    combination = fusion.get("combination")
    if (
        not isinstance(raw_naive, dict)
        or not isinstance(raw_magento, dict)
        or not isinstance(raw_competent, dict)
    ):
        raise ConfigError("experiment config is missing fixed BM25 profiles")
    if not isinstance(raw_candidates, list) or not all(
        isinstance(candidate, dict) for candidate in raw_candidates
    ):
        raise ConfigError("experiment config has invalid BM25 tuning candidates")
    if not isinstance(weight_grid, list) or not weight_grid:
        raise ConfigError("experiment config has no hybrid lexical weight grid")
    if (
        not all(isinstance(weight, float | int) for weight in weight_grid)
        or any(not 0.0 <= float(weight) <= 1.0 for weight in weight_grid)
        or len({float(weight) for weight in weight_grid}) != len(weight_grid)
    ):
        raise ConfigError("hybrid lexical weights must be unique values in [0, 1]")
    if (
        not isinstance(rrf_rank_constants, list)
        or not rrf_rank_constants
        or not all(
            isinstance(rank_constant, int) and not isinstance(rank_constant, bool)
            for rank_constant in rrf_rank_constants
        )
        or any(rank_constant < 1 for rank_constant in rrf_rank_constants)
        or len(set(rrf_rank_constants)) != len(rrf_rank_constants)
    ):
        raise ConfigError("RRF rank constants must be unique positive integers")
    if normalization != "min_max":
        raise ConfigError("registered fusion normalization must be min_max")
    if combination != "arithmetic_mean":
        raise ConfigError("registered fusion combination must be arithmetic_mean")
    primary_metric = decision.get("primary_metric")
    minimum_paired_delta = decision.get("minimum_paired_delta")
    alpha = decision.get("alpha")
    correction = decision.get("multiple_testing_correction")
    maximum_trec_orphan_rate = decision.get("maximum_trec_orphan_rate")
    minimum_judged_at_10 = decision.get("minimum_judged_at_10")
    minimum_ann_recall_at_100 = decision.get("minimum_ann_recall_at_100")
    ef_search_grid = ann.get("ef_search_grid")
    if primary_metric != "ndcg@10":
        raise ConfigError("registered primary metric must be ndcg@10")
    if not isinstance(minimum_paired_delta, float | int) or minimum_paired_delta <= 0:
        raise ConfigError("minimum paired delta must be positive")
    if not isinstance(alpha, float | int) or not 0 < alpha < 1:
        raise ConfigError("decision alpha must be in (0, 1)")
    if correction != "holm":
        raise ConfigError("registered multiple-testing correction must be holm")
    if (
        not isinstance(maximum_trec_orphan_rate, float | int)
        or isinstance(maximum_trec_orphan_rate, bool)
        or not 0.0 <= float(maximum_trec_orphan_rate) <= 1.0
    ):
        raise ConfigError("maximum TREC orphan rate must be in [0, 1]")
    if (
        not isinstance(minimum_judged_at_10, float | int)
        or isinstance(minimum_judged_at_10, bool)
        or not 0.0 <= float(minimum_judged_at_10) <= 1.0
    ):
        raise ConfigError("minimum judged@10 must be in [0, 1]")
    if (
        not isinstance(minimum_ann_recall_at_100, float | int)
        or isinstance(minimum_ann_recall_at_100, bool)
        or not 0.0 <= float(minimum_ann_recall_at_100) <= 1.0
    ):
        raise ConfigError("minimum ANN recall@100 must be in [0, 1]")
    if (
        not isinstance(ef_search_grid, list)
        or not ef_search_grid
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 100
            for value in ef_search_grid
        )
        or ef_search_grid != sorted(set(ef_search_grid))
    ):
        raise ConfigError("ANN ef_search grid must be sorted unique integers >= 100")

    naive = BM25Profile.from_mapping(raw_naive)
    magento = BM25Profile.from_mapping(raw_magento)
    competent = BM25Profile.from_mapping(raw_competent)
    candidates = tuple(BM25Profile.from_mapping(candidate) for candidate in raw_candidates)
    if len(candidates) != len(weight_grid) or len(candidates) != len(rrf_rank_constants):
        raise ConfigError("BM25 and hybrid query-layer tuning budgets must be equal")
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ConfigError("BM25 tuning candidate names must be unique")
    if candidates[0] != magento:
        raise ConfigError("the first BM25 tuning candidate must be Magento emulation")
    return BM25Experiments(
        naive=naive,
        magento_emulation=magento,
        competent_integrator=competent,
        tuning_candidates=candidates,
        hybrid_weight_count=len(weight_grid),
        lexical_weight_grid=tuple(float(weight) for weight in weight_grid),
        rrf_rank_constants=tuple(rrf_rank_constants),
        fusion_normalization=normalization,
        fusion_combination=combination,
        primary_metric=primary_metric,
        minimum_paired_delta=float(minimum_paired_delta),
        alpha=float(alpha),
        multiple_testing_correction=correction,
        maximum_trec_orphan_rate=float(maximum_trec_orphan_rate),
        minimum_judged_at_10=float(minimum_judged_at_10),
        minimum_ann_recall_at_100=float(minimum_ann_recall_at_100),
        ann_ef_search_grid=tuple(ef_search_grid),
    )
