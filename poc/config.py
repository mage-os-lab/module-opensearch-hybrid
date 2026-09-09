from __future__ import annotations

import re
import string
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when benchmark configuration is ambiguous or non-reproducible."""


_PINNED_REVISION = re.compile(r"[0-9a-f]{40}")
_REQUIRED_MODEL_KEYS = {
    "hf_id",
    "revision",
    "dims",
    "normalize",
    "max_seq_length",
    "bulk_dtype",
    "use_memory_efficient_attention",
    "trust_remote_code",
    "query_prefix",
    "document_prefix",
    "query_template",
    "document_template",
    "license",
    "decision_eligible",
    "contamination",
}

WANDS_FINALIST_MODEL_NAMES = (
    "gte_modernbert_base",
    "granite_embedding_english_r2",
    "arctic_embed_m_v2",
)

WANDS_INT8_MODEL_NAMES = (
    "gte_modernbert_base",
    "granite_embedding_english_r2",
    "arctic_embed_m_v2",
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    hf_id: str
    revision: str
    dims: int
    normalize: bool
    max_seq_length: int
    bulk_dtype: str
    use_memory_efficient_attention: bool
    trust_remote_code: bool
    query_prefix: str
    document_prefix: str
    query_template: str
    document_template: str
    license: str
    decision_eligible: bool
    contamination: str

    @classmethod
    def from_mapping(cls, name: str, values: dict[str, Any]) -> ModelSpec:
        missing = sorted(_REQUIRED_MODEL_KEYS - values.keys())
        if missing:
            raise ConfigError(f"model {name!r} is missing mandatory keys: {', '.join(missing)}")

        revision = _require_str(name, "revision", values["revision"])
        if _PINNED_REVISION.fullmatch(revision) is None:
            raise ConfigError(
                f"model {name!r} revision must be an immutable 40-character commit SHA"
            )

        dims = values["dims"]
        if not isinstance(dims, int) or isinstance(dims, bool) or dims <= 0:
            raise ConfigError(f"model {name!r} dims must be a positive integer")

        normalize = _require_bool(name, "normalize", values["normalize"])
        max_seq_length = _require_positive_int(name, "max_seq_length", values["max_seq_length"])
        bulk_dtype = _require_str(name, "bulk_dtype", values["bulk_dtype"])
        if bulk_dtype != "float32":
            raise ConfigError(f"model {name!r} bulk_dtype must be float32 for the MPS preflight")
        use_memory_efficient_attention = _require_bool(
            name,
            "use_memory_efficient_attention",
            values["use_memory_efficient_attention"],
        )
        trust_remote_code = _require_bool(name, "trust_remote_code", values["trust_remote_code"])
        decision_eligible = _require_bool(name, "decision_eligible", values["decision_eligible"])
        query_template = _require_str(name, "query_template", values["query_template"])
        document_template = _require_str(name, "document_template", values["document_template"])
        _validate_template(name, "query_template", query_template, {"query"})
        _validate_template(
            name,
            "document_template",
            document_template,
            {"title", "description", "attributes"},
        )

        return cls(
            name=name,
            hf_id=_require_str(name, "hf_id", values["hf_id"]),
            revision=revision,
            dims=dims,
            normalize=normalize,
            max_seq_length=max_seq_length,
            bulk_dtype=bulk_dtype,
            use_memory_efficient_attention=use_memory_efficient_attention,
            trust_remote_code=trust_remote_code,
            query_prefix=_require_str(name, "query_prefix", values["query_prefix"]),
            document_prefix=_require_str(name, "document_prefix", values["document_prefix"]),
            query_template=query_template,
            document_template=document_template,
            license=_require_str(name, "license", values["license"]),
            decision_eligible=decision_eligible,
            contamination=_require_str(name, "contamination", values["contamination"]),
        )

    def render_query(self, query: str) -> str:
        return self.query_prefix + self.query_template.format(query=query)

    def render_document(self, *, title: str, description: str = "", attributes: str = "") -> str:
        return self.document_prefix + self.document_template.format(
            title=title,
            description=description,
            attributes=attributes,
        )


def load_model_registry(path: Path | str = Path("config/models.toml")) -> dict[str, ModelSpec]:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    if raw.get("schema_version") != 1:
        raise ConfigError(f"unsupported model registry schema in {config_path}")
    raw_models = raw.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise ConfigError(f"model registry {config_path} contains no models")

    registry: dict[str, ModelSpec] = {}
    for name, values in raw_models.items():
        if not isinstance(name, str) or not isinstance(values, dict):
            raise ConfigError(f"invalid model entry in {config_path}")
        registry[name] = ModelSpec.from_mapping(name, values)
    return registry


def _require_str(model: str, key: str, value: object) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"model {model!r} {key} must be a string")
    return value


def _require_bool(model: str, key: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"model {model!r} {key} must be a boolean")
    return value


def _require_positive_int(model: str, key: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"model {model!r} {key} must be a positive integer")
    return value


def _validate_template(model: str, key: str, template: str, allowed: set[str]) -> None:
    fields = {
        field_name
        for _, field_name, _, _ in string.Formatter().parse(template)
        if field_name is not None
    }
    unknown = sorted(fields - allowed)
    if unknown:
        raise ConfigError(f"model {model!r} {key} has unknown fields: {', '.join(unknown)}")
    if not fields:
        raise ConfigError(f"model {model!r} {key} must render at least one input field")
