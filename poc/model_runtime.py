from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.metadata
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from poc.config import ConfigError, ModelSpec
from poc.datasets import DatasetIntegrityError

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ModelArtifactPin:
    hf_id: str
    revision: str
    snapshot_sha256: str
    files: tuple[str, ...]


class SentenceTransformerBackend:
    def __init__(self, *, root: Path, model: ModelSpec, device: str) -> None:
        metal_validation_disabled = False
        if is_mps_device(device):
            metal_validation_disabled = prepare_mps_environment()
        cache_root = root / "data/cache/models"
        sentence_transformer_cache = cache_root / "sentence-transformers"
        module_cache = cache_root / "modules"
        sentence_transformer_cache.mkdir(parents=True, exist_ok=True)
        module_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_root / "huggingface-home"))
        os.environ.setdefault("HF_MODULES_CACHE", str(module_cache))

        sentence_transformers = importlib.import_module("sentence_transformers")
        sentence_transformer_class = sentence_transformers.SentenceTransformer
        model_source, local_files_only = resolve_cached_model_source(
            sentence_transformer_cache,
            model,
            root=root,
        )
        self._encoder: Any = sentence_transformer_class(
            model_source,
            revision=model.revision,
            cache_folder=str(sentence_transformer_cache),
            trust_remote_code=model.trust_remote_code,
            device=device,
            model_kwargs={"dtype": model.bulk_dtype},
            config_kwargs=(
                {"use_memory_efficient_attention": model.use_memory_efficient_attention}
                if model.trust_remote_code
                else None
            ),
            local_files_only=local_files_only,
        )
        self._encoder.max_seq_length = model.max_seq_length
        dimensions = self._encoder.get_embedding_dimension()
        if dimensions != model.dims:
            raise DatasetIntegrityError(
                f"{model.name} reports {dimensions} dimensions, expected {model.dims}"
            )
        self._model = model
        self.device = str(self._encoder.device)
        patched_layer_norms = 0
        if is_mps_device(self.device):
            patched_layer_norms = patch_biasless_layer_norms(self._encoder)
        self.runtime = (
            f"sentence-transformers/{importlib.metadata.version('sentence-transformers')} "
            f"torch/{importlib.metadata.version('torch')} "
            f"mps_zero_bias_layer_norms/{patched_layer_norms} "
            f"metal_debug_validation_disabled/{int(metal_validation_disabled)}"
        )
        snapshot = _find_snapshot(sentence_transformer_cache, model)
        self.model_artifact_sha256 = verify_registered_model_snapshot(
            root=root,
            model=model,
            snapshot=snapshot,
        )

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, self._model.dims), dtype=np.float32)
        values = self._encoder.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self._model.normalize,
        )
        return np.asarray(values, dtype=np.float32)


class TransformersClsBackend:
    def __init__(self, *, root: Path, model: ModelSpec, device: str) -> None:
        metal_validation_disabled = False
        if is_mps_device(device):
            metal_validation_disabled = prepare_mps_environment()
        cache_root = root / "data/cache/models"
        transformer_cache = cache_root / "sentence-transformers"
        module_cache = cache_root / "modules"
        transformer_cache.mkdir(parents=True, exist_ok=True)
        module_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_root / "huggingface-home"))
        os.environ.setdefault("HF_MODULES_CACHE", str(module_cache))

        transformers = importlib.import_module("transformers")
        torch = importlib.import_module("torch")
        model_source, local_files_only = resolve_cached_model_source(
            transformer_cache,
            model,
            root=root,
            required_files=("config.json", "model.safetensors", "tokenizer.json"),
        )
        common = {
            "revision": model.revision,
            "cache_dir": str(transformer_cache),
            "trust_remote_code": model.trust_remote_code,
            "local_files_only": local_files_only,
        }
        config = transformers.AutoConfig.from_pretrained(
            model_source,
            use_memory_efficient_attention=model.use_memory_efficient_attention,
            **common,
        )
        self._tokenizer: Any = transformers.AutoTokenizer.from_pretrained(
            model_source,
            **common,
        )
        self._encoder: Any = transformers.AutoModel.from_pretrained(
            model_source,
            config=config,
            add_pooling_layer=False,
            dtype=model.bulk_dtype,
            **common,
        )
        reset_position_ids = reset_position_id_buffers(self._encoder)
        self._encoder.eval()
        self._encoder.to(device)
        self._torch = torch
        self._model = model
        self.device = str(next(self._encoder.parameters()).device)
        patched_layer_norms = 0
        if is_mps_device(self.device):
            patched_layer_norms = patch_biasless_layer_norms(self._encoder)
        self.runtime = (
            f"transformers/{importlib.metadata.version('transformers')} "
            f"torch/{importlib.metadata.version('torch')} cls_mrl/{model.dims} "
            f"position_id_buffers_reset/{reset_position_ids} "
            f"mps_zero_bias_layer_norms/{patched_layer_norms} "
            f"metal_debug_validation_disabled/{int(metal_validation_disabled)}"
        )
        self.model_artifact_sha256 = verify_registered_model_snapshot(
            root=root,
            model=model,
            snapshot=_find_snapshot(transformer_cache, model),
        )

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, self._model.dims), dtype=np.float32)
        tokens = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._model.max_seq_length,
            return_tensors="pt",
        )
        device_tokens = {name: value.to(self.device) for name, value in tokens.items()}
        with self._torch.no_grad():
            hidden_state = self._encoder(**device_tokens)[0]
        return finalize_cls_embeddings(
            hidden_state,
            dims=self._model.dims,
            normalize=self._model.normalize,
        )

    def onnx_export_components(self) -> tuple[Any, Any, Any]:
        return self._encoder, self._tokenizer, self._torch


def create_bulk_backend(*, root: Path, model: ModelSpec, device: str) -> Any:
    if model.name == "arctic_embed_m_v2":
        return TransformersClsBackend(root=root, model=model, device=device)
    return SentenceTransformerBackend(root=root, model=model, device=device)


def resolve_cached_model_source(
    cache_root: Path,
    model: ModelSpec,
    *,
    root: Path | None = None,
    required_files: tuple[str, ...] = (),
) -> tuple[str, bool]:
    expected_snapshot = _expected_snapshot(cache_root, model)
    cached = expected_snapshot.is_dir() and all(
        (expected_snapshot / name).exists() for name in required_files
    )
    if cached and root is not None:
        verify_registered_model_snapshot(
            root=root,
            model=model,
            snapshot=expected_snapshot,
        )
    return (str(expected_snapshot), True) if cached else (model.hf_id, False)


def select_bulk_device(requested: str = "auto") -> str:
    if requested not in {"auto", "mps", "cpu"}:
        raise ValueError(f"unsupported embedding device {requested!r}")
    prepare_mps_environment()
    torch = importlib.import_module("torch")
    mps_available = bool(torch.backends.mps.is_available())
    if requested == "mps" and not mps_available:
        raise DatasetIntegrityError("MPS was requested but is not available")
    if requested == "auto":
        return "mps" if mps_available else "cpu"
    return requested


def is_mps_device(device: str) -> bool:
    return device == "mps" or device.startswith("mps:")


def prepare_mps_environment() -> bool:
    marker = "OPENSEARCH_HYBRID_METAL_VALIDATION_DISABLED"
    if os.environ.pop("METAL_DEVICE_WRAPPER_TYPE", None) is not None:
        os.environ[marker] = "1"
    return os.environ.get(marker) == "1"


def release_device_memory() -> None:
    gc.collect()
    try:
        torch = importlib.import_module("torch")
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        return


def patch_biasless_layer_norms(model: Any) -> int:
    torch = importlib.import_module("torch")
    patched = 0
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm) and module.bias is None:
            if module.weight is None:
                raise DatasetIntegrityError("cannot patch LayerNorm without a weight tensor")
            zeros = torch.zeros(
                module.normalized_shape,
                dtype=module.weight.dtype,
                device=module.weight.device,
            )
            module.bias = torch.nn.Parameter(zeros, requires_grad=False)
            patched += 1
    return patched


def reset_position_id_buffers(model: Any) -> int:
    torch = importlib.import_module("torch")
    reset = 0
    for module in model.modules():
        position_ids = module._buffers.get("position_ids")
        if position_ids is None:
            continue
        if position_ids.ndim != 1:
            raise DatasetIntegrityError("position_ids buffer must be one-dimensional")
        module.position_ids = torch.arange(
            position_ids.shape[0],
            dtype=position_ids.dtype,
            device=position_ids.device,
        )
        reset += 1
    return reset


def compare_embedding_vectors(
    reference: NDArray[np.float32],
    candidate: NDArray[np.float32],
) -> dict[str, float]:
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise DatasetIntegrityError(
            f"embedding parity shapes differ: {reference.shape} versus {candidate.shape}"
        )
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise DatasetIntegrityError("embedding parity inputs contain non-finite values")
    reference_norms = np.linalg.norm(reference, axis=1)
    candidate_norms = np.linalg.norm(candidate, axis=1)
    similarities = np.sum(reference * candidate, axis=1) / (reference_norms * candidate_norms)
    return {
        "maximum_absolute_delta": float(np.max(np.abs(reference - candidate))),
        "minimum_cosine_similarity": float(np.min(similarities)),
        "mean_cosine_similarity": float(np.mean(similarities)),
    }


def finalize_cls_embeddings(
    hidden_state: Any,
    *,
    dims: int,
    normalize: bool,
) -> NDArray[np.float32]:
    torch = importlib.import_module("torch")
    vectors = hidden_state[:, 0, :dims]
    if normalize:
        vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
    return np.asarray(vectors.detach().to(device="cpu", dtype=torch.float32).numpy())


def hash_model_snapshot(snapshot: Path) -> str:
    if not snapshot.is_dir():
        raise DatasetIntegrityError(f"model snapshot does not exist: {snapshot}")
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    if not files:
        raise DatasetIntegrityError(f"model snapshot has no files: {snapshot}")
    digest = hashlib.sha256()
    for path in files:
        relative = str(path.relative_to(snapshot)).encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def load_model_artifact_pins(path: Path) -> dict[str, ModelArtifactPin]:
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot load model artifact registry {path}: {error}") from error
    models = raw.get("models")
    if raw.get("schema_version") != 1 or not isinstance(models, dict) or not models:
        raise ConfigError("model artifact registry schema differs")
    pins: dict[str, ModelArtifactPin] = {}
    for name, value in models.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            raise ConfigError("model artifact registry entry differs")
        hf_id = value.get("hf_id")
        revision = value.get("revision")
        snapshot_sha256 = value.get("snapshot_sha256")
        files = value.get("files")
        if (
            not isinstance(hf_id, str)
            or not hf_id
            or not isinstance(revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", revision) is None
            or not isinstance(snapshot_sha256, str)
            or _SHA256.fullmatch(snapshot_sha256) is None
            or not isinstance(files, list)
            or not files
            or not all(isinstance(item, str) and item for item in files)
            or len(files) != len(set(files))
            or files != sorted(files)
        ):
            raise ConfigError(f"model artifact pin differs for {name}")
        pins[name] = ModelArtifactPin(
            hf_id=hf_id,
            revision=revision,
            snapshot_sha256=snapshot_sha256,
            files=tuple(files),
        )
    return pins


def verify_registered_model_snapshot(
    *,
    root: Path,
    model: ModelSpec,
    snapshot: Path | None = None,
) -> str:
    pins = load_model_artifact_pins(root / "config/model_artifacts.toml")
    pin = pins.get(model.name)
    if pin is None:
        raise DatasetIntegrityError(
            f"model {model.name} has no registered snapshot artifact"
        )
    if pin.hf_id != model.hf_id or pin.revision != model.revision:
        raise DatasetIntegrityError(
            f"registered snapshot identity differs for {model.name}"
        )
    snapshot = snapshot or model_snapshot_directory(root, model)
    actual_files = tuple(
        sorted(
            path.relative_to(snapshot).as_posix()
            for path in snapshot.rglob("*")
            if path.is_file()
        )
    )
    if actual_files != pin.files:
        raise DatasetIntegrityError(
            f"registered snapshot file set differs for {model.name}"
        )
    actual_sha256 = hash_model_snapshot(snapshot)
    if actual_sha256 != pin.snapshot_sha256:
        raise DatasetIntegrityError(
            f"registered snapshot bytes differ for {model.name}"
        )
    return actual_sha256


def model_snapshot_directory(root: Path, model: ModelSpec) -> Path:
    cache_root = root / "data/cache/models/sentence-transformers"
    return _expected_snapshot(cache_root, model)


def _find_snapshot(cache_root: Path, model: ModelSpec) -> Path:
    snapshot = _expected_snapshot(cache_root, model)
    if not snapshot.is_dir():
        raise DatasetIntegrityError(
            f"pinned model snapshot was not cached at expected revision: {snapshot}"
        )
    return snapshot


def _expected_snapshot(cache_root: Path, model: ModelSpec) -> Path:
    repository_directory = "models--" + model.hf_id.replace("/", "--")
    return cache_root / repository_directory / "snapshots" / model.revision
