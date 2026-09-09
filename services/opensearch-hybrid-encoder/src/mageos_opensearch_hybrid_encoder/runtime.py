"""Fail-closed ONNX runtime for a sealed production encoder identity."""

from __future__ import annotations

import importlib.metadata
import platform
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .identity import EXPECTED_MODEL, EXPECTED_RECIPE, seal_identity

EXPECTED_INPUTS = {"attention_mask", "input_ids"}
EXPECTED_OUTPUT = "sentence_embedding"
EXECUTION_PROVIDER = "CPUExecutionProvider"


class ProductionRuntimeError(RuntimeError):
    """The local process cannot prove the sealed production runtime contract."""


class Encoding(Protocol):
    ids: list[int]
    attention_mask: list[int]


class Tokenizer(Protocol):
    def encode_batch(self, texts: list[str]) -> Sequence[Encoding]: ...


class Node(Protocol):
    name: str


class Session(Protocol):
    def get_inputs(self) -> Sequence[Node]: ...

    def get_outputs(self) -> Sequence[Node]: ...

    def get_providers(self) -> list[str]: ...

    def run(
        self,
        output_names: list[str],
        input_feed: dict[str, NDArray[np.int64]],
    ) -> Sequence[Any]: ...


class ProductionEncoderRuntime:
    """Encode frozen Mage-OS query and document text through sealed ONNX bytes."""

    def __init__(
        self,
        *,
        identity: dict[str, object],
        tokenizer: Tokenizer,
        query_session: Session,
        document_session: Session,
    ) -> None:
        if (
            identity.get("production_eligible") is not True
            and identity.get("qualification_candidate") is not True
        ):
            raise ProductionRuntimeError("the sealed encoder identity is not production eligible")
        if identity.get("dimension") != EXPECTED_MODEL["dimension"]:
            raise ProductionRuntimeError("the sealed encoder dimension does not match the runtime")
        self._validate_session(query_session, "query")
        self._validate_session(document_session, "document")
        self.identity = identity
        self._tokenizer = tokenizer
        self._query_session = query_session
        self._document_session = document_session

    @classmethod
    def load(
        cls,
        manifest_path: Path,
        artifact_root: Path,
        expected_deployment_digest: str,
    ) -> ProductionEncoderRuntime:
        """Seal all release bytes, verify the process, and load both ONNX sessions."""
        identity = seal_identity(manifest_path, artifact_root)
        verify_runtime_environment(
            identity,
            expected_deployment_digest=expected_deployment_digest,
        )
        artifacts = cast(dict[str, dict[str, object]], identity["artifacts"])
        tokenizer_path = cls.artifact_path(
            artifact_root,
            artifacts["tokenizer"],
            "tokenizer",
        )
        query_model_path = cls.artifact_path(
            artifact_root,
            artifacts["query_model"],
            "query_model",
        )
        document_model_path = cls.artifact_path(
            artifact_root,
            artifacts["document_model"],
            "document_model",
        )

        return cls._load_from_paths(
            identity=identity,
            tokenizer_path=tokenizer_path,
            query_model_path=query_model_path,
            document_model_path=document_model_path,
        )

    @classmethod
    def load_qualification_artifacts(
        cls,
        *,
        tokenizer_path: Path,
        query_model_path: Path,
        document_model_path: Path,
    ) -> ProductionEncoderRuntime:
        """Load exact candidate bytes for offline qualification, never HTTP serving."""
        for label, path in {
            "tokenizer": tokenizer_path,
            "query_model": query_model_path,
            "document_model": document_model_path,
        }.items():
            if path.is_symlink() or not path.is_file():
                raise ProductionRuntimeError(
                    f"qualification {label} must be a regular non-symlink file"
                )
        candidate_identity: dict[str, object] = {
            "production_eligible": False,
            "qualification_candidate": True,
            "dimension": EXPECTED_MODEL["dimension"],
        }
        return cls._load_from_paths(
            identity=candidate_identity,
            tokenizer_path=tokenizer_path,
            query_model_path=query_model_path,
            document_model_path=document_model_path,
        )

    @classmethod
    def _load_from_paths(
        cls,
        *,
        identity: dict[str, object],
        tokenizer_path: Path,
        query_model_path: Path,
        document_model_path: Path,
    ) -> ProductionEncoderRuntime:
        """Load runtime components shared by sealed serving and qualification."""

        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
            from tokenizers import Tokenizer as HuggingFaceTokenizer  # type: ignore[import-untyped]

            options = ort.SessionOptions()
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            tokenizer = HuggingFaceTokenizer.from_file(str(tokenizer_path))
            query_session = ort.InferenceSession(
                str(query_model_path),
                sess_options=options,
                providers=[EXECUTION_PROVIDER],
            )
            document_session = (
                query_session
                if query_model_path.resolve() == document_model_path.resolve()
                else ort.InferenceSession(
                    str(document_model_path),
                    sess_options=options,
                    providers=[EXECUTION_PROVIDER],
                )
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            raise ProductionRuntimeError(
                f"unable to load sealed encoder artifacts: {error}"
            ) from error

        return cls(
            identity=identity,
            tokenizer=cast(Tokenizer, tokenizer),
            query_session=cast(Session, query_session),
            document_session=cast(Session, document_session),
        )

    @staticmethod
    def artifact_path(
        artifact_root: Path,
        record: dict[str, object],
        role: str,
    ) -> Path:
        """Resolve one already-sealed artifact without permitting path substitution."""
        relative = record.get("path")
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ProductionRuntimeError(f"{role} artifact path is invalid")
        pure_path = PurePosixPath(relative)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise ProductionRuntimeError(f"{role} artifact path is invalid")
        root = artifact_root.resolve()
        candidate = artifact_root.joinpath(*pure_path.parts)
        resolved = candidate.resolve()
        if (
            not resolved.is_relative_to(root)
            or candidate.is_symlink()
            or not candidate.is_file()
        ):
            raise ProductionRuntimeError(f"{role} artifact is not a regular file under the root")
        return candidate

    def encode_query(self, query: str) -> list[float]:
        prefix = cast(str, EXPECTED_RECIPE["query_prefix"])
        template = cast(str, EXPECTED_RECIPE["query_template"])
        rendered = prefix + template.format(query=query)
        return self._encode([rendered], self._query_session)[0]

    def encode_documents(self, documents: list[str]) -> list[list[float]]:
        if not documents or len(documents) > 64:
            raise ProductionRuntimeError("document batch must contain 1 to 64 values")
        prefix = cast(str, EXPECTED_RECIPE["document_prefix"])
        return self._encode([prefix + document for document in documents], self._document_session)

    def _encode(self, texts: list[str], session: Session) -> list[list[float]]:
        try:
            encodings = self._tokenizer.encode_batch(texts)
            if len(encodings) != len(texts):
                raise ProductionRuntimeError("tokenizer returned the wrong batch size")
            input_ids = np.asarray([encoding.ids for encoding in encodings], dtype=np.int64)
            attention_mask = np.asarray(
                [encoding.attention_mask for encoding in encodings],
                dtype=np.int64,
            )
            outputs = session.run(
                [EXPECTED_OUTPUT],
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )
            if len(outputs) != 1:
                raise ProductionRuntimeError("ONNX runtime returned an unexpected output set")
            vectors = np.asarray(outputs[0], dtype=np.float32)
        except ProductionRuntimeError:
            raise
        except (TypeError, ValueError, RuntimeError) as error:
            raise ProductionRuntimeError(f"encoder inference failed: {error}") from error

        expected_shape = (len(texts), cast(int, EXPECTED_MODEL["dimension"]))
        if vectors.shape != expected_shape or not np.isfinite(vectors).all():
            raise ProductionRuntimeError("encoder output shape or numeric values are invalid")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms <= 0.0):
            raise ProductionRuntimeError("encoder returned a zero or non-finite vector")
        normalized = np.asarray(vectors / norms, dtype=np.float32)
        return cast(list[list[float]], normalized.tolist())

    @staticmethod
    def _validate_session(session: Session, label: str) -> None:
        inputs = {node.name for node in session.get_inputs()}
        outputs = [node.name for node in session.get_outputs()]
        providers = session.get_providers()
        if inputs != EXPECTED_INPUTS or outputs != [EXPECTED_OUTPUT]:
            raise ProductionRuntimeError(f"{label} ONNX graph input or output contract is invalid")
        if providers != [EXECUTION_PROVIDER]:
            raise ProductionRuntimeError(
                f"{label} ONNX session must use only {EXECUTION_PROVIDER}"
            )


def verify_runtime_environment(
    identity: dict[str, object],
    *,
    expected_deployment_digest: str,
    system: str | None = None,
    machine: str | None = None,
    python_version: str | None = None,
    package_versions: Mapping[str, str] | None = None,
) -> None:
    """Fail startup if process, package, or deployed-image identity has drifted."""
    actual_system = system if system is not None else platform.system()
    actual_machine = machine if machine is not None else platform.machine()
    if actual_system != "Linux" or actual_machine.lower() not in {"amd64", "x86_64"}:
        raise ProductionRuntimeError("production encoder requires Linux amd64")

    deployment = identity.get("deployment")
    if (
        not isinstance(deployment, dict)
        or deployment.get("artifact_digest") != expected_deployment_digest
    ):
        raise ProductionRuntimeError("deployment digest does not match the sealed identity")

    runtime = identity.get("runtime")
    versions = runtime.get("versions") if isinstance(runtime, dict) else None
    if not isinstance(versions, dict):
        raise ProductionRuntimeError("sealed runtime versions are missing")
    actual_python = python_version if python_version is not None else platform.python_version()
    actual_packages = package_versions
    if actual_packages is None:
        try:
            actual_packages = {
                "numpy": importlib.metadata.version("numpy"),
                "onnxruntime": importlib.metadata.version("onnxruntime"),
                "tokenizers": importlib.metadata.version("tokenizers"),
            }
        except importlib.metadata.PackageNotFoundError as error:
            raise ProductionRuntimeError("a sealed runtime package is not installed") from error
    actual = {
        "python": actual_python,
        "numpy": actual_packages.get("numpy"),
        "onnxruntime": actual_packages.get("onnxruntime"),
        "tokenizers": actual_packages.get("tokenizers"),
    }
    if actual != versions:
        raise ProductionRuntimeError("installed runtime versions do not match the sealed identity")
