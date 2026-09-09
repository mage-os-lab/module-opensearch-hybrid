from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from mageos_opensearch_hybrid_encoder.runtime import (
    ProductionEncoderRuntime,
    ProductionRuntimeError,
    verify_runtime_environment,
)


@dataclass
class FakeEncoding:
    ids: list[int]
    attention_mask: list[int]


class FakeTokenizer:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def encode_batch(self, texts: list[str]) -> list[FakeEncoding]:
        self.batches.append(texts)
        width = max(len(text) for text in texts)
        return [
            FakeEncoding(
                ids=[len(text)] + ([0] * (width - 1)),
                attention_mask=[1] + ([0] * (width - 1)),
            )
            for text in texts
        ]


@dataclass
class FakeNode:
    name: str


class FakeSession:
    def __init__(self) -> None:
        self.feeds: list[dict[str, NDArray[np.int64]]] = []

    def get_inputs(self) -> list[FakeNode]:
        return [FakeNode("input_ids"), FakeNode("attention_mask")]

    def get_outputs(self) -> list[FakeNode]:
        return [FakeNode("sentence_embedding")]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(
        self,
        output_names: list[str],
        feed: dict[str, NDArray[np.int64]],
    ) -> list[np.ndarray[Any, Any]]:
        assert output_names == ["sentence_embedding"]
        self.feeds.append(feed)
        rows = len(feed["input_ids"])
        result = np.zeros((rows, 256), dtype=np.float32)
        result[:, 0] = 3.0
        result[:, 1] = 4.0
        return [result]


def sealed_identity() -> dict[str, object]:
    return {
        "model_id": "Snowflake/snowflake-arctic-embed-m-v2.0",
        "model_revision": "95c2741480856aa9666782eb4afe11959938017f",
        "dimension": 256,
        "recipe_version": "mageos-v1",
        "encoder_identity_digest": "c" * 64,
        "production_eligible": True,
        "runtime": {
            "versions": {
                "python": "3.13.15",
                "numpy": "2.5.2",
                "onnxruntime": "1.28.0",
                "tokenizers": "0.22.2",
            }
        },
        "deployment": {"artifact_digest": "sha256:" + ("a" * 64)},
    }


def test_production_runtime_applies_query_recipe_and_normalizes_vectors() -> None:
    tokenizer = FakeTokenizer()
    query_session = FakeSession()
    document_session = FakeSession()
    runtime = ProductionEncoderRuntime(
        identity=sealed_identity(),
        tokenizer=tokenizer,
        query_session=query_session,
        document_session=document_session,
    )

    query = runtime.encode_query("trail shoe")
    documents = runtime.encode_documents(["Boot. Waterproof.", "Shoe. Red."])

    assert tokenizer.batches == [
        ["query: trail shoe"],
        ["Boot. Waterproof.", "Shoe. Red."],
    ]
    assert query[:2] == pytest.approx([0.6, 0.8])
    assert documents[0][:2] == pytest.approx([0.6, 0.8])
    assert len(query) == 256
    assert len(documents) == 2
    assert query_session.feeds[0]["input_ids"].dtype == np.int64
    assert document_session.feeds[0]["attention_mask"].dtype == np.int64


def test_production_runtime_rejects_unqualified_session_contract() -> None:
    session = FakeSession()
    session.get_providers = lambda: ["CoreMLExecutionProvider"]  # type: ignore[method-assign]

    with pytest.raises(ProductionRuntimeError, match="CPUExecutionProvider"):
        ProductionEncoderRuntime(
            identity=sealed_identity(),
            tokenizer=FakeTokenizer(),
            query_session=session,
            document_session=FakeSession(),
        )


def test_qualification_candidate_remains_explicitly_production_ineligible() -> None:
    identity: dict[str, object] = {
        "production_eligible": False,
        "qualification_candidate": True,
        "dimension": 256,
    }

    runtime = ProductionEncoderRuntime(
        identity=identity,
        tokenizer=FakeTokenizer(),
        query_session=FakeSession(),
        document_session=FakeSession(),
    )

    assert runtime.identity["production_eligible"] is False
    assert runtime.identity["qualification_candidate"] is True


def test_runtime_environment_is_bound_to_amd64_versions_and_deployment_digest() -> None:
    verify_runtime_environment(
        sealed_identity(),
        expected_deployment_digest="sha256:" + ("a" * 64),
        system="Linux",
        machine="x86_64",
        python_version="3.13.15",
        package_versions={
            "numpy": "2.5.2",
            "onnxruntime": "1.28.0",
            "tokenizers": "0.22.2",
        },
    )

    with pytest.raises(ProductionRuntimeError, match="deployment digest"):
        verify_runtime_environment(
            sealed_identity(),
            expected_deployment_digest="sha256:" + ("b" * 64),
            system="Linux",
            machine="x86_64",
            python_version="3.13.15",
            package_versions={
                "numpy": "2.5.2",
                "onnxruntime": "1.28.0",
                "tokenizers": "0.22.2",
            },
        )


def test_runtime_loader_requires_regular_artifacts_under_the_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"model")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "query.onnx").symlink_to(outside)

    with pytest.raises(ProductionRuntimeError, match="regular file"):
        ProductionEncoderRuntime.artifact_path(
            artifact_root,
            {"path": "query.onnx", "sha256": "a" * 64, "size": 5},
            "query_model",
        )


def test_runtime_loads_one_session_for_the_shared_fp32_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_bytes(b"tokenizer")
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    sessions: list[tuple[str, object, list[str]]] = []

    class SessionOptions:
        pass

    def inference_session(
        path: str,
        *,
        sess_options: object,
        providers: list[str],
    ) -> FakeSession:
        sessions.append((path, sess_options, providers))
        return FakeSession()

    class Tokenizer:
        @staticmethod
        def from_file(path: str) -> FakeTokenizer:
            assert path == str(tokenizer_path)
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            SessionOptions=SessionOptions,
            ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
            InferenceSession=inference_session,
        ),
    )
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=Tokenizer))

    runtime = ProductionEncoderRuntime.load_qualification_artifacts(
        tokenizer_path=tokenizer_path,
        query_model_path=model_path,
        document_model_path=model_path,
    )

    assert len(sessions) == 1
    assert runtime._query_session is runtime._document_session
