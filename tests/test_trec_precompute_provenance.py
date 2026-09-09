from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pytest import MonkeyPatch

import poc.manifest as manifest_module
from poc.datasets import DatasetIntegrityError

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/32_precompute_trec_sparse.py"


class _FakeModel:
    def eval(self) -> _FakeModel:
        return self

    def float(self) -> _FakeModel:
        return self


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("precompute_trec_sparse", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_captures_generation_provenance_before_package_work(
    monkeypatch: MonkeyPatch,
) -> None:
    module = _load_script()
    generation_provenance: dict[str, object] = {
        "code_revision": {"git_commit": "start-revision"}
    }
    events: list[object] = []
    received_provenance: list[object] = []

    def collect_provenance(root: Path, **kwargs: object) -> dict[str, object]:
        assert root == ROOT
        events.append("collect_provenance")
        return generation_provenance

    def load_config(path: Path) -> object:
        events.append("load_config")
        return object()

    def verify_package(spec: object) -> None:
        events.append("verify_package")

    def precompute(
        spec: object,
        *,
        output: Path,
        manifest_path: Path,
        limit: int | None,
        generation_provenance: object | None = None,
    ) -> dict[str, object]:
        received_provenance.append(generation_provenance)
        return {"records": 1, "documents_per_second": 1.0}

    monkeypatch.setattr(
        module, "collect_manifest_provenance", collect_provenance, raising=False
    )
    monkeypatch.setattr(module, "load_trec_sparse_config", load_config)
    monkeypatch.setattr(module, "_verify_official_package", verify_package)
    monkeypatch.setattr(module, "precompute", precompute)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--preflight-documents", "1"])

    module.main()

    assert events == ["collect_provenance", "load_config", "verify_package"]
    assert received_provenance == [generation_provenance]


def test_precompute_writes_start_snapshot_as_generation_provenance(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    module = _load_script()
    start_snapshot = {"code_revision": {"git_commit": "start-revision"}}
    completion_snapshot = {"code_revision": {"git_commit": "completion-revision"}}
    tokenizer = SimpleNamespace(
        enable_truncation=lambda **kwargs: None,
        enable_padding=lambda: None,
        get_vocab_size=lambda: 0,
        id_to_token=lambda index: None,
    )
    model = _FakeModel()
    fake_modules = {
        "torch": SimpleNamespace(
            jit=SimpleNamespace(load=lambda *args, **kwargs: model)
        ),
        "tokenizers": SimpleNamespace(
            Tokenizer=SimpleNamespace(from_file=lambda path: tokenizer)
        ),
    }
    spec: Any = SimpleNamespace(
        maximum_token_length=512,
        inference_batch_size=16,
        maximum_value_ratio=0.1,
        minimum_preflight_documents_per_second=1.0,
        document_model_name="test-model",
        document_model_version="1",
        document_model_format="TORCH_SCRIPT",
        document_model_url="https://example.test/model.zip",
    )
    output = tmp_path / "embeddings.jsonl"
    manifest_path = tmp_path / "precompute-manifest.json"

    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: fake_modules[name],
    )
    monkeypatch.setattr(module, "iter_trec_products", lambda path: iter(()))
    monkeypatch.setattr(
        module,
        "_artifact",
        lambda path: {"path": path.name, "sha256": "test", "bytes": 0},
    )
    monkeypatch.setattr(
        manifest_module,
        "collect_manifest_provenance",
        lambda root: completion_snapshot,
    )

    module.precompute(
        spec,
        output=output,
        manifest_path=manifest_path,
        limit=0,
        generation_provenance=start_snapshot,
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 2
    assert manifest["generation_provenance"] == start_snapshot
    assert manifest["benchmark_provenance"] == start_snapshot
    assert manifest["checkpoint_resume_verified"] is False
    assert manifest["quality_evidence_eligible_for_decision"] is False


def test_precompute_rejects_an_unbound_resume_checkpoint(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    module = _load_script()
    output = tmp_path / "embeddings.jsonl"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text('{"embedding":{"token":1.0},"product_id":"doc-1"}\n')
    _prepare_empty_precompute(module, monkeypatch)
    monkeypatch.setattr(
        module,
        "iter_trec_products",
        lambda path: iter((("doc-1", {"title": "one", "description": ""}),)),
    )

    with pytest.raises(DatasetIntegrityError, match="checkpoint sidecar"):
        module.precompute(
            _precompute_spec(),
            output=output,
            manifest_path=tmp_path / "manifest.json",
            limit=1,
            generation_provenance={"code_revision": {"git_commit": "start"}},
        )


def test_precompute_rejects_a_mismatched_resume_checkpoint_binding(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    module = _load_script()
    output = tmp_path / "embeddings.jsonl"
    temporary = output.with_name(f".{output.name}.tmp")
    sidecar = output.with_name(f".{output.name}.checkpoint.json")
    temporary.write_text('{"embedding":{"token":1.0},"product_id":"doc-1"}\n')
    sidecar.write_text('{"schema_version":1,"generation_provenance":{}}\n')
    _prepare_empty_precompute(module, monkeypatch)
    monkeypatch.setattr(
        module,
        "iter_trec_products",
        lambda path: iter((("doc-1", {"title": "one", "description": ""}),)),
    )

    with pytest.raises(DatasetIntegrityError, match="checkpoint binding"):
        module.precompute(
            _precompute_spec(),
            output=output,
            manifest_path=tmp_path / "manifest.json",
            limit=1,
            generation_provenance={"code_revision": {"git_commit": "start"}},
        )


def test_precompute_accepts_only_a_matching_attested_checkpoint_prefix(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    module = _load_script()
    output = tmp_path / "embeddings.jsonl"
    temporary = output.with_name(f".{output.name}.tmp")
    sidecar = output.with_name(f".{output.name}.checkpoint.json")
    line = b'{"embedding":{"token":1.0},"product_id":"doc-1"}\n'
    temporary.write_bytes(line)
    _prepare_empty_precompute(module, monkeypatch)
    spec = _precompute_spec()
    provenance = {"code_revision": {"git_commit": "start"}}
    sidecar.write_text(
        json.dumps(
            module._checkpoint_payload(
                module._checkpoint_binding(
                    spec,
                    output=output,
                    limit=1,
                    generation_provenance=provenance,
                ),
                records=1,
                bytes_written=len(line),
                stream_sha256=hashlib.sha256(line).hexdigest(),
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    monkeypatch.setattr(
        module,
        "iter_trec_products",
        lambda path: iter((("doc-1", {"title": "one", "description": ""}),)),
    )

    result = module.precompute(
        spec,
        output=output,
        manifest_path=tmp_path / "manifest.json",
        limit=1,
        generation_provenance=provenance,
    )

    assert result["schema_version"] == 2
    assert result["resumed_records"] == 1
    assert result["encoded_records_this_session"] == 0
    assert result["checkpoint_resume_verified"] is True


def test_precompute_rejects_a_tampered_attested_checkpoint_prefix(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    module = _load_script()
    output = tmp_path / "embeddings.jsonl"
    temporary = output.with_name(f".{output.name}.tmp")
    sidecar = output.with_name(f".{output.name}.checkpoint.json")
    original = b'{"embedding":{"token":1.0},"product_id":"doc-1"}\n'
    tampered = b'{"embedding":{"token":2.0},"product_id":"doc-1"}\n'
    assert len(original) == len(tampered)
    temporary.write_bytes(tampered)
    _prepare_empty_precompute(module, monkeypatch)
    spec = _precompute_spec()
    provenance = {"code_revision": {"git_commit": "start"}}
    sidecar.write_text(
        json.dumps(
            module._checkpoint_payload(
                module._checkpoint_binding(
                    spec,
                    output=output,
                    limit=1,
                    generation_provenance=provenance,
                ),
                records=1,
                bytes_written=len(original),
                stream_sha256=hashlib.sha256(original).hexdigest(),
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    monkeypatch.setattr(
        module,
        "iter_trec_products",
        lambda path: iter((("doc-1", {"title": "one", "description": ""}),)),
    )

    with pytest.raises(DatasetIntegrityError, match="prefix attestation"):
        module.precompute(
            spec,
            output=output,
            manifest_path=tmp_path / "manifest.json",
            limit=1,
            generation_provenance=provenance,
        )


@pytest.mark.parametrize("provenance_eligible", [False, True])
def test_precompute_completion_quality_follows_start_provenance(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    provenance_eligible: bool,
) -> None:
    module = _load_script()
    output = tmp_path / "embeddings.jsonl"
    _prepare_empty_precompute(module, monkeypatch)
    monkeypatch.setattr(module, "iter_trec_products", lambda path: iter(()))
    requirements: list[bool] = []

    def verify(*args: object, **kwargs: object) -> bool:
        requirements.append(
            kwargs.get("require_current_code_revision") is True
        )
        return provenance_eligible

    monkeypatch.setattr(module, "verify_decision_provenance", verify)

    result = module.precompute(
        _precompute_spec(),
        output=output,
        manifest_path=tmp_path / "manifest.json",
        limit=None,
        generation_provenance={"code_revision": {"git_commit": "start"}},
    )

    assert result["quality_evidence_eligible_for_decision"] is provenance_eligible
    assert requirements == [True]


def _prepare_empty_precompute(module: ModuleType, monkeypatch: MonkeyPatch) -> None:
    tokenizer = SimpleNamespace(
        enable_truncation=lambda **kwargs: None,
        enable_padding=lambda: None,
        get_vocab_size=lambda: 0,
        id_to_token=lambda index: None,
    )
    model = _FakeModel()
    fake_modules = {
        "torch": SimpleNamespace(jit=SimpleNamespace(load=lambda *args, **kwargs: model)),
        "tokenizers": SimpleNamespace(
            Tokenizer=SimpleNamespace(from_file=lambda path: tokenizer)
        ),
    }
    monkeypatch.setattr(module.importlib, "import_module", lambda name: fake_modules[name])
    monkeypatch.setattr(
        module,
        "_artifact",
        lambda path: {"path": path.name, "sha256": "test", "bytes": 0},
    )


def _precompute_spec() -> Any:
    return SimpleNamespace(
        maximum_token_length=512,
        inference_batch_size=16,
        maximum_value_ratio=0.1,
        minimum_preflight_documents_per_second=1.0,
        document_model_name="test-model",
        document_model_version="1",
        document_model_format="TORCH_SCRIPT",
        document_model_url="https://example.test/model.zip",
    )
