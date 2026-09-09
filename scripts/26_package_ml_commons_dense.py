from __future__ import annotations

import argparse
import json
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, cast

import onnx

from poc.config import load_model_registry
from poc.datasets import file_facts
from poc.manifest import read_json, write_json
from poc.ml_commons import wrap_prepooled_output_for_cls
from poc.model_runtime import model_snapshot_directory
from poc.query_runtime import (
    load_query_runtime_spec,
    onnx_artifact_directory,
    onnx_quantized_filename,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "arctic_embed_m_v2"
OUTPUT_DIRECTORY = ROOT / "data/cache/models/ml-commons"
TOKENIZER_FILES = (
    "config.json",
    "config_sentence_transformers.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)
PACKAGE_SERVER_SERVICE = "ml-commons-package-server"
PACKAGE_SERVER_PORT = 8080
PACKAGE_SERVER_ORIGIN = f"http://{PACKAGE_SERVER_SERVICE}:{PACKAGE_SERVER_PORT}"
SUPPORTED_OPENSEARCH_VERSIONS = frozenset({"2.19.6", "3.8.0"})


def package_server_binding(filename: str) -> dict[str, object]:
    if not filename or Path(filename).name != filename:
        raise ValueError("ML Commons package filename must be a single path component")
    url = f"{PACKAGE_SERVER_ORIGIN}/{filename}"
    return {
        "service": PACKAGE_SERVER_SERVICE,
        "port": PACKAGE_SERVER_PORT,
        "network_scope": "compose_internal",
        "url": url,
        "trusted_url_regex": f"^{re.escape(url)}$",
    }


def _write_member(archive: zipfile.ZipFile, source: Path, name: str) -> None:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED if name == "model.onnx" else zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, source.read_bytes())


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package Arctic for ML Commons")
    parser.add_argument("--opensearch-version", default="3.8.0")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.opensearch_version not in SUPPORTED_OPENSEARCH_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_OPENSEARCH_VERSIONS))
        raise ValueError(
            f"unsupported OpenSearch version {args.opensearch_version}; expected {supported}"
        )
    major = int(args.opensearch_version.split(".", maxsplit=1)[0])
    compatibility_219 = major < 3
    output_name = (
        f"{MODEL_NAME}-int8-onnx-opensearch-2.x-cls.zip"
        if compatibility_219
        else f"{MODEL_NAME}-int8-onnx-arm64.zip"
    )
    output = OUTPUT_DIRECTORY / output_name
    manifest_path = (
        ROOT / "results/wands/ml-commons/arctic-package-opensearch-2.x.json"
        if compatibility_219
        else ROOT / "results/wands/ml-commons/arctic-package.json"
    )
    registry = load_model_registry(ROOT / "config/models.toml")
    runtime = load_query_runtime_spec(ROOT / "config/query_runtime.toml")
    model = registry[MODEL_NAME]
    snapshot = model_snapshot_directory(ROOT, model)
    source_manifest_path = (
        ROOT
        / "results/wands/query-runtime"
        / f"{model.name}.{runtime.quantization_config}.manifest.json"
    )
    source_manifest = cast(dict[str, Any], read_json(source_manifest_path))
    onnx_path = (
        onnx_artifact_directory(ROOT, model, runtime) / onnx_quantized_filename(runtime)
    )
    missing = [name for name in TOKENIZER_FILES if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Arctic ML Commons package inputs are missing: {missing}")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    packaged_onnx_facts: dict[str, object]
    with tempfile.TemporaryDirectory(
        prefix="opensearch-hybrid-ml-commons-package-"
    ) as temporary_directory:
        packaged_onnx = onnx_path
        if compatibility_219:
            packaged_onnx = Path(temporary_directory) / "model.onnx"
            wrapped = wrap_prepooled_output_for_cls(onnx.load(onnx_path))
            onnx.save(wrapped, packaged_onnx)
        packaged_facts = file_facts(packaged_onnx)
        packaged_onnx_facts = {
            "sha256": packaged_facts.sha256,
            "bytes": packaged_facts.bytes,
        }
        temporary = output.with_suffix(".tmp")
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            _write_member(archive, packaged_onnx, "model.onnx")
            for name in TOKENIZER_FILES:
                _write_member(archive, snapshot / name, name)
        temporary.replace(output)
    config = json.loads((snapshot / "config.json").read_text())
    package_facts = file_facts(output)
    package_server = package_server_binding(output.name)
    registration_body = {
        "name": (
            "opensearch-hybrid/arctic-embed-m-v2-int8-opensearch-2.x-cls"
            if compatibility_219
            else "opensearch-hybrid/arctic-embed-m-v2-int8-arm64"
        ),
        "version": model.revision,
        "description": "OpenSearch Hybrid Arctic CLS MRL 256 dynamic-int8 smoke artifact",
        "model_format": "ONNX",
        "function_name": "TEXT_EMBEDDING",
        "model_content_hash_value": package_facts.sha256,
        "model_config": {
            "model_type": config["model_type"],
            "embedding_dimension": model.dims,
            "framework_type": "sentence_transformers",
            "all_config": json.dumps(config, separators=(",", ":"), sort_keys=True),
            "pooling_mode": "CLS" if compatibility_219 else "none",
            "normalize_result": False,
            "additional_config": {"space_type": "cosinesimil"},
        },
        "url": package_server["url"],
    }
    write_json(
        manifest_path,
        {
            "schema_version": 2,
            "model": MODEL_NAME,
            "source_int8_artifact_sha256": source_manifest["artifact_sha256"],
            "source_onnx": {
                "path": str(onnx_path.relative_to(ROOT)),
                "sha256": file_facts(onnx_path).sha256,
                "bytes": file_facts(onnx_path).bytes,
            },
            "package": {
                "path": str(output.relative_to(ROOT)),
                "sha256": package_facts.sha256,
                "bytes": package_facts.bytes,
                "members": ["model.onnx", *TOKENIZER_FILES],
            },
            "package_server": package_server,
            "packaged_onnx": packaged_onnx_facts,
            "opensearch_target": args.opensearch_version,
            "compatibility_transform": (
                "unsqueeze_prepooled_output_then_ml_commons_cls_pooling"
                if compatibility_219
                else "none"
            ),
            "registration_body": registration_body,
            "target": "OpenSearch ML Commons local ONNX text embedding",
            "eligible_for_decision": False,
            "ineligibility_reason": (
                "diagnostic packaging artifact without benchmark quality or latency evidence"
            ),
        },
    )
    print(
        f"packaged Arctic int8 ONNX for ML Commons {args.opensearch_version}: {output} "
        f"({package_facts.bytes} bytes)"
    )


if __name__ == "__main__":
    main()
