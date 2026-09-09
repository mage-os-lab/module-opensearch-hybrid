from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Any, cast

MODULE_COMPONENT = "mage-os/module-opensearch-hybrid"
SUPPORTED_FORMAT = "CycloneDX"
SUPPORTED_SPEC_VERSION = "1.6"


def verify_resolved_module_sbom(
    path: Path,
    module_component: str = MODULE_COMPONENT,
) -> dict[str, str | int]:
    document = _load_document(path)
    if document.get("bomFormat") != SUPPORTED_FORMAT:
        raise ValueError(f"SBOM format must be {SUPPORTED_FORMAT}")
    if document.get("specVersion") != SUPPORTED_SPEC_VERSION:
        raise ValueError(f"SBOM spec version must be {SUPPORTED_SPEC_VERSION}")

    components = _objects(document.get("components"), "SBOM components")
    metadata = document.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("component"), dict):
        raise ValueError("SBOM metadata must identify its root component")
    root = cast(dict[str, Any], metadata["component"])
    root_ref = _reference(root, "SBOM root component")
    _reject_local_references([root, *components])

    known_refs = {root_ref}
    module: dict[str, Any] | None = None
    for component in components:
        reference = _reference(component, "SBOM component")
        if reference in known_refs:
            raise ValueError(f"SBOM component reference is duplicated: {reference}")
        known_refs.add(reference)
        version = component.get("version")
        if not _is_resolved_version(version):
            raise ValueError(f"SBOM component must have a resolved version: {reference}")
        purl = component.get("purl")
        if (
            not isinstance(purl, str)
            or not purl.startswith("pkg:composer/")
            or f"@{version}" not in purl
        ):
            raise ValueError(
                f"SBOM Composer component must bind its resolved version in a purl: {reference}"
            )
        if _composer_name(component) == module_component:
            if module is not None:
                raise ValueError("SBOM contains more than one module component")
            module = component

    if module is None:
        raise ValueError(f"SBOM is missing module component {module_component}")

    dependencies = _objects(document.get("dependencies"), "SBOM dependencies")
    dependency_by_ref: dict[str, list[str]] = {}
    for dependency in dependencies:
        reference = _reference(dependency, "SBOM dependency")
        if reference in dependency_by_ref:
            raise ValueError(f"SBOM dependency reference is duplicated: {reference}")
        raw_dependencies = dependency.get("dependsOn", [])
        if not isinstance(raw_dependencies, list) or not all(
            isinstance(item, str) for item in raw_dependencies
        ):
            raise ValueError(f"SBOM dependency list is invalid: {reference}")
        resolved_dependencies = cast(list[str], raw_dependencies)
        unknown = sorted(set(resolved_dependencies) - known_refs)
        if unknown:
            raise ValueError(
                f"SBOM dependency references an unknown component: {reference} -> {unknown[0]}"
            )
        dependency_by_ref[reference] = resolved_dependencies

    module_ref = _reference(module, "SBOM module component")
    if module_ref not in dependency_by_ref or not dependency_by_ref[module_ref]:
        raise ValueError("SBOM module dependency graph is empty")
    if module_ref not in dependency_by_ref.get(root_ref, []):
        raise ValueError("SBOM root dependency graph does not include the module")

    return {
        "component": module_component,
        "component_count": len(components),
        "dependency_count": len(dependencies),
        "module_ref": module_ref,
        "module_version": cast(str, module["version"]),
        "spec_version": SUPPORTED_SPEC_VERSION,
    }


def normalize_module_sbom(source: Path, destination: Path) -> int:
    document = _load_document(source)
    metadata = document.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("component"), dict):
        raise ValueError("SBOM metadata must identify its root component")
    components = _objects(document.get("components"), "SBOM components")
    subjects = [cast(dict[str, Any], metadata["component"]), *components]
    removed = 0
    for subject in subjects:
        references = subject.get("externalReferences")
        if references is None:
            continue
        if not isinstance(references, list) or not all(
            isinstance(reference, dict) for reference in references
        ):
            raise ValueError("SBOM external references must be an array of objects")
        retained = [
            reference
            for reference in cast(list[dict[str, Any]], references)
            if not _is_local_reference(reference.get("url"))
        ]
        removed += len(references) - len(retained)
        if retained:
            subject["externalReferences"] = retained
        else:
            subject.pop("externalReferences", None)

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
                + b"\n"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return removed


def _objects(value: object, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{label} must be an array of objects")

    return cast(list[dict[str, Any]], value)


def _load_document(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("SBOM must be a JSON object")

    return cast(dict[str, Any], document)


def _reject_local_references(subjects: list[dict[str, Any]]) -> None:
    for subject in subjects:
        references = subject.get("externalReferences", [])
        if not isinstance(references, list) or not all(
            isinstance(reference, dict) for reference in references
        ):
            raise ValueError("SBOM external references must be an array of objects")
        for reference in cast(list[dict[str, Any]], references):
            if _is_local_reference(reference.get("url")):
                raise ValueError("SBOM contains a local filesystem reference")


def _is_local_reference(value: object) -> bool:
    if not isinstance(value, str):
        return False

    return (
        value.lower().startswith("file:")
        or Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
    )


def _reference(component: dict[str, Any], label: str) -> str:
    reference = component.get("bom-ref") or component.get("ref")
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"{label} must have a non-empty reference")

    return reference


def _composer_name(component: dict[str, Any]) -> str:
    group = component.get("group")
    name = component.get("name")
    if not isinstance(name, str):
        return ""
    if "/" in name:
        return name

    return f"{group}/{name}" if isinstance(group, str) and group else name


def _is_resolved_version(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False

    return not any(marker in value for marker in ("*", "^", "~", "||", ">", "<", " "))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the resolved CycloneDX dependency graph for the Mage-OS module"
    )
    parser.add_argument("--sbom", type=Path, required=True)
    parser.add_argument("--component", default=MODULE_COMPONENT)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="atomically remove environment-local external references before verification",
    )
    args = parser.parse_args()

    if args.normalize:
        normalize_module_sbom(args.sbom, args.sbom)
    summary = verify_resolved_module_sbom(args.sbom, args.component)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
