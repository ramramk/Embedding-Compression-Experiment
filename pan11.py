from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET
import math
import random


TEXT_SUFFIXES = {".txt", ".text", ".xml"}


@dataclass(frozen=True)
class Document:
    doc_id: str
    path: Path
    text: str


@dataclass(frozen=True)
class Pan11Dataset:
    source_documents: list[Document]
    suspicious_documents: list[Document]
    ground_truth: dict[str, set[str]]


def normalize_doc_id(value: str | Path) -> str:
    name = Path(str(value).replace("\\", "/")).name
    return Path(name).stem


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding, errors="strict")
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _is_text_document(path: Path) -> bool:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return False
    lower = path.name.lower()
    if lower.endswith(".xml"):
        return False
    return True


def _find_documents(root: Path, kind: str) -> list[Path]:
    candidates: list[Path] = []
    kind_terms = {
        "source": ("source", "src"),
        "suspicious": ("susp", "suspicious"),
    }[kind]
    for path in root.rglob("*"):
        if not path.is_file() or not _is_text_document(path):
            continue
        searchable = str(path.relative_to(root)).lower()
        if any(term in searchable for term in kind_terms):
            candidates.append(path)
    return sorted(set(candidates))


def _paths_by_id(root: Path, kind: str) -> dict[str, Path]:
    return {normalize_doc_id(path.name): path for path in _find_documents(root, kind)}


def _load_paths(paths: Iterable[Path]) -> list[Document]:
    return [Document(normalize_doc_id(path.name), path, _read_text(path)) for path in sorted(paths)]


def load_documents(
    root: Path,
    kind: str,
    limit: int | None = None,
    required_doc_ids: set[str] | None = None,
) -> list[Document]:
    paths = list(_paths_by_id(root, kind).values())
    if limit is not None or required_doc_ids:
        required_doc_ids = required_doc_ids or set()
        required_paths = [path for path in paths if normalize_doc_id(path.name) in required_doc_ids]
        distractor_paths = [path for path in paths if normalize_doc_id(path.name) not in required_doc_ids]
        if limit is not None:
            distractor_paths = distractor_paths[:limit]
        paths = sorted(set(required_paths + distractor_paths))
    docs = _load_paths(paths)
    if not docs:
        raise FileNotFoundError(
            f"No {kind} text documents found under {root}. Expected file or folder names containing {kind!r}."
        )
    return docs


def _iter_xml_features(xml_path: Path) -> Iterable[tuple[str | None, str | None]]:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return

    doc_ref = (
        root.attrib.get("reference")
        or root.attrib.get("this_reference")
        or root.attrib.get("this-reference")
        or xml_path.stem
    )
    susp_id = normalize_doc_id(doc_ref)
    for elem in root.iter():
        attrs = elem.attrib
        source_ref = (
            attrs.get("source_reference")
            or attrs.get("source-reference")
            or attrs.get("source")
            or attrs.get("source_document")
            or attrs.get("source-document")
        )
        feature_name = attrs.get("name", "").lower()
        if source_ref and ("plagiarism" in feature_name or elem.tag.lower().endswith("feature")):
            yield susp_id, normalize_doc_id(source_ref)


def parse_ground_truth(root: Path, target_suspicious_ids: set[str] | None = None) -> dict[str, set[str]]:
    truth: dict[str, set[str]] = {}
    for xml_path in sorted(root.rglob("*.xml")):
        if target_suspicious_ids is not None and normalize_doc_id(xml_path.name) not in target_suspicious_ids:
            continue
        for suspicious_id, source_id in _iter_xml_features(xml_path):
            if suspicious_id and source_id:
                truth.setdefault(suspicious_id, set()).add(source_id)
    if not truth:
        raise FileNotFoundError(
            f"No PAN-style plagiarism annotations found under {root}. Expected XML features with source_reference."
        )
    return truth


def _sample_dataset(root: Path, sample_fraction: float, seed: int) -> Pan11Dataset:
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in the interval (0, 1]")

    source_paths = _paths_by_id(root, "source")
    suspicious_paths = _paths_by_id(root, "suspicious")
    rng = random.Random(seed)
    all_suspicious_ids = sorted(suspicious_paths)
    suspicious_count = max(1, math.ceil(len(all_suspicious_ids) * sample_fraction))
    selected_suspicious_ids = set(rng.sample(all_suspicious_ids, suspicious_count))
    truth = parse_ground_truth(root, selected_suspicious_ids)
    selected_truth = {
        susp_id: {source_id for source_id in truth[susp_id] if source_id in source_paths}
        for susp_id in selected_suspicious_ids
        if susp_id in truth
    }
    selected_truth = {susp_id: source_ids for susp_id, source_ids in selected_truth.items() if source_ids}
    if not selected_truth:
        raise ValueError("The sampled suspicious documents had no matching true sources. Try a larger sample fraction.")

    required_source_ids = {source_id for source_ids in selected_truth.values() for source_id in source_ids}
    target_source_count = max(len(required_source_ids), math.ceil(len(source_paths) * sample_fraction))
    distractor_candidates = sorted(set(source_paths) - required_source_ids)
    distractor_count = max(0, target_source_count - len(required_source_ids))
    selected_source_ids = set(required_source_ids)
    if distractor_count:
        selected_source_ids.update(rng.sample(distractor_candidates, min(distractor_count, len(distractor_candidates))))

    sources = _load_paths(source_paths[source_id] for source_id in selected_source_ids)
    suspicious = _load_paths(suspicious_paths[susp_id] for susp_id in selected_truth)
    return Pan11Dataset(sources, suspicious, selected_truth)


def load_pan11_dataset(
    root: Path,
    debug_limit: int | None = None,
    debug_source_limit: int | None = None,
    sample_fraction: float | None = None,
    seed: int = 13,
) -> Pan11Dataset:
    if sample_fraction is not None:
        return _sample_dataset(root, sample_fraction, seed)

    suspicious = load_documents(root, "suspicious", debug_limit)
    suspicious_ids = {doc.doc_id for doc in suspicious}
    truth = parse_ground_truth(root, suspicious_ids)
    required_source_ids = {source_id for source_ids in truth.values() for source_id in source_ids}
    sources = load_documents(root, "source", debug_source_limit, required_source_ids)
    source_ids = {doc.doc_id for doc in sources}
    truth = {
        susp_id: {src_id for src_id in src_ids if src_id in source_ids}
        for susp_id, src_ids in truth.items()
        if susp_id in suspicious_ids
    }
    truth = {susp_id: src_ids for susp_id, src_ids in truth.items() if src_ids}
    suspicious = [doc for doc in suspicious if doc.doc_id in truth]
    if not suspicious:
        raise ValueError("Ground truth did not match the loaded suspicious/source document IDs.")
    return Pan11Dataset(sources, suspicious, truth)
