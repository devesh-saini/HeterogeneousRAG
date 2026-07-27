from __future__ import annotations

import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from config.domains import PROJECT_ROOT


CACHE_ROOT = PROJECT_ROOT / ".cache" / "ingestion"

QASPER_TRAIN_DEV_URL = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
)
QASPER_FILES = {
    "train": "qasper-train-v0.3.json",
    "validation": "qasper-dev-v0.3.json",
}

CUAD_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
CUAD_FILES = {
    "train": "train_separate_questions.json",
    "test": "test.json",
}


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urllib.request.urlretrieve(url, destination)
    return destination


def _read_qasper_json(split: str = "train") -> dict[str, Any]:
    try:
        filename = QASPER_FILES[split]
    except KeyError as exc:
        valid = ", ".join(sorted(QASPER_FILES))
        raise ValueError(f"Unknown QASPER split '{split}'. Expected one of: {valid}") from exc

    archive = _download(QASPER_TRAIN_DEV_URL, CACHE_ROOT / "qasper-train-dev-v0.3.tgz")
    with tarfile.open(archive, "r:gz") as tar:
        member = next((m for m in tar.getmembers() if Path(m.name).name == filename), None)
        if member is None:
            raise FileNotFoundError(f"Could not find {filename} in {archive}")
        extracted = tar.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"Could not extract {filename} from {archive}")
        return json.loads(extracted.read().decode("utf-8"))


def load_qasper_documents(split: str = "train") -> list[dict[str, Any]]:
    data = _read_qasper_json(split)
    documents: list[dict[str, Any]] = []
    for paper_id, row in data.items():
        if isinstance(row, dict):
            row["id"] = paper_id
            documents.append(row)
    return documents


def _read_cuad_json(split: str = "train") -> dict[str, Any]:
    try:
        filename = CUAD_FILES[split]
    except KeyError as exc:
        valid = ", ".join(sorted(CUAD_FILES))
        raise ValueError(f"Unknown CUAD split '{split}'. Expected one of: {valid}") from exc

    archive = _download(CUAD_URL, CACHE_ROOT / "cuad-data.zip")
    with zipfile.ZipFile(archive) as zipped:
        member = next((name for name in zipped.namelist() if Path(name).name == filename), None)
        if member is None:
            raise FileNotFoundError(f"Could not find {filename} in {archive}")
        with zipped.open(member) as extracted:
            return json.loads(extracted.read().decode("utf-8"))


def load_cuad_contexts(split: str = "train") -> list[str]:
    data = _read_cuad_json(split)
    contexts: list[str] = []
    for example in data.get("data", []):
        if not isinstance(example, dict):
            continue
        for paragraph in example.get("paragraphs", []):
            if isinstance(paragraph, dict):
                contexts.append(str(paragraph.get("context", "")))
    return contexts
