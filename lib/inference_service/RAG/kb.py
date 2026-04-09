from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Any


RAG_DIR = Path(__file__).resolve().parent
CHROMA_PATH = RAG_DIR / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

KB_CONFIG = {
    "tuberculosis": {
        "csv_path": RAG_DIR / "tb_normal.csv",
        "collection_name": "tuberculosis_reports_tb_normal_v2",
        "label_prefix": "tb-report",
    },
    "brain-tumor": {
        "csv_path": RAG_DIR / "brain_tumor.csv",
        "collection_name": "brain_tumor_reports_v1",
        "label_prefix": "brain-report",
    },
}


def _normalize_field(value: str | None) -> str:
    text = (value or "").replace("XXXX", "unspecified").strip()
    return re.sub(r"\s+", " ", text)


def _get_kb_config(disease_key: str) -> dict[str, Path | str]:
    if disease_key not in KB_CONFIG:
        raise ValueError(f"Unsupported RAG disease key: {disease_key}")
    return KB_CONFIG[disease_key]


def _load_rows(disease_key: str) -> list[dict[str, str]]:
    config = _get_kb_config(disease_key)
    csv_path = Path(config["csv_path"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Knowledge base CSV not found: {csv_path}")

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for index, row in enumerate(reader):
            findings = _normalize_field(row.get("findings"))
            impression = _normalize_field(row.get("impression"))
            disease_type = _normalize_field(row.get("disease_type"))
            class_label = _normalize_field(row.get("class_label"))
            location = _normalize_field(row.get("location"))
            report_id = _normalize_field(row.get("id")) or f"{config['label_prefix']}-{index:05d}"

            if not any([findings, impression, disease_type, class_label, location]):
                continue

            document_parts = [
                f"Disease Type: {disease_type}",
                f"Class Label: {class_label}",
            ]
            if location:
                document_parts.append(f"Location: {location}")
            if findings:
                document_parts.append(f"Findings: {findings}")
            if impression:
                document_parts.append(f"Impression: {impression}")

            rows.append(
                {
                    "id": report_id,
                    "disease_type": disease_type,
                    "class_label": class_label,
                    "location": location,
                    "findings": findings,
                    "impression": impression,
                    "document": "\n".join(document_parts).strip(),
                }
            )
    return rows


def _import_chroma() -> tuple[Any, Any]:
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return chromadb, SentenceTransformerEmbeddingFunction


def _get_client() -> Any:
    chromadb, _ = _import_chroma()
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def _create_collection(client: Any, disease_key: str) -> Any:
    _, sentence_transformer_embedding = _import_chroma()
    embedding_function = sentence_transformer_embedding(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(
        name=str(_get_kb_config(disease_key)["collection_name"]),
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


def _csv_hash(disease_key: str) -> str:
    csv_path = Path(_get_kb_config(disease_key)["csv_path"])
    return hashlib.md5(csv_path.read_bytes()).hexdigest()


def _hash_path(disease_key: str) -> Path:
    return CHROMA_PATH / f"{disease_key}_csv_hash.txt"


def ensure_vector_store(disease_key: str) -> int:
    rows = _load_rows(disease_key)
    client = _get_client()
    collection = _create_collection(client, disease_key)

    hash_path = _hash_path(disease_key)
    current_hash = _csv_hash(disease_key)
    stored_hash = hash_path.read_text(encoding="utf-8").strip() if hash_path.exists() else ""

    if collection.count() == len(rows) and stored_hash == current_hash:
        return collection.count()

    try:
        client.delete_collection(str(_get_kb_config(disease_key)["collection_name"]))
    except Exception:
        pass

    collection = _create_collection(client, disease_key)
    collection.upsert(
        ids=[row["id"] for row in rows],
        documents=[row["document"] for row in rows],
        metadatas=[
            {
                "report_id": row["id"],
                "disease_type": row["disease_type"],
                "class_label": row["class_label"],
                "location": row["location"],
                "findings": row["findings"],
                "impression": row["impression"],
            }
            for row in rows
        ],
    )
    hash_path.write_text(current_hash, encoding="utf-8")
    return collection.count()


def get_collection(disease_key: str) -> Any:
    ensure_vector_store(disease_key)
    client = _get_client()
    return _create_collection(client, disease_key)


def ensure_tb_vector_store() -> int:
    return ensure_vector_store("tuberculosis")


def ensure_brain_tumor_vector_store() -> int:
    return ensure_vector_store("brain-tumor")
