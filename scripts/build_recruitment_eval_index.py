#!/usr/bin/env python
import argparse
import shutil
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict

from src.evaluation.dataset import load_recruitment_eval_dataset, validate_recruitment_eval_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


def build_recruitment_eval_index_dry_run(
    dataset_dir: str | Path,
    index_dir: str | Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Dict[str, Any]:
    index_path = Path(index_dir)
    _reject_chroma_db(index_path)
    dataset = load_recruitment_eval_dataset(dataset_dir)
    validation = validate_recruitment_eval_dataset(dataset)
    if not validation.valid:
        return {
            "status": "failed",
            "error_hint": "dataset_validation_failed",
            "validation": validation.to_dict(),
            "summary_only": True,
        }
    document_count = len(dataset.candidates)
    chunk_count_estimate = sum(max(1, len(candidate.resume_text) // 450 + 1) for candidate in dataset.candidates)
    return {
        "status": "ok",
        "dry_run": True,
        "candidate_count": len(dataset.candidates),
        "document_count": document_count,
        "chunk_count_estimate": chunk_count_estimate,
        "index_dir": str(index_path),
        "embedding_model": embedding_model,
        "source_dataset_version": dataset.manifest.get("dataset_version"),
        "dataset_name": dataset.manifest.get("dataset_name"),
        "would_write_existing_chroma_db": False,
        "summary_only": True,
    }


def build_recruitment_eval_index(
    dataset_dir: str | Path,
    index_dir: str | Path,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    force: bool = False,
    index_writer=None,
) -> Dict[str, Any]:
    index_path = Path(index_dir)
    _reject_chroma_db(index_path)
    dataset = load_recruitment_eval_dataset(dataset_dir)
    validation = validate_recruitment_eval_dataset(dataset)
    if not validation.valid:
        return {
            "status": "failed",
            "error_hint": "dataset_validation_failed",
            "validation": validation.to_dict(),
            "summary_only": True,
        }
    if index_path.exists():
        if not force:
            raise FileExistsError(f"index directory already exists: {index_path}")
        shutil.rmtree(index_path)
    index_path.mkdir(parents=True, exist_ok=True)
    documents = [
        {
            "text": candidate.resume_text,
            "metadata": {
                "candidate_id": candidate.candidate_id,
                "source_document_id": candidate.candidate_id,
                "source_file_name": candidate.source_file_name,
                "dataset_version": dataset.manifest.get("dataset_version"),
                "special_case_type": candidate.special_case_type,
                "is_special_case": candidate.is_special_case,
            },
        }
        for candidate in dataset.candidates
    ]
    writer = index_writer or _real_llama_index_writer
    writer_result = writer(
        documents=documents,
        index_dir=index_path,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunk_count = int(writer_result.get("chunk_count") or len(documents))
    manifest = {
        "index_version": "phase13b-v1",
        "dataset_name": dataset.manifest.get("dataset_name"),
        "dataset_version": dataset.manifest.get("dataset_version"),
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "document_count": len(documents),
        "chunk_count": chunk_count,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "contains_relevance_labels": False,
        "contains_ideal_ranking": False,
        "summary_only": True,
    }
    (index_path / "eval_index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "dry_run": False,
        "dataset_version": dataset.manifest.get("dataset_version"),
        "candidate_count": len(dataset.candidates),
        "document_count": len(documents),
        "chunk_count": chunk_count,
        "index_dir": str(index_path),
        "embedding_model": embedding_model,
        "existing_chroma_db_modified": False,
        "index_manifest": manifest,
        "summary_only": True,
    }


def _reject_chroma_db(index_path: Path) -> None:
    resolved = index_path.resolve()
    chroma = (PROJECT_ROOT / "chroma_db").resolve()
    if resolved == chroma or chroma in resolved.parents or index_path.name == "chroma_db":
        raise ValueError("refusing to use existing chroma_db for recruitment evaluation index")


def _real_llama_index_writer(
    *,
    documents,
    index_dir: Path,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Dict[str, Any]:
    from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb

    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
    Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chroma_client = chromadb.PersistentClient(path=str(index_dir))
    collection = chroma_client.get_or_create_collection("recruitment_eval_v1")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    llama_documents = [Document(text=item["text"], metadata=item["metadata"]) for item in documents]
    index = VectorStoreIndex.from_documents(llama_documents, storage_context=storage_context, show_progress=False)
    index.storage_context.persist(persist_dir=str(index_dir))
    return {"chunk_count": collection.count(), "summary_only": True}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare recruitment evaluation index dry-run.")
    parser.add_argument("--dataset-dir", default="evaluation_data/v1")
    parser.add_argument("--index-dir", default="evaluation_indexes/recruitment_eval_v1_chroma")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        result = build_recruitment_eval_index(
            args.dataset_dir,
            args.index_dir,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            force=args.force,
        )
    else:
        result = build_recruitment_eval_index_dry_run(args.dataset_dir, args.index_dir, args.embedding_model)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"{result['status']}: dry_run={result.get('dry_run', False)}")
    return 0 if result["status"] in {"ok", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
