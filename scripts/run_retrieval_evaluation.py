#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.evaluation.retrieval_export import export_retrieval_report_json, export_retrieval_report_text
from src.evaluation.retrieval_runner import RetrievalEvaluationConfig, RetrievalEvaluationRunner


class RealEvaluationRetriever:
    def __init__(self, index_dir: str | Path, embedding_model: str, top_k: int):
        from llama_index.core import StorageContext, load_index_from_storage
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.vector_stores.chroma import ChromaVectorStore
        import chromadb

        self.top_k = top_k
        chroma_client = chromadb.PersistentClient(path=str(index_dir))
        collection = chroma_client.get_or_create_collection("recruitment_eval_v1")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(persist_dir=str(index_dir), vector_store=vector_store)
        index = load_index_from_storage(
            storage_context,
            embed_model=HuggingFaceEmbedding(model_name=embedding_model),
        )
        self.retriever = index.as_retriever(similarity_top_k=top_k)

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        nodes = self.retriever.retrieve(query)
        results = []
        for idx, node in enumerate(nodes, start=1):
            if idx > k:
                break
            metadata = dict(getattr(node, "metadata", {}) or {})
            results.append(
                {
                    "candidate_id": metadata.get("candidate_id", ""),
                    "source_document_id": metadata.get("source_document_id", ""),
                    "chunk_id": getattr(node, "node_id", "") or metadata.get("chunk_id", ""),
                    "score": getattr(node, "score", None),
                    "rank": idx,
                    "metadata": {
                        "candidate_id": metadata.get("candidate_id", ""),
                        "source_document_id": metadata.get("source_document_id", ""),
                        "dataset_version": metadata.get("dataset_version", ""),
                    },
                }
            )
        return results


def build_real_eval_retriever_factory(index_dir: str | Path, embedding_model: str, top_k: int):
    def factory(_config: RetrievalEvaluationConfig) -> RealEvaluationRetriever:
        return RealEvaluationRetriever(index_dir=index_dir, embedding_model=embedding_model, top_k=top_k)

    return factory


def main() -> int:
    parser = argparse.ArgumentParser(description="Run candidate-level retrieval evaluation.")
    parser.add_argument("--dataset-dir", default="evaluation_data/v1")
    parser.add_argument("--index-dir", default="evaluation_indexes/recruitment_eval_v1_chroma")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--query-mode", choices=["raw_jd", "structured"], default="raw_jd")
    parser.add_argument("--top-k", default="5,10")
    parser.add_argument("--output", default="")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    top_k_values = [int(item.strip()) for item in args.top_k.split(",") if item.strip()]
    config = RetrievalEvaluationConfig(
        dataset_dir=args.dataset_dir,
        index_dir=args.index_dir,
        query_mode=args.query_mode,
        top_k_values=top_k_values,
        index_version="phase13b-v1",
        embedding_model=args.embedding_model,
    )
    runner = RetrievalEvaluationRunner(config)
    try:
        report = runner.run_with_retriever_factory(
            build_real_eval_retriever_factory(args.index_dir, args.embedding_model, max(top_k_values))
        )
        payload = export_retrieval_report_json(report) if args.json else export_retrieval_report_text(report)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 0
    except Exception as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_hint": "retrieval_evaluation_failed",
            "summary_only": True,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
