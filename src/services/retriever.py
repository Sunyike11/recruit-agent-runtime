
import os
from typing import List, Dict, Any
from src.config import get_settings
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.readers.
# from llama_index.readers.file import PyPDFReader  # 或使用 SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core import SimpleDirectoryReader

import chromadb
import fitz



class CustomPDFReader:
    """使用 PyMuPDF 提取文本，解决乱码问题"""
    def load_data(self, file_path: str, extra_info: dict = None) -> List[Document]:
        text = ""
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    page_text = page.get_text("text")
                    if page_text:
                        text += page_text + "\n"
            # 清洗文本
            text = clean_text(text)
            # 清理非法字符
            text = text.replace('\x00', '')
            # 如果提取的文本为空，可能是扫描件，可考虑后续增加 OCR
            if len(text.strip()) == 0:
                print(f"警告：{file_path} 提取的文本为空，可能为扫描件")
            # 构建 Document
            doc = Document(
                text=text,
                metadata={"file_name": os.path.basename(file_path)}
            )
            return [doc]
        except Exception as e:
            print(f"读取文件 {file_path} 出错: {e}")
            return []

import re

def clean_text(text: str) -> str:
    """
    清洗文本：
    - 去除开头和结尾的空白字符
    - 将连续的多个空格、制表符替换为单个空格
    - 将连续的多个换行符替换为单个换行符（保留段落结构）
    """
    # 先去除首尾空白
    text = text.strip()
    # 将多个空白（包括空格、制表符）替换为单个空格
    text = re.sub(r'[ \t]+', ' ', text)
    # 将多个连续的换行符（包括\r\n）替换为单个换行符
    text = re.sub(r'[\n\r]+', '\n', text)
    # 可选：移除每行首尾空格（已经在上面的全局空格压缩中处理了）
    return text

class ResumeRetriever:
    def __init__(self, persist_dir: str = None, model_name: str = None):
        settings = get_settings()
        self.persist_dir = persist_dir or str(settings.chroma_dir)
        self.embed_model = HuggingFaceEmbedding(model_name=model_name or settings.embedding_model)
        self.index = None
        self.vector_store = None
        # self.file_extractor = {".pdf": custom_pdf_reader}

        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            self._load_index()
        else:
            print(f"向量库目录 {self.persist_dir} 不存在或为空，请先运行构建脚本。")

    def _load_index(self):
        if not os.path.exists(self.persist_dir):
            print(f"持久化目录 {self.persist_dir} 不存在，无法加载索引。")
            return

        # 创建 Chroma 客户端和向量存储
        chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        chroma_collection = chroma_client.get_or_create_collection("resumes")
        count = chroma_collection.count()
        print(f"Chroma 向量库中实际存储的记录数: {count}")
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # 创建存储上下文，指定持久化目录和向量存储
        storage_context = StorageContext.from_defaults(
            persist_dir=self.persist_dir,
            vector_store=self.vector_store
        )
        self.index = load_index_from_storage(storage_context, embed_model=self.embed_model)
        # 统计信息
        print(f"已从 {self.persist_dir} 加载索引")
        # print(f"文档存储中的节点数: {len(self.index.docstore.docs)}")

        # 打印统计信息
        # print(f"已从 {self.persist_dir} 加载索引，包含 {len(self.index.docstore.docs)} 个文档。")
        # all_nodes = list(self.index.docstore.docs.values())
        # print(f"索引中共有 {len(all_nodes)} 个节点")
        # unique_files = set()
        # for doc in all_nodes:
        #     file_name = doc.metadata.get("file_name", doc.metadata.get("source", ""))
        #     if file_name:
        #         unique_files.add(file_name)
        # print(f"唯一简历文件数: {len(unique_files)}")

    def build_index_from_pdfs(self, pdf_dir: str):
        # 读取 PDF

        file_extractor = {".pdf": CustomPDFReader()}  # 注意：实例化类
        reader = SimpleDirectoryReader(input_dir=pdf_dir, file_extractor=file_extractor, required_exts=[".pdf"], recursive=True)
        documents = reader.load_data()
        print(f"成功加载 {len(documents)} 个 pdf 文件，每个文件作为一个 Document")

        # 添加元数据
        for doc in documents:
            doc.metadata["source"] = doc.metadata.get("file_name", "unknown")
            # print(f"文件: {doc.metadata.get('file_name')}")
            # print(f"文本预览: {doc.text[:500]}")
            # print("-" * 50)

        # 创建 Chroma 客户端和向量存储
        chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        chroma_collection = chroma_client.get_or_create_collection("resumes")
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # 创建存储上下文，明确指定向量存储、文档存储和索引存储
        from llama_index.core.storage.docstore import SimpleDocumentStore
        from llama_index.core.storage.index_store import SimpleIndexStore
        docstore = SimpleDocumentStore()
        index_store = SimpleIndexStore()
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=docstore,
            index_store=index_store
        )

        # 创建索引
        self.index = VectorStoreIndex.from_documents(
            documents,
            embed_model=self.embed_model,
            storage_context=storage_context,
            show_progress=True
        )

        # 持久化所有存储（向量存储已由 Chroma 管理，这里保存 docstore 和 index_store）
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        print(f"已成功构建并持久化 {len(documents)} 份简历的索引到 {self.persist_dir}")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """检索匹配的简历"""
        if not self.index:
            raise ValueError("索引未加载，请先构建索引或确保持久化目录存在。")
        retriever = self.index.as_retriever(similarity_top_k=k)
        nodes = retriever.retrieve(query)
        # 按文件名去重，保留第一次出现（通常分数最高的节点先出现）
        seen = set()
        results = []
        for node in nodes:
            # 获取唯一标识：通常使用文件名或文件路径
            file_name = node.metadata.get("file_name", node.metadata.get("source", ""))
            if file_name and file_name not in seen:
                seen.add(file_name)
                results.append({
                    "text": node.text,
                    "metadata": node.metadata,
                    "score": node.score if hasattr(node, "score") else None
                })

        return results

    def add_pdfs(self, pdf_paths: List[str]):
        """
        增量添加新的 PDF 文件到已有索引
        :param pdf_paths: PDF 文件路径列表
        """
        if not self.index:
            raise ValueError("索引未加载，请先构建索引或确保持久化目录存在。")

        # 读取新 PDF 文件
        documents = []
        for path in pdf_paths:
            # 使用 SimpleDirectoryReader 读取单个文件
            reader = SimpleDirectoryReader(input_files=[path])
            docs = reader.load_data()
            documents.extend(docs)

        if not documents:
            print("没有新文档需要添加")
            return

        # 将新文档插入到现有索引
        for doc in documents:
            self.index.insert(doc)

        # 持久化更新后的索引
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        print(f"已成功添加 {len(documents)} 个新文档到索引")
