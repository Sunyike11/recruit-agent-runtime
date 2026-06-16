
from langchain_core.messages import AIMessage
from src.services.retriever import ResumeRetriever
from src.core.state import RecruitState
from src.config import get_settings
from typing import Optional

# 问题 不应该检索简历吧,不应该是岗位吗

class RetrieverAgent:
    def __init__(self, retriever=None, top_k: Optional[int] = None):
        settings = get_settings()
        self.retriever = retriever or ResumeRetriever(persist_dir=str(settings.chroma_dir))
        self.top_k = top_k or settings.retriever_top_k

    def __call__(self, state: RecruitState):
        query = state["extracted_jd"].get("search_query", "")
        if not query:
            return {
                "candidate_pool": [],
                "next_action": "match_evaluation",
                "messages": [AIMessage(content="未提供搜索词，跳过检索。")]
            }
        matches = self.retriever.search(query, k=self.top_k)
        candidate_pool = [{"text": m["text"], "metadata": m["metadata"]} for m in matches]
        return {
            "candidate_pool": candidate_pool,
            "next_action": "match_evaluation",
            "messages": [AIMessage(content=f"已根据需求检索到 {len(candidate_pool)} 名匹配候选人。")]
        }
