from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage


class RecruitState(TypedDict, total=False):
    # 使用 Annotated 确保消息是增量追加的，而不是覆盖
    messages: Annotated[List[dict], add_messages]

    # 当前处理的结构化招聘需求
    extracted_jd: dict

    # 候选人池：存储初步筛选出的简历路径或简要画像
    candidate_pool: List[dict]

    # 最终匹配报告：包含匹配度分数和理由
    final_reports: List[dict]

    # 新增：记录尝试次数
    loop_count: int
    # 新增：Matcher 给出的反思建议
    refinement_advice: str

    # 状态控制：下一步去哪？
    next_action: str

    # 新增：人工审核反馈
    human_feedback: str


def create_initial_state(jd_text: str) -> RecruitState:
    """Create a complete initial graph state for one recruiting request."""
    return {
        "messages": [HumanMessage(content=jd_text)],
        "extracted_jd": {},
        "candidate_pool": [],
        "final_reports": [],
        "loop_count": 0,
        "refinement_advice": "",
        "next_action": "",
        "human_feedback": "",
    }
