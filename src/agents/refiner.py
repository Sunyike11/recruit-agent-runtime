from src.utils.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate
from src.core.state import RecruitState
from langchain_core.messages import AIMessage

class RefinerAgent:
    def __init__(self):
        self.llm = get_llm()
        self.prompt = ChatPromptTemplate.from_template("""
        你是一位搜索优化专家。
        目前的招聘搜索结果不佳，建议如下：{advice}
        原始搜索词：{old_query}

        请结合原始 JD 需求，生成一个更具“泛化性”的新搜索词（例如：缩减不必要的限制、增加同义词）。
        仅输出新的搜索词，不要有其他文字。
        """)

    def __call__(self, state: RecruitState):
        # 1. 优先使用人工反馈
        user_feedback = state.get("human_feedback")
        if user_feedback and user_feedback.strip():
            # 直接使用用户提供的搜索词（假设用户已经输入了完整的新搜索词）
            new_query = user_feedback
            print(f"[Refiner] 使用人工反馈: {new_query}")
        else:
            # 2. 否则由 LLM 生成优化搜索词
            chain = self.prompt | self.llm
            new_query = chain.invoke({
                "advice": state["refinement_advice"],
                "old_query": state["extracted_jd"].get("search_query", "")
            }).content.strip()
            print(f"[Refiner] LLM 生成: {new_query}")

        # 更新 extracted_jd 中的搜索词
        updated_jd = state["extracted_jd"].copy()
        updated_jd["search_query"] = new_query

        return {
            "extracted_jd": updated_jd,
            "messages": [AIMessage(content=f"优化搜索词为：{new_query}")]
        }