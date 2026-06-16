from langchain_core.prompts import ChatPromptTemplate
from src.utils.llm_factory import get_llm
from src.core.state import RecruitState
from langchain_core.messages import AIMessage
from src.utils.structured_output import parse_json_object_or_default


class PlannerAgent:
    def __init__(self):
        self.llm = get_llm()
        self.prompt = ChatPromptTemplate.from_template("""
        你是一位资深技术猎头规划师。
        任务：分析下方的招聘需求（JD），提取出核心技术关键词、学历要求和加分项。

        JD内容: {jd_text}

        请输出 JSON 格式：
        {{
            "tech_stack": ["关键词1", "关键词2"],
            "education": "学历要求",
            "must_have": ["必须具备的经验"],
            "search_query": "用于检索简历的搜索词"
        }}
        """)

    def __call__(self, state: RecruitState):
        # 假设最新的一条消息是原始 JD
        jd_text = state["messages"][-1].content
        chain = self.prompt | self.llm
        result = chain.invoke({"jd_text": jd_text})

        # 从 AIMessage.content 中提取 JSON 内容
        content = result.content
        default_extracted = {
            "tech_stack": [],
            "education": "",
            "must_have": [],
            "search_query": ""
        }
        extracted = parse_json_object_or_default(content, default_extracted)

        # 确保字段存在
        tech_stack = extracted.get("tech_stack", [])
        education = extracted.get("education", "")
        must_have = extracted.get("must_have", [])
        search_query = extracted.get("search_query", "")

        # 更新状态：存入提取的结构化 JD，并指引下一步去检索
        return {
            "extracted_jd": extracted,
            "next_action": "retrieve_candidates",
            "messages": [AIMessage(content=f"已完成需求分析，提取关键词：{tech_stack}")]
        }
