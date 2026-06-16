import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from src.core.state import RecruitState
from src.config import get_settings
from src.utils.llm_factory import get_llm
from src.utils.structured_output import coerce_score, parse_json_object


logger = logging.getLogger(__name__)


class MatcherAgent:
    def __init__(self):
        self.llm = get_llm()
        self.loop_limit = get_settings().matcher_loop_limit
        self.prompt = ChatPromptTemplate.from_template("""
        你是一位专业的技术面试官。请对比【招聘需求】与【候选人简历】，严格按以下逻辑评分：

        ### 第一步：硬性准入检查 (Education/Degree)
        - 如果 JD 要求“硕士”而候选人是“本科”，或专业完全不相关，则直接判定为 0 分，不再进行后续分析。
        - 注意：学历符合仅代表“通过准入”，不额外加分。

        ### 第二步：核心能力评估 (满分 80 分)
        - 技术栈匹配 (40分)：候选人是否掌握了 JD 要求的核心工具（如 PyTorch, Java）？
        - 项目相关性 (40分)：候选人是否有过类似的落地项目？项目的深度、解决的问题是否具有说服力？
        - 如果此项得分低于 50 分，则判定为“不合格”。

        ### 第三步：亮点加分 (满分 20 分)
        - 仅在核心能力评估“合格”的情况下计分。
        - 包含：顶会论文 (CVPR/ICCV 等)、大厂实习经历、开源贡献等。

        ---
        【招聘需求 (JD)】:
        {jd_info}
        
        【候选人简历】:
        {candidate_text}

        请输出 JSON：
        {{
            "candidate_name": "姓名",
            "is_hard_filter_passed": true/false,
            "core_score": 0-80,
            "bonus_score": 0-20,
            "total_score": 0-100,
            "reasoning": "请详细说明：1.学历是否合规；2.技术项目是否达到岗位要求；3.有哪些亮点加分项",
            "final_verdict": "REJECTED(不合格) / QUALIFIED(合格) / OUTSTANDING(优秀)"
        }}
        """)

    def __call__(self, state: RecruitState):
        jd_info = state["extracted_jd"]
        candidates = state["candidate_pool"]

        # 存储本轮所有候选人的匹配报告
        all_reports = []

        for cand in candidates:
            # 这里的 cand 是从 Retriever 拿到的字典，包含 text 和 metadata
            cand_text = cand.get("text", "")
            # print(f"候选人文本预览: {cand_text[:200]}...")

            chain = self.prompt | self.llm
            response = chain.invoke({
                "jd_info": json.dumps(jd_info, ensure_ascii=False),
                "candidate_text": cand_text
            })

            content = response.content
            try:
                report = parse_json_object(content)
                report["total_score"] = coerce_score(report.get("total_score", 0))
                logger.debug(
                    "Matcher report parsed",
                    extra={
                        "candidate_name_present": bool(report.get("candidate_name")),
                        "score_present": "total_score" in report,
                        "total_score": report.get("total_score", 0),
                        "summary_only": True,
                    },
                )
                all_reports.append(report)
            except Exception as e:
                # print(f"匹配报告解析失败: {e}")
                logger.warning(
                    "Matcher report parse failed",
                    extra={
                        "error_type": type(e).__name__,
                        "content_length": len(str(content or "")),
                        "summary_only": True,
                    },
                )

        # 将评分最高的报告排在前面
        all_reports.sort(key=lambda x: coerce_score(x.get("total_score", 0)), reverse=True)

        # 判定逻辑
        max_score = max([coerce_score(r.get("total_score", 0)) for r in all_reports]) if all_reports else 0

        if max_score < 60 and state.get("loop_count", 0) < self.loop_limit:
            # 触发反思：为什么没找到合适的人？
            advice = f"当前最高分仅 {max_score}。可能原因：JD 要求的 {state['extracted_jd'].get('tech_stack')} 过于严苛，建议扩大搜索关键词范围。"
            return {
                "final_reports": all_reports,
                "next_action": "refine",  # 标记为需要优化
                "refinement_advice": advice,
                "loop_count": state.get("loop_count", 0) + 1,
                "messages": [AIMessage(content=f"匹配度不足（最高{max_score}分），准备优化搜索策略。")]
            }
        else:
            return {
                "final_reports": all_reports,
                "next_action": "end",
                "messages": [AIMessage(content="匹配完成，已找到合适人选或已达重试上限。")]
            }

        # return {
        #     "final_reports": all_reports,
        #     "next_action": "end",
        #     "messages": [AIMessage(content=f"已完成对 {len(all_reports)} 名候选人的深度匹配评估。")]
        # }
