import fitz  # PyMuPDF
from src.utils.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate
from src.utils.structured_output import parse_json_object


class ResumeReader:
    def __init__(self):
        self.llm = get_llm()
        self.prompt = ChatPromptTemplate.from_template("""
        你是一位专业的HR助手。请从下方的简历文本中提取关键信息，并输出为JSON格式。

        简历文本: {resume_text}

        请严格按以下格式输出JSON：
        {{
            "name": "姓名",
            "university": "毕业院校",
            "degree": "学位",
            "skills": ["技能1", "技能2"],
            "project_experience": ["项目描述1", "项目描述2"],
            "summary": "个人核心竞争力简述"
        }}
        """)

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """从PDF中提取纯文本内容"""
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text

    def parse_resume(self, pdf_path: str) -> dict:
        """解析PDF简历并返回结构化数据"""
        raw_text = self._extract_text_from_pdf(pdf_path)
        chain = self.prompt | self.llm
        result = chain.invoke({"resume_text": raw_text})

        content = result.content

        try:
            return parse_json_object(content)
        except:
            print(f"简历解析结构化失败: {pdf_path}")
            return {"name": "未知", "raw_text": raw_text}  # 兜底逻辑
