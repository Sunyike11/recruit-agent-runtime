import os
from typing import Optional
from langchain_openai import ChatOpenAI
from src.config import get_settings
# 如果你想用其他模型，可以按需导入，例如：
# from langchain_community.chat_models import ChatOllama

def get_llm(model_name: Optional[str] = None, temperature: Optional[float] = None):
    """
    统一获取 LLM 实例的工厂方法
    """
    # 建议在项目根目录创建 .env 文件并使用 python-dotenv 加载
    settings = get_settings()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE") # 如果使用代理或国产大模型转发

    if not api_key:
        raise ValueError("请在环境变量中设置 OPENAI_API_KEY")

    return ChatOpenAI(
        model=model_name or settings.llm_model,
        temperature=settings.llm_temperature if temperature is None else temperature,
        api_key=api_key,
        base_url=base_url
    )
