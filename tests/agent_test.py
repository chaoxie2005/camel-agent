import os
import asyncio
import dotenv

from camel_agent.agent_components.agent.agent import ChatAgentClient
from camel_agent.agent_components.llm.model import ModelClient
from camel_agent.agent_components.toolkits.rag_search import make_rag_search_tool
from camel_agent.rag_components.retriever import build_retriever_from_env


dotenv.load_dotenv()

async def test_agent_chat():
    """agent测试"""
    model_client = ModelClient(
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # type: ignore
        url=os.getenv("DEEPSEEK_URL"),  # type: ignore
    )
    retriever = build_retriever_from_env()
    tools = [make_rag_search_tool(retriever)]
    system_message = "你是一个专业的助手，你的任务是回答用户的问题。"
    agent_client = ChatAgentClient(model_client, system_prompt=system_message, tools=tools)
    response = await agent_client.run_chat("CPU使用率过高告警处理方案是什么？")
    print(response)


if __name__ == "__main__":
    asyncio.run(test_agent_chat())
