"""MongoMilvusMemory 接入示例

MongoDB 持久化聊天历史 + Milvus 语义召回。
前置：docker compose up -d（mongo:27018、milvus:19530）
"""
import asyncio
import os

import dotenv

from camel_agent.agent_components.agent.agent import ChatAgentClient
from camel_agent.agent_components.llm.model import ModelClient
from camel_agent.agent_components.memory.memory import MongoMilvusMemory

dotenv.load_dotenv()


async def main():
    # 1. 创建记忆（一个会话独占一组 collection，避免历史串号）
    with MongoMilvusMemory(
        mongo_db_name="camel_memory",
        mongo_collection_name="demo_session",
        milvus_db_name="demo_session_memory",
    ) as memory:
        # 2. 挂到 ChatAgent
        # 注意：传入 memory 后 ChatAgent 忽略 message_window_size，
        # 上下文裁剪由 memory 内的 ScoreBasedContextCreator(4096 token) 负责
        agent_client = ChatAgentClient(
            model_client=ModelClient(
                api_key=os.getenv("DEEPSEEK_API_KEY"),  # type: ignore
                url=os.getenv("DEEPSEEK_URL"),  # type: ignore
            ),
            system_prompt="你是一个专业的助手。",
            memory=memory,
        )

        # 3. 多轮对话
        answer1 = await agent_client.run_chat("我正在学习CAMEL框架，请介绍一下它。")
        print("第一轮:", answer1)
        answer2 = await agent_client.run_chat("它和LangChain有什么区别？")
        print("第二轮:", answer2)


if __name__ == "__main__":
    asyncio.run(main())
