"""需要真实服务的 RAG 手动验证脚本。"""
import asyncio

import dotenv

from camel_agent.rag_components.retriever import build_retriever_from_env


async def main():
    dotenv.load_dotenv()
    retriever = build_retriever_from_env()
    try:
        docs = await retriever.ainvoke("CPU使用率过高告警处理方案 ## 告警名称")
        print(docs)
    finally:
        await retriever.client.close()


if __name__ == "__main__":
    asyncio.run(main())
