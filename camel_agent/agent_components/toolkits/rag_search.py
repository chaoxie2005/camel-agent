from camel.toolkits import FunctionTool
from langchain_core.retrievers import BaseRetriever


def make_rag_search_tool(retriever: BaseRetriever) -> FunctionTool:
    """绑定检索器，仅向 LLM 暴露 query 参数。"""

    async def rag_search_tool(query: str) -> str:
        """检索内部知识库中的相关文档。

        Args:
            query (str): 检索查询文本。

        Returns:
            str: 相关文档片段及相关性分数，未命中时返回提示。
        """
        docs = await retriever.ainvoke(query)
        if not docs:
            return "未检索到相关文档"
        return "\n\n".join(
            f"[score={doc.metadata.get('score', '?')}]\n{doc.page_content}"
            for doc in docs
        )

    return FunctionTool(rag_search_tool)
