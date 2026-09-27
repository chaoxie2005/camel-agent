from camel_agent.rag_components.retriever import MilvusRetriever
from langchain_core.documents import Document
from typing import List
from camel.toolkits import FunctionTool



def rag_search(
    query: str,
    retriever: MilvusRetriever,
) -> List[Document]:
    """
    RAG 搜索
    """
    return retriever.invoke(query)


def make_rag_search_tool(retriever: MilvusRetriever) -> FunctionTool:
    """构造 RAG 检索工具（绑定 retriever，仅向 LLM 暴露 query 参数）"""

    def rag_search_tool(query: str) -> str:
        r"""RAG 知识库检索工具。

        当用户提问涉及内部知识库/文档/运维方案等内容，
        需要查询已入库文档时调用此工具。

        Args:
            query (str): 检索查询文本。

        Returns:
            str: 相关文档片段（含相关性分数），未命中时返回提示。
        """
        docs = retriever._get_relevant_documents(query)
        if not docs:
            return "未检索到相关文档"
        return "\n\n".join(
            f"[score={doc.metadata.get('score', '?')}]\n{doc.page_content}"
            for doc in docs
        )

    return FunctionTool(rag_search_tool)
