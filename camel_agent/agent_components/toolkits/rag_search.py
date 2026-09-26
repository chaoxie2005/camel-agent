from camel_agent.rag_components.retriever import MilvusRetriever
from langchain_core.documents import Document
from typing import List



def rag_search(
    query: str,
    retriever: MilvusRetriever,
) -> List[Document]:
    """
    RAG 搜索
    """
    return retriever.invoke(query)
