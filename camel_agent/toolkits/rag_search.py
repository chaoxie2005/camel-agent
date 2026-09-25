from camel_agent.components.retriever import MilvusRetriever
from langchain_core.documents import Document
from typing import List



def rag_search(
    query: str,
    retriever: MilvusRetriever,
) -> List[Document]:
    """
    RAG 搜索
    """
    return retriever._get_relevant_documents(query)
