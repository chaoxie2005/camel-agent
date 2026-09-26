from typing import List
import os
import dotenv


from camel_agent.toolkits.rag_search import rag_search
from camel_agent.components.retriever import MilvusRetriever
from camel_agent.components.rerank import RerankModel
from langchain_core.documents import Document
from camel_agent.components.store import AMilvusClient
from camel_agent.components.embedding import EmbeddingModel

dotenv.load_dotenv()


def test_rag_search(query: str) -> List[Document]:
    """
    测试RAG搜索
    """
    embedding = EmbeddingModel(
        model_name=os.getenv("MODEL_NAME"),
        base_url=os.getenv("BASE_URL"),
        key=os.getenv("MODEL_API_KEY"),
    )
    rerank = RerankModel(
        model_name=os.getenv("RERANK_MODEL_NAME"),
        base_url=os.getenv("BASE_URL"),
        key=os.getenv("MODEL_API_KEY"),
        endpoint=os.getenv("RERANK_BASE_URL"),
    )
    client = AMilvusClient(
        url="http://localhost:19530",
    )
    retriever = MilvusRetriever(
        client=client,
        embedding_function=embedding,
        rerank_function=rerank,
        collection_name="camel_agent",
        score_threshold=0.3,
    )
    return retriever._get_relevant_documents(query)


if __name__ == "__main__":
    query = " CPU使用率过高告警处理方案 ## 告警名称"
    rag_documents = test_rag_search(query)
    print(rag_documents)
