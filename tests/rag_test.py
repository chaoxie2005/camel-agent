from typing import List
import os
import dotenv


from camel_agent.toolkits.rag_search import rag_search
from camel_agent.components.retriever import MilvusRetriever
from langchain_core.documents import Document
from camel_agent.components.store import MilvusStore
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
    milvus_store = MilvusStore(
        embedding_function=embedding,
        collection_name="camel_agent",
        url="http://localhost:19530",
    )
    retriever = MilvusRetriever(
        milvus_store=milvus_store,
        score_threshold=0.3
    )
    return retriever._get_relevant_documents(query)


if __name__ == "__main__":
    query = " CPU使用率过高告警处理方案 ## 告警名称"
    rag_documents = test_rag_search(query)
    print(rag_documents)
