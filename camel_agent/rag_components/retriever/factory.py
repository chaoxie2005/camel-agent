import os

from camel_agent.rag_components.embedding import EmbeddingModel
from camel_agent.rag_components.rerank import RerankModel
from camel_agent.rag_components.retriever.retriever import MilvusRetriever
from camel_agent.rag_components.store import AMilvusClient


def build_retriever_from_env(
    milvus_url: str = "http://localhost:19530",
    collection_name: str = "camel_agent",
    score_threshold: float = 0.3,
) -> MilvusRetriever:
    """从 .env 配置构造 MilvusRetriever"""
    embedding = EmbeddingModel(
        model_name=os.getenv("MODEL_NAME"),  # type: ignore
        base_url=os.getenv("BASE_URL"),  # type: ignore
        key=os.getenv("MODEL_API_KEY"),  # type: ignore
    )
    rerank = RerankModel(
        model_name=os.getenv("RERANK_MODEL_NAME"),  # type: ignore
        base_url=os.getenv("BASE_URL"),  # type: ignore
        key=os.getenv("MODEL_API_KEY"),  # type: ignore
        endpoint=os.getenv("RERANK_BASE_URL"),
    )
    client = AMilvusClient(url=milvus_url)
    return MilvusRetriever(
        client=client,
        embedding_function=embedding,
        rerank_function=rerank,
        collection_name=collection_name,
        score_threshold=score_threshold,
    )
