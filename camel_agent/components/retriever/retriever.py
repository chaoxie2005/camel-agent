from camel_agent.components.store import MilvusStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class MilvusRetriever(BaseRetriever):
    """Milvus 纯向量召回器"""

    milvus_store: MilvusStore
    top_k: int = 3
    score_threshold: float | None = None
    partition_name: str = "_default"

    def embedding_search(self, query: str) -> list[tuple[Document, float]]:
        """
        纯向量检索，返回 (文档, 分数) 列表
        """
        results = self.milvus_store.similarity_search_with_score_by_vector(
            query,
            k=self.top_k,
            partition_name=self.partition_name,
        )
        if self.score_threshold is not None:
            results = [
                (doc, score)
                for doc, score in results
                if score >= self.score_threshold
            ]
        return results

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> list[Document]:
        """
        LangChain 标准召回接口
        """
        docs = []
        for doc, score in self.embedding_search(query):
            doc.metadata["score"] = score
            docs.append(doc)
        docs.sort(key=lambda x: x.metadata["score"], reverse=True)
        return docs
