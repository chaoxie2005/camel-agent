import asyncio

from camel_agent.rag_components.store import AMilvusClient
from camel_agent.rag_components.embedding import EmbeddingModel
from camel_agent.rag_components.rerank import RerankModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pymilvus import AnnSearchRequest, RRFRanker

class MilvusRetriever(BaseRetriever):
    """Milvus 混合召回器(dense + BM25 全文 + 重排)"""

    client: AMilvusClient
    embedding_function: EmbeddingModel
    rerank_function: RerankModel | None = None
    collection_name: str = "camel_agent"
    top_k: int = 3
    score_threshold: float | None = None
    partition_name: str = "_default"
    candidate_factor: int = 4

    def embedding_search(self, query: str) -> list[tuple[Document, float]]:
        """
        纯向量检索，返回 (文档, 分数) 列表
        """
        embedding = self.embedding_function.embed_query(query)
        results = asyncio.run(
            self.client.vector_search(
                collection_name=self.collection_name,
                query=embedding,
                partition_names=[self.partition_name],
                top_k=self.top_k,
                output_fields=["child_content", "child_metadata"],
            )
        )
        docs: list[tuple[Document, float]] = []
        for hit in results[0]:
            entity = hit.get("entity") or {}
            doc = Document(
                page_content=entity.get("child_content", ""),
                metadata=entity.get("child_metadata") or {},
            )
            docs.append((doc, hit.get("distance", 0.0)))
        if self.score_threshold is not None:
            results = [
                (doc, score)
                for doc, score in docs
                if score >= self.score_threshold
            ]
            return results
        return docs

    def hybrid_search(self, query: str) -> list[tuple[Document, float]]:
        """
        混合检索: dense 向量与 BM25 全文双路召回，RRF 融合后经重排模型精排

        Args:
            query: 查询文本

        Returns:
            [(文档, 分数)]，分数为重排相关性分数，
            未配置重排模型时为 RRF 融合分数
        """
        embedding = self.embedding_function.embed_query(query)
        candidate_k = self.top_k * self.candidate_factor
        ann_requests = [
            AnnSearchRequest(
                data=[embedding],
                anns_field="dense_vector",
                param={"metric_type": "COSINE"},
                limit=candidate_k,
            ),
            AnnSearchRequest(
                data=[query],
                anns_field="sparse_vector",
                param={"metric_type": "BM25"},
                limit=candidate_k,
            ),
        ]
        results = asyncio.run(
            self.client.hybrid_search(
                collection_name=self.collection_name,
                requests=ann_requests,
                ranker=RRFRanker(),
                partition_names=[self.partition_name],
                top_k=candidate_k,
                output_fields=["child_content", "child_metadata"],
            )
        )
        candidates: list[tuple[Document, float]] = []
        for hit in results[0]:
            entity = hit.get("entity") or {}
            doc = Document(
                page_content=entity.get("child_content", ""),
                metadata=entity.get("child_metadata") or {},
            )
            candidates.append((doc, hit.get("distance", 0.0)))
        if self.rerank_function is not None and candidates:
            ranked = self.rerank_function.rerank(
                query=query,
                documents=[doc.page_content for doc, _ in candidates],
                top_n=self.top_k,
            )
            docs = [(candidates[index][0], score) for index, score in ranked]
        else:
            docs = candidates[: self.top_k]
        if self.score_threshold is not None:
            return [
                (doc, score)
                for doc, score in docs
                if score >= self.score_threshold
            ]
        return docs

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> list[Document]:
        """
        检索与查询最相关的文档

        流程:
        1. 混合检索: dense 向量(COSINE)与 BM25 全文双路召回候选集，
           候选池大小为 top_k * candidate_factor
        2. RRF 融合两路召回结果
        3. 若配置重排模型，调用重排模型对候选精排并截取 top_k
        4. 按 score_threshold 过滤低分文档
        5. 分数写入 metadata["score"]，按分数降序返回

        Args:
            query: 用户查询文本
            run_manager: LangChain 回调管理器，此处未使用

        Returns:
            按相关性分数降序排列的文档列表
        """
        docs = []
        for doc, score in self.hybrid_search(query):
            doc.metadata["score"] = score
            docs.append(doc)
        docs.sort(key=lambda x: x.metadata["score"], reverse=True)
        return docs
