from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
from camel_agent.components.embedding import EmbeddingModel

class MilvusStore:
    """
    Milvus 存储
    """

    def __init__(
        self,
        embedding_function: EmbeddingModel,
        collection_name: str = "camel_agent",
        url: str = "http://localhost:19530",
    ):
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        # TODO: 使用pymilvus自己实现Milvus类，避免依赖langchain_milvus
        self.milvus = Milvus(
            embedding_function=embedding_function,
            connection_args={"url": url},
            collection_name=collection_name,
        )

    # TODO: 实现Milvus集合是否存在检查
    def is_collection_exists(self) -> None:
        """
        检查Milvus集合是否存在
        """
        pass

    # TODO: 实现Milvus集合创建
    def create_collection(self) -> None:
        """
        创建Milvus集合
        """
        pass

    # TODO: 实现Milvus集合删除
    def delete_collection(self) -> None:
        """
        删除Milvus集合
        """
        pass

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        添加文档到 Milvus 数据库
        """
        return self.milvus.add_documents(documents)

    async def aadd_documents(self, documents: list[Document]) -> list[str]:
        """
        异步添加文档到 Milvus 数据库
        """
        return await self.milvus.aadd_documents(documents)

    def query(self, query: str) -> list[Document]:
        """
        查询Milvus数据库 纯相似度搜索
        """
        return self.milvus.similarity_search(query)

    def similarity_search_with_score_by_vector(
        self,
        query: str,
        k: int = 4,
        partition_name: str = "_default",
    ) -> list[tuple[Document, float]]:
        """
        查询Milvus数据库 包含相似度分数
        """
        embedding = self.embedding_function.embed_query(query)
        return self.milvus.similarity_search_with_score_by_vector(
            embedding,
            k=k,
            partition_names=[partition_name],
        )
