import os

import dotenv
from camel.embeddings import OpenAICompatibleEmbedding
from camel.memories import (
        ChatHistoryBlock,  # 聊天历史记忆块：保存最近的对话记录
        ContextRecord,
        LongtermAgentMemory,  # 长期 Agent 记忆：统一管理不同类型的记忆
        ScoreBasedContextCreator,  # 基于评分选择记忆，构建最终上下文
    VectorDBBlock,  # 向量数据库记忆块：通过向量检索相关历史记忆
)
from camel.storages.vectordb_storages import MilvusStorage
from camel.types import ModelType, OpenAIBackendRole
from camel.utils import OpenAITokenCounter

from .mongo_store import MongoKeyValueStorage
from .sanitize import SanitizingContextCreator

dotenv.load_dotenv()


class MongoMilvusMemory(LongtermAgentMemory):
    """
    基于 MongoDB 和 Milvus 的长期 Agent 记忆
    """

    def __init__(
        self,
        mongo_db_name: str,
        mongo_collection_name: str,
        milvus_db_name: str,
        context_creator: ScoreBasedContextCreator | None = None,
        embedding: OpenAICompatibleEmbedding | None = None,
        mongo_url: str | None = None,
        mongo_server_selection_timeout_ms: int = 5000,
        vector_db_url: str | None = None,
        vector_db_timeout: float | None = None,
        retrieve_limit: int = 3,
        agent_id: str | None = None,
    ):
        """
        Args:
            db_name (str): MongoDB 数据库名称。
            collection_name (str): MongoDB 集合名称。
            vector_db_name (str): Milvus 集合名称。
            context_creator (ScoreBasedContextCreator | None): 上下文创建器，
            embedding (OpenAICompatibleEmbedding | None): 向量化模型
            mongo_url (str | None): MongoDB地址
            mongo_server_selection_timeout_ms (int): MongoDB 服务器选择超时（毫秒）。
            vector_db_url (str | None): Milvus 连接地址
            vector_db_timeout (float | None): Milvus 请求超时（秒）
            retrieve_limit (int): 向量检索召回条数。
            agent_id (str | None): Agent 标识。
        """
        self.mongo_store = MongoKeyValueStorage(
            mongo_db_name,
            mongo_collection_name,
            mongo_url or "mongodb://localhost:27018",
            mongo_server_selection_timeout_ms,
        )

        self.embedding = embedding or OpenAICompatibleEmbedding(
            model_type=os.getenv("MODEL_EMBEDDING_NAME", ""),
            api_key=os.getenv("MODEL_API_KEY"),
            url=os.getenv("BASE_URL"),
        )

        self.milvus_db = MilvusStorage(
            vector_dim=self.embedding.get_output_dim(),
            url_and_api_key=(vector_db_url or "http://localhost:19530", ""),
            collection_name=milvus_db_name,
            timeout=vector_db_timeout,
        )
        # Milvus 集合须加载进内存后才能检索（VectorDBBlock 不会自动调用）
        self.milvus_db.load()

        super().__init__(
            context_creator=context_creator
            or SanitizingContextCreator(
                token_counter=OpenAITokenCounter(ModelType.DEEPSEEK_CHAT),
                token_limit=int(os.getenv("TOKEN_LIMIT", 4096)),
            ),
            chat_history_block=ChatHistoryBlock(storage=self.mongo_store),
            vector_db_block=VectorDBBlock(
                storage=self.milvus_db,
                embedding=self.embedding,
            ),
            retrieve_limit=retrieve_limit,
            agent_id=agent_id,
        )

    def retrieve(self) -> list[ContextRecord]:
        """覆写 LongtermAgentMemory.retrieve()：

        1. 向量召回结果按 uuid 与聊天历史去重（原实现直接拼接，重复记录
           会导致 assistant(tool_calls) 连发、配对被打破）；
        2. 丢弃召回的 FUNCTION/TOOL 记录及带 tool_calls 的 assistant 记录
           —— Milvus 侧（VectorDBBlock 过滤空 content）本就存不进配对的
           tool 结果，召回它们必然产生孤儿 tool_calls。
        3. topic 为空时跳过向量检索：embedding API 对空串返回 2560 维，
           与集合 1024 维不匹配会直接报 dimension mismatch。
        保持 camel 原有返回结构 chat[:1] + vector + chat[1:] 不变。
        """
        chat_history = self.chat_history_block.retrieve()
        vector_hits: list[ContextRecord] = []
        if self._current_topic and self._current_topic.strip():
            vector_hits = self.vector_db_block.retrieve(
                self._current_topic, self.retrieve_limit
            )

        history_uuids = {r.memory_record.uuid for r in chat_history}
        filtered: list[ContextRecord] = []
        for record in vector_hits:
            mem = record.memory_record
            if mem.uuid in history_uuids:
                continue
            if mem.role_at_backend in (
                OpenAIBackendRole.FUNCTION,
                OpenAIBackendRole.TOOL,
            ):
                continue
            if mem.role_at_backend == OpenAIBackendRole.ASSISTANT:
                meta = mem.message.meta_dict or {}
                if meta.get("tool_calls"):
                    continue
            filtered.append(record)

        return chat_history[:1] + filtered + chat_history[1:]

    # ===================== 生命周期 =====================

    def close(self) -> None:
        """关闭 MongoDB 连接"""
        self.mongo_store.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def clear(self) -> None:
        """
        清空记忆

        ChatAgent.init_messages() 在已有历史时会调用 clear()，
        而 MilvusStorage.clear() 是删表重建，新表处于 NotLoad 状态，
        必须重新 load()，否则后续检索报 collection not loaded。
        """
        super().clear()
        self.milvus_db.load()

