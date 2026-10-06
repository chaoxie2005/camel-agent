import os
from uuid import uuid4

from bson import ObjectId
from pymongo import MongoClient, ReturnDocument

from camel.storages.key_value_storages.base import BaseKeyValueStorage


class MongoKeyValueStorage(BaseKeyValueStorage):
    """
    基于 MongoDB 的键值存储系统
    """

    def __init__(
        self,
        db_name: str,
        collection_name: str,
        url: str | None = None,
        server_selection_timeout_ms: int = 5000,
    ):
        """
        Args:
            db_name (str): 数据库名称。
            collection_name (str): 集合名称。
            url (str | None): MongoDB地址
            server_selection_timeout_ms (int): 服务器选择超时（毫秒），
        """
        self.db_name = db_name
        self.collection_name = collection_name
        self.url = url or os.getenv(
            "MONGO_URI", "mongodb://localhost:27018"
        )
        self.client = MongoClient(
            self.url,
            serverSelectionTimeoutMS=server_selection_timeout_ms,
        )
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.counters = self.db[f"{collection_name}__seq"]

    # ===================== BaseKeyValueStorage 契约 =====================

    def save(self, records: list[dict]) -> None:
        """
        批量保存记录（按列表顺序注入单调递增的 seq，保证 load() 顺序稳定）

        外来 _id 会被剥离并重新生成，避免 DuplicateKeyError；
        外来 seq 会被覆盖为新分配的连续值。

        Args:
            records (list[dict]): 待保存的记录列表。
        """
        if not records:
            return
        documents = []
        for record in records:
            document = dict(record)
            document.pop("_id", None)
            document.pop("seq", None)
            document["_id"] = str(uuid4())
            documents.append(document)
        start_seq = self._next_seq(len(documents))
        for offset, document in enumerate(documents):
            document["seq"] = start_seq + offset
        self.collection.insert_many(documents)

    def load(self) -> list[dict]:
        """
        全量读取记录（按 seq 升序，即写入顺序）

        Returns:
            list[dict]: 记录列表，_id 统一为 str，保证 JSON 可序列化。
        """
        records = list(self.collection.find({}).sort([("seq", 1)]))
        for record in records:
            record["_id"] = self._to_str_id(record.get("_id"))
        return records

    def clear(self) -> None:
        """
        清空集合中的全部记录
        """
        self.collection.delete_many({})

    # ===================== 连接生命周期 =====================

    def close(self) -> None:
        """
        关闭 MongoDB 连接
        """
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ===================== CRUD 扩展方法 =====================

    def insert_one_document(self, document: dict) -> None:
        """
        插入一条数据到 MongoDB 中

        Args:
            document (dict): 文档内容，无 _id 时自动生成 str 类型 _id。
        """
        document = dict(document)
        document.setdefault("_id", str(uuid4()))
        document["seq"] = self._next_seq(1)
        self.collection.insert_one(document)

    def insert_many_documents(self, documents: list[dict]) -> None:
        """
        插入多条数据到 MongoDB 中

        Args:
            documents (list[dict]): 文档列表，无 _id 时自动生成
                str 类型 _id。
        """
        if not documents:
            return
        prepared = []
        for document in documents:
            document = dict(document)
            document.setdefault("_id", str(uuid4()))
            document.pop("seq", None)
            prepared.append(document)
        start_seq = self._next_seq(len(prepared))
        for offset, document in enumerate(prepared):
            document["seq"] = start_seq + offset
        self.collection.insert_many(prepared)

    def delete_one_document(self, document_id: str) -> None:
        """
        删除一条数据

        Args:
            document_id (str): 文档 _id。
        """
        self.collection.delete_one({"_id": document_id})

    def delete_many_documents(self, document_ids: list[str]) -> None:
        """
        删除多条数据

        Args:
            document_ids (list[str]): 文档 _id 列表。
        """
        if not document_ids:
            return
        self.collection.delete_many({"_id": {"$in": document_ids}})

    def update_one_document(
        self, document_id: str, update_data: dict
    ) -> None:
        """
        更新一条数据

        Args:
            document_id (str): 文档 _id。
            update_data (dict): 待更新字段（$set 语义）。
        """
        self.collection.update_one(
            {"_id": document_id}, {"$set": update_data}
        )

    def update_many_documents(
        self, document_ids: list[str], update_data: dict
    ) -> None:
        """
        更新多条数据

        Args:
            document_ids (list[str]): 文档 _id 列表。
            update_data (dict): 待更新字段（$set 语义）。
        """
        if not document_ids:
            return
        self.collection.update_many(
            {"_id": {"$in": document_ids}}, {"$set": update_data}
        )

    def find_many_documents(self, query: dict) -> list[dict]:
        """
        查找多条数据

        Args:
            query (dict): MongoDB 查询条件。

        Returns:
            list[dict]: 匹配的文档列表，_id 统一转为 str。
        """
        documents = list(self.collection.find(query))
        for document in documents:
            document["_id"] = self._to_str_id(document.get("_id"))
        return documents

    # ===================== 内部工具 =====================

    def _next_seq(self, count: int) -> int:
        """
        原子分配 count 个连续 seq，返回本批次起始值

        用独立计数器集合 + $inc，避免用 count()/时间戳排序的竞态与并列。

        Args:
            count (int): 本次需要分配的序号数量。

        Returns:
            int: 本批次第一个 seq（后续值依次 +1）。
        """
        doc = self.counters.find_one_and_update(
            {"_id": "seq"},
            {"$inc": {"value": count}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return doc["value"] - count + 1 

    @staticmethod
    def _to_str_id(value) -> str:
        """
        将 _id 统一转为字符串（兼容外部写入的 ObjectId 数据）

        Args:
            value: 原始 _id，可能是 str 或 ObjectId。

        Returns:
            str: 字符串形式的 _id。
        """
        if isinstance(value, str):
            return value
        return str(value)
