from pathlib import Path
from typing import List, Any
import uuid

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class ParentChildChunker:
    """父子文档切分器"""
    def __init__(
        self,
        embedding_model,
        parent_chunk_size: int = 2000,
        parent_min_chunk_size: int = 500,
        child_chunk_size: int = 800,
        child_min_chunk_size: int = 200,
        chunk_overlap_rate: float = 0.25,
    ):
        self.embedding_model = embedding_model

        self.parent_chunk_size = parent_chunk_size # 父块最大字符数
        self.parent_min_chunk_size = parent_min_chunk_size # 父块最小字符数

        self.child_chunk_size = child_chunk_size # 子块最大字符数
        self.child_min_chunk_size = child_min_chunk_size # 子块最小字符数

        self.chunk_overlap_rate = chunk_overlap_rate # 文本切分重叠率

        # 1. 父块语义切分器
        self.parent_semantic_splitter = SemanticChunker(
            embeddings=self.embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=6.0,
            min_chunk_size=self.parent_min_chunk_size,
            sentence_split_regex=r"(?<=[。！？；：.!?;:])",
        )

        # 2. 子块语义切分器
        self.child_semantic_splitter = SemanticChunker(
            embeddings=self.embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=3.0,
            min_chunk_size=self.child_min_chunk_size,
            sentence_split_regex=r"(?<=[。！？；：.!?;:])",
        )

        # 3. 父块过大时使用的兜底切分器
        self.parent_recursive_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",
                "\n",
                "。",
                "！",
                "？",
                "；",
                ".",
                "!",
                "?",
                ";",
                "，",
                ",",
                " ",
                "",
            ],
            chunk_size=self.parent_chunk_size,
            chunk_overlap=int(self.parent_chunk_size * chunk_overlap_rate),
            length_function=len,
        )

        # 4. 子块过大时使用的兜底切分器
        self.child_recursive_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",
                "\n",
                "。",
                "！",
                "？",
                "；",
                ".",
                "!",
                "?",
                ";",
                "，",
                ",",
                " ",
                "",
            ],
            chunk_size=self.child_chunk_size,
            chunk_overlap=int(self.child_chunk_size * chunk_overlap_rate),
            length_function=len,
        )

    def split_text(self, text: Document, metadata: dict[str, Any] | None = None) -> List[dict[str, Any]]:
        """
        对文档进行父子切分, 适用于解析成普通文本文档， 先按语义切分，再按字符切分

        Args:
            text: 待切分的文档，来源信息从 text.metadata 读取（source/page）
            metadata: 额外元数据，会透传到每个分块；file_name/page 由来源自动归一化

        Returns:
            切分后的文档列表，每项包含 parent 和 children
        """
        base_metadata = {**text.metadata, **(metadata or {})}
        source = base_metadata.get("source")
        raw_page = base_metadata.get("page")

        # 归一化 file_name/page
        if source is None:
            raise ValueError("source 不能为空")
        if raw_page is None:
            raise ValueError("page 不能为空")
        chunk_metadata: dict[str, Any] = {
            key: value
            for key, value in (metadata or {}).items()
            if key not in ("file_name", "page")
        }
        chunk_metadata["file_name"] = Path(source).name if source else ""
        chunk_metadata["page"] = int(raw_page) + 1 if raw_page is not None else 1

        # 第一层：切 Parent
        semantic_parent_docs = self.parent_semantic_splitter.create_documents(
            texts=[text.page_content],
            metadatas=[chunk_metadata],
        )

        parent_docs: list[Document] = []

        for parent_doc in semantic_parent_docs:
            if len(parent_doc.page_content) >= self.parent_chunk_size:
                recursive_docs = self.parent_recursive_splitter.split_documents(
                    [parent_doc]
                )

                parent_docs.extend(recursive_docs)
            else:
                parent_docs.append(parent_doc)

        # 第二层：Parent → Child
        results = []

        for parent_doc in parent_docs:

            parent_id = str(uuid.uuid4())

            # 给 Parent 添加 ID
            parent_doc.metadata["parent_id"] = parent_id

            # 对 Parent 进行语义子切分
            semantic_child_docs = self.child_semantic_splitter.create_documents(
                texts=[parent_doc.page_content],
                metadatas=[parent_doc.metadata],
            )

            child_docs: list[Document] = []

            for child_doc in semantic_child_docs:
                if len(child_doc.page_content) >= self.child_chunk_size:
                    recursive_docs = self.child_recursive_splitter.split_documents(
                        [child_doc]
                    )
                    child_docs.extend(recursive_docs)
                else:
                    child_docs.append(child_doc)

            # 添加 Child 元数据
            for child_doc in child_docs:
                child_doc.metadata["parent_id"] = parent_id
                child_doc.metadata["child_id"] = f"{parent_id}_child_{str(uuid.uuid4())}"

            results.append(
            {
                "parent": parent_doc,
                "children": child_docs,
            }
        )
        return results
