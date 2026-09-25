"""
文档解析 文档切分 文档向量化 插入 Milvus 数据库 脚本
"""
from typing import List
from pathlib import Path
import os
import dotenv

from langchain_core.documents import Document
from camel_agent.components.parser import Parser
from camel_agent.components.chunk import ParentChildChunker, MarkdownChunkSplitter
from camel_agent.components.embedding import EmbeddingModel
from camel_agent.components.store import MilvusStore
from camel_agent.components.minio import MinioClient

dotenv.load_dotenv()


embedding_model = EmbeddingModel(
    model_name=os.getenv("MODEL_NAME"),
    base_url=os.getenv("BASE_URL"),
    key=os.getenv("MODEL_API_KEY"),
)


def load_documents(file_path) -> List[Document]:
    """
    解析文档
    
    Args:
        file_path: 文件路径
    
    Returns:
        List[Document]: 文档列表
    """
    paser = Parser(file_path)
    return paser.parse()


def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100):
    """
    切分Markdown文档并存入 Milvus 数据库
    
    Args:
        documents: 文档列表
        chunk_size: 切分块大小
        chunk_overlap: 切分块重叠大小
        
    Returns:
        List[Document]: 切分后的文档列表
    """
    chunker = MarkdownChunkSplitter(
        embedding_model=embedding_model,
    )
    results = []
    for document in documents:
        parent_child_documents = chunker.split_text(document)
        results.extend(parent_child_documents)
    return results


def flatten_chunks(chunked_results) -> List[Document]:
    """
    将父子块结果展平为子块 Document 列表，父块内容写入 metadata["parent_content"]
    只有子块 page_content 会被向量化
    
    Args:
        chunked_results: 父子块结果列表
        
    Returns:
        List[Document]: 展平后的子块文档列表
    """
    docs: List[Document] = []
    for item in chunked_results:
        parent_content = item["parent"].page_content
        for child in item["children"]:
            child.metadata["parent_content"] = parent_content
            docs.append(child)
    return docs


if __name__ == "__main__":
    minio_client = MinioClient(
        endpoint="127.0.0.1:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False, 
    )
    bucket_name = "camel-agent-rag"
    download_dir = Path("./download_files")
    downloaded_files = minio_client.download_files(bucket_name, download_dir)
    print(f"下载完成 {len(downloaded_files)} 个文件")

    all_chunks: List[Document] = []
    for file_path in downloaded_files:
        documents = load_documents(file_path)
        object_name = file_path.relative_to(download_dir).as_posix()
        for document in documents:
            document.metadata["source_url"] = f"http://127.0.0.1:9000/{bucket_name}/{object_name}"
        chunked_documents = chunk_documents(documents)
        all_chunks.extend(flatten_chunks(chunked_documents))

    store = MilvusStore(
        embedding_function=embedding_model,
        collection_name="camel_agent",
        uri="http://localhost:19530",
    )
    ids = store.add_documents(all_chunks)
    print(f"写入 Milvus {len(ids)} 条")