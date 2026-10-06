"""无需数据库和模型服务的模块边界回归测试。"""
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from camel_agent.agent_components.toolkits.rag_search import make_rag_search_tool
from camel_agent.rag_components.embedding import EmbeddingModel
from camel_agent.rag_components.milvus import AMilvusClient
from camel_agent.rag_components.rerank import RerankModel
from camel_agent.rag_components.retriever import MilvusRetriever, build_retriever_from_env
from camel_agent.rag_components.chunk import MarkdownChunkSplitter, ParentChildChunker
from scripts.insert_milvus import flatten_chunks


class RetrieverTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.hits = [[
            {"entity": {"child_content": "A", "child_metadata": {"source": "a"}}, "distance": 0.8},
            {"entity": {"child_content": "B", "child_metadata": {}}, "distance": 0.7},
        ]]
        self.client = Mock(spec=AMilvusClient)
        self.client.hybrid_search_sync.return_value = self.hits
        self.client.hybrid_search = AsyncMock(return_value=self.hits)
        self.client.vector_search_sync.return_value = self.hits
        self.client.vector_search = AsyncMock(return_value=self.hits)
        self.embedding = Mock(spec=EmbeddingModel)
        self.embedding.embed_query.return_value = [0.1, 0.2]
        self.embedding.aembed_query = AsyncMock(return_value=[0.1, 0.2])
        self.rerank = Mock(spec=RerankModel)
        self.rerank.rerank.return_value = [(1, 0.9), (0, 0.2)]
        self.rerank.arerank = AsyncMock(return_value=[(1, 0.9), (0, 0.2)])
        self.retriever = MilvusRetriever(
            client=self.client, embedding_function=self.embedding,
            rerank_function=self.rerank, score_threshold=0.3,
        )

    async def test_sync_async_rerank_and_threshold(self):
        sync_docs = self.retriever.invoke("question")
        async_docs = await self.retriever.ainvoke("question")
        self.assertEqual(sync_docs, async_docs)
        self.assertEqual([doc.page_content for doc in async_docs], ["B"])
        self.assertEqual(async_docs[0].metadata["score"], 0.9)
        self.embedding.aembed_query.assert_awaited_once_with("question")
        self.client.hybrid_search.assert_awaited_once()
        self.rerank.arerank.assert_awaited_once()
        options = self.client.hybrid_search.call_args.kwargs
        self.assertEqual(options["top_k"], 12)
        self.assertEqual(options["output_fields"], ["child_content", "child_metadata"])
        self.assertNotIn("score", self.hits[0][1]["entity"]["child_metadata"])

    async def test_no_rerank_and_empty_results(self):
        self.retriever.rerank_function = None
        self.retriever.top_k = 1
        self.assertEqual((await self.retriever.ainvoke("q"))[0].page_content, "A")
        self.client.hybrid_search.return_value = []
        self.client.hybrid_search_sync.return_value = [[]]
        self.assertEqual(await self.retriever.ainvoke("q"), [])
        self.assertEqual(self.retriever.invoke("q"), [])
        self.rerank.arerank.assert_not_awaited()

    async def test_vector_paths(self):
        self.retriever.score_threshold = 0.75
        self.assertEqual(self.retriever.embedding_search("q"),
                         await self.retriever.aembedding_search("q"))
        self.assertEqual(len(await self.retriever.aembedding_search("q")), 1)

    async def test_tool_accepts_generic_retriever(self):
        retriever = Mock(spec=BaseRetriever)
        retriever.ainvoke = AsyncMock(return_value=[Document(page_content="answer", metadata={"score": 0.8})])
        tool = make_rag_search_tool(retriever)
        self.assertEqual(await tool.async_call(query="q"), "[score=0.8]\nanswer")
        retriever.ainvoke.assert_awaited_once_with("q")
        retriever.ainvoke.return_value = []
        self.assertEqual(await tool.async_call(query="q"), "未检索到相关文档")

    async def test_async_rerank_does_not_block_event_loop(self):
        import threading
        loop_thread = threading.get_ident()
        model = RerankModel("model", "http://example.invalid", "test")
        def rerank(*args):
            self.assertNotEqual(threading.get_ident(), loop_thread)
            return [(0, 0.9)]
        with patch.object(model, "rerank", side_effect=rerank):
            self.assertEqual(await model.arerank("q", ["a"], 1), [(0, 0.9)])


class MilvusAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_lazy_sync_connection_mapping_and_cleanup(self):
        module = "camel_agent.rag_components.milvus.milvus_client"
        with patch(f"{module}.AsyncMilvusClient") as async_type, patch(f"{module}.MilvusClient") as sync_type:
            async_type.return_value.close = AsyncMock()
            client = AMilvusClient(url="http://example.invalid", db_name="test")
            sync_type.assert_not_called()
            client.vector_search_sync(collection_name="docs", query=[0.1], top_k=2)
            sync_type.return_value.search.assert_called_once_with(
                collection_name="docs", data=[[0.1]], limit=2, anns_field="dense_vector"
            )
            client.hybrid_search_sync(collection_name="docs", requests=[], ranker="ranker", top_k=4)
            sync_type.return_value.hybrid_search.assert_called_once_with(
                collection_name="docs", reqs=[], ranker="ranker", limit=4
            )
            sync_type.assert_called_once()
            self.assertEqual(sync_type.call_args.kwargs["db_name"], "test")
            await client.close()
            async_type.return_value.close.assert_awaited_once()
            sync_type.return_value.close.assert_called_once()


class ConfigurationAndChunkTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_config_fails_before_client_creation(self):
        with patch("camel_agent.rag_components.retriever.factory.AMilvusClient") as client:
            with self.assertRaisesRegex(ValueError, "MODEL_EMBEDDING_NAME"):
                build_retriever_from_env()
            client.assert_not_called()

    def test_legacy_embedding_env_alias(self):
        from camel_agent.rag_components.retriever.factory import _required_env
        with patch.dict(os.environ, {"MODEL_NAME": "legacy"}, clear=True):
            self.assertEqual(_required_env("MODEL_EMBEDDING_NAME", "MODEL_NAME"), "legacy")
        with patch.dict(os.environ, {"MODEL_NAME": "legacy", "MODEL_EMBEDDING_NAME": "current"}, clear=True):
            self.assertEqual(_required_env("MODEL_EMBEDDING_NAME", "MODEL_NAME"), "current")

    def test_parent_child_contract_and_ingestion(self):
        embedding = Mock(spec=EmbeddingModel)
        for cls in (ParentChildChunker, MarkdownChunkSplitter):
            with self.subTest(chunker=cls.__name__):
                chunker = cls(embedding)
                def split(*, texts, metadatas=None):
                    return [Document(page_content=text, metadata=metadatas[i] if metadatas else {})
                            for i, text in enumerate(texts)]
                with patch.object(chunker.parent_semantic_splitter, "create_documents", side_effect=split), \
                     patch.object(chunker.child_semantic_splitter, "create_documents", side_effect=split):
                    groups = chunker.split_text(Document(page_content="Example text", metadata={"source": "sample.md", "page": 0}))
                self.assertEqual(len(groups), 1)
                parent, children = groups[0]["parent"], groups[0]["children"]
                self.assertEqual(children[0].metadata["parent_id"], parent.metadata["parent_id"])
                self.assertEqual(children[0].metadata["page"], 1)
                self.assertEqual(flatten_chunks(groups)[0].metadata["parent_content"], parent.page_content)
                embedding.embed_documents.assert_not_called()


if __name__ == "__main__":
    unittest.main()
