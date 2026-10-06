# camel-agent

基于 [CAMEL-AI](https://github.com/camel-ai/camel) 与 LangChain 的 RAG 智能体项目：
将本地/MinIO 中的文档解析、切分、向量化后写入 Milvus，再把检索能力作为工具挂载给
CAMEL ChatAgent，实现带私有知识库的问答对话。

## 项目概述

完整链路：**文档解析 → 父子块切分 → 向量化 → 写入 Milvus → 检索 + 重排 → Agent 工具调用**

- 文档解析：支持 Markdown / PDF，PDF 可走 MinerU 云端解析
- 切分策略：`MarkdownChunkSplitter` 父子块切分，子块向量化入库，父块内容写入 `metadata["parent_content"]` 用于召回还原
- 向量存储：pymilvus 封装的 `AMilvusClient`，默认 collection 为 `camel_agent`
- 对象存储：MinIO 存放原始文档，并生成可访问的 `source_url`
- 检索增强：`MilvusRetriever` 向量召回 + `RerankModel` 重排，带分数阈值过滤
- Agent：`ModelClient` 基于 DeepSeek 平台创建模型，`ChatAgentClient` 封装 CAMEL `ChatAgent`，支持异步多轮对话与工具调用

## 目录结构

```text
camel-agent/
├── camel_agent/
│   ├── agent_components/
│   │   ├── agent/agent.py          # ChatAgentClient（CAMEL ChatAgent 封装）
│   │   ├── llm/model.py            # ModelClient（DeepSeek 模型工厂）
│   │   └── toolkits/rag_search.py  # RAG 检索工具（暴露给 LLM）
│   └── rag_components/
│       ├── parser/                 # Parser / MineruClient 文档解析
│       ├── chunk/                  # MarkdownChunkSplitter / ParentChildChunker
│       ├── embedding/              # EmbeddingModel 向量模型
│       ├── retriever/              # MilvusRetriever + build_retriever_from_env
│       ├── rerank/                 # RerankModel 重排模型
│       ├── milvus/                 # AMilvusClient（异步 Milvus 客户端）
│       └── minio/                  # MinioClient 对象存储
├── scripts/insert_milvus.py        # 文档入库脚本
├── tests/                          # 可直接运行的示例/验证脚本
├── docker-compose.yml              # Milvus / etcd / MinIO / MongoDB
├── pyproject.toml
└── uv.lock
```

## 环境要求

- Python >= 3.12（见 `.python-version`）
- [uv](https://docs.astral.sh/uv/) 依赖管理
- Docker + Docker Compose（用于启动 Milvus 等基础设施）

## 安装

### 1. 克隆并安装依赖

```bash
git clone https://github.com/chaoxie2005/camel-agent.git
cd camel-agent
uv sync
```

### 2. 配置环境变量

在项目根目录创建 `.env`（已被 `.gitignore` 忽略，不会提交）：

| 变量 | 说明 |
| --- | --- |
| `MODEL_EMBEDDING_NAME` | Embedding 向量模型名；兼容旧变量 `MODEL_NAME`，新变量优先 |
| `BASE_URL` | 向量/重排模型服务地址 |
| `MODEL_API_KEY` | 向量/重排模型 API Key |
| `RERANK_MODEL_NAME` | 重排模型名 |
| `RERANK_BASE_URL` | 重排服务 endpoint |
| `MINERU_URL` | MinerU 解析服务地址 |
| `MINERU_API_KEY` | MinerU API Key |
| `DEEPSEEK_API_KEY` | DeepSeek LLM API Key |
| `DEEPSEEK_URL` | DeepSeek LLM 服务地址 |

### 3. 启动基础设施

```bash
docker compose up -d
```

启动内容与端口：

| 服务 | 端口 | 用途 |
| --- | --- | --- |
| Milvus | 19530 / 9091 | 向量数据库 |
| MinIO (files) | 9000 / 9001 | 原始文档对象存储 |
| MinIO (milvus 内部) | 不对外 | Milvus 元数据 |
| etcd | 不对外 | Milvus 元数据 |
| MongoDB | 27018 | 持久化存储 |

## 使用

### 1. 文档入库

从 MinIO 桶 `camel-agent-rag` 下载文档，解析、切分、向量化后写入 Milvus
collection `camel_agent`：

```bash
uv run scripts/insert_milvus.py
```

### 2. 验证 RAG 检索

```bash
uv run tests/rag_test.py
```

### 3. Agent 对话（挂载 RAG 工具）

```bash
uv run tests/agent_test.py
```

Agent 会根据用户问题自主决定是否调用 `rag_search` 工具检索知识库。

> 前置条件：Milvus 已启动且已完成数据入库，并在 `.env` 中配置了
> DeepSeek 与向量模型的凭证。

## 开发调试

项目提供 VSCode 调试配置 `.vscode/launch.json`：

- **调试当前文件 (venv)**：使用 `.venv/bin/python` 运行当前打开的文件

## License

暂未提供。

## 模块接口与本地验证

- 检索器同步入口为 `retriever.invoke(query)`，异步入口为 `await retriever.ainvoke(query)`。
  同步查询使用独立、按需创建的 Milvus 连接；使用完后调用 `await retriever.client.close()` 释放连接。
- `make_rag_search_tool()` 接收 LangChain `BaseRetriever`，生成异步 CAMEL 工具。
  原 `rag_search()` 转发函数已移除，直接使用检索器公开入口即可。
- 入库和检索通过 `build_embedding_from_env()` 共用向量模型配置。
- 切分结果使用 `ParentChildChunks` 类型，保留 `item["parent"]` 和 `item["children"]` 的访问方式。
- 异步重排通过线程运行现有同步 HTTP 请求，共用请求与响应解析逻辑。

无需真实数据库或模型服务的回归测试：

```bash
uv run python -m unittest discover -s tests -p test_retriever.py -v
```

`tests/agent_test.py` 和 `tests/rag_test.py` 仍是需要真实服务的手动验证脚本。
