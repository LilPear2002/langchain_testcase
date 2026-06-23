# Test Case Agent

基于 LangChain 的测试用例自动生成后端服务。上传需求文档，AI Agent 自动分析并生成高质量测试用例。

## 功能特性

- **文档解析**: 基于 MinerU 的高精度文档解析，支持 PDF/Word/Excel，中文友好
- **RAG 检索**: 上传需求文档，向量化存储，语义检索相关片段
- **Agent 生成**: ReAct 模式 Agent，自动拆解功能点并生成测试用例
- **LLM Judge**: 5 维评分（完整性/清晰度/正确性/覆盖度/优先级），低分自动重生成
- **多类型覆盖**: functional / boundary / exception / negative 四种用例类型

## 技术栈

| 组件 | 技术 |
|------|------|
| Web 框架 | FastAPI |
| AI 框架 | LangChain 1.0 + LangGraph |
| LLM | 通义千问 (DashScope) |
| 文档解析 | MinerU (MinerULoader) |
| 向量库 | ChromaDB |
| 数据库 | PostgreSQL + SQLAlchemy |

## 快速开始

### 1. 环境准备

```bash
# Python 3.11+
pip install -r requirements.txt
```

### 2. 配置

```bash
cp .env.example .env
# 编辑 .env 填入：
# - DATABASE_URL: PostgreSQL 连接串
# - DASHSCOPE_API_KEY: 阿里云百炼 API Key
# - MINERU_TOKEN: MinerU 精准模式 Token（可选，flash 模式无需）
```

### 3. 启动 PostgreSQL

```bash
# 创建数据库
createdb testcase_agent
```

### 4. 启动服务

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

服务启动后访问 http://localhost:8000/docs 查看 API 文档。

## API 概览

| 端点 | 说明 |
|------|------|
| `POST /api/projects` | 创建项目 |
| `POST /api/projects/{id}/documents` | 上传需求文档 |
| `POST /api/projects/{id}/agent/chat` | Agent 对话生成用例 |
| `GET /api/projects/{id}/test-cases` | 获取测试用例列表 |
| `GET /api/projects/{id}/dashboard` | 数据看板 |

## 项目结构

```
backend/
├── app/
│   ├── agent/          # ReAct Agent 实现
│   ├── api/            # FastAPI 路由
│   ├── core/           # LLM 配置、日志
│   ├── db/             # 数据库模型与初始化
│   ├── judge/          # LLM Judge 评分模块
│   ├── rag/            # RAG 文档处理链
│   ├── schemas/        # Pydantic 数据模型
│   ├── services/       # 业务逻辑
│   ├── config.py       # 配置管理
│   └── main.py         # FastAPI 入口
├── data/               # 上传文件与向量库数据
├── eval/               # 评估脚本与数据集
└── requirements.txt
```
