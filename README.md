# Test Case Agent

基于 LangChain 的测试用例自动生成后端服务。上传需求文档，AI Agent 自动分析并生成高质量测试用例。

## 功能特性

### ReAct Agent

采用 LangChain 的 ReAct（Reasoning + Acting）范式，Agent 严格遵循 **思考 → 行动 → 观察 → 再思考** 循环：

```
用户需求 → 思考(理解意图) → 行动(调用工具) → 观察(获取结果) → 思考(决策) → ... → 输出
```

每轮只允许调用一个工具，确保推理过程可控、可追溯。

### 自定义 Tools

为测试用例场景定制 5 个专用工具：

| 工具 | 功能 |
|------|------|
| `retrieve_requirement` | RAG 语义检索需求文档 |
| `analyze_requirement_points` | LLM 拆解需求为可测功能点 |
| `generate_test_case` | 生成单条测试用例（内置 Judge 评分 + 低分重生成） |
| `save_test_cases` | 批量持久化到 PostgreSQL |
| `list_existing_cases` | 查询已有用例，避免重复生成 |

### LLM Judge 评分体系

LangGraph 驱动的 **生成 → 评分 → 重生成** 状态机：

```
START ──► generate ──► judge ──┐
                           ▲    │
                           │    ├─ passed or attempt>=max ─► END
                           │    │
                       regenerate ◄──┘
```

5 维评分（各 1-5 分，满分 25，合格线 18）：
- **completeness**: 前置条件/步骤/预期是否齐全
- **clarity**: 步骤是否可直接执行
- **correctness**: 是否与需求一致
- **coverage**: 是否体现指定 case_type 测试意图
- **priority**: 优先级设置是否合理

### 多轮对话记忆

基于 PostgreSQL 持久化的 `PostgresChatMessageHistory`，支持跨会话的多轮对话：
- 每轮对话自动存储到 `chat_messages` 表
- Agent 可引用历史上下文继续生成用例
- 支持会话管理（创建/查询/清空）

### Agent Harness（可观测性 + 运行时护栏）

通过 Middleware 机制实现 Agent 行为控制与观测：

```python
@before_agent    # 生命周期钩子：敏感词拦截
@wrap_model_call # LLM 调用计时
@wrap_tool_call  # 工具调用护栏：单次保存上限 50 条
@after_agent     # 统计工具调用次数
```

- **输入护栏**: 拦截 `drop table`、`删除所有用例` 等危险指令
- **输出护栏**: `save_test_cases` 单次最多 50 条，防止误操作
- **LangSmith 集成**: 可选开启 LangSmith tracing，追踪完整调用链路

### 文档解析与 RAG

- **MinerU**: 高精度文档解析，支持 PDF/Word/Excel，中文优化
- **ChromaDB**: 向量存储，语义检索相关需求片段
- **文本分割**: 可配置 chunk_size / chunk_overlap

### RESTful API

基于 FastAPI 构建，遵循 REST 规范：
- 自动生成 OpenAPI 文档（Swagger UI）
- Pydantic 请求/响应校验
- 统一异常处理与错误码
- CORS 中间件支持跨域

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
