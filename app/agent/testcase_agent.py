"""测试用例生成 Agent（ReAct 模式）。"""
from langchain.agents import create_agent

from app.agent.middleware import AGENT_MIDDLEWARE
from app.agent.tools import make_tools
from app.core.llm import get_generation_llm

SYSTEM_PROMPT = """你是一名资深测试用例设计专家，严格遵循 ReAct（思考→行动→观察→再思考）模式工作。
**每一轮只允许调用一个工具**，禁止一次并行调用多个工具。

你的任务：根据用户给出的需求主题或指令，为指定项目生成高质量的测试用例并保存到数据库。

标准工作流程：
1. 思考：理解用户要做什么。
2. 行动：调用 `list_existing_cases` 查看已有用例，避免重复。
3. 思考：是否需要补充资料。
4. 行动：调用 `retrieve_requirement` 检索相关需求文档。
5. 行动：调用 `analyze_requirement_points` 把需求拆成可测功能点。
6. 对关键功能点，逐个调用 `generate_test_case`（覆盖多种 case_type：
   functional / boundary / exception / negative）。该工具内部已内置 LLM Judge
   评分与低分重生成（rubric 5 维、满分 25、合格线 18），返回结果里带
   `score` / `_judge_comment` / `_attempts` 字段，你无需自己再评分。
7. 聚合所有用例后，调用 `save_test_cases` 批量保存。**保留 score 字段**
   一起传入，让数据库留下评审分数。
8. 最后用自然语言向用户总结：生成了多少条用例、各自分数与覆盖类型、
   有没有被重生成过、以及建议的后续补充。

请在每次调用工具前，用简短中文说明你的思考和决策，让用户看到推理过程。"""


def build_testcase_agent(project_id: int):
    """为指定项目构建一个测试用例生成 Agent。

    invoke 入参: {"messages": [{"role": "user", "content": "..."}]}
    """
    return create_agent(
        model=get_generation_llm(),
        tools=make_tools(project_id),
        system_prompt=SYSTEM_PROMPT,
        middleware=AGENT_MIDDLEWARE,
    )
