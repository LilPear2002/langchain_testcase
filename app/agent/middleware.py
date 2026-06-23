"""Agent Middleware：可观测性 + 行为控制。

按笔记约定使用装饰器写法：
- before_agent / after_agent：记录 agent 生命周期
- wrap_model_call：计时 + 记录每次 LLM 调用
- wrap_tool_call：记录工具调用、并做运行时护栏

护栏示例：
1. 输入敏感词：发现 "drop table" / "删除所有用例" / "rm -rf" 直接拦截
2. save_test_cases 单次最多保存 50 条，超限返回错误 ToolMessage
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import (
    after_agent,
    before_agent,
    before_model,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.runtime import Runtime

logger = logging.getLogger("agent.middleware")

# ---------- 配置 ----------

_SENSITIVE_PATTERNS = [
    r"drop\s+table",
    r"删除所有用例",
    r"删除全部用例",
    r"rm\s+-rf",
    r"truncate\s+table",
]
_SENSITIVE_RE = re.compile("|".join(_SENSITIVE_PATTERNS), re.IGNORECASE)

MAX_BATCH_SAVE = 50  # save_test_cases 单次最多保存条数


# ---------- 生命周期钩子 ----------

@before_agent
def log_before_agent(state: AgentState, runtime: Runtime) -> dict | None:
    msgs = state["messages"]
    logger.info("[before_agent] 启动 | messages=%d", len(msgs))
    # 敏感词守卫：检查最后一条 user 消息
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            text = m.content if isinstance(m.content, str) else str(m.content)
            if _SENSITIVE_RE.search(text):
                logger.warning("[before_agent] 检测到敏感指令，已拦截：%s", text[:80])
                raise ValueError(
                    "输入包含敏感/危险指令，已被安全策略拦截。请重新描述需求。"
                )
            break
    return None


@after_agent
def log_after_agent(state: AgentState, runtime: Runtime) -> None:
    msgs = state["messages"]
    tool_calls = sum(
        1 for m in msgs if getattr(m, "tool_calls", None)
    )
    tool_results = sum(1 for m in msgs if isinstance(m, ToolMessage))
    logger.info(
        "[after_agent] 结束 | 总消息=%d | tool_call轮=%d | tool_result=%d",
        len(msgs), tool_calls, tool_results,
    )


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> None:
    logger.debug("[before_model] context messages=%d", len(state["messages"]))


@wrap_model_call
def time_model_call(request, handler):
    """计时 + 记录 LLM 调用耗时。"""
    t0 = time.perf_counter()
    try:
        response = handler(request)
        return response
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000
        logger.info("[model_call] 耗时 %.0f ms", dt_ms)


# ---------- 工具拦截 ----------

def _reject_tool(request, reason: str) -> ToolMessage:
    """构造一条错误 ToolMessage，替代真实执行。"""
    return ToolMessage(
        content=f"error: {reason}",
        name=request.tool_call["name"],
        tool_call_id=request.tool_call["id"],
    )


@wrap_tool_call
def guard_and_log_tool(request, handler):
    name = request.tool_call["name"]
    args = request.tool_call.get("args", {}) or {}
    logger.info("[tool_call] %s args=%s", name, _short_repr(args))

    # 护栏：限制 save_test_cases 单次保存数量
    if name == "save_test_cases":
        raw = args.get("cases_json", "")
        approx_count = _rough_count_cases(raw)
        if approx_count > MAX_BATCH_SAVE:
            logger.warning(
                "[tool_call] 拦截 save_test_cases：%d > %d", approx_count, MAX_BATCH_SAVE
            )
            return _reject_tool(
                request,
                f"单次最多保存 {MAX_BATCH_SAVE} 条用例，本次 {approx_count} 条，请分批。",
            )

    t0 = time.perf_counter()
    result = handler(request)
    dt_ms = (time.perf_counter() - t0) * 1000
    logger.info("[tool_result] %s 耗时 %.0f ms", name, dt_ms)
    return result


def _short_repr(obj: Any, limit: int = 200) -> str:
    s = repr(obj)
    return s if len(s) <= limit else s[:limit] + "...<truncated>"


def _rough_count_cases(cases_json: str) -> int:
    """粗略估算 cases_json 里数组元素个数，避免完整 JSON parse 开销。"""
    if not isinstance(cases_json, str):
        return 0
    try:
        import json
        data = json.loads(cases_json)
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            for k in ("cases", "items"):
                if isinstance(data.get(k), list):
                    return len(data[k])
            return 1
    except Exception:
        return cases_json.count('"title"')  # 兜底估算
    return 0


# 汇总成一个 list 便于 create_agent(middleware=[...]) 引用
AGENT_MIDDLEWARE = [
    log_before_agent,
    log_before_model,
    time_model_call,
    guard_and_log_tool,
    log_after_agent,
]
