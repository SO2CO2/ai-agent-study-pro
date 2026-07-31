"""
Day 12: Observable Agent With JSONL Trace Logs

Run:
    python3 day12_observable_agent.py

Try:
    北京今天适合跑步吗？
    深圳今天适合户外运动吗？我的 api_key 是 sk-test-123456
    今天适合跑步吗？

Commands:
    trace          Show the latest trace summary again.
    :traces        Show recent trace summaries from agent_traces.jsonl.
    :clear-traces  Clear local trace logs.
    exit           Quit.

Today's key idea:
    Without trace logs, an Agent is hard to debug, evaluate, and trust.

This script stays offline so students can focus on observability:
trace_id, event_type, step, JSONL, redaction, error events, and summaries.
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict


TRACE_FILE = Path(__file__).with_name("agent_traces.jsonl")
SUPPORTED_CITIES = ["北京", "上海", "深圳", "广州"]
MAX_STEPS = 6
RECENT_TRACE_LIMIT = 5


class ObservableAgentError(Exception):
    """A readable error for this learning script."""


class TraceEvent(TypedDict):
    trace_id: str
    timestamp: str
    step: int
    event_type: str
    message: str
    data: dict[str, Any]


class AgentState(TypedDict):
    user_goal: str
    city: str | None
    weather: str | None
    current_time: str | None
    final_answer: str | None
    error: str | None


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def create_trace_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"trace_{timestamp}_{suffix}"


def redact_sensitive_data(value: Any) -> Any:
    """
    Redact sensitive data before writing logs.

    Teaching point:
    Trace is for debugging, not for leaking secrets. This small function is not
    a complete privacy system, but it shows where redaction belongs.
    """
    sensitive_keys = {"api_key", "apikey", "token", "password", "secret", "authorization"}

    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            if key.casefold() in sensitive_keys or any(word in key.casefold() for word in sensitive_keys):
                safe[key] = "***redacted***"
            else:
                safe[key] = redact_sensitive_data(item)
        return safe

    if isinstance(value, list):
        return [redact_sensitive_data(item) for item in value]

    if isinstance(value, str):
        text = value
        text = re.sub(r"sk-[A-Za-z0-9_-]{6,}", "sk-***redacted***", text)
        text = re.sub(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*\S+", r"\1=***redacted***", text)
        text = re.sub(r"1[3-9]\d{9}", "1**********", text)
        text = re.sub(
            r"([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
            r"\1***\2",
            text,
        )
        return text

    return value


class TraceLogger:
    """Append one JSON event per line and keep this run's events in memory."""

    def __init__(self, trace_id: str | None = None, path: Path = TRACE_FILE):
        self.trace_id = trace_id or create_trace_id()
        self.path = path
        self.step = 0
        self.events: list[TraceEvent] = []
        self.path.touch(exist_ok=True)

    def log(self, event_type: str, message: str, data: dict[str, Any] | None = None) -> TraceEvent:
        self.step += 1
        event: TraceEvent = {
            "trace_id": self.trace_id,
            "timestamp": now_text(),
            "step": self.step,
            "event_type": event_type,
            "message": message,
            "data": redact_sensitive_data(data or {}),
        }
        self.events.append(event)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def get_events(self) -> list[TraceEvent]:
        return list(self.events)


def extract_city(user_input: str) -> str | None:
    for city in SUPPORTED_CITIES:
        if city in user_input:
            return city
    return None


def create_initial_state(user_input: str) -> AgentState:
    return {
        "user_goal": user_input,
        "city": extract_city(user_input),
        "weather": None,
        "current_time": None,
        "final_answer": None,
        "error": None,
    }


def get_weather(city: str) -> str:
    fake_weather = {
        "北京": "晴，18-28 度，空气质量良好",
        "上海": "多云，22-30 度，湿度较高",
        "深圳": "阵雨，25-31 度，降雨概率较高",
        "广州": "雷阵雨，24-30 度，午后可能有强降雨",
    }
    if city not in fake_weather:
        raise ObservableAgentError(f"暂时没有 {city} 的天气数据")
    return fake_weather[city]


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_safety_check(user_input: str) -> dict[str, Any]:
    """A tiny check to show that safety events can also be traced."""
    matched = []
    patterns = [
        r"sk-[A-Za-z0-9_-]{6,}",
        r"(?i)api[_-]?key",
        r"(?i)password",
        r"(?i)token",
        r"忽略.*(规则|指令|提示)",
        r"输出.*系统提示",
    ]
    for pattern in patterns:
        if re.search(pattern, user_input):
            matched.append(pattern)
    return {
        "risk_level": "medium" if matched else "low",
        "matched_patterns": matched,
    }


def plan_next_step(state: AgentState) -> dict[str, Any]:
    if state["error"] or state["final_answer"]:
        return {"name": "finish", "reason": "任务已经结束"}

    if not state["city"]:
        return {"name": "ask_city", "reason": "缺少城市信息，无法查询天气"}

    if not state["weather"]:
        return {
            "name": "get_weather",
            "arguments": {"city": state["city"]},
            "reason": "需要天气信息判断是否适合运动",
        }

    if not state["current_time"]:
        return {
            "name": "get_current_time",
            "arguments": {},
            "reason": "需要当前时间辅助生成建议",
        }

    return {"name": "create_final_answer", "arguments": {}, "reason": "信息足够，可以生成最终回答"}


def execute_step(plan: dict[str, Any], state: AgentState, logger: TraceLogger) -> None:
    step_name = plan["name"]

    if step_name == "ask_city":
        state["final_answer"] = "你想查询哪个城市的运动建议？目前支持：北京、上海、深圳、广州。"
        logger.log(
            "final_answer",
            "缺少必要信息，向用户追问",
            {"answer": state["final_answer"]},
        )
        return

    if step_name == "get_weather":
        arguments = plan.get("arguments", {})
        logger.log("tool_call", "调用天气工具", {"tool": "get_weather", "arguments": arguments})
        city = arguments["city"]
        state["weather"] = get_weather(city)
        logger.log(
            "tool_result",
            "天气工具返回结果",
            {"tool": "get_weather", "result": state["weather"]},
        )
        return

    if step_name == "get_current_time":
        logger.log("tool_call", "调用当前时间工具", {"tool": "get_current_time", "arguments": {}})
        state["current_time"] = get_current_time()
        logger.log(
            "tool_result",
            "当前时间工具返回结果",
            {"tool": "get_current_time", "result": state["current_time"]},
        )
        return

    if step_name == "create_final_answer":
        state["final_answer"] = create_final_answer(state)
        logger.log(
            "final_answer",
            "已生成最终回答",
            {"answer": state["final_answer"]},
        )
        return

    if step_name == "finish":
        logger.log("finish", "任务结束", {})
        return

    raise ObservableAgentError(f"未知步骤：{step_name}")


def create_final_answer(state: AgentState) -> str:
    city = state["city"]
    weather = state["weather"]
    current_time = state["current_time"]
    if not city or not weather or not current_time:
        return "信息还不完整，暂时无法给出可靠建议。"

    risk_notes = []
    suggestion = "比较适合运动"
    if any(word in weather for word in ["阵雨", "雷阵雨", "强降雨"]):
        suggestion = "不太适合户外跑步"
        risk_notes.append("有降雨风险，路面湿滑")
    if any(word in weather for word in ["30 度", "31 度"]):
        risk_notes.append("气温偏高，注意补水和防晒")
    if "湿度较高" in weather:
        risk_notes.append("湿度较高，体感会更闷")

    risk_text = "；".join(risk_notes) if risk_notes else "整体条件比较友好"
    advice = "建议选择轻松强度，控制在 30-45 分钟。"
    if suggestion == "不太适合户外跑步":
        advice = "建议改为室内训练，或等天气稳定后再进行短距离慢跑。"

    return (
        f"{city}现在的天气信息是：{weather}；当前时间是：{current_time}。\n"
        f"综合判断：{suggestion}。\n"
        f"原因：{risk_text}。\n"
        f"{advice}"
    )


def observable_agent(user_input: str) -> tuple[str, list[TraceEvent]]:
    logger = TraceLogger()
    state = create_initial_state(user_input)

    logger.log("user_input", "收到用户输入", {"input": user_input})
    safety = run_safety_check(user_input)
    logger.log("safety_check", "完成基础安全检查", safety)
    logger.log(
        "state_created",
        "初始化任务状态",
        {"city": state["city"], "has_city": state["city"] is not None},
    )

    try:
        for step_index in range(1, MAX_STEPS + 1):
            plan = plan_next_step(state)
            logger.log(
                "planner_decision",
                "Planner 决定下一步",
                {"loop_step": step_index, "plan": plan},
            )
            execute_step(plan, state, logger)
            if should_stop(state):
                break
        else:
            state["error"] = "达到最大步骤数，任务停止。"
            logger.log("error", "达到最大步骤数", {"max_steps": MAX_STEPS})
    except Exception as error:
        state["error"] = str(error)
        logger.log(
            "error",
            "执行过程中出现错误",
            {"error_type": type(error).__name__, "error": str(error)},
        )

    if state["error"]:
        answer = f"执行过程中出现问题：{state['error']}"
        if not state["final_answer"]:
            logger.log("final_answer", "生成错误兜底回答", {"answer": answer})
    else:
        answer = state["final_answer"] or "任务结束，但没有生成回答。"

    return answer, logger.get_events()


def should_stop(state: AgentState) -> bool:
    return state["final_answer"] is not None or state["error"] is not None


def print_trace_summary(events: list[TraceEvent]) -> None:
    if not events:
        print("Trace：无事件。")
        return
    trace_id = events[0]["trace_id"]
    print(f"\nTrace Summary：{trace_id}")
    for event in events:
        details = summarize_event_data(event)
        suffix = f"：{details}" if details else ""
        print(f"{event['step']}. {event['event_type']} - {event['message']}{suffix}")
    print(f"完整 JSONL 已写入：{TRACE_FILE.name}")


def summarize_event_data(event: TraceEvent) -> str:
    data = event["data"]
    event_type = event["event_type"]
    if event_type == "tool_call":
        return f"{data.get('tool')}({format_arguments(data.get('arguments', {}))})"
    if event_type == "tool_result":
        return str(data.get("result", ""))[:80]
    if event_type == "planner_decision":
        plan = data.get("plan", {})
        if isinstance(plan, dict):
            return f"{plan.get('name')}，原因：{plan.get('reason')}"
    if event_type == "safety_check":
        return f"risk={data.get('risk_level')}"
    if event_type == "state_created":
        return f"city={data.get('city')}"
    if event_type == "error":
        return str(data.get("error", ""))[:80]
    return ""


def format_arguments(arguments: Any) -> str:
    if not isinstance(arguments, dict) or not arguments:
        return ""
    return ", ".join(f"{key}={value}" for key, value in arguments.items())


def load_trace_events(path: Path = TRACE_FILE) -> list[TraceEvent]:
    if not path.exists():
        return []
    events: list[TraceEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and {"trace_id", "step", "event_type", "message", "data"}.issubset(item):
            events.append(item)  # type: ignore[arg-type]
    return events


def print_recent_traces(limit: int = RECENT_TRACE_LIMIT) -> None:
    events = load_trace_events()
    if not events:
        print("还没有 trace 记录。")
        return

    traces: dict[str, list[TraceEvent]] = {}
    for event in events:
        traces.setdefault(event["trace_id"], []).append(event)

    print(f"最近 {min(limit, len(traces))} 条 Trace：")
    for trace_id, trace_events in list(traces.items())[-limit:]:
        final_events = [event for event in trace_events if event["event_type"] == "final_answer"]
        status = "error" if any(event["event_type"] == "error" for event in trace_events) else "ok"
        final_preview = ""
        if final_events:
            answer = final_events[-1]["data"].get("answer", "")
            final_preview = str(answer).replace("\n", " ")[:60]
        print(f"- {trace_id}：{len(trace_events)} events，status={status}，final={final_preview}")


def clear_traces(path: Path = TRACE_FILE) -> None:
    path.write_text("", encoding="utf-8")
    print(f"已清空 {path.name}。")


def main() -> None:
    print("Day 12 Observable Agent")
    print("命令：trace、:traces、:clear-traces、exit")
    latest_events: list[TraceEvent] = []

    while True:
        try:
            user_input = input("\n你：").strip()
        except EOFError:
            print("\n已退出。")
            break

        if user_input.lower() in {"exit", "quit", "退出"}:
            print("Agent：已退出。明天可以学习错误处理、重试与降级。")
            break
        if user_input == "trace":
            print_trace_summary(latest_events)
            continue
        if user_input == ":traces":
            print_recent_traces()
            continue
        if user_input == ":clear-traces":
            clear_traces()
            latest_events = []
            continue
        if not user_input:
            print("Agent：请输入问题，例如：北京今天适合跑步吗？")
            continue

        answer, latest_events = observable_agent(user_input)
        print("\nAgent：")
        print(answer)
        print_trace_summary(latest_events)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出。")
        sys.exit(0)
