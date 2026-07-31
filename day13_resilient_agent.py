"""
Day 13: Resilient Agent With Retry, Fallback, and Partial Answers

Run:
    python3 day13_resilient_agent.py

Try:
    北京今天适合跑步吗？      -> normal success
    深圳今天适合跑步吗？      -> transient failure, retry succeeds
    上海今天适合跑步吗？      -> live weather fails, cached fallback succeeds
    广州今天适合跑步吗？      -> live weather fails, cache misses, partial answer
    今天适合跑步吗？          -> asks user for missing city
    北京今天适合跑步吗？时间失败 -> weather succeeds, time tool fails, partial answer

Commands:
    trace          Show the latest trace summary again.
    :traces        Show recent trace summaries from agent_traces.jsonl.
    :clear-traces  Clear local trace logs.
    exit           Quit.

Today's key idea:
    A reliable Agent is not an Agent that never fails. It is an Agent that knows
    when to retry, when to fall back, when to stop, and how to record the path.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from typing import Any, Callable, TypedDict

from day12_observable_agent import (
    TraceEvent,
    TraceLogger,
    clear_traces,
    print_recent_traces,
)


SUPPORTED_CITIES = ["北京", "上海", "深圳", "广州"]
MAX_LIVE_WEATHER_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 0.1


class AgentError(Exception):
    """Base class for expected Agent errors."""


class UserInputError(AgentError):
    """The user request is missing information that the Agent needs."""


class ValidationError(AgentError):
    """A model plan or tool argument is structurally invalid."""


class TransientToolError(AgentError):
    """A temporary tool failure that may succeed after retry."""


class FatalToolError(AgentError):
    """A non-retryable tool failure."""


class FallbackError(AgentError):
    """The fallback path is also unavailable."""


class MaxStepsError(AgentError):
    """The Agent reached its step limit."""


class WeatherResult(TypedDict):
    value: str | None
    source: str | None
    fallback_used: bool
    error: str | None


class AgentState(TypedDict):
    user_goal: str
    city: str | None
    weather: str | None
    weather_source: str | None
    weather_error: str | None
    fallback_used: bool
    current_time: str | None
    time_error: str | None
    final_answer: str | None
    warnings: list[str]


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
        "weather_source": None,
        "weather_error": None,
        "fallback_used": False,
        "current_time": None,
        "time_error": None,
        "final_answer": None,
        "warnings": [],
    }


def retry_tool_call(
    tool_name: str,
    func: Callable[[int], Any],
    logger: TraceLogger,
    max_attempts: int = 3,
) -> Any:
    """
    Retry a tool only for transient failures.

    The callable receives the attempt number so this lesson can simulate tools
    that fail on the first attempt and recover later.
    """
    for attempt in range(1, max_attempts + 1):
        logger.log(
            "tool_call",
            "调用工具",
            {"tool": tool_name, "attempt": attempt, "max_attempts": max_attempts},
        )
        try:
            result = func(attempt)
            logger.log(
                "tool_result",
                "工具调用成功",
                {"tool": tool_name, "attempt": attempt, "result": result},
            )
            return result
        except TransientToolError as error:
            logger.log(
                "tool_error",
                "工具出现可重试错误",
                {
                    "tool": tool_name,
                    "attempt": attempt,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            if attempt >= max_attempts:
                logger.log(
                    "retry_exhausted",
                    "工具重试次数已用尽",
                    {"tool": tool_name, "max_attempts": max_attempts},
                )
                raise
            delay = RETRY_BASE_DELAY_SECONDS * attempt
            logger.log(
                "retry",
                "准备重试工具",
                {"tool": tool_name, "next_attempt": attempt + 1, "delay_seconds": delay},
            )
            time.sleep(delay)
        except AgentError:
            raise

    raise MaxStepsError(f"{tool_name} reached an unexpected retry state")


def get_weather_live(city: str, attempt: int) -> str:
    """
    Simulated live weather tool.

    Beijing succeeds. Shenzhen fails once then succeeds. Shanghai and Guangzhou
    keep failing so the Agent can demonstrate fallback behavior.
    """
    live_weather = {
        "北京": "实时天气：晴，18-28 度，空气质量良好",
        "深圳": "实时天气：阵雨，25-31 度，降雨概率较高",
    }

    if city == "深圳" and attempt == 1:
        raise TransientToolError("天气服务临时超时")
    if city in {"上海", "广州"}:
        raise TransientToolError("实时天气服务暂时不可用")
    if city not in live_weather:
        raise FatalToolError(f"不支持的城市：{city}")
    return live_weather[city]


def get_weather_cached(city: str) -> str:
    cached_weather = {
        "北京": "缓存天气：晴，18-28 度，空气质量良好",
        "上海": "缓存天气：多云，22-30 度，湿度较高",
        "深圳": "缓存天气：阵雨，25-31 度，降雨概率较高",
    }
    if city not in cached_weather:
        raise FallbackError(f"没有 {city} 的缓存天气")
    return cached_weather[city]


def get_weather_with_fallback(city: str, logger: TraceLogger) -> WeatherResult:
    """Try live weather first; fall back to cached weather when retry fails."""
    try:
        weather = retry_tool_call(
            "get_weather_live",
            lambda attempt: get_weather_live(city, attempt),
            logger,
            max_attempts=MAX_LIVE_WEATHER_ATTEMPTS,
        )
        return {
            "value": weather,
            "source": "live_weather",
            "fallback_used": False,
            "error": None,
        }
    except FatalToolError as error:
        logger.log(
            "error",
            "实时天气工具发生不可重试错误",
            {"error_type": type(error).__name__, "error": str(error)},
        )
        return {"value": None, "source": None, "fallback_used": False, "error": str(error)}
    except TransientToolError as error:
        logger.log(
            "fallback",
            "实时天气失败，切换到缓存天气",
            {"from": "live_weather", "to": "cached_weather", "reason": str(error)},
        )

    try:
        logger.log("tool_call", "调用缓存天气工具", {"tool": "get_weather_cached", "city": city})
        cached = get_weather_cached(city)
        logger.log(
            "tool_result",
            "缓存天气工具返回结果",
            {"tool": "get_weather_cached", "result": cached},
        )
        return {
            "value": cached,
            "source": "cached_weather",
            "fallback_used": True,
            "error": None,
        }
    except FallbackError as error:
        logger.log(
            "tool_error",
            "缓存天气也不可用",
            {
                "tool": "get_weather_cached",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        return {"value": None, "source": None, "fallback_used": True, "error": str(error)}


def get_current_time_safe(user_input: str, logger: TraceLogger) -> tuple[str | None, str | None]:
    """Return current time, or record a recoverable partial-answer failure."""
    logger.log("tool_call", "调用当前时间工具", {"tool": "get_current_time"})
    if "时间失败" in user_input:
        error = "当前时间工具模拟失败"
        logger.log(
            "tool_error",
            "当前时间工具失败，允许生成部分回答",
            {"tool": "get_current_time", "error_type": "TransientToolError", "error": error},
        )
        return None, error

    result = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.log("tool_result", "当前时间工具返回结果", {"tool": "get_current_time", "result": result})
    return result, None


def create_final_answer(state: AgentState) -> str:
    city = state["city"]
    weather = state["weather"]
    current_time = state["current_time"]

    if not city:
        return "我需要知道你所在的城市，才能判断是否适合运动。目前支持：北京、上海、深圳、广州。"

    if not weather:
        return (
            f"我尝试获取 {city} 的实时天气，并在失败后切换到缓存天气，但仍然没有足够可靠的天气信息。\n"
            "因此我不能负责任地判断是否适合户外跑步。你可以稍后再试，或换一个城市。"
        )

    notes = []
    if state["fallback_used"]:
        notes.append("实时天气不可用，本次使用了缓存天气作为降级数据")
    if state["time_error"]:
        notes.append("当前时间工具失败，因此回答不包含精确当前时间")

    suggestion = "比较适合运动"
    risk_notes = []
    if any(word in weather for word in ["阵雨", "雷阵雨", "强降雨"]):
        suggestion = "不太适合户外跑步"
        risk_notes.append("有降雨风险，路面湿滑")
    if any(word in weather for word in ["30 度", "31 度"]):
        risk_notes.append("气温偏高，注意补水和防晒")
    if "湿度较高" in weather:
        risk_notes.append("湿度较高，体感会更闷")

    risk_text = "；".join(risk_notes) if risk_notes else "整体条件比较友好"
    time_text = f"当前时间是：{current_time}。" if current_time else "当前时间暂时不可用。"
    note_text = f"\n说明：{'；'.join(notes)}。" if notes else ""

    advice = "建议选择轻松强度，控制在 30-45 分钟。"
    if suggestion == "不太适合户外跑步":
        advice = "建议改为室内训练，或等天气稳定后再进行短距离慢跑。"

    return (
        f"{city}天气信息：{weather}；{time_text}\n"
        f"综合判断：{suggestion}。\n"
        f"原因：{risk_text}。\n"
        f"{advice}"
        f"{note_text}"
    )


def resilient_agent(user_input: str) -> tuple[str, list[TraceEvent]]:
    logger = TraceLogger()
    state = create_initial_state(user_input)

    logger.log("user_input", "收到用户输入", {"input": user_input})
    logger.log("state_created", "初始化任务状态", {"city": state["city"]})

    try:
        if not state["city"]:
            raise UserInputError("缺少城市信息")

        logger.log(
            "planner_decision",
            "Planner 决定查询天气并准备失败策略",
            {
                "plan": {
                    "primary_tool": "get_weather_live",
                    "retry": True,
                    "fallback": "get_weather_cached",
                }
            },
        )
        weather_result = get_weather_with_fallback(state["city"], logger)
        state["weather"] = weather_result["value"]
        state["weather_source"] = weather_result["source"]
        state["fallback_used"] = weather_result["fallback_used"]
        state["weather_error"] = weather_result["error"]
        if state["weather_error"]:
            state["warnings"].append(state["weather_error"])

        logger.log(
            "planner_decision",
            "Planner 决定查询当前时间",
            {"plan": {"tool": "get_current_time", "partial_answer_if_failed": True}},
        )
        current_time, time_error = get_current_time_safe(user_input, logger)
        state["current_time"] = current_time
        state["time_error"] = time_error
        if time_error:
            state["warnings"].append(time_error)

    except UserInputError as error:
        logger.log(
            "ask_user",
            "用户输入缺少必要信息",
            {"error_type": type(error).__name__, "error": str(error)},
        )
    except AgentError as error:
        logger.log(
            "error",
            "Agent 遇到无法恢复的错误",
            {"error_type": type(error).__name__, "error": str(error)},
        )
        state["warnings"].append(str(error))
    except Exception as error:
        logger.log(
            "error",
            "未知错误，安全停止",
            {"error_type": type(error).__name__, "error": str(error)},
        )
        state["warnings"].append("出现未知错误")

    answer_type = "partial_answer" if state["warnings"] else "normal_answer"
    if state["city"] and not state["weather"]:
        answer_type = "safe_stop"
    if not state["city"]:
        answer_type = "ask_user"

    state["final_answer"] = create_final_answer(state)
    logger.log(
        "final_answer",
        "生成最终回答",
        {
            "answer_type": answer_type,
            "weather_source": state["weather_source"],
            "fallback_used": state["fallback_used"],
            "warnings": state["warnings"],
            "answer": state["final_answer"],
        },
    )
    return state["final_answer"], logger.get_events()


def summarize_event_data(event: TraceEvent) -> str:
    data = event["data"]
    event_type = event["event_type"]
    if event_type == "tool_call":
        tool = data.get("tool")
        attempt = data.get("attempt")
        if attempt:
            return f"{tool} attempt={attempt}/{data.get('max_attempts')}"
        return str(tool)
    if event_type == "tool_result":
        return str(data.get("result", ""))[:80]
    if event_type == "tool_error":
        return f"{data.get('error_type')}: {data.get('error')}"
    if event_type == "retry":
        return f"next_attempt={data.get('next_attempt')} delay={data.get('delay_seconds')}s"
    if event_type == "fallback":
        return f"{data.get('from')} -> {data.get('to')}"
    if event_type == "planner_decision":
        return str(data.get("plan", ""))[:100]
    if event_type == "final_answer":
        return f"type={data.get('answer_type')}, fallback={data.get('fallback_used')}"
    if event_type in {"error", "ask_user"}:
        return str(data.get("error", ""))[:80]
    if event_type == "state_created":
        return f"city={data.get('city')}"
    return ""


def print_trace_summary(events: list[TraceEvent]) -> None:
    if not events:
        print("Trace：无事件。")
        return
    print(f"\nTrace Summary：{events[0]['trace_id']}")
    for event in events:
        details = summarize_event_data(event)
        suffix = f"：{details}" if details else ""
        print(f"{event['step']}. {event['event_type']} - {event['message']}{suffix}")
    print("完整 JSONL 已写入：agent_traces.jsonl")


def main() -> None:
    print("Day 13 Resilient Agent")
    print("命令：trace、:traces、:clear-traces、exit")
    print("示例：北京正常；深圳重试成功；上海降级成功；广州降级失败；输入“时间失败”模拟部分回答。")
    latest_events: list[TraceEvent] = []

    while True:
        try:
            user_input = input("\n你：").strip()
        except EOFError:
            print("\n已退出。")
            break

        if user_input.lower() in {"exit", "quit", "退出"}:
            print("Agent：已退出。明天可以做第二周综合项目。")
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

        answer, latest_events = resilient_agent(user_input)
        print("\nAgent：")
        print(answer)
        print_trace_summary(latest_events)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出。")
        sys.exit(0)
