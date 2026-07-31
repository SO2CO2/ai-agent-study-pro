"""
Day 6: Real LLM Planner With Multi-step Agent State

Run:
    export OPENAI_API_KEY="your_api_key"
    python3 day6_llm_planner_agent.py

Optional:
    export OPENAI_MODEL="gpt-4.1-mini"

Day 4 taught:
    state + Plan -> Act -> Observe loop

Day 5 taught:
    real LLM + tool calling

Day 6 combines them:
    The LLM plans the next step from state.
    Python validates the plan, executes tools, updates state, and controls stop.

Important boundary:
    LLM Planner decides what should happen next.
    Python decides whether that plan is valid and performs the real action.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, TypedDict


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-4.1-mini"
SUPPORTED_CITIES = ["北京", "上海", "深圳", "广州"]
MAX_STEPS = 5
ALLOWED_STEPS = [
    "get_weather",
    "get_current_time",
    "create_final_answer",
    "ask_user",
    "finish",
]


class AgentError(Exception):
    """A readable error for this learning script."""


class AgentState(TypedDict):
    # State is the agent's working memory for this single task.
    user_goal: str
    city: str | None
    weather: str | None
    current_time: str | None
    steps: list[str]
    observations: list[str]
    planner_reasons: list[str]
    final_answer: str | None
    error: str | None


def get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise AgentError(
            "缺少 OPENAI_API_KEY。请先运行：\n"
            'export OPENAI_API_KEY="你的 API key"'
        )
    return api_key


def call_openai_text(instructions: str, user_content: str) -> str:
    """
    Call the OpenAI Responses API using only Python standard library.

    This keeps the Day 6 file dependency-free. The planner and final answer
    generator both use this helper.
    """
    api_key = get_api_key()
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    payload = {
        "model": model,
        "instructions": instructions,
        "input": [
            {
                "role": "user",
                "content": user_content,
            }
        ],
    }

    request = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise AgentError(f"OpenAI API 请求失败：HTTP {error.code}\n{details}") from error
    except urllib.error.URLError as error:
        raise AgentError(f"无法连接 OpenAI API：{error.reason}") from error
    except TimeoutError as error:
        raise AgentError("OpenAI API 请求超时，请稍后重试。") from error

    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    # Fallback parser for message content blocks.
    parts: list[str] = []
    for item in data.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)

    return "\n".join(parts).strip()


def create_initial_state(user_input: str) -> AgentState:
    return {
        "user_goal": user_input,
        "city": extract_city_by_rule(user_input),
        "weather": None,
        "current_time": None,
        "steps": [],
        "observations": [],
        "planner_reasons": [],
        "final_answer": None,
        "error": None,
    }


def extract_city_by_rule(user_input: str) -> str | None:
    """
    A tiny helper, not the main lesson.

    Day 6 lets the LLM Planner infer city too. If this rule does not find a
    city, the planner may still choose get_weather with a city argument.
    """
    aliases = {
        "帝都": "北京",
        "魔都": "上海",
        "羊城": "广州",
        "鹏城": "深圳",
    }
    for alias, city in aliases.items():
        if alias in user_input:
            return city

    for city in SUPPORTED_CITIES:
        if city in user_input:
            return city
    return None


def get_weather(city: str) -> str:
    # Tool function: the LLM can ask for this, but Python executes it.
    fake_weather = {
        "北京": "晴，18-28 度，空气质量良好，适合户外活动",
        "上海": "多云，22-30 度，湿度较高，体感偏闷",
        "深圳": "阵雨，25-31 度，降雨概率较高，出门建议带伞",
        "广州": "雷阵雨，24-30 度，午后可能有强降雨",
    }
    return fake_weather.get(city, f"暂时没有 {city} 的天气数据")


def get_current_time() -> str:
    # Tool function: the LLM can ask for this, but Python executes it.
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def state_for_prompt(state: AgentState) -> dict[str, Any]:
    """
    Keep only the information the planner needs.
    Avoid sending unnecessary internal fields.
    """
    return {
        "user_goal": state["user_goal"],
        "city": state["city"],
        "weather": state["weather"],
        "current_time": state["current_time"],
        "steps": state["steps"],
        "observations": state["observations"],
    }


def build_planner_prompt(state: AgentState) -> str:
    """
    The planner prompt is an interface contract.

    The model should not answer the user here. It should return a JSON plan.
    """
    state_json = json.dumps(state_for_prompt(state), ensure_ascii=False, indent=2)
    allowed_steps = ", ".join(ALLOWED_STEPS)

    return f"""
你是一个 Agent Planner，不是聊天助手。
你的任务是根据当前 state 决定下一步动作。

你只能选择以下 next_step 之一：
{allowed_steps}

动作含义：
- get_weather: 查询指定城市天气。arguments 必须是 {{"city": "城市名"}}。
- get_current_time: 查询当前本地时间。arguments 必须是 {{}}。
- create_final_answer: 信息足够，生成最终运动建议。arguments 必须是 {{}}。
- ask_user: 缺少必要信息，需要追问用户。arguments 必须是 {{"question": "问题"}}。
- finish: 任务已经完成。arguments 必须是 {{}}。

规划规则：
- 如果缺少城市，并且无法从 user_goal 推断城市，选择 ask_user。
- 如果可以从 user_goal 推断城市，即使 state.city 为 null，也可以选择 get_weather 并提供城市。
- 如果缺少 weather，优先选择 get_weather。
- 如果已有 weather 但缺少 current_time，选择 get_current_time。
- 只有 weather 和 current_time 都存在时，才选择 create_final_answer。
- 不要重复已经完成且已有结果的步骤。
- 你必须只返回 JSON，不要返回 Markdown，不要返回解释文字。

返回格式：
{{
  "next_step": "get_weather",
  "arguments": {{"city": "北京"}},
  "reason": "需要先查询天气，才能判断是否适合运动。"
}}

当前 state：
{state_json}
""".strip()


PLANNER_INSTRUCTIONS = """
你是一个严格的 Agent Planner。
只输出合法 JSON object，不输出 Markdown，不输出额外解释。
""".strip()


FINAL_ANSWER_INSTRUCTIONS = """
你是一个中文运动建议助手。
根据用户目标、天气、当前时间和执行轨迹，给出简洁、可靠、自然的最终建议。
不要编造没有提供的信息。
""".strip()


def parse_json_output(text: str) -> dict[str, Any]:
    """
    Parse model JSON. Also tolerates accidental ```json fences for teaching.
    """
    cleaned = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise AgentError(f"Planner 没有返回合法 JSON：{text}") from error

    if not isinstance(data, dict):
        raise AgentError("Planner 返回值必须是 JSON object。")

    return data


def call_llm_for_plan(state: AgentState) -> dict[str, Any]:
    prompt = build_planner_prompt(state)
    raw_output = call_openai_text(
        instructions=PLANNER_INSTRUCTIONS,
        user_content=prompt,
    )
    return parse_json_output(raw_output)


def validate_plan(plan: dict[str, Any], state: AgentState) -> None:
    """
    Treat model output as a proposal, not a command.
    Code must validate before executing anything.
    """
    next_step = plan.get("next_step")
    arguments = plan.get("arguments")
    reason = plan.get("reason")

    if next_step not in ALLOWED_STEPS:
        raise AgentError(f"不允许的 next_step：{next_step}")

    if not isinstance(arguments, dict):
        raise AgentError("plan.arguments 必须是 object。")

    if not isinstance(reason, str) or not reason.strip():
        raise AgentError("plan.reason 必须是非空字符串。")

    if next_step == "get_weather":
        city = arguments.get("city")
        if not isinstance(city, str) or not city.strip():
            raise AgentError("get_weather 需要字符串参数 city。")
        if city not in SUPPORTED_CITIES:
            raise AgentError(f"暂不支持城市：{city}")
        if state["weather"] is not None:
            raise AgentError("weather 已存在，不应重复调用 get_weather。")

    if next_step == "get_current_time":
        if arguments:
            raise AgentError("get_current_time 不需要参数。")
        if state["current_time"] is not None:
            raise AgentError("current_time 已存在，不应重复调用 get_current_time。")

    if next_step == "create_final_answer":
        if arguments:
            raise AgentError("create_final_answer 不需要参数。")
        if not state["weather"] or not state["current_time"]:
            raise AgentError("weather 和 current_time 都存在后才能生成最终回答。")

    if next_step == "ask_user":
        question = arguments.get("question")
        if not isinstance(question, str) or not question.strip():
            raise AgentError("ask_user 需要字符串参数 question。")

    if next_step == "finish" and not state["final_answer"]:
        raise AgentError("没有 final_answer 时不能 finish。")


def create_final_answer_with_llm(state: AgentState) -> str:
    final_input = {
        "user_goal": state["user_goal"],
        "city": state["city"],
        "weather": state["weather"],
        "current_time": state["current_time"],
        "observations": state["observations"],
    }
    prompt = json.dumps(final_input, ensure_ascii=False, indent=2)
    answer = call_openai_text(
        instructions=FINAL_ANSWER_INSTRUCTIONS,
        user_content=prompt,
    )
    return answer or "信息已获取，但模型没有生成最终回答。"


def execute_step(plan: dict[str, Any], state: AgentState) -> None:
    """
    Act + Observe.

    The LLM planned the next step. Python now executes the validated step and
    writes the observation back into state.
    """
    next_step = plan["next_step"]
    arguments = plan["arguments"]
    reason = plan["reason"]

    state["planner_reasons"].append(reason)

    if next_step == "get_weather":
        city = arguments["city"]
        state["city"] = city
        state["weather"] = get_weather(city)
        state["steps"].append("get_weather")
        state["observations"].append(f"{city}天气：{state['weather']}")
        return

    if next_step == "get_current_time":
        state["current_time"] = get_current_time()
        state["steps"].append("get_current_time")
        state["observations"].append(f"当前时间：{state['current_time']}")
        return

    if next_step == "create_final_answer":
        state["final_answer"] = create_final_answer_with_llm(state)
        state["steps"].append("create_final_answer")
        state["observations"].append("信息已足够，生成最终回答")
        return

    if next_step == "ask_user":
        state["final_answer"] = arguments["question"]
        state["steps"].append("ask_user")
        state["observations"].append("缺少必要信息，需要追问用户")
        return

    if next_step == "finish":
        state["steps"].append("finish")
        state["observations"].append("Planner 判断任务结束")
        return

    state["error"] = f"未知步骤：{next_step}"


def should_stop(state: AgentState, step_count: int) -> bool:
    if state["final_answer"] is not None:
        return True

    if state["error"] is not None:
        return True

    return step_count >= MAX_STEPS


def format_trace(state: AgentState) -> str:
    lines = ["执行轨迹："]
    for index, observation in enumerate(state["observations"], start=1):
        step_name = state["steps"][index - 1]
        reason = state["planner_reasons"][index - 1]
        lines.append(f"{index}. Plan: {step_name}")
        lines.append(f"   Reason: {reason}")
        lines.append(f"   Observe: {observation}")
    return "\n".join(lines)


def agent(user_input: str, show_trace: bool = True) -> str:
    state = create_initial_state(user_input)

    for step_count in range(1, MAX_STEPS + 1):
        try:
            plan = call_llm_for_plan(state)
            validate_plan(plan, state)
            execute_step(plan, state)
        except AgentError as error:
            state["error"] = str(error)
            break

        if should_stop(state, step_count):
            break

    if state["error"]:
        answer = f"执行过程中出现问题：{state['error']}"
    elif state["final_answer"]:
        answer = state["final_answer"]
    else:
        answer = "已经达到最大步骤数，任务还没有完成。"

    if show_trace:
        return f"{format_trace(state)}\n\n最终回答：\n{answer}"

    return answer


def main() -> None:
    print("Day 6 LLM Planner Agent")
    print("你可以问：我今天在北京适合跑步吗？深圳现在出门跑步要带伞吗？")
    print("输入 trace 可以切换是否显示执行轨迹。")
    print("输入 exit / quit / 退出 结束。")

    show_trace = True

    while True:
        try:
            user_input = input("\n你：").strip()
        except EOFError:
            print("\n已退出。")
            break

        if user_input.lower() in ["exit", "quit", "退出"]:
            print("Agent：明天继续学习记忆和上下文管理。")
            break

        if user_input.lower() == "trace":
            show_trace = not show_trace
            status = "开启" if show_trace else "关闭"
            print(f"Agent：执行轨迹已{status}。")
            continue

        if not user_input:
            print("Agent：请输入一个运动建议问题。")
            continue

        try:
            print("Agent：")
            print(agent(user_input, show_trace=show_trace))
        except AgentError as error:
            print("Agent 出错：")
            print(error)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出。")
        sys.exit(0)
