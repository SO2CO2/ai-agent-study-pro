"""
Day 5: Real LLM Tool Calling Agent

Run:
    export OPENAI_API_KEY="your_api_key"
    python3 day5_real_llm_tool_calling_agent.py

Optional:
    export OPENAI_MODEL="gpt-4.1-mini"

This lesson replaces Day 3's fake_llm() with a real model call.

Core flow:
    1. Send user input + tool schemas to the model.
    2. The model returns zero or more function_call items.
    3. Your code validates and executes the requested tools.
    4. Send function_call_output items back to the model.
    5. The model writes the final answer.

Important:
    The model decides which tool to call, but Python executes the tool.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Callable


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-4.1-mini"


# 1. Tool schemas: these are shown to the model.
# Responses API function tools use this shape:
#   {"type": "function", "name": "...", "description": "...", "parameters": ...}
TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": (
            "查询指定城市的天气。适合回答天气、温度、下雨、带伞、"
            "是否适合出门或跑步等问题。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名，例如北京、上海、深圳、广州",
                }
            },
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_current_time",
        "description": "查询当前本地时间。适合回答现在几点、当前时间等问题。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


# 2. Real tool functions: these are executed by Python, not by the model.
def get_weather(city: str) -> str:
    # To keep Day 5 focused on tool calling, this still uses fake weather data.
    # Later you can replace this function with a real weather API call.
    fake_weather = {
        "北京": "晴，18-28 度，空气质量良好，适合户外活动",
        "上海": "多云，22-30 度，湿度较高，体感偏闷",
        "深圳": "阵雨，25-31 度，出门建议带伞",
        "广州": "雷阵雨，24-30 度，午后可能有强降雨",
    }
    return fake_weather.get(city, f"暂时没有 {city} 的天气数据")


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {
    "get_weather": get_weather,
    "get_current_time": get_current_time,
}


SYSTEM_INSTRUCTIONS = """
你是一个适合新手学习 Agent 开发的中文助手。

你可以使用工具查询天气和当前时间。
如果用户问题需要实时或外部信息，优先调用工具。
如果工具结果已经给出，请基于工具结果回答，不要编造工具没有提供的数据。
回答要简洁、自然，并说明关键依据。
""".strip()


class AgentError(Exception):
    """A readable error for this learning script."""


def get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise AgentError(
            "缺少 OPENAI_API_KEY。请先运行：\n"
            'export OPENAI_API_KEY="你的 API key"'
        )
    return api_key


def call_openai_responses(input_items: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Call the OpenAI Responses API using only Python standard library.

    This avoids requiring students to install the openai package on Day 5.
    """
    api_key = get_api_key()
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    payload = {
        "model": model,
        "instructions": SYSTEM_INSTRUCTIONS,
        "input": input_items,
        "tools": TOOLS,
        # For teaching, keep one tool call per turn so the flow is easier to see.
        "parallel_tool_calls": False,
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
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise AgentError(f"OpenAI API 请求失败：HTTP {error.code}\n{details}") from error
    except urllib.error.URLError as error:
        raise AgentError(f"无法连接 OpenAI API：{error.reason}") from error
    except TimeoutError as error:
        raise AgentError("OpenAI API 请求超时，请稍后重试。") from error


def get_output_items(response: dict[str, Any]) -> list[dict[str, Any]]:
    output = response.get("output")
    if not isinstance(output, list):
        return []
    return [item for item in output if isinstance(item, dict)]


def get_function_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in get_output_items(response)
        if item.get("type") == "function_call"
    ]


def get_output_text(response: dict[str, Any]) -> str:
    """
    Responses API usually provides output_text.
    This fallback also handles message content blocks.
    """
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: list[str] = []
    for item in get_output_items(response):
        if item.get("type") != "message":
            continue

        content = item.get("content")
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in ["output_text", "text"]:
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)

    return "\n".join(parts).strip()


def parse_arguments(arguments_text: Any) -> dict[str, Any]:
    if isinstance(arguments_text, dict):
        return arguments_text

    if not isinstance(arguments_text, str):
        raise AgentError("工具参数不是 JSON 字符串。")

    try:
        arguments = json.loads(arguments_text or "{}")
    except json.JSONDecodeError as error:
        raise AgentError(f"工具参数 JSON 解析失败：{arguments_text}") from error

    if not isinstance(arguments, dict):
        raise AgentError("工具参数必须是 JSON object。")

    return arguments


def validate_tool_call(tool_name: str, arguments: dict[str, Any]) -> None:
    """
    Never execute model-generated tool calls blindly.
    The model can request a call, but your program owns the safety checks.
    """
    if tool_name not in TOOL_FUNCTIONS:
        raise AgentError(f"未知工具：{tool_name}")

    if tool_name == "get_weather":
        city = arguments.get("city")
        if not isinstance(city, str) or not city.strip():
            raise AgentError("get_weather 需要字符串参数 city。")

    if tool_name == "get_current_time" and arguments:
        raise AgentError("get_current_time 不需要参数。")


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    validate_tool_call(tool_name, arguments)

    try:
        if tool_name == "get_current_time":
            return TOOL_FUNCTIONS[tool_name]()
        return TOOL_FUNCTIONS[tool_name](**arguments)
    except Exception as error:
        raise AgentError(f"工具 {tool_name} 执行失败：{error}") from error


def build_tool_outputs(function_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert model function_call items into function_call_output items.
    These outputs will be sent back to the model in the second request.
    """
    outputs: list[dict[str, Any]] = []

    for function_call in function_calls:
        tool_name = function_call.get("name")
        call_id = function_call.get("call_id")

        if not isinstance(tool_name, str) or not tool_name:
            raise AgentError(f"工具调用缺少 name：{function_call}")

        if not isinstance(call_id, str) or not call_id:
            raise AgentError(f"工具调用缺少 call_id：{function_call}")

        arguments = parse_arguments(function_call.get("arguments"))
        tool_result = execute_tool(tool_name, arguments)

        outputs.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": tool_result,
            }
        )

    return outputs


def agent(user_input: str, show_trace: bool = False) -> str:
    """
    One full real-LLM tool calling round.

    Compared with Day 3:
        Day 3: fake_llm() decides the tool call.
        Day 5: OpenAI model decides the tool call.
    """
    input_items: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": user_input,
        }
    ]

    first_response = call_openai_responses(input_items)
    function_calls = get_function_calls(first_response)

    if not function_calls:
        text = get_output_text(first_response)
        return text or "模型没有返回工具调用，也没有返回文本回答。"

    # Preserve the model's function_call items before appending tool outputs.
    # This is required because the second request needs the conversation context.
    input_items.extend(get_output_items(first_response))

    tool_outputs = build_tool_outputs(function_calls)
    input_items.extend(tool_outputs)

    second_response = call_openai_responses(input_items)
    final_answer = get_output_text(second_response)

    if not final_answer:
        final_answer = "工具已执行，但模型没有生成最终文本回答。"

    if not show_trace:
        return final_answer

    trace_lines = ["执行轨迹："]
    for function_call, tool_output in zip(function_calls, tool_outputs):
        trace_lines.append(
            f"- 模型请求工具：{function_call.get('name')} "
            f"参数：{function_call.get('arguments')}"
        )
        trace_lines.append(f"- 工具返回：{tool_output['output']}")

    return "\n".join(trace_lines) + "\n\n最终回答：\n" + final_answer


def main() -> None:
    print("Day 5 Real LLM Tool Calling Agent")
    print("你可以问：北京天气怎么样？现在几点？深圳出门要带伞吗？")
    print("输入 trace 可以切换是否显示工具调用轨迹。")
    print("输入 exit / quit / 退出 结束。")

    show_trace = True

    while True:
        try:
            user_input = input("\n你：").strip()
        except EOFError:
            print("\n已退出。")
            break

        if user_input.lower() in ["exit", "quit", "退出"]:
            print("Agent：明天继续，把真实 LLM 接到多步骤 state 循环里。")
            break

        if user_input.lower() == "trace":
            show_trace = not show_trace
            status = "开启" if show_trace else "关闭"
            print(f"Agent：工具调用轨迹已{status}。")
            continue

        if not user_input:
            print("Agent：请输入一个问题。")
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
