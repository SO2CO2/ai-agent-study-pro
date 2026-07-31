"""
Day 3: Minimal Tool Calling Agent

Run:
    python3 day3_tool_calling_agent.py

This file uses a fake LLM so you can learn the tool-calling flow without
setting up an API key. Later, you can replace fake_llm() with a real model API.
"""

from datetime import datetime
from typing import Any


# 1. Tool schemas: this is how we describe available tools to the model.
TOOLS = [
    {
        "name": "get_weather",
        "description": "查询指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名，例如北京、上海、深圳",
                }
            },
            "required": ["city"],
        },
    },
    {
        "name": "get_current_time",
        "description": "查询当前本地时间",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# 2. Real tools: these functions are executed by your program, not by the LLM.
def get_weather(city: str) -> str:
    fake_weather = {
        "北京": "晴，18-28 度，适合户外活动",
        "上海": "多云，22-30 度，体感有点闷",
        "深圳": "阵雨，25-31 度，出门建议带伞",
        "广州": "雷阵雨，24-30 度，注意防雨",
    }
    return fake_weather.get(city, f"暂时没有 {city} 的天气数据")


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "get_current_time": get_current_time,
}


# 3. Fake LLM: simulates a model deciding whether to call a tool.
def fake_llm(user_input: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    """
    A real LLM would receive user_input + tools and return either:
    - a normal answer, or
    - a tool call with tool name and arguments.

    This fake version uses simple rules so you can focus on the architecture.
    """
    del tools  # Kept in the signature to match the real tool-calling shape.

    if any(word in user_input for word in ["几点", "时间", "现在"]):
        return {
            "type": "tool_call",
            "tool_name": "get_current_time",
            "arguments": {},
        }

    city_names = ["北京", "上海", "深圳", "广州"]
    mentions_weather_need = any(
        word in user_input
        for word in ["天气", "温度", "下雨", "带伞", "出门", "跑步", "户外"]
    )

    for city in city_names:
        if city in user_input and mentions_weather_need:
            return {
                "type": "tool_call",
                "tool_name": "get_weather",
                "arguments": {"city": city},
            }

    if mentions_weather_need:
        return {
            "type": "final_answer",
            "content": "你想查询哪个城市的天气？目前我支持北京、上海、深圳、广州。",
        }

    return {
        "type": "final_answer",
        "content": "我是第三天版本的学习 Agent，可以查询天气和当前时间。",
    }


# 4. Validation: never execute a tool call blindly.
def validate_tool_call(tool_call: dict[str, Any]) -> tuple[bool, str]:
    tool_name = tool_call.get("tool_name")
    arguments = tool_call.get("arguments")

    if tool_name not in TOOL_FUNCTIONS:
        return False, f"未知工具：{tool_name}"

    if not isinstance(arguments, dict):
        return False, "工具参数必须是一个字典"

    if tool_name == "get_weather":
        city = arguments.get("city")
        if not isinstance(city, str) or not city.strip():
            return False, "get_weather 需要 city 参数"

    return True, "ok"


# 5. Tool executor: routes a valid tool call to the matching Python function.
def execute_tool(tool_call: dict[str, Any]) -> str:
    tool_name = tool_call["tool_name"]
    arguments = tool_call["arguments"]
    tool_function = TOOL_FUNCTIONS[tool_name]

    if tool_name == "get_current_time":
        return tool_function()

    return tool_function(**arguments)


# 6. Final response: in a real app, this can be another LLM call.
def create_final_answer(user_input: str, tool_name: str, tool_result: str) -> str:
    if tool_name == "get_weather":
        return f"根据查询结果：{tool_result}。你的问题是：{user_input}"

    if tool_name == "get_current_time":
        return f"现在的本地时间是：{tool_result}"

    return f"工具结果：{tool_result}"


def agent(user_input: str) -> str:
    model_response = fake_llm(user_input, TOOLS)

    if model_response["type"] == "final_answer":
        return model_response["content"]

    if model_response["type"] != "tool_call":
        return "模型返回了无法识别的响应类型。"

    is_valid, message = validate_tool_call(model_response)
    if not is_valid:
        return f"工具调用参数有问题：{message}"

    tool_result = execute_tool(model_response)
    return create_final_answer(
        user_input=user_input,
        tool_name=model_response["tool_name"],
        tool_result=tool_result,
    )


def main() -> None:
    print("Day 3 Tool Calling Agent")
    print("你可以问：北京天气怎么样？现在几点？深圳出门要带伞吗？")
    print("输入 exit / quit / 退出 结束。")

    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() in ["exit", "quit", "退出"]:
            print("Agent：明天见，继续升级。")
            break

        if not user_input:
            print("Agent：请输入一个问题。")
            continue

        print("Agent：", agent(user_input))


if __name__ == "__main__":
    main()
