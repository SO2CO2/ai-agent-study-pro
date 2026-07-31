"""
Day 4: Multi-step Agent With Task State

Run:
    python3 day4_multi_step_agent.py

This lesson upgrades Day 3's single tool call into a small task-solving agent.
The key idea is:

    Agent = goal + state + Plan -> Act -> Observe loop + final answer

The code still uses simple rules instead of a real LLM, so beginners can focus
on the agent structure before adding a model API.

Important upgrade points:
    - extract_city() can later be replaced by an LLM information extractor.
    - plan_next_step() can later be replaced by an LLM planner.
    - create_final_answer() can later be replaced by an LLM response generator.
    - get_weather() and get_current_time() should still be executed by code.
"""

from datetime import datetime
from typing import Any, TypedDict


SUPPORTED_CITIES = ["北京", "上海", "深圳", "广州"]
MAX_STEPS = 5


class AgentState(TypedDict):
    # State is the agent's working memory for this task.
    # Every step reads from and writes to this object.
    user_goal: str
    city: str | None
    weather: str | None
    current_time: str | None
    steps: list[str]
    observations: list[str]
    final_answer: str | None
    error: str | None


def create_initial_state(user_input: str) -> AgentState:
    # Course note:
    # For Day 4, we extract city with simple rules.
    # Later, this can become:
    #   LLM(user_input) -> {"city": "...", "task_type": "exercise_advice"}
    return {
        "user_goal": user_input,
        "city": extract_city(user_input),
        "weather": None,
        "current_time": None,
        "steps": [],
        "observations": [],
        "final_answer": None,
        "error": None,
    }


def extract_city(user_input: str) -> str | None:
    # Upgrade point:
    # This is a rule-based information extractor.
    # In a real agent, an LLM can extract structured fields from flexible text:
    #   "明天在魔都跑步合适吗" -> {"city": "上海"}
    # The output should still be structured, not free-form text.
    for city in SUPPORTED_CITIES:
        if city in user_input:
            return city
    return None


def get_weather(city: str) -> str:
    # Tool function:
    # This is intentionally normal Python code. Even with a real LLM, the model
    # should not "pretend" to know the weather. The program should call a real
    # weather API here, then put the result into state.
    fake_weather = {
        "北京": "晴，18-28 度，空气质量良好",
        "上海": "多云，22-30 度，湿度较高",
        "深圳": "阵雨，25-31 度，降雨概率较高",
        "广州": "雷阵雨，24-30 度，午后可能有强降雨",
    }
    return fake_weather.get(city, f"暂时没有 {city} 的天气数据")


def get_current_time() -> str:
    # Tool function:
    # This can stay as code. The LLM may decide to call it, but code executes it.
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def plan_next_step(state: AgentState) -> dict[str, Any]:
    """
    Plan: decide the next action based on current state.

    In a real agent, this planner can be an LLM. Here we use rules to keep the
    decision process transparent for beginners.

    Possible future LLM planner output:
        {"name": "get_weather", "arguments": {"city": "北京"}}
        {"name": "get_current_time", "arguments": {}}
        {"name": "create_final_answer", "arguments": {}}

    Teaching point:
        The planner only decides the next action. It does not execute tools.
    """
    if state["error"]:
        return {"name": "finish"}

    if state["final_answer"]:
        return {"name": "finish"}

    if not state["city"]:
        return {"name": "ask_city"}

    if not state["weather"]:
        return {
            "name": "get_weather",
            "arguments": {"city": state["city"]},
        }

    if not state["current_time"]:
        return {
            "name": "get_current_time",
            "arguments": {},
        }

    return {"name": "create_final_answer"}


def execute_step(step: dict[str, Any], state: AgentState) -> None:
    """
    Act + Observe: execute the planned action and store the observation in state.

    This part usually remains code even after adding a real LLM. The model may
    request a tool call, but your program must validate and execute it.
    """
    step_name = step["name"]

    if step_name == "ask_city":
        state["final_answer"] = (
            "你想查询哪个城市的运动建议？目前支持：北京、上海、深圳、广州。"
        )
        state["steps"].append("ask_city")
        state["observations"].append("缺少城市信息，需要向用户追问")
        return

    if step_name == "get_weather":
        city = step["arguments"]["city"]
        state["weather"] = get_weather(city)
        state["steps"].append("get_weather")
        state["observations"].append(f"{city}天气：{state['weather']}")
        return

    if step_name == "get_current_time":
        state["current_time"] = get_current_time()
        state["steps"].append("get_current_time")
        state["observations"].append(f"当前时间：{state['current_time']}")
        return

    if step_name == "create_final_answer":
        state["final_answer"] = create_final_answer(state)
        state["steps"].append("create_final_answer")
        state["observations"].append("信息已足够，生成最终运动建议")
        return

    if step_name == "finish":
        return

    state["error"] = f"无法识别的步骤：{step_name}"


def should_stop(state: AgentState, step_count: int) -> bool:
    # Safety/control point:
    # Real agents must have stopping conditions. Without them, an agent may keep
    # planning and calling tools forever.
    if state["final_answer"] is not None:
        return True

    if state["error"] is not None:
        return True

    return step_count >= MAX_STEPS


def create_final_answer(state: AgentState) -> str:
    # Upgrade point:
    # This is a rule-based answer generator.
    # Later, you can replace it with an LLM call:
    #   prompt = user_goal + weather + current_time + constraints
    #   LLM(prompt) -> polished final answer
    #
    # Keep this principle:
    #   The final answer should solve the user's goal, not just paste tool data.
    city = state["city"]
    weather = state["weather"]
    current_time = state["current_time"]
    user_goal = state["user_goal"]

    if not city or not weather or not current_time:
        return "信息还不完整，暂时无法给出可靠建议。"

    risk_notes = []
    suggestion = "可以运动"

    if any(word in weather for word in ["阵雨", "雷阵雨", "强降雨"]):
        suggestion = "不太适合户外跑步"
        risk_notes.append("有降雨风险，路面湿滑")

    if any(word in weather for word in ["30 度", "31 度"]):
        risk_notes.append("气温偏高，注意补水和防晒")

    if "湿度较高" in weather:
        risk_notes.append("湿度较高，体感会更闷")

    advice = "建议选择轻松强度，控制在 30-45 分钟。"
    if suggestion == "不太适合户外跑步":
        advice = "建议改为室内训练，或等降雨结束后再进行短距离慢跑。"

    risk_text = "；".join(risk_notes) if risk_notes else "整体条件比较友好"

    return (
        f"针对你的目标「{user_goal}」，我的建议是：{suggestion}。\n"
        f"{city}当前天气信息是：{weather}；当前时间是：{current_time}。\n"
        f"判断依据：{risk_text}。\n"
        f"{advice}"
    )


def format_trace(state: AgentState) -> str:
    # Teaching helper:
    # The trace is not required for a product, but it is very useful in a course.
    # It lets students see the Plan -> Act -> Observe process step by step.
    lines = ["执行轨迹："]
    for index, observation in enumerate(state["observations"], start=1):
        step_name = state["steps"][index - 1]
        lines.append(f"{index}. {step_name} -> {observation}")
    return "\n".join(lines)


def agent(user_input: str, show_trace: bool = False) -> str:
    state = create_initial_state(user_input)

    # Main agent loop:
    # 1. Plan the next step.
    # 2. Act by executing that step.
    # 3. Observe by writing the result back into state.
    # 4. Stop when the task is done or the step limit is reached.
    for step_count in range(1, MAX_STEPS + 1):
        step = plan_next_step(state)
        execute_step(step, state)

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
    print("Day 4 Multi-step Agent")
    print("你可以问：我今天在北京适合跑步吗？深圳现在出门跑步要带伞吗？")
    print("输入 trace 可以切换是否显示执行轨迹。")
    print("输入 exit / quit / 退出 结束。")

    show_trace = True

    while True:
        user_input = input("\n你：").strip()

        if user_input.lower() in ["exit", "quit", "退出"]:
            print("Agent：明天继续，把这个结构接到真实 LLM 上。")
            break

        if user_input.lower() == "trace":
            show_trace = not show_trace
            status = "开启" if show_trace else "关闭"
            print(f"Agent：执行轨迹已{status}。")
            continue

        if not user_input:
            print("Agent：请输入一个运动建议问题。")
            continue

        print("Agent：")
        print(agent(user_input, show_trace=show_trace))


if __name__ == "__main__":
    main()
