"""
Day 7: Memory and Context Management for an LLM Planner Agent

Run:
    export OPENAI_API_KEY="your_api_key"
    python3 day7_memory_agent.py

Optional:
    export OPENAI_MODEL="gpt-4.1-mini"

What is new compared with Day 6:
    - conversation_history: short-term memory for this running session
    - memory.json: long-term memory that survives program restarts
    - retrieval: only relevant long-term memories enter the LLM context

Important boundary:
    State tracks ONE task.
    History tracks THIS conversation.
    Long-term memory stores facts useful in FUTURE conversations.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-4.1-mini"
MEMORY_FILE = Path(__file__).with_name("memory.json")
SUPPORTED_CITIES = ["北京", "上海", "深圳", "广州"]
MAX_STEPS = 5
MAX_HISTORY_MESSAGES = 6
MAX_RETRIEVED_MEMORIES = 3
ALLOWED_STEPS = ["get_weather", "get_current_time", "create_final_answer", "ask_user"]


class AgentError(Exception):
    """A readable error for this learning script."""


class Memory(TypedDict):
    id: str
    content: str
    category: str
    importance: int
    created_at: str
    updated_at: str


class Message(TypedDict):
    role: str
    content: str


class AgentState(TypedDict):
    # State is working memory for one user request, not persistent memory.
    user_goal: str
    city: str | None
    weather: str | None
    current_time: str | None
    steps: list[str]
    observations: list[str]
    planner_reasons: list[str]
    final_answer: str | None
    error: str | None


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise AgentError(
            "缺少 OPENAI_API_KEY。请先运行：\n"
            'export OPENAI_API_KEY="你的 API key"'
        )
    return api_key


def call_openai_text(instructions: str, user_content: str) -> str:
    """Call the Responses API using only Python standard library."""
    payload = {
        "model": os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        "instructions": instructions,
        "input": [{"role": "user", "content": user_content}],
    }
    request = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {get_api_key()}",
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

    parts: list[str] = []
    for item in data.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for block in item.get("content", []):
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Long-term memory: stored in a local JSON file.
# ---------------------------------------------------------------------------

def load_memories() -> list[Memory]:
    """Read long-term memories. A missing file simply means no memories yet."""
    if not MEMORY_FILE.exists():
        return []

    try:
        raw_data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AgentError(f"无法读取记忆文件 {MEMORY_FILE.name}：{error}") from error

    if not isinstance(raw_data, list):
        raise AgentError(f"记忆文件 {MEMORY_FILE.name} 必须是 JSON array。")

    memories: list[Memory] = []
    required_keys = {"id", "content", "category", "importance", "created_at", "updated_at"}
    for item in raw_data:
        if not isinstance(item, dict) or not required_keys.issubset(item):
            continue  # Ignore malformed entries so one bad item does not end a lesson.
        if not isinstance(item["content"], str) or not isinstance(item["importance"], int):
            continue
        memories.append(item)  # type: ignore[arg-type]
    return memories


def save_memories(memories: list[Memory]) -> None:
    """Persist memories in readable JSON so learners can inspect the data."""
    try:
        MEMORY_FILE.write_text(
            json.dumps(memories, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise AgentError(f"无法写入记忆文件 {MEMORY_FILE.name}：{error}") from error


def keyword_set(text: str) -> set[str]:
    """A deliberately simple retrieval feature for Day 7, not semantic search."""
    normalized = re.sub(r"[^\w\u4e00-\u9fff]", "", text.lower())
    chinese_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)
    latin_words = re.findall(r"[a-z0-9_]{2,}", normalized)
    keywords = set(latin_words)

    # Chinese has no spaces. Add 2-character slices as a minimal learning aid.
    for chunk in chinese_chunks:
        keywords.add(chunk)
        keywords.update(chunk[index : index + 2] for index in range(len(chunk) - 1))
    return keywords


def retrieve_relevant_memories(query: str, memories: list[Memory]) -> list[Memory]:
    """
    Retrieve only the memories useful for this query.

    Score = keyword overlap * 10 + importance.
    Later, replace keyword_set() with Embedding/vector retrieval without changing
    the rest of the agent flow.
    """
    query_keywords = keyword_set(query)
    scored: list[tuple[int, Memory]] = []
    for memory in memories:
        overlap = len(query_keywords & keyword_set(memory["content"]))
        score = overlap * 10 + memory["importance"]
        if overlap > 0:
            scored.append((score, memory))

    scored.sort(key=lambda item: (item[0], item[1]["updated_at"]), reverse=True)
    return [memory for _, memory in scored[:MAX_RETRIEVED_MEMORIES]]


def extract_memories_from_turn(user_input: str, final_answer: str) -> list[dict[str, Any]]:
    """
    Decide what is worth remembering after one turn.

    This is intentionally rule-based for Day 7: its behavior is visible and
    testable. In a later lesson, this function is a good place to call an LLM
    and require structured JSON such as {content, category, importance}.
    """
    candidates: list[dict[str, Any]] = []

    name_match = re.search(r"(?:我叫|我的名字是)\s*([\u4e00-\u9fffA-Za-z]{2,20})", user_input)
    if name_match:
        candidates.append(
            {
                "content": f"用户的名字是{name_match.group(1)}。",
                "category": "user_profile",
                "importance": 5,
            }
        )

    preference_match = re.search(r"(?:我喜欢|我偏好|请)([^。！？!?]{2,30})(?:回答|回复|风格)?", user_input)
    if preference_match and any(word in user_input for word in ["简洁", "详细", "中文", "英文"]):
        candidates.append(
            {
                "content": f"用户偏好：{preference_match.group(0).strip()}。",
                "category": "user_preference",
                "importance": 4,
            }
        )

    if "Agent" in user_input or "agent" in user_input or "智能体" in user_input:
        candidates.append(
            {
                "content": "用户正在学习 Agent 开发。",
                "category": "learning_progress",
                "importance": 5,
            }
        )

    # Do not save temporary weather or a whole answer as long-term memory.
    # final_answer is kept as an argument to show where an LLM extractor would
    # consider both sides of a conversation in a later version.
    _ = final_answer
    return candidates


def upsert_memories(memories: list[Memory], candidates: list[dict[str, Any]]) -> list[Memory]:
    """Add new facts, or refresh an identical fact instead of duplicating it."""
    for candidate in candidates:
        content = candidate["content"]
        existing = next((memory for memory in memories if memory["content"] == content), None)
        if existing:
            existing["importance"] = max(existing["importance"], candidate["importance"])
            existing["updated_at"] = now_text()
            continue

        timestamp = now_text()
        memories.append(
            {
                "id": f"memory_{len(memories) + 1:03d}",
                "content": content,
                "category": candidate["category"],
                "importance": candidate["importance"],
                "created_at": timestamp,
                "updated_at": timestamp,
            }
        )
    return memories


# ---------------------------------------------------------------------------
# Agent State, Planner, and tools.
# ---------------------------------------------------------------------------

def extract_city_by_rule(text: str) -> str | None:
    aliases = {"帝都": "北京", "魔都": "上海", "羊城": "广州", "鹏城": "深圳"}
    for alias, city in aliases.items():
        if alias in text:
            return city
    return next((city for city in SUPPORTED_CITIES if city in text), None)


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


def get_weather(city: str) -> str:
    fake_weather = {
        "北京": "晴，18-28 度，空气质量良好，适合户外活动",
        "上海": "多云，22-30 度，湿度较高，体感偏闷",
        "深圳": "阵雨，25-31 度，降雨概率较高，出门建议带伞",
        "广州": "雷阵雨，24-30 度，午后可能有强降雨",
    }
    return fake_weather.get(city, f"暂时没有 {city} 的天气数据")


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_agent_context(
    state: AgentState,
    history: list[Message],
    relevant_memories: list[Memory],
) -> dict[str, Any]:
    """Build a bounded context instead of sending all historical data."""
    return {
        "current_task_state": {
            "user_goal": state["user_goal"],
            "city": state["city"],
            "weather": state["weather"],
            "current_time": state["current_time"],
            "steps": state["steps"],
            "observations": state["observations"],
        },
        "recent_conversation": history[-MAX_HISTORY_MESSAGES:],
        "relevant_long_term_memories": [
            {
                "content": memory["content"],
                "category": memory["category"],
                "importance": memory["importance"],
            }
            for memory in relevant_memories
        ],
    }


PLANNER_INSTRUCTIONS = """
你是一个严格的 Agent Planner。只输出合法 JSON object，不输出 Markdown 或额外解释。
""".strip()


def build_planner_prompt(context: dict[str, Any]) -> str:
    context_json = json.dumps(context, ensure_ascii=False, indent=2)
    return f"""
你是一个 Agent Planner，不是聊天助手。请根据 current_task_state、recent_conversation
和 relevant_long_term_memories 决定下一步动作。会话历史可能包含省略信息，例如“那明天呢”
中的城市；你可以据此选择工具参数，但不要凭空编造城市。

你只能选择：get_weather、get_current_time、create_final_answer、ask_user。
- get_weather: arguments 必须是 {{"city": "北京/上海/深圳/广州"}}。
- get_current_time: arguments 必须是 {{}}。
- create_final_answer: weather 和 current_time 都存在时使用，arguments 必须是 {{}}。
- ask_user: 缺少必要信息时使用，arguments 必须是 {{"question": "..."}}。

规则：
- 没有城市时，先从近期会话和相关记忆中查找；仍找不到才 ask_user。
- 没有天气时，调用 get_weather；没有当前时间时，调用 get_current_time。
- 不重复调用已得到结果的工具。
- 只返回下列 JSON 格式：
{{"next_step": "get_weather", "arguments": {{"city": "北京"}}, "reason": "..."}}

上下文：
{context_json}
""".strip()


def parse_json_output(text: str) -> dict[str, Any]:
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


def call_llm_for_plan(context: dict[str, Any]) -> dict[str, Any]:
    raw_output = call_openai_text(PLANNER_INSTRUCTIONS, build_planner_prompt(context))
    return parse_json_output(raw_output)


def validate_plan(plan: dict[str, Any], state: AgentState) -> None:
    """The LLM proposes a plan; Python validates it before execution."""
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
        if city not in SUPPORTED_CITIES:
            raise AgentError(f"get_weather 需要受支持的 city：{SUPPORTED_CITIES}")
        if state["weather"] is not None:
            raise AgentError("weather 已存在，不应重复调用 get_weather。")
    elif next_step == "get_current_time":
        if arguments or state["current_time"] is not None:
            raise AgentError("get_current_time 不需要参数且不能重复调用。")
    elif next_step == "create_final_answer":
        if arguments or not state["weather"] or not state["current_time"]:
            raise AgentError("天气和时间齐全后才能生成最终回答。")
    else:
        question = arguments.get("question")
        if not isinstance(question, str) or not question.strip():
            raise AgentError("ask_user 需要非空 question。")


FINAL_ANSWER_INSTRUCTIONS = """
你是中文运动建议助手。根据提供的任务状态、近期会话和相关长期记忆给出简洁、可靠的回答。
不要编造没有提供的信息，也不要提及内部 state、memory 或 planner。
""".strip()


def create_final_answer_with_llm(context: dict[str, Any]) -> str:
    prompt = json.dumps(context, ensure_ascii=False, indent=2)
    answer = call_openai_text(FINAL_ANSWER_INSTRUCTIONS, prompt)
    return answer or "信息已获取，但模型没有生成最终回答。"


def execute_step(plan: dict[str, Any], state: AgentState, context: dict[str, Any]) -> None:
    next_step = plan["next_step"]
    arguments = plan["arguments"]
    state["planner_reasons"].append(plan["reason"])

    if next_step == "get_weather":
        city = arguments["city"]
        state["city"] = city
        state["weather"] = get_weather(city)
        state["steps"].append(next_step)
        state["observations"].append(f"{city}天气：{state['weather']}")
    elif next_step == "get_current_time":
        state["current_time"] = get_current_time()
        state["steps"].append(next_step)
        state["observations"].append(f"当前时间：{state['current_time']}")
    elif next_step == "create_final_answer":
        # Rebuild context so the final-answer LLM sees tool observations too.
        context["current_task_state"]["weather"] = state["weather"]
        context["current_task_state"]["current_time"] = state["current_time"]
        context["current_task_state"]["observations"] = state["observations"]
        state["final_answer"] = create_final_answer_with_llm(context)
        state["steps"].append(next_step)
        state["observations"].append("信息已足够，生成最终回答")
    else:
        state["final_answer"] = arguments["question"]
        state["steps"].append(next_step)
        state["observations"].append("缺少必要信息，需要追问用户")


def format_trace(state: AgentState, relevant_memories: list[Memory]) -> str:
    lines = ["本轮检索到的长期记忆："]
    lines.extend(f"- {memory['content']}" for memory in relevant_memories)
    if not relevant_memories:
        lines.append("- 无")
    lines.append("执行轨迹：")
    for index, observation in enumerate(state["observations"], start=1):
        lines.append(f"{index}. Plan: {state['steps'][index - 1]}")
        lines.append(f"   Reason: {state['planner_reasons'][index - 1]}")
        lines.append(f"   Observe: {observation}")
    return "\n".join(lines)


def agent(user_input: str, history: list[Message], show_trace: bool = True) -> str:
    """Run one task, then update the caller-owned conversation history."""
    state = create_initial_state(user_input)
    memories = load_memories()
    relevant_memories = retrieve_relevant_memories(user_input, memories)

    for _ in range(MAX_STEPS):
        context = build_agent_context(state, history, relevant_memories)
        try:
            plan = call_llm_for_plan(context)
            validate_plan(plan, state)
            execute_step(plan, state, context)
        except AgentError as error:
            state["error"] = str(error)
            break
        if state["final_answer"] is not None:
            break

    if state["error"]:
        answer = f"执行过程中出现问题：{state['error']}"
    elif state["final_answer"]:
        answer = state["final_answer"]
    else:
        answer = "已经达到最大步骤数，任务还没有完成。"

    # Short-term memory: keep this session bounded in RAM.
    history.extend([{"role": "user", "content": user_input}, {"role": "assistant", "content": answer}])
    del history[:-MAX_HISTORY_MESSAGES]

    # Long-term memory: save only selected facts after a successful turn.
    if not state["error"]:
        updated_memories = upsert_memories(memories, extract_memories_from_turn(user_input, answer))
        save_memories(updated_memories)

    if show_trace:
        return f"{format_trace(state, relevant_memories)}\n\n最终回答：\n{answer}"
    return answer


def print_memories() -> None:
    memories = load_memories()
    if not memories:
        print("Agent：还没有长期记忆。")
        return
    print("Agent：长期记忆：")
    for memory in memories:
        print(f"- {memory['id']} [{memory['category']}, 重要性 {memory['importance']}] {memory['content']}")


def handle_memory_command(user_input: str) -> bool:
    """Return True when input was a management command and has been handled."""
    if user_input == ":memories":
        print_memories()
        return True
    if user_input == ":clear-memories":
        save_memories([])
        print("Agent：长期记忆已清空。")
        return True
    if user_input.startswith(":delete-memory "):
        memory_id = user_input.removeprefix(":delete-memory ").strip()
        memories = load_memories()
        remaining = [memory for memory in memories if memory["id"] != memory_id]
        save_memories(remaining)
        message = "已删除。" if len(remaining) < len(memories) else "没有找到该记忆 ID。"
        print(f"Agent：{message}")
        return True
    return False


def main() -> None:
    print("Day 7 Memory Agent")
    print("你可以问：我在北京适合跑步吗？然后继续问：那明天呢？")
    print("命令：trace、:memories、:delete-memory memory_001、:clear-memories、exit")

    history: list[Message] = []
    show_trace = True
    while True:
        try:
            user_input = input("\n你：").strip()
        except EOFError:
            print("\n已退出。")
            break

        if user_input.lower() in ["exit", "quit", "退出"]:
            print("Agent：已保存长期记忆，明天继续学习语义检索。")
            break
        if user_input.lower() == "trace":
            show_trace = not show_trace
            print(f"Agent：执行轨迹已{'开启' if show_trace else '关闭'}。")
            continue
        if handle_memory_command(user_input):
            continue
        if not user_input:
            print("Agent：请输入一个问题。")
            continue

        try:
            print("Agent：")
            print(agent(user_input, history, show_trace))
        except AgentError as error:
            print(f"Agent 出错：\n{error}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出。")
        sys.exit(0)
