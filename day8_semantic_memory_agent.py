"""
Day 8: Semantic Memory Retrieval With Embeddings

Run:
    export OPENAI_API_KEY="your_api_key"
    python3 day8_semantic_memory_agent.py

Optional:
    export OPENAI_MODEL="gpt-4.1-mini"
    export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"

Day 7 used keyword overlap to retrieve memories. Day 8 upgrades retrieval:
    text -> embedding vector -> cosine similarity -> Top K relevant memories

The program deliberately uses a JSON file instead of a vector database. This
makes every stored vector and every retrieval decision visible for learning.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
MEMORY_FILE = Path(__file__).with_name("memory_with_embeddings.json")
SUPPORTED_CITIES = ["北京", "上海", "深圳", "广州"]
MAX_STEPS = 5
MAX_HISTORY_MESSAGES = 6
MAX_RETRIEVED_MEMORIES = 3
MIN_SIMILARITY = 0.45
ALLOWED_STEPS = ["get_weather", "get_current_time", "create_final_answer", "ask_user"]


class AgentError(Exception):
    """A readable error for this learning script."""


class Memory(TypedDict):
    id: str
    content: str
    category: str
    importance: int
    embedding: list[float]
    created_at: str
    updated_at: str


class Message(TypedDict):
    role: str
    content: str


class AgentState(TypedDict):
    # State belongs to one task. It is different from persistent memory.
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


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Shared HTTP helper for Responses and Embeddings API requests."""
    request = urllib.request.Request(
        url,
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

    if not isinstance(data, dict):
        raise AgentError("OpenAI API 返回了意外的数据格式。")
    return data


def call_openai_text(instructions: str, user_content: str) -> str:
    """Use the Responses API for planning and final natural-language answers."""
    data = post_json(
        OPENAI_RESPONSES_URL,
        {
            "model": os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            "instructions": instructions,
            "input": [{"role": "user", "content": user_content}],
        },
    )
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
# Embeddings and semantic retrieval
# ---------------------------------------------------------------------------

def create_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Convert text into vectors with the Embeddings API.

    Sending a list indexes multiple new memories in one request. The API returns
    an object with data[i].embedding, and index restores the input order.
    """
    if not texts or any(not text.strip() for text in texts):
        raise AgentError("生成 Embedding 的文本不能为空。")

    data = post_json(
        OPENAI_EMBEDDINGS_URL,
        {
            "model": os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            "input": texts,
            "encoding_format": "float",
        },
    )
    raw_items = data.get("data")
    if not isinstance(raw_items, list) or len(raw_items) != len(texts):
        raise AgentError("Embedding API 返回数量与输入文本数量不一致。")

    vectors: list[list[float] | None] = [None] * len(texts)
    for item in raw_items:
        if not isinstance(item, dict):
            raise AgentError("Embedding API 返回了无效向量项。")
        index = item.get("index")
        vector = item.get("embedding")
        if not isinstance(index, int) or not 0 <= index < len(texts):
            raise AgentError("Embedding API 返回了无效 index。")
        if not isinstance(vector, list) or not vector or not all(
            isinstance(value, (int, float)) for value in vector
        ):
            raise AgentError("Embedding API 返回了无效向量。")
        vectors[index] = [float(value) for value in vector]

    if any(vector is None for vector in vectors):
        raise AgentError("Embedding API 返回缺少部分向量。")
    return [vector for vector in vectors if vector is not None]


def create_embedding(text: str) -> list[float]:
    """Small single-text wrapper used when embedding the user's query."""
    return create_embeddings([text])[0]


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """
    Return semantic similarity in [-1, 1]. Higher means more related.

    This explicit implementation is intentionally kept in the lesson. A vector
    database can later do the same search faster over much larger collections.
    """
    if len(vector_a) != len(vector_b) or not vector_a:
        raise AgentError("两个 Embedding 向量必须非空且维度相同。")

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))
    if norm_a == 0 or norm_b == 0:
        raise AgentError("Embedding 向量不能是零向量。")
    return dot_product / (norm_a * norm_b)


def retrieve_semantic_memories(
    query: str,
    memories: list[Memory],
    top_k: int = MAX_RETRIEVED_MEMORIES,
    min_similarity: float = MIN_SIMILARITY,
) -> list[tuple[float, Memory]]:
    """
    Return the Top K memories whose meaning is sufficiently close to the query.

    Importance breaks ties only. This keeps the core Day 8 lesson focused on
    semantic similarity instead of silently mixing several ranking strategies.
    """
    if not memories:
        return []

    query_vector = create_embedding(query)
    scored: list[tuple[float, Memory]] = []
    for memory in memories:
        similarity = cosine_similarity(query_vector, memory["embedding"])
        if similarity >= min_similarity:
            scored.append((similarity, memory))

    scored.sort(
        key=lambda item: (item[0], item[1]["importance"], item[1]["updated_at"]),
        reverse=True,
    )
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Long-term memory stored in JSON
# ---------------------------------------------------------------------------

def load_memories() -> list[Memory]:
    if not MEMORY_FILE.exists():
        return []
    try:
        raw_data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AgentError(f"无法读取记忆文件 {MEMORY_FILE.name}：{error}") from error
    if not isinstance(raw_data, list):
        raise AgentError(f"记忆文件 {MEMORY_FILE.name} 必须是 JSON array。")

    required = {"id", "content", "category", "importance", "embedding", "created_at", "updated_at"}
    memories: list[Memory] = []
    for item in raw_data:
        if not isinstance(item, dict) or not required.issubset(item):
            continue
        embedding = item["embedding"]
        if (
            not isinstance(item["content"], str)
            or not isinstance(item["importance"], int)
            or not isinstance(embedding, list)
            or not embedding
            or not all(isinstance(value, (int, float)) for value in embedding)
        ):
            continue
        item["embedding"] = [float(value) for value in embedding]
        memories.append(item)  # type: ignore[arg-type]
    return memories


def save_memories(memories: list[Memory]) -> None:
    try:
        MEMORY_FILE.write_text(
            json.dumps(memories, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise AgentError(f"无法写入记忆文件 {MEMORY_FILE.name}：{error}") from error


def extract_memories_from_turn(user_input: str, final_answer: str) -> list[dict[str, Any]]:
    """
    Keep Day 7's transparent rule-based memory extraction.

    A future upgrade can ask an LLM to output validated memory JSON. Retrieval
    remains semantic even while this extraction rule stays simple.
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
    _ = final_answer
    return candidates


def upsert_memories(memories: list[Memory], candidates: list[dict[str, Any]]) -> list[Memory]:
    """Embed only genuinely new facts; refresh an identical fact without cost."""
    new_candidates = [
        candidate
        for candidate in candidates
        if not any(memory["content"] == candidate["content"] for memory in memories)
    ]
    if new_candidates:
        vectors = create_embeddings([candidate["content"] for candidate in new_candidates])
        for candidate, vector in zip(new_candidates, vectors):
            timestamp = now_text()
            memories.append(
                {
                    "id": f"memory_{len(memories) + 1:03d}",
                    "content": candidate["content"],
                    "category": candidate["category"],
                    "importance": candidate["importance"],
                    "embedding": vector,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                }
            )

    for candidate in candidates:
        for memory in memories:
            if memory["content"] == candidate["content"]:
                memory["importance"] = max(memory["importance"], candidate["importance"])
                memory["updated_at"] = now_text()
    return memories


# ---------------------------------------------------------------------------
# The Day 6 Planner -> Act -> Observe loop, augmented with semantic memories.
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
    retrieved: list[tuple[float, Memory]],
) -> dict[str, Any]:
    """Bounded context: task state + recent chat + semantic Top K results."""
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
        "semantic_memories": [
            {
                "content": memory["content"],
                "category": memory["category"],
                "similarity": round(similarity, 3),
            }
            for similarity, memory in retrieved
        ],
    }


PLANNER_INSTRUCTIONS = """
你是一个严格的 Agent Planner。只输出合法 JSON object，不输出 Markdown 或额外解释。
""".strip()


def build_planner_prompt(context: dict[str, Any]) -> str:
    return f"""
你是一个 Agent Planner，不是聊天助手。根据 current_task_state、recent_conversation
和 semantic_memories 决定下一步。记忆只是参考；不要把记忆中没有明确支持的内容当作事实。

你只能选择：get_weather、get_current_time、create_final_answer、ask_user。
- get_weather: arguments 必须是 {{"city": "北京/上海/深圳/广州"}}。
- get_current_time: arguments 必须是 {{}}。
- create_final_answer: weather 和 current_time 都存在时使用，arguments 必须是 {{}}。
- ask_user: 缺少必要信息时使用，arguments 必须是 {{"question": "..."}}。

规则：先从 state、近期对话和语义记忆寻找城市；仍找不到才 ask_user。
没有天气时调用 get_weather，没有当前时间时调用 get_current_time，不重复工具调用。
只返回：{{"next_step": "get_weather", "arguments": {{"city": "北京"}}, "reason": "..."}}

上下文：
{json.dumps(context, ensure_ascii=False, indent=2)}
""".strip()


def parse_json_output(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    try:
        plan = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise AgentError(f"Planner 没有返回合法 JSON：{text}") from error
    if not isinstance(plan, dict):
        raise AgentError("Planner 返回值必须是 JSON object。")
    return plan


def call_llm_for_plan(context: dict[str, Any]) -> dict[str, Any]:
    return parse_json_output(call_openai_text(PLANNER_INSTRUCTIONS, build_planner_prompt(context)))


def validate_plan(plan: dict[str, Any], state: AgentState) -> None:
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
        if arguments.get("city") not in SUPPORTED_CITIES or state["weather"] is not None:
            raise AgentError("get_weather 需要受支持的 city，且不能重复调用。")
    elif next_step == "get_current_time":
        if arguments or state["current_time"] is not None:
            raise AgentError("get_current_time 不需要参数且不能重复调用。")
    elif next_step == "create_final_answer":
        if arguments or not state["weather"] or not state["current_time"]:
            raise AgentError("天气和时间齐全后才能生成最终回答。")
    else:
        if not isinstance(arguments.get("question"), str) or not arguments["question"].strip():
            raise AgentError("ask_user 需要非空 question。")


FINAL_ANSWER_INSTRUCTIONS = """
你是中文运动建议助手。根据提供的任务状态、近期会话和语义检索到的记忆回答。
保持简洁可靠，不编造信息，不提及内部 memory、embedding 或 planner。
""".strip()


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
        context["current_task_state"].update(
            {
                "city": state["city"],
                "weather": state["weather"],
                "current_time": state["current_time"],
                "observations": state["observations"],
            }
        )
        state["final_answer"] = call_openai_text(
            FINAL_ANSWER_INSTRUCTIONS,
            json.dumps(context, ensure_ascii=False, indent=2),
        ) or "信息已获取，但模型没有生成最终回答。"
        state["steps"].append(next_step)
        state["observations"].append("信息已足够，生成最终回答")
    else:
        state["final_answer"] = arguments["question"]
        state["steps"].append(next_step)
        state["observations"].append("缺少必要信息，需要追问用户")


def format_trace(state: AgentState, retrieved: list[tuple[float, Memory]]) -> str:
    lines = ["本轮语义检索结果："]
    if retrieved:
        lines.extend(f"- {similarity:.3f} | {memory['content']}" for similarity, memory in retrieved)
    else:
        lines.append("- 无（没有记忆或没有达到相似度阈值）")
    lines.append("执行轨迹：")
    for index, observation in enumerate(state["observations"], start=1):
        lines.append(f"{index}. Plan: {state['steps'][index - 1]}")
        lines.append(f"   Reason: {state['planner_reasons'][index - 1]}")
        lines.append(f"   Observe: {observation}")
    return "\n".join(lines)


def agent(user_input: str, history: list[Message], show_trace: bool = True) -> str:
    state = create_initial_state(user_input)
    memories = load_memories()
    retrieved = retrieve_semantic_memories(user_input, memories)

    for _ in range(MAX_STEPS):
        context = build_agent_context(state, history, retrieved)
        try:
            plan = call_llm_for_plan(context)
            validate_plan(plan, state)
            execute_step(plan, state, context)
        except AgentError as error:
            state["error"] = str(error)
            break
        if state["final_answer"] is not None:
            break

    answer = (
        f"执行过程中出现问题：{state['error']}"
        if state["error"]
        else state["final_answer"] or "已经达到最大步骤数，任务还没有完成。"
    )
    history.extend([{"role": "user", "content": user_input}, {"role": "assistant", "content": answer}])
    del history[:-MAX_HISTORY_MESSAGES]

    if not state["error"]:
        updated = upsert_memories(memories, extract_memories_from_turn(user_input, answer))
        save_memories(updated)

    return f"{format_trace(state, retrieved)}\n\n最终回答：\n{answer}" if show_trace else answer


def print_memories() -> None:
    memories = load_memories()
    if not memories:
        print("Agent：还没有语义长期记忆。")
        return
    print("Agent：长期记忆（向量仅显示维度，避免刷屏）：")
    for memory in memories:
        print(
            f"- {memory['id']} [{memory['category']}, 重要性 {memory['importance']}, "
            f"维度 {len(memory['embedding'])}] {memory['content']}"
        )


def handle_memory_command(user_input: str) -> bool:
    if user_input == ":memories":
        print_memories()
        return True
    if user_input == ":clear-memories":
        save_memories([])
        print("Agent：语义长期记忆已清空。")
        return True
    if user_input.startswith(":delete-memory "):
        memory_id = user_input.removeprefix(":delete-memory ").strip()
        memories = load_memories()
        remaining = [memory for memory in memories if memory["id"] != memory_id]
        save_memories(remaining)
        print("Agent：已删除。" if len(remaining) < len(memories) else "Agent：没有找到该记忆 ID。")
        return True
    return False


def main() -> None:
    print("Day 8 Semantic Memory Agent")
    print("先说：我正在学习 Agent 开发。再问：接着上次的内容讲。")
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
            print("Agent：已保存语义长期记忆，明天进入 RAG。")
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
