"""
Day 21: Agent Workbench

Run:
    python3 day21_agent_workbench.py

Try:
    帮我准备 Agent 第三周复习材料
    面向新手，用 Markdown，重点讲项目实践
    :run-session session_001
    :run-task task_001
    :run-task task_001
    :run-task task_001
    :run-task task_001
    :run-task task_001
    :confirm yes
    :outputs

Commands:
    :sessions                    Show workflow sessions.
    :tasks                       Show tasks.
    :outputs                     Show saved outputs.
    :show-session session_001     Show one workflow session.
    :show-task task_001           Show one task.
    :run-session session_001      Generate a plan and create a task.
    :run-task task_001            Advance one task by one step.
    :resume-task task_001         Resume a cancelled or failed task.
    :cancel-task task_001         Cancel a task.
    :confirm yes / :confirm no    Approve or reject the pending action.
    :clear-workbench              Clear workbench_state.json.
    trace                         Show latest trace summary.
    :traces                       Show recent trace summaries from agent_traces.jsonl.
    :clear-traces                 Clear local trace logs.
    exit                          Quit.

Today's key idea:
    The third-week project is not about adding more features. It is about
    connecting Workflow, Task Queue, Tool Registry, Human Confirmation,
    API/UI thinking, and Trace into one small explainable system.

This project stays offline. The tool calls are simulated, but the architecture
is the same shape you would use when real tools and a real LLM are connected.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict

from day12_observable_agent import (
    TraceEvent,
    TraceLogger,
    clear_traces,
    print_recent_traces,
)


STATE_FILE = Path(__file__).with_name("workbench_state.json")
REQUIRED_FIELDS = ["audience", "output_format", "focus"]

SessionStatus = Literal["clarifying", "ready", "planned", "task_created", "completed", "cancelled"]
TaskStatus = Literal["pending", "running", "waiting_confirmation", "completed", "failed", "cancelled"]
StepStatus = Literal["pending", "running", "waiting_confirmation", "completed", "failed", "skipped"]


class WorkbenchError(Exception):
    """A readable error for the Day 21 learning script."""


class ToolSpec(TypedDict):
    function: Callable[[dict[str, Any], "WorkbenchState"], dict[str, Any]]
    description: str
    risk: str
    side_effect: bool
    confirm_required: bool
    required_args: list[str]


class WorkbenchMessage(TypedDict):
    role: str
    content: str
    timestamp: str


class WorkbenchEvent(TypedDict):
    timestamp: str
    event_type: str
    message: str
    data: dict[str, Any]


class WorkbenchSession(TypedDict):
    id: str
    goal: str
    status: SessionStatus
    required_fields: list[str]
    collected_context: dict[str, str]
    missing_fields: list[str]
    messages: list[WorkbenchMessage]
    plan: list[str]
    task_id: str | None
    created_at: str
    updated_at: str
    events: list[WorkbenchEvent]


class WorkbenchStep(TypedDict):
    id: str
    description: str
    tool: str
    arguments: dict[str, Any]
    status: StepStatus
    result: str | None
    error: str | None
    started_at: str | None
    completed_at: str | None


class WorkbenchTask(TypedDict):
    id: str
    session_id: str
    title: str
    status: TaskStatus
    steps: list[WorkbenchStep]
    final_result: str | None
    created_at: str
    updated_at: str
    events: list[WorkbenchEvent]


class PendingAction(TypedDict):
    id: str
    tool: str
    arguments: dict[str, Any]
    risk: str
    side_effect: bool
    status: str
    summary: str
    task_id: str | None
    step_id: str | None
    created_at: str


class OutputRecord(TypedDict):
    id: str
    title: str
    content: str
    source_task_id: str | None
    source_session_id: str | None
    created_at: str


class WorkbenchState(TypedDict):
    sessions: list[WorkbenchSession]
    tasks: list[WorkbenchTask]
    pending_action: PendingAction | None
    outputs: list[OutputRecord]
    updated_at: str


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def empty_state() -> WorkbenchState:
    return {
        "sessions": [],
        "tasks": [],
        "pending_action": None,
        "outputs": [],
        "updated_at": now_text(),
    }


def ensure_state_file() -> None:
    if not STATE_FILE.exists():
        save_state(empty_state())


def load_state() -> WorkbenchState:
    ensure_state_file()
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise WorkbenchError(f"{STATE_FILE.name} 格式无效：{error}") from error
    if not isinstance(data, dict):
        raise WorkbenchError(f"{STATE_FILE.name} 必须是 JSON 对象。")

    state = empty_state()
    for key in state:
        if key in data:
            state[key] = data[key]  # type: ignore[literal-required]
    return state


def save_state(state: WorkbenchState) -> None:
    state["updated_at"] = now_text()
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clear_workbench() -> None:
    save_state(empty_state())
    print(f"已清空 {STATE_FILE.name}。")


def next_id(items: list[dict[str, Any]], prefix: str) -> str:
    max_number = 0
    pattern = re.compile(rf"{re.escape(prefix)}_(\d{{3}})")
    for item in items:
        item_id = item.get("id")
        if not isinstance(item_id, str):
            continue
        match = pattern.fullmatch(item_id)
        if match:
            max_number = max(max_number, int(match.group(1)))
    return f"{prefix}_{max_number + 1:03d}"


def extract_id(text: str, prefix: str) -> str | None:
    match = re.search(rf"{re.escape(prefix)}_\d{{3}}", text)
    return match.group(0) if match else None


def append_message(session: WorkbenchSession, role: str, content: str) -> None:
    session["messages"].append({"role": role, "content": content, "timestamp": now_text()})
    session["updated_at"] = now_text()


def append_event(target: WorkbenchSession | WorkbenchTask, event_type: str, message: str, data: dict[str, Any] | None = None) -> None:
    target["events"].append(
        {
            "timestamp": now_text(),
            "event_type": event_type,
            "message": message,
            "data": data or {},
        }
    )
    target["updated_at"] = now_text()


def extract_initial_goal(user_input: str) -> str:
    text = user_input.strip(" 。.，,")
    patterns = [
        r"^(?:帮我|请帮我|我要|我想)\s*(.+)$",
        r"^新建工作台任务[:：]\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip(" 。.，,")
    return text


def extract_topic(goal: str) -> str:
    if "复习" in goal:
        return "Agent 第三周复习材料"
    if "报告" in goal:
        before_report = goal.split("报告", 1)[0].strip("一份 个的")
        return f"{before_report}报告" if before_report else "学习报告"
    if "计划" in goal:
        return "学习计划"
    return goal[:40]


def extract_context_from_message(message: str) -> dict[str, str]:
    """
    Extract workflow slots from user input.

    LLM replacement point:
    Later this can call a model to extract structured fields. The rest of the
    workbench should still receive a plain dict and update explicit state.
    """
    text = message.strip()
    context: dict[str, str] = {}

    audience_match = re.search(r"面向([^，,。]+)", text) or re.search(r"给([^，,。]+)看", text)
    if audience_match:
        context["audience"] = audience_match.group(1).strip()
    elif "新手" in text:
        context["audience"] = "新手"
    elif "开发者" in text:
        context["audience"] = "开发者"

    if re.search(r"markdown|md", text, flags=re.IGNORECASE):
        context["output_format"] = "Markdown"
    elif "PPT" in text.upper() or "幻灯片" in text:
        context["output_format"] = "PPT"
    elif "网页" in text or "HTML" in text.upper():
        context["output_format"] = "HTML"
    elif "大纲" in text:
        context["output_format"] = "大纲"

    focus_match = (
        re.search(r"重点(?:讲|放在|偏)?([^，,。]+)", text)
        or re.search(r"侧重([^，,。]+)", text)
        or re.search(r"聚焦([^，,。]+)", text)
    )
    if focus_match:
        context["focus"] = focus_match.group(1).strip()
    elif "项目" in text:
        context["focus"] = "项目实践"
    elif "原理" in text and "代码" in text:
        context["focus"] = "原理和代码结合"

    length_match = re.search(r"(\d+)\s*(?:字|页|分钟)", text)
    if length_match:
        context["length"] = length_match.group(0)
    return context


def find_missing_fields(session: WorkbenchSession) -> list[str]:
    return [field for field in session["required_fields"] if not session["collected_context"].get(field)]


def create_clarifying_question(session: WorkbenchSession) -> str:
    labels = {
        "audience": "面向谁",
        "output_format": "输出格式",
        "focus": "重点方向",
    }
    missing = "、".join(labels.get(field, field) for field in session["missing_fields"])
    return f"我已经记录目标：{session['goal']}。\n还需要确认：{missing}。"


def create_session(user_input: str, state: WorkbenchState, logger: TraceLogger) -> WorkbenchSession:
    goal = extract_initial_goal(user_input)
    context = extract_context_from_message(user_input)
    context.setdefault("topic", extract_topic(goal))
    session: WorkbenchSession = {
        "id": next_id(state["sessions"], "session"),
        "goal": goal,
        "status": "clarifying",
        "required_fields": list(REQUIRED_FIELDS),
        "collected_context": context,
        "missing_fields": [],
        "messages": [],
        "plan": [],
        "task_id": None,
        "created_at": now_text(),
        "updated_at": now_text(),
        "events": [],
    }
    append_message(session, "user", user_input)
    update_session_status(session)
    answer = create_session_answer(session)
    append_message(session, "agent", answer)
    append_event(session, "session_created", "创建 workflow session", summarize_session(session))
    state["sessions"].append(session)
    save_state(state)
    logger.log("session_created", "工作台创建 workflow session", {"session": summarize_session(session)})
    return session


def update_session_status(session: WorkbenchSession) -> None:
    session["missing_fields"] = find_missing_fields(session)
    if session["missing_fields"]:
        session["status"] = "clarifying"
    elif session["task_id"]:
        session["status"] = "task_created"
    elif session["plan"]:
        session["status"] = "planned"
    else:
        session["status"] = "ready"
    session["updated_at"] = now_text()


def create_session_answer(session: WorkbenchSession) -> str:
    if session["status"] == "clarifying":
        return create_clarifying_question(session)
    if session["status"] == "ready":
        return f"{session['id']} 的信息已齐全，可以输入 :run-session {session['id']} 生成计划并创建任务。"
    if session["status"] == "planned":
        return f"{session['id']} 已有计划，可以输入 :run-session {session['id']} 创建任务。"
    return f"{session['id']} 已创建任务：{session['task_id']}。"


def find_session(state: WorkbenchState, session_id: str) -> WorkbenchSession:
    for session in state["sessions"]:
        if session["id"] == session_id:
            return session
    raise WorkbenchError(f"没有找到 session：{session_id}")


def find_task(state: WorkbenchState, task_id: str) -> WorkbenchTask:
    for task in state["tasks"]:
        if task["id"] == task_id:
            return task
    raise WorkbenchError(f"没有找到 task：{task_id}")


def update_session_with_message(session: WorkbenchSession, message: str, logger: TraceLogger) -> str:
    append_message(session, "user", message)
    extracted = extract_context_from_message(message)
    session["collected_context"].update(extracted)
    update_session_status(session)
    answer = create_session_answer(session)
    append_message(session, "agent", answer)
    append_event(session, "context_updated", "用户补充信息后更新 session", {"extracted": extracted})
    logger.log("context_updated", "工作台更新 workflow context", {"session": summarize_session(session), "extracted": extracted})
    return answer


def create_plan_from_session(session: WorkbenchSession) -> list[str]:
    """
    Create a workbench plan.

    LLM replacement point:
    Later this can ask a real model to generate steps from the collected
    context. Keep the output structured so it can become a task queue.
    """
    context = session["collected_context"]
    topic = context.get("topic", session["goal"])
    audience = context.get("audience", "学习者")
    output_format = context.get("output_format", "Markdown")
    focus = context.get("focus", "项目实践")
    return [
        f"读取课程大纲，定位第三周主题和能力边界",
        f"为{audience}提取重点：{focus}",
        f"整理成{output_format}结构的复习材料框架",
        "汇总任务进度、关键概念和练习建议",
        f"保存最终输出：{topic}",
    ]


def create_task_from_session(session: WorkbenchSession, state: WorkbenchState, logger: TraceLogger) -> WorkbenchTask:
    if session["missing_fields"]:
        raise WorkbenchError(f"{session['id']} 还缺少信息：{', '.join(session['missing_fields'])}")
    if session["task_id"]:
        return find_task(state, session["task_id"])

    if not session["plan"]:
        session["plan"] = create_plan_from_session(session)
        append_event(session, "plan_created", "根据 workflow context 生成计划", {"plan": session["plan"]})

    context = session["collected_context"]
    steps: list[WorkbenchStep] = [
        create_step("step_001", session["plan"][0], "read_course_outline", {"query": context.get("topic", session["goal"])}),
        create_step("step_002", session["plan"][1], "summarize_week3_capabilities", {"focus": context.get("focus", "")}),
        create_step("step_003", session["plan"][2], "create_review_outline", {"format": context.get("output_format", "Markdown")}),
        create_step("step_004", session["plan"][3], "summarize_progress", {"session_id": session["id"]}),
        create_step(
            "step_005",
            session["plan"][4],
            "write_final_report",
            {"title": context.get("topic", session["goal"]), "session_id": session["id"]},
        ),
    ]
    task: WorkbenchTask = {
        "id": next_id(state["tasks"], "task"),
        "session_id": session["id"],
        "title": context.get("topic", session["goal"]),
        "status": "pending",
        "steps": steps,
        "final_result": None,
        "created_at": now_text(),
        "updated_at": now_text(),
        "events": [],
    }
    session["task_id"] = task["id"]
    update_session_status(session)
    append_event(session, "task_created", "从 session 创建 task", {"task_id": task["id"]})
    append_event(task, "task_created", "创建工作台任务", {"session_id": session["id"], "step_count": len(steps)})
    state["tasks"].append(task)
    save_state(state)
    logger.log("task_created", "工作台从 workflow 创建任务", {"session": summarize_session(session), "task": summarize_task(task)})
    return task


def create_step(step_id: str, description: str, tool: str, arguments: dict[str, Any]) -> WorkbenchStep:
    return {
        "id": step_id,
        "description": description,
        "tool": tool,
        "arguments": arguments,
        "status": "pending",
        "result": None,
        "error": None,
        "started_at": None,
        "completed_at": None,
    }


def get_next_step(task: WorkbenchTask) -> WorkbenchStep | None:
    for step in task["steps"]:
        if step["status"] in {"pending", "failed"}:
            return step
    return None


def register_tools() -> dict[str, ToolSpec]:
    return {
        "read_course_outline": {
            "function": tool_read_course_outline,
            "description": "读取课程大纲摘要",
            "risk": "low",
            "side_effect": False,
            "confirm_required": False,
            "required_args": ["query"],
        },
        "summarize_week3_capabilities": {
            "function": tool_summarize_week3_capabilities,
            "description": "总结第三周能力",
            "risk": "low",
            "side_effect": False,
            "confirm_required": False,
            "required_args": ["focus"],
        },
        "create_review_outline": {
            "function": tool_create_review_outline,
            "description": "生成复习材料结构",
            "risk": "low",
            "side_effect": False,
            "confirm_required": False,
            "required_args": ["format"],
        },
        "summarize_progress": {
            "function": tool_summarize_progress,
            "description": "汇总当前工作台进度",
            "risk": "low",
            "side_effect": False,
            "confirm_required": False,
            "required_args": ["session_id"],
        },
        "write_final_report": {
            "function": tool_write_final_report,
            "description": "把最终材料写入 workbench outputs",
            "risk": "high",
            "side_effect": True,
            "confirm_required": True,
            "required_args": ["title", "session_id"],
        },
    }


def validate_tool_call(tool_name: str, arguments: dict[str, Any]) -> ToolSpec:
    registry = register_tools()
    if tool_name not in registry:
        raise WorkbenchError(f"未知工具：{tool_name}")
    spec = registry[tool_name]
    for arg_name in spec["required_args"]:
        if arg_name not in arguments:
            raise WorkbenchError(f"{tool_name} 缺少参数：{arg_name}")
    return spec


def execute_tool(tool_name: str, arguments: dict[str, Any], state: WorkbenchState) -> dict[str, Any]:
    spec = validate_tool_call(tool_name, arguments)
    return spec["function"](arguments, state)


def tool_read_course_outline(arguments: dict[str, Any], state: WorkbenchState) -> dict[str, Any]:
    del state
    query = arguments["query"]
    return {
        "summary": (
            f"围绕“{query}”读取课程主线：第 15 天多工具，第 16 天人类确认，"
            "第 17 天任务队列，第 18 天 API，第 19 天前端，第 20 天多轮工作流。"
        )
    }


def tool_summarize_week3_capabilities(arguments: dict[str, Any], state: WorkbenchState) -> dict[str, Any]:
    del state
    focus = arguments.get("focus") or "应用整合"
    return {
        "summary": (
            f"第三周重点是{focus}：让 Agent 具备工具组合、风险确认、长任务状态、"
            "服务化接口、前端操作和多轮协作。"
        )
    }


def tool_create_review_outline(arguments: dict[str, Any], state: WorkbenchState) -> dict[str, Any]:
    del state
    output_format = arguments.get("format", "Markdown")
    return {
        "outline": [
            f"输出格式：{output_format}",
            "一、第三周解决什么问题",
            "二、Workflow、Task Queue、Tool Registry 的边界",
            "三、人类确认和 Trace 如何提升可靠性",
            "四、把 Agent 做成可使用应用的练习建议",
        ]
    }


def tool_summarize_progress(arguments: dict[str, Any], state: WorkbenchState) -> dict[str, Any]:
    session_id = arguments["session_id"]
    session = find_session(state, session_id)
    task = find_task(state, session["task_id"]) if session.get("task_id") else None
    progress = summarize_task(task)["progress"] if task else "0/0"
    return {
        "summary": f"{session_id} 当前状态：{session['status']}；任务进度：{progress}；输出数量：{len(state['outputs'])}。"
    }


def tool_write_final_report(arguments: dict[str, Any], state: WorkbenchState) -> dict[str, Any]:
    session = find_session(state, arguments["session_id"])
    task = find_task(state, session["task_id"]) if session.get("task_id") else None
    if not task:
        raise WorkbenchError("没有关联任务，无法写入最终报告。")

    completed_results = [step["result"] for step in task["steps"] if step["result"]]
    content = (
        f"# {arguments['title']}\n\n"
        "## 工作台输出\n"
        + "\n".join(f"- {result}" for result in completed_results)
        + "\n\n## 结论\n第三周综合项目已经把 workflow、task、tool、confirmation 和 trace 串成一个最小系统。"
    )
    output: OutputRecord = {
        "id": next_id(state["outputs"], "output"),
        "title": arguments["title"],
        "content": content,
        "source_task_id": task["id"],
        "source_session_id": session["id"],
        "created_at": now_text(),
    }
    state["outputs"].append(output)
    return {"output_id": output["id"], "title": output["title"], "preview": content[:120]}


def create_pending_action(
    tool_name: str,
    arguments: dict[str, Any],
    summary: str,
    state: WorkbenchState,
    task_id: str | None,
    step_id: str | None,
) -> PendingAction:
    spec = validate_tool_call(tool_name, arguments)
    action: PendingAction = {
        "id": f"action_{datetime.now().strftime('%H%M%S')}",
        "tool": tool_name,
        "arguments": arguments,
        "risk": spec["risk"],
        "side_effect": spec["side_effect"],
        "status": "pending",
        "summary": summary,
        "task_id": task_id,
        "step_id": step_id,
        "created_at": now_text(),
    }
    state["pending_action"] = action
    return action


def advance_task(task_id: str, state: WorkbenchState, logger: TraceLogger, resume: bool = False) -> WorkbenchTask:
    task = find_task(state, task_id)
    if task["status"] == "completed":
        logger.log("task_noop", "任务已经完成", {"task_id": task_id})
        return task
    if task["status"] == "cancelled" and not resume:
        raise WorkbenchError(f"{task_id} 已取消。如需继续，请使用 :resume-task {task_id}。")
    if state["pending_action"]:
        raise WorkbenchError(f"当前有待确认动作：{state['pending_action']['id']}。请先输入 :confirm yes 或 :confirm no。")

    if resume and task["status"] in {"cancelled", "failed"}:
        task["status"] = "pending"
        append_event(task, "task_resumed", "恢复任务", {})

    step = get_next_step(task)
    if not step:
        task["status"] = "completed"
        task["final_result"] = "任务已完成。"
        append_event(task, "task_completed", "任务所有步骤已完成", {})
        save_state(state)
        logger.log("task_completed", "工作台任务已完成", {"task": summarize_task(task)})
        return task

    step["status"] = "running"
    step["started_at"] = now_text()
    step["error"] = None
    task["status"] = "running"
    append_event(task, "step_started", "开始执行步骤", {"step_id": step["id"], "tool": step["tool"]})
    logger.log("step_started", "工作台推进任务步骤", {"task_id": task_id, "step_id": step["id"], "tool": step["tool"]})

    try:
        spec = validate_tool_call(step["tool"], step["arguments"])
        if spec["risk"] == "blocked":
            raise WorkbenchError(f"工具 {step['tool']} 已被禁用。")
        if spec["confirm_required"]:
            action = create_pending_action(
                step["tool"],
                step["arguments"],
                f"执行 {step['tool']}：{step['description']}",
                state,
                task["id"],
                step["id"],
            )
            step["status"] = "waiting_confirmation"
            task["status"] = "waiting_confirmation"
            append_event(task, "pending_action_created", "创建待确认动作", {"action": action})
            save_state(state)
            logger.log("pending_action_created", "工作台创建待确认动作", {"action": action})
            return task

        result = execute_tool(step["tool"], step["arguments"], state)
        step["status"] = "completed"
        step["result"] = format_tool_result(result)
        step["completed_at"] = now_text()
        append_event(task, "step_completed", "步骤执行完成", {"step_id": step["id"], "result": result})
    except Exception as error:
        step["status"] = "failed"
        step["error"] = str(error)
        step["completed_at"] = now_text()
        task["status"] = "failed"
        append_event(task, "step_failed", "步骤执行失败", {"step_id": step["id"], "error": str(error)})
        save_state(state)
        logger.log("step_failed", "工作台任务步骤失败", {"task_id": task_id, "step_id": step["id"], "error": str(error)})
        return task

    if get_next_step(task):
        task["status"] = "pending"
    else:
        task["status"] = "completed"
        task["final_result"] = "所有步骤已完成。"
        mark_session_completed_if_needed(state, task)
        append_event(task, "task_completed", "任务所有步骤已完成", {})
    save_state(state)
    logger.log("step_completed", "工作台任务步骤完成", {"task": summarize_task(task), "step_id": step["id"]})
    return task


def handle_confirmation(user_input: str, state: WorkbenchState, logger: TraceLogger) -> str:
    action = state["pending_action"]
    if not action:
        return "当前没有待确认动作。"
    normalized = user_input.strip().casefold()
    if normalized not in {"yes", "y", "确认", "同意", "no", "n", "取消", "拒绝"}:
        return "请明确输入 :confirm yes 或 :confirm no。"

    task = find_task(state, action["task_id"]) if action["task_id"] else None
    step = find_step(task, action["step_id"]) if task and action["step_id"] else None

    if normalized in {"no", "n", "取消", "拒绝"}:
        action["status"] = "rejected"
        if step:
            step["status"] = "pending"
        if task:
            task["status"] = "pending"
            append_event(task, "confirmation_rejected", "用户拒绝待确认动作", {"action_id": action["id"]})
        state["pending_action"] = None
        save_state(state)
        logger.log("confirmation_rejected", "用户拒绝工作台待确认动作", {"action_id": action["id"]})
        return f"已取消动作：{action['summary']}。"

    result = execute_tool(action["tool"], action["arguments"], state)
    action["status"] = "executed"
    if step:
        step["status"] = "completed"
        step["result"] = format_tool_result(result)
        step["completed_at"] = now_text()
    if task:
        task["status"] = "completed" if get_next_step(task) is None else "pending"
        task["final_result"] = "已确认并完成最终输出。" if task["status"] == "completed" else task["final_result"]
        append_event(task, "confirmation_approved", "用户确认并执行待确认动作", {"action_id": action["id"], "result": result})
        if task["status"] == "completed":
            mark_session_completed_if_needed(state, task)
    state["pending_action"] = None
    save_state(state)
    logger.log("confirmation_approved", "用户确认工作台待确认动作", {"action_id": action["id"], "result": result})
    return f"已确认并执行：{action['summary']}。\n结果：{format_tool_result(result)}"


def find_step(task: WorkbenchTask, step_id: str | None) -> WorkbenchStep | None:
    for step in task["steps"]:
        if step["id"] == step_id:
            return step
    return None


def mark_session_completed_if_needed(state: WorkbenchState, task: WorkbenchTask) -> None:
    session = find_session(state, task["session_id"])
    if task["status"] == "completed":
        session["status"] = "completed"
        append_event(session, "session_completed", "关联任务完成，session 标记完成", {"task_id": task["id"]})


def cancel_task(task_id: str, state: WorkbenchState, logger: TraceLogger) -> WorkbenchTask:
    task = find_task(state, task_id)
    if task["status"] == "completed":
        raise WorkbenchError("已完成任务不能取消。")
    task["status"] = "cancelled"
    append_event(task, "task_cancelled", "用户取消任务", {})
    save_state(state)
    logger.log("task_cancelled", "工作台取消任务", {"task": summarize_task(task)})
    return task


def format_tool_result(result: dict[str, Any]) -> str:
    if "summary" in result:
        return str(result["summary"])
    if "outline" in result:
        return "；".join(str(item) for item in result["outline"])
    if "output_id" in result:
        return f"已保存输出 {result['output_id']}：{result.get('title')}"
    return json.dumps(result, ensure_ascii=False)


def summarize_session(session: WorkbenchSession) -> dict[str, Any]:
    return {
        "id": session["id"],
        "goal": session["goal"],
        "status": session["status"],
        "missing_fields": session["missing_fields"],
        "task_id": session["task_id"],
        "has_plan": bool(session["plan"]),
        "updated_at": session["updated_at"],
    }


def summarize_task(task: WorkbenchTask | None) -> dict[str, Any]:
    if not task:
        return {"progress": "0/0"}
    completed = sum(1 for step in task["steps"] if step["status"] == "completed")
    failed = sum(1 for step in task["steps"] if step["status"] == "failed")
    return {
        "id": task["id"],
        "session_id": task["session_id"],
        "title": task["title"],
        "status": task["status"],
        "progress": f"{completed}/{len(task['steps'])}",
        "failed_steps": failed,
        "updated_at": task["updated_at"],
    }


def format_sessions(state: WorkbenchState) -> str:
    if not state["sessions"]:
        return "暂无 session。你可以输入：帮我准备 Agent 第三周复习材料"
    lines = ["Workflow Sessions："]
    for session in state["sessions"]:
        missing = "、".join(session["missing_fields"]) if session["missing_fields"] else "无"
        lines.append(f"- {session['id']} [{session['status']}] {session['goal']} missing={missing} task={session['task_id']}")
    return "\n".join(lines)


def format_tasks(state: WorkbenchState) -> str:
    if not state["tasks"]:
        return "暂无 task。先创建 session，再输入 :run-session session_001。"
    lines = ["Tasks："]
    for task in state["tasks"]:
        summary = summarize_task(task)
        lines.append(f"- {task['id']} [{task['status']}] {task['title']} progress={summary['progress']} session={task['session_id']}")
    return "\n".join(lines)


def format_outputs(state: WorkbenchState) -> str:
    if not state["outputs"]:
        return "暂无输出。最终写入动作确认后会保存 output。"
    lines = ["Outputs："]
    for output in state["outputs"]:
        preview = output["content"].replace("\n", " ")[:90]
        lines.append(f"- {output['id']} {output['title']} ({output['created_at']}) {preview}")
    return "\n".join(lines)


def format_pending_action(action: PendingAction | None) -> str:
    if not action:
        return "当前没有待确认动作。"
    return (
        "需要确认一个高风险动作：\n"
        f"- action_id：{action['id']}\n"
        f"- tool：{action['tool']}\n"
        f"- risk：{action['risk']}\n"
        f"- side_effect：{action['side_effect']}\n"
        f"- summary：{action['summary']}\n"
        f"- arguments：{json.dumps(action['arguments'], ensure_ascii=False)}\n"
        "请输入 :confirm yes 执行，或 :confirm no 取消。"
    )


def format_session_detail(session: WorkbenchSession) -> str:
    missing = "、".join(session["missing_fields"]) if session["missing_fields"] else "无"
    lines = [
        f"Session：{session['id']}",
        f"目标：{session['goal']}",
        f"状态：{session['status']}",
        f"缺失字段：{missing}",
        f"关联任务：{session['task_id']}",
        "已收集信息：",
    ]
    lines.extend(f"- {key}: {value}" for key, value in session["collected_context"].items())
    lines.append("计划：")
    if session["plan"]:
        lines.extend(f"{index}. {item}" for index, item in enumerate(session["plan"], start=1))
    else:
        lines.append("暂无")
    lines.append("最近消息：")
    lines.extend(f"- {message['role']}: {message['content']}" for message in session["messages"][-6:])
    return "\n".join(lines)


def format_task_detail(task: WorkbenchTask) -> str:
    summary = summarize_task(task)
    lines = [
        f"Task：{task['id']}",
        f"标题：{task['title']}",
        f"状态：{task['status']}",
        f"进度：{summary['progress']}",
        f"关联 session：{task['session_id']}",
        "步骤：",
    ]
    for step in task["steps"]:
        suffix = ""
        if step["result"]:
            suffix = f" -> {step['result']}"
        if step["error"]:
            suffix = f" error={step['error']}"
        lines.append(f"- {step['id']} [{step['status']}] {step['description']} tool={step['tool']}{suffix}")
    if task["final_result"]:
        lines.append(f"最终结果：{task['final_result']}")
    return "\n".join(lines)


def latest_open_session(state: WorkbenchState) -> WorkbenchSession | None:
    for session in reversed(state["sessions"]):
        if session["status"] == "clarifying":
            return session
    return None


def looks_like_new_goal(text: str) -> bool:
    return any(word in text for word in ["帮我", "请帮我", "准备", "生成", "整理", "写"])


def summarize_event_data(event: TraceEvent) -> str:
    data = event["data"]
    for key in ["session", "task"]:
        value = data.get(key)
        if isinstance(value, dict):
            return f"{value.get('id')} status={value.get('status')}"
    if "action" in data:
        action = data["action"]
        if isinstance(action, dict):
            return f"{action.get('id')} {action.get('tool')}"
    if "error" in data:
        return str(data["error"])[:80]
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


def handle_workbench_input(user_input: str, state: WorkbenchState) -> tuple[str, list[TraceEvent]]:
    logger = TraceLogger()
    logger.log("user_input", "收到用户输入", {"input": user_input})
    try:
        if user_input.startswith(":show-session"):
            session_id = extract_id(user_input, "session")
            if not session_id:
                raise WorkbenchError("请指定 session_id，例如：:show-session session_001")
            answer = format_session_detail(find_session(state, session_id))
        elif user_input.startswith(":show-task"):
            task_id = extract_id(user_input, "task")
            if not task_id:
                raise WorkbenchError("请指定 task_id，例如：:show-task task_001")
            answer = format_task_detail(find_task(state, task_id))
        elif user_input.startswith(":run-session"):
            session_id = extract_id(user_input, "session")
            if not session_id:
                raise WorkbenchError("请指定 session_id，例如：:run-session session_001")
            session = find_session(state, session_id)
            task = create_task_from_session(session, state, logger)
            answer = f"已从 {session_id} 创建/获取任务：\n{format_task_detail(task)}"
        elif user_input.startswith(":run-task"):
            task_id = extract_id(user_input, "task")
            if not task_id:
                raise WorkbenchError("请指定 task_id，例如：:run-task task_001")
            task = advance_task(task_id, state, logger)
            answer = format_task_detail(task)
            if state["pending_action"]:
                answer += "\n\n" + format_pending_action(state["pending_action"])
        elif user_input.startswith(":resume-task"):
            task_id = extract_id(user_input, "task")
            if not task_id:
                raise WorkbenchError("请指定 task_id，例如：:resume-task task_001")
            answer = format_task_detail(advance_task(task_id, state, logger, resume=True))
        elif user_input.startswith(":cancel-task"):
            task_id = extract_id(user_input, "task")
            if not task_id:
                raise WorkbenchError("请指定 task_id，例如：:cancel-task task_001")
            answer = format_task_detail(cancel_task(task_id, state, logger))
        elif user_input.startswith(":confirm"):
            decision = user_input.replace(":confirm", "", 1).strip()
            answer = handle_confirmation(decision, state, logger)
        else:
            session = latest_open_session(state)
            if session and not looks_like_new_goal(user_input):
                answer_text = update_session_with_message(session, user_input, logger)
                save_state(state)
                answer = f"{answer_text}\n\n{format_session_detail(session)}"
            else:
                session = create_session(user_input, state, logger)
                answer = format_session_detail(session)

        logger.log("final_answer", "返回工作台结果", {"answer": answer})
        return answer, logger.get_events()
    except Exception as error:
        logger.log("error", "工作台操作失败", {"error_type": type(error).__name__, "error": str(error)})
        answer = f"工作台操作失败：{error}"
        logger.log("final_answer", "返回错误结果", {"answer": answer})
        return answer, logger.get_events()


def main() -> None:
    print("Day 21 Agent Workbench")
    print("命令：:sessions、:tasks、:outputs、:show-session session_001、:show-task task_001、:run-session session_001、:run-task task_001、:confirm yes/no、:clear-workbench、trace、:traces、:clear-traces、exit")
    latest_events: list[TraceEvent] = []

    while True:
        try:
            user_input = input("\n你：").strip()
        except EOFError:
            print("\n已退出。")
            break

        state = load_state()
        if user_input.lower() in {"exit", "quit", "退出"}:
            print("Agent：已退出。第 22 天可以进入第四周，学习 Agent 测试体系。")
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
        if user_input == ":clear-workbench":
            clear_workbench()
            continue
        if user_input == ":sessions":
            print(format_sessions(state))
            continue
        if user_input == ":tasks":
            print(format_tasks(state))
            continue
        if user_input == ":outputs":
            print(format_outputs(state))
            continue
        if not user_input:
            print("Agent：请输入目标，例如：帮我准备 Agent 第三周复习材料")
            continue

        answer, latest_events = handle_workbench_input(user_input, state)
        print("\nAgent：")
        print(answer)
        print_trace_summary(latest_events)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出。")
        sys.exit(0)
