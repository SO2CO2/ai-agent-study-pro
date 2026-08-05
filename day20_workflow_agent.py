"""
Day 20: Multi-turn Workflow Agent

Run:
    python3 day20_workflow_agent.py

Try:
    帮我准备一份 Agent 学习报告
    面向新手，用 Markdown，重点讲原理和代码结合
    :run session_001
    :revise session_001 增加 RAG 和安全部分
    :complete session_001

Commands:
    :sessions                    Show all workflow sessions.
    :show session_001             Show one session in detail.
    :run session_001              Generate a plan or draft for the session.
    :revise session_001 修改要求   Revise the draft with user feedback.
    :complete session_001         Mark a reviewed session completed.
    :clear-sessions               Clear workflow_sessions.json.
    trace                         Show latest trace summary.
    :traces                       Show recent trace summaries from agent_traces.jsonl.
    :clear-traces                 Clear local trace logs.
    exit                          Quit.

Today's key idea:
    A multi-turn workflow Agent is not just a longer chat. After every turn it
    updates state and decides whether to clarify, plan, execute, wait for
    review, revise, or finish.

This script stays offline and rule-based so students can focus on workflow
state. Comments mark the places that can later be replaced with LLM extraction,
planning, drafting, and revision.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

from day12_observable_agent import (
    TraceEvent,
    TraceLogger,
    clear_traces,
    print_recent_traces,
)


SESSION_FILE = Path(__file__).with_name("workflow_sessions.json")
REQUIRED_FIELDS = ["audience", "output_format", "focus"]

WorkflowStatus = Literal[
    "clarifying",
    "ready_to_plan",
    "planning",
    "ready_to_draft",
    "drafting",
    "reviewing",
    "revising",
    "completed",
    "cancelled",
]


class WorkflowAgentError(Exception):
    """A readable error for the Day 20 learning script."""


class WorkflowMessage(TypedDict):
    role: str
    content: str
    timestamp: str


class WorkflowEvent(TypedDict):
    timestamp: str
    event_type: str
    message: str
    data: dict[str, Any]


class WorkflowSession(TypedDict):
    id: str
    goal: str
    status: WorkflowStatus
    current_stage: str
    required_fields: list[str]
    collected_context: dict[str, str]
    missing_fields: list[str]
    messages: list[WorkflowMessage]
    plan: list[str]
    draft: str | None
    final_result: str | None
    created_at: str
    updated_at: str
    events: list[WorkflowEvent]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_session_file() -> None:
    if not SESSION_FILE.exists():
        SESSION_FILE.write_text("[]\n", encoding="utf-8")


def load_sessions() -> list[WorkflowSession]:
    ensure_session_file()
    try:
        data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise WorkflowAgentError(f"{SESSION_FILE.name} 格式无效：{error}") from error
    if not isinstance(data, list):
        raise WorkflowAgentError(f"{SESSION_FILE.name} 必须是 JSON 数组。")

    sessions: list[WorkflowSession] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            sessions.append(item)  # type: ignore[arg-type]
    return sessions


def save_sessions(sessions: list[WorkflowSession]) -> None:
    SESSION_FILE.write_text(json.dumps(sessions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clear_sessions() -> None:
    save_sessions([])
    print(f"已清空 {SESSION_FILE.name}。")


def next_session_id(sessions: list[WorkflowSession]) -> str:
    max_number = 0
    for session in sessions:
        match = re.fullmatch(r"session_(\d{3})", session["id"])
        if match:
            max_number = max(max_number, int(match.group(1)))
    return f"session_{max_number + 1:03d}"


def extract_session_id(text: str) -> str | None:
    match = re.search(r"session_\d{3}", text)
    return match.group(0) if match else None


def extract_initial_goal(user_input: str) -> str:
    text = user_input.strip(" 。.，,")
    patterns = [
        r"^(?:帮我|请帮我|我要|我想)\s*(.+)$",
        r"^新建工作流[:：]\s*(.+)$",
        r"^创建工作流[:：]\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip(" 。.，,")
    return text


def extract_topic(goal: str) -> str:
    if "报告" in goal:
        before_report = goal.split("报告", 1)[0].strip("一份 个的")
        return f"{before_report}报告" if before_report else "学习报告"
    if "计划" in goal:
        return "学习计划"
    if "总结" in goal:
        return "总结材料"
    return goal[:40]


def extract_context_from_message(message: str) -> dict[str, str]:
    """
    Extract workflow slots from a user message.

    LLM replacement point:
    A production Agent can ask a model to extract structured fields such as
    audience, output_format, focus, length, tone, deadline, and constraints.
    Here we use simple rules so students can inspect every state transition.
    """
    text = message.strip()
    context: dict[str, str] = {}

    audience_patterns = [
        (r"面向([^，,。]+)", "audience"),
        (r"给([^，,。]+)看", "audience"),
        (r"适合([^，,。]+)", "audience"),
    ]
    for pattern, field in audience_patterns:
        match = re.search(pattern, text)
        if match:
            context[field] = match.group(1).strip()
            break
    if "新手" in text and "audience" not in context:
        context["audience"] = "新手"
    elif "开发者" in text and "audience" not in context:
        context["audience"] = "开发者"

    if re.search(r"markdown|md", text, flags=re.IGNORECASE):
        context["output_format"] = "Markdown"
    elif "PPT" in text.upper() or "幻灯片" in text:
        context["output_format"] = "PPT"
    elif "网页" in text or "HTML" in text.upper():
        context["output_format"] = "HTML"
    elif "大纲" in text:
        context["output_format"] = "大纲"

    focus_patterns = [
        r"重点(?:讲|放在|偏)?([^，,。]+)",
        r"侧重([^，,。]+)",
        r"聚焦([^，,。]+)",
    ]
    for pattern in focus_patterns:
        match = re.search(pattern, text)
        if match:
            context["focus"] = match.group(1).strip()
            break
    if "原理" in text and "代码" in text and "focus" not in context:
        context["focus"] = "原理和代码结合"
    elif "项目" in text and "focus" not in context:
        context["focus"] = "项目实践"

    length_match = re.search(r"(\d+)\s*(?:字|页|分钟)", text)
    if length_match:
        context["length"] = length_match.group(0)

    return context


def append_message(session: WorkflowSession, role: str, content: str) -> None:
    session["messages"].append({"role": role, "content": content, "timestamp": now_text()})
    session["updated_at"] = now_text()


def append_session_event(
    session: WorkflowSession,
    event_type: str,
    message: str,
    data: dict[str, Any] | None = None,
) -> None:
    session["events"].append(
        {
            "timestamp": now_text(),
            "event_type": event_type,
            "message": message,
            "data": data or {},
        }
    )
    session["updated_at"] = now_text()


def find_missing_fields(session: WorkflowSession) -> list[str]:
    return [field for field in session["required_fields"] if not session["collected_context"].get(field)]


def create_clarifying_question(session: WorkflowSession) -> str:
    labels = {
        "audience": "这份内容面向谁",
        "output_format": "希望输出成什么格式",
        "focus": "重点偏原理、代码还是项目",
    }
    missing_text = "、".join(labels.get(field, field) for field in session["missing_fields"])
    return f"我已经记录了目标：{session['goal']}。\n还需要确认：{missing_text}。"


def maybe_advance_workflow(session: WorkflowSession) -> str:
    """
    Move the workflow to the next stage if conditions are satisfied.

    LLM replacement point:
    Later, an LLM can decide whether the workflow should clarify, plan, execute,
    wait for review, or revise. The important engineering rule stays the same:
    decision result must be written into explicit state.
    """
    session["missing_fields"] = find_missing_fields(session)
    if session["missing_fields"]:
        session["status"] = "clarifying"
        session["current_stage"] = "collect_requirements"
        return create_clarifying_question(session)

    if not session["plan"]:
        session["status"] = "ready_to_plan"
        session["current_stage"] = "ready_to_plan"
        return f"{session['id']} 的关键信息已经齐全，可以输入 :run {session['id']} 生成计划。"

    if not session["draft"]:
        session["status"] = "ready_to_draft"
        session["current_stage"] = "ready_to_draft"
        return f"{session['id']} 已有计划，可以输入 :run {session['id']} 生成草稿。"

    session["status"] = "reviewing"
    session["current_stage"] = "waiting_user_review"
    return f"{session['id']} 已生成草稿。你可以输入 :revise {session['id']} 修改要求，或 :complete {session['id']} 完成。"


def create_session(user_input: str, logger: TraceLogger) -> WorkflowSession:
    sessions = load_sessions()
    goal = extract_initial_goal(user_input)
    context = extract_context_from_message(user_input)
    context.setdefault("topic", extract_topic(goal))

    session: WorkflowSession = {
        "id": next_session_id(sessions),
        "goal": goal,
        "status": "clarifying",
        "current_stage": "collect_requirements",
        "required_fields": list(REQUIRED_FIELDS),
        "collected_context": context,
        "missing_fields": [],
        "messages": [],
        "plan": [],
        "draft": None,
        "final_result": None,
        "created_at": now_text(),
        "updated_at": now_text(),
        "events": [],
    }
    append_message(session, "user", user_input)
    answer = maybe_advance_workflow(session)
    append_message(session, "agent", answer)
    append_session_event(session, "session_created", "创建 workflow session", summarize_session(session))
    sessions.append(session)
    save_sessions(sessions)
    logger.log("session_created", "创建多轮工作流会话", {"session": summarize_session(session)})
    return session


def find_session(sessions: list[WorkflowSession], session_id: str) -> WorkflowSession:
    for session in sessions:
        if session["id"] == session_id:
            return session
    raise WorkflowAgentError(f"没有找到会话：{session_id}")


def update_session_with_user_message(session_id: str, message: str, logger: TraceLogger) -> WorkflowSession:
    sessions = load_sessions()
    session = find_session(sessions, session_id)
    if session["status"] in {"completed", "cancelled"}:
        raise WorkflowAgentError(f"{session_id} 已结束，不能继续补充信息。")

    append_message(session, "user", message)
    extracted = extract_context_from_message(message)
    session["collected_context"].update(extracted)
    append_session_event(session, "context_updated", "根据用户补充更新上下文", {"extracted": extracted})
    answer = maybe_advance_workflow(session)
    append_message(session, "agent", answer)
    save_sessions(sessions)
    logger.log(
        "context_updated",
        "多轮会话吸收用户补充信息",
        {"session": summarize_session(session), "extracted": extracted},
    )
    return session


def create_plan(session: WorkflowSession) -> list[str]:
    """
    Create a plan from collected context.

    LLM replacement point:
    A model can generate a personalized plan from topic, audience, format,
    focus, course history, and user preferences. This deterministic template is
    easier for Day 20 students to read.
    """
    context = session["collected_context"]
    topic = context.get("topic", session["goal"])
    audience = context.get("audience", "目标用户")
    focus = context.get("focus", "核心概念")
    return [
        f"说明 {topic} 的目标和适用受众：{audience}",
        f"解释关键概念，并突出重点：{focus}",
        "结合课程前面内容给出结构化案例",
        "整理常见误区、检查清单和下一步练习",
    ]


def create_draft(session: WorkflowSession) -> str:
    """
    Generate a draft from the current workflow state.

    LLM replacement point:
    Later this can call a real model to produce polished content. The important
    habit is to use collected_context and plan, not just the latest user turn.
    """
    context = session["collected_context"]
    topic = context.get("topic", session["goal"])
    audience = context.get("audience", "学习者")
    output_format = context.get("output_format", "Markdown")
    focus = context.get("focus", "核心概念")
    length = context.get("length", "适中篇幅")
    plan_text = "\n".join(f"{index}. {item}" for index, item in enumerate(session["plan"], start=1))
    return (
        f"# {topic}\n\n"
        f"- 受众：{audience}\n"
        f"- 格式：{output_format}\n"
        f"- 重点：{focus}\n"
        f"- 篇幅：{length}\n\n"
        "## 计划\n"
        f"{plan_text}\n\n"
        "## 草稿\n"
        f"这份内容面向{audience}，围绕“{topic}”展开。它会先建立基本概念，"
        f"再结合具体例子说明{focus}，最后给出可以继续练习的任务。"
    )


def run_workflow(session_id: str, logger: TraceLogger) -> WorkflowSession:
    sessions = load_sessions()
    session = find_session(sessions, session_id)
    if session["status"] in {"completed", "cancelled"}:
        raise WorkflowAgentError(f"{session_id} 已结束，不能继续运行。")

    session["missing_fields"] = find_missing_fields(session)
    if session["missing_fields"]:
        session["status"] = "clarifying"
        session["current_stage"] = "collect_requirements"
        answer = create_clarifying_question(session)
        append_message(session, "agent", answer)
        append_session_event(session, "clarification_needed", "运行前发现信息不足", {"missing_fields": session["missing_fields"]})
        save_sessions(sessions)
        logger.log("clarification_needed", "多轮工作流需要继续追问", {"session": summarize_session(session)})
        return session

    if not session["plan"]:
        session["status"] = "planning"
        session["current_stage"] = "planning"
        append_session_event(session, "planning_started", "开始生成计划", {})
        session["plan"] = create_plan(session)
        session["status"] = "ready_to_draft"
        session["current_stage"] = "ready_to_draft"
        answer = f"已生成计划。再次输入 :run {session_id} 可以生成草稿。"
        append_message(session, "agent", answer)
        append_session_event(session, "plan_created", "计划已生成", {"plan": session["plan"]})
        save_sessions(sessions)
        logger.log("plan_created", "多轮工作流生成计划", {"session": summarize_session(session), "plan": session["plan"]})
        return session

    if not session["draft"]:
        session["status"] = "drafting"
        session["current_stage"] = "drafting"
        append_session_event(session, "drafting_started", "开始生成草稿", {})
        session["draft"] = create_draft(session)
        session["status"] = "reviewing"
        session["current_stage"] = "waiting_user_review"
        answer = f"已生成草稿。可以输入 :revise {session_id} 修改要求，或 :complete {session_id} 完成。"
        append_message(session, "agent", answer)
        append_session_event(session, "draft_created", "草稿已生成", {"draft_preview": session["draft"][:120]})
        save_sessions(sessions)
        logger.log("draft_created", "多轮工作流生成草稿", {"session": summarize_session(session)})
        return session

    session["status"] = "reviewing"
    session["current_stage"] = "waiting_user_review"
    append_session_event(session, "waiting_review", "草稿已存在，等待用户反馈", {})
    save_sessions(sessions)
    logger.log("waiting_review", "多轮工作流等待用户审阅", {"session": summarize_session(session)})
    return session


def revise_workflow(session_id: str, instruction: str, logger: TraceLogger) -> WorkflowSession:
    sessions = load_sessions()
    session = find_session(sessions, session_id)
    if not session["draft"]:
        raise WorkflowAgentError(f"{session_id} 还没有草稿，请先运行 :run {session_id}。")
    if session["status"] in {"completed", "cancelled"}:
        raise WorkflowAgentError(f"{session_id} 已结束，不能修改。")

    session["status"] = "revising"
    session["current_stage"] = "revising"
    append_message(session, "user", f"修改要求：{instruction}")
    append_session_event(session, "revision_requested", "用户请求修改草稿", {"instruction": instruction})

    # LLM replacement point: this can call a model to rewrite the draft based on
    # the instruction. The simple append keeps revision state visible offline.
    session["draft"] = (
        f"{session['draft']}\n\n"
        "## 根据反馈补充\n"
        f"- 修改要求：{instruction}\n"
        "- 已根据反馈补充重点内容，并保留原始结构，方便继续审阅。"
    )
    session["status"] = "reviewing"
    session["current_stage"] = "waiting_user_review"
    append_message(session, "agent", f"已根据反馈修改草稿。可以继续 :revise {session_id}，或 :complete {session_id}。")
    append_session_event(session, "draft_revised", "草稿已根据反馈修改", {"instruction": instruction})
    save_sessions(sessions)
    logger.log("draft_revised", "多轮工作流修改草稿", {"session": summarize_session(session), "instruction": instruction})
    return session


def complete_workflow(session_id: str, logger: TraceLogger) -> WorkflowSession:
    sessions = load_sessions()
    session = find_session(sessions, session_id)
    if session["status"] == "completed":
        logger.log("workflow_noop", "会话已经完成", {"session_id": session_id})
        return session
    if not session["draft"]:
        raise WorkflowAgentError(f"{session_id} 还没有草稿，不能完成。")

    session["status"] = "completed"
    session["current_stage"] = "completed"
    session["final_result"] = session["draft"]
    append_message(session, "agent", "工作流已完成，最终结果已保存。")
    append_session_event(session, "workflow_completed", "工作流完成", {"final_preview": session["final_result"][:120]})
    save_sessions(sessions)
    logger.log("workflow_completed", "多轮工作流完成", {"session": summarize_session(session)})
    return session


def summarize_session(session: WorkflowSession) -> dict[str, Any]:
    return {
        "id": session["id"],
        "goal": session["goal"],
        "status": session["status"],
        "current_stage": session["current_stage"],
        "missing_fields": list(session["missing_fields"]),
        "has_plan": bool(session["plan"]),
        "has_draft": bool(session["draft"]),
        "updated_at": session["updated_at"],
    }


def format_sessions(sessions: list[WorkflowSession]) -> str:
    if not sessions:
        return "暂无 workflow session。你可以输入：帮我准备一份 Agent 学习报告"
    lines = ["当前 Workflow Sessions："]
    for session in sessions:
        summary = summarize_session(session)
        missing = "、".join(summary["missing_fields"]) if summary["missing_fields"] else "无"
        lines.append(
            f"- {summary['id']} [{summary['status']}] {summary['goal']} "
            f"stage={summary['current_stage']} missing={missing}"
        )
    return "\n".join(lines)


def format_context(context: dict[str, str]) -> str:
    if not context:
        return "无"
    return "\n".join(f"- {key}: {value}" for key, value in context.items())


def format_session_detail(session: WorkflowSession) -> str:
    missing = "、".join(session["missing_fields"]) if session["missing_fields"] else "无"
    lines = [
        f"会话：{session['id']}",
        f"目标：{session['goal']}",
        f"状态：{session['status']}",
        f"阶段：{session['current_stage']}",
        f"缺失字段：{missing}",
        "已收集信息：",
        format_context(session["collected_context"]),
        "",
        "计划：",
    ]
    if session["plan"]:
        lines.extend(f"{index}. {item}" for index, item in enumerate(session["plan"], start=1))
    else:
        lines.append("暂无")

    lines.extend(["", "最近消息："])
    for message in session["messages"][-6:]:
        lines.append(f"- {message['role']}: {message['content']}")

    if session["draft"]:
        lines.extend(["", "草稿：", session["draft"]])
    if session["final_result"]:
        lines.extend(["", "最终结果已保存。"])
    return "\n".join(lines)


def get_latest_open_session() -> WorkflowSession | None:
    sessions = load_sessions()
    open_sessions = [session for session in sessions if session["status"] not in {"completed", "cancelled"}]
    return open_sessions[-1] if open_sessions else None


def looks_like_new_goal(user_input: str) -> bool:
    return any(word in user_input for word in ["帮我", "请帮我", "准备", "生成", "写", "整理", "新建工作流", "创建工作流"])


def summarize_event_data(event: TraceEvent) -> str:
    data = event["data"]
    event_type = event["event_type"]
    if event_type in {"session_created", "context_updated", "plan_created", "draft_created", "draft_revised", "workflow_completed"}:
        session = data.get("session", {})
        if isinstance(session, dict):
            return f"{session.get('id')} status={session.get('status')} stage={session.get('current_stage')}"
    if event_type == "clarification_needed":
        session = data.get("session", {})
        if isinstance(session, dict):
            return f"{session.get('id')} missing={session.get('missing_fields')}"
    if event_type == "error":
        return str(data.get("error", ""))[:80]
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


def handle_user_input(user_input: str) -> tuple[str, list[TraceEvent]]:
    logger = TraceLogger()
    logger.log("user_input", "收到用户输入", {"input": user_input})

    try:
        if user_input.startswith(":run"):
            session_id = extract_session_id(user_input)
            if not session_id:
                raise WorkflowAgentError("请指定 session_id，例如：:run session_001")
            session = run_workflow(session_id, logger)
            answer = format_session_detail(session)
            logger.log("final_answer", "返回 workflow 运行结果", {"answer": answer})
            return answer, logger.get_events()

        if user_input.startswith(":revise"):
            session_id = extract_session_id(user_input)
            if not session_id:
                raise WorkflowAgentError("请指定 session_id，例如：:revise session_001 增加 RAG 部分")
            instruction = user_input.split(session_id, 1)[1].strip()
            if not instruction:
                raise WorkflowAgentError("请提供修改要求。")
            session = revise_workflow(session_id, instruction, logger)
            answer = format_session_detail(session)
            logger.log("final_answer", "返回 workflow 修改结果", {"answer": answer})
            return answer, logger.get_events()

        if user_input.startswith(":complete"):
            session_id = extract_session_id(user_input)
            if not session_id:
                raise WorkflowAgentError("请指定 session_id，例如：:complete session_001")
            session = complete_workflow(session_id, logger)
            answer = format_session_detail(session)
            logger.log("final_answer", "返回 workflow 完成结果", {"answer": answer})
            return answer, logger.get_events()

        if user_input.startswith(":show"):
            session_id = extract_session_id(user_input)
            if not session_id:
                raise WorkflowAgentError("请指定 session_id，例如：:show session_001")
            session = find_session(load_sessions(), session_id)
            logger.log("session_inspected", "查看 workflow session", {"session": summarize_session(session)})
            answer = format_session_detail(session)
            logger.log("final_answer", "返回 workflow 详情", {"answer": answer})
            return answer, logger.get_events()

        current = get_latest_open_session()
        if current and current["status"] == "clarifying" and not looks_like_new_goal(user_input):
            session = update_session_with_user_message(current["id"], user_input, logger)
            answer = format_session_detail(session)
            logger.log("final_answer", "返回补充信息后的 workflow 状态", {"answer": answer})
            return answer, logger.get_events()

        session = create_session(user_input, logger)
        answer = format_session_detail(session)
        logger.log("final_answer", "返回 workflow 创建结果", {"answer": answer})
        return answer, logger.get_events()

    except Exception as error:
        logger.log("error", "workflow 操作失败", {"error_type": type(error).__name__, "error": str(error)})
        answer = f"workflow 操作失败：{error}"
        logger.log("final_answer", "返回错误结果", {"answer": answer})
        return answer, logger.get_events()


def main() -> None:
    print("Day 20 Multi-turn Workflow Agent")
    print("命令：:sessions、:show session_001、:run session_001、:revise session_001 修改要求、:complete session_001、:clear-sessions、trace、:traces、:clear-traces、exit")
    latest_events: list[TraceEvent] = []

    while True:
        try:
            user_input = input("\n你：").strip()
        except EOFError:
            print("\n已退出。")
            break

        if user_input.lower() in {"exit", "quit", "退出"}:
            print("Agent：已退出。第 21 天可以做第三周综合项目。")
            break
        if user_input == "trace":
            print_trace_summary(latest_events)
            continue
        if user_input == ":traces":
            print_recent_traces()
            continue
        if user_input == ":sessions":
            print(format_sessions(load_sessions()))
            continue
        if user_input == ":clear-sessions":
            clear_sessions()
            continue
        if user_input == ":clear-traces":
            clear_traces()
            latest_events = []
            continue
        if not user_input:
            print("Agent：请输入目标，例如：帮我准备一份 Agent 学习报告")
            continue

        answer, latest_events = handle_user_input(user_input)
        print("\nAgent：")
        print(answer)
        print_trace_summary(latest_events)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出。")
        sys.exit(0)
