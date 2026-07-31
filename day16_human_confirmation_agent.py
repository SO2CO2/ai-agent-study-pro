"""
Day 16: Human Confirmation Agent

Run:
    python3 day16_human_confirmation_agent.py

Try:
    请记住：我喜欢早晨运动
    :memories
    删除 memory_001
    no
    删除 memory_001
    yes
    发送邮件给 test@example.com，主题：你好，内容：明天开会
    yes
    运行 rm -rf /

Commands:
    yes / no       Approve or reject the current pending action.
    trace          Show latest trace summary.
    :traces        Show recent trace summaries from agent_traces.jsonl.
    :memories      Show saved memories.
    :clear-memory  Clear multi_tool_memory.json.
    :clear-traces  Clear local trace logs.
    exit           Quit.

Today's key idea:
    The model may suggest a high-risk action, but only explicit user
    confirmation allows program code to execute it.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypedDict

from day12_observable_agent import (
    TraceEvent,
    TraceLogger,
    clear_traces,
    print_recent_traces,
)


MEMORY_FILE = Path(__file__).with_name("multi_tool_memory.json")


class ConfirmationAgentError(Exception):
    """A readable error for the Day 16 learning script."""


class ToolSpec(TypedDict):
    function: Callable[[dict[str, Any]], dict[str, Any]]
    description: str
    risk: str
    side_effect: bool
    confirm_required: bool
    required_args: list[str]


class PendingAction(TypedDict):
    id: str
    tool: str
    arguments: dict[str, Any]
    risk: str
    side_effect: bool
    status: str
    summary: str
    created_at: str


def ensure_memory_file() -> None:
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text("[]\n", encoding="utf-8")


def load_memory() -> list[dict[str, Any]]:
    ensure_memory_file()
    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfirmationAgentError(f"{MEMORY_FILE.name} 格式无效：{error}") from error
    if not isinstance(data, list):
        raise ConfirmationAgentError(f"{MEMORY_FILE.name} 必须是 JSON 数组。")
    return [item for item in data if isinstance(item, dict)]


def save_memory(memories: list[dict[str, Any]]) -> None:
    MEMORY_FILE.write_text(json.dumps(memories, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clear_memory() -> None:
    save_memory([])
    print(f"已清空 {MEMORY_FILE.name}。")


def extract_memory_text(user_input: str) -> str | None:
    patterns = [
        r"请记住[:：]?\s*(.+)",
        r"记住[:：]?\s*(.+)",
        r"我喜欢(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_input)
        if not match:
            continue
        text = match.group(1).strip(" 。.，,")
        if pattern.startswith("我喜欢"):
            text = f"用户喜欢{text}"
        return text
    return None


def extract_memory_id(user_input: str) -> str | None:
    match = re.search(r"memory_\d{3}", user_input)
    return match.group(0) if match else None


def extract_email_fields(user_input: str) -> dict[str, str] | None:
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", user_input)
    if not email_match:
        return None
    to = email_match.group(0)

    subject = "无主题"
    subject_match = re.search(r"主题[:：]?\s*([^，,。]+)", user_input)
    if subject_match:
        subject = subject_match.group(1).strip()

    body = "无正文"
    body_match = re.search(r"内容[:：]?\s*(.+)", user_input)
    if body_match:
        body = body_match.group(1).strip()
    return {"to": to, "subject": subject, "body": body}


def tool_write_memory(arguments: dict[str, Any]) -> dict[str, Any]:
    memory_text = arguments["memory_text"].strip()
    risky_words = ["管理员", "admin", "权限", "执行所有工具", "最高权限", "root", "system"]
    if any(word in memory_text.casefold() for word in risky_words):
        raise ConfirmationAgentError("拒绝保存权限类记忆。记忆不能提升用户权限。")

    memories = load_memory()
    memory = {
        "id": f"memory_{len(memories) + 1:03d}",
        "text": memory_text,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "day16_human_confirmation_agent",
    }
    memories.append(memory)
    save_memory(memories)
    return {"saved": memory}


def tool_delete_memory(arguments: dict[str, Any]) -> dict[str, Any]:
    memory_id = arguments["memory_id"]
    memories = load_memory()
    remaining = [memory for memory in memories if memory.get("id") != memory_id]
    if len(remaining) == len(memories):
        raise ConfirmationAgentError(f"没有找到记忆：{memory_id}")
    save_memory(remaining)
    return {"deleted_memory_id": memory_id}


def tool_send_email(arguments: dict[str, Any]) -> dict[str, Any]:
    # Teaching safety: this only simulates sending. It never calls a real email API.
    return {
        "sent": True,
        "simulated": True,
        "to": arguments["to"],
        "subject": arguments["subject"],
        "body_preview": arguments["body"][:80],
    }


def tool_run_shell_command(arguments: dict[str, Any]) -> dict[str, Any]:
    del arguments
    raise ConfirmationAgentError("run_shell_command 已被安全策略禁用。")


def register_tools() -> dict[str, ToolSpec]:
    return {
        "write_memory": {
            "function": tool_write_memory,
            "description": "保存用户偏好类记忆",
            "risk": "medium",
            "side_effect": True,
            "confirm_required": False,
            "required_args": ["memory_text"],
        },
        "delete_memory": {
            "function": tool_delete_memory,
            "description": "删除指定记忆",
            "risk": "high",
            "side_effect": True,
            "confirm_required": True,
            "required_args": ["memory_id"],
        },
        "send_email": {
            "function": tool_send_email,
            "description": "模拟发送邮件",
            "risk": "high",
            "side_effect": True,
            "confirm_required": True,
            "required_args": ["to", "subject", "body"],
        },
        "run_shell_command": {
            "function": tool_run_shell_command,
            "description": "执行 shell 命令",
            "risk": "blocked",
            "side_effect": True,
            "confirm_required": True,
            "required_args": ["command"],
        },
    }


def classify_tool_risk(tool_name: str) -> str:
    return register_tools().get(tool_name, {}).get("risk", "unknown")  # type: ignore[union-attr]


def requires_confirmation(tool_name: str) -> bool:
    tool = register_tools().get(tool_name)
    return bool(tool and tool["confirm_required"])


def validate_tool_call(tool_name: str, arguments: dict[str, Any]) -> None:
    registry = register_tools()
    if tool_name not in registry:
        raise ConfirmationAgentError(f"未知工具：{tool_name}")
    if not isinstance(arguments, dict):
        raise ConfirmationAgentError("工具参数必须是对象。")
    for arg_name in registry[tool_name]["required_args"]:
        if arg_name not in arguments:
            raise ConfirmationAgentError(f"{tool_name} 缺少必要参数：{arg_name}")
        if not isinstance(arguments[arg_name], str) or not arguments[arg_name].strip():
            raise ConfirmationAgentError(f"{tool_name}.{arg_name} 必须是非空字符串。")


def create_pending_action(tool_name: str, arguments: dict[str, Any], summary: str) -> PendingAction:
    tool = register_tools()[tool_name]
    return {
        "id": f"action_{datetime.now().strftime('%H%M%S')}",
        "tool": tool_name,
        "arguments": arguments,
        "risk": tool["risk"],
        "side_effect": tool["side_effect"],
        "status": "pending",
        "summary": summary,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def format_pending_action(action: PendingAction) -> str:
    args_text = json.dumps(action["arguments"], ensure_ascii=False)
    return (
        "需要你确认一个高风险动作：\n"
        f"- action_id：{action['id']}\n"
        f"- tool：{action['tool']}\n"
        f"- summary：{action['summary']}\n"
        f"- risk：{action['risk']}\n"
        f"- side_effect：{action['side_effect']}\n"
        f"- arguments：{args_text}\n"
        "请输入 yes 确认执行，或 no 取消。"
    )


def reject_blocked_tool(tool_name: str, arguments: dict[str, Any], logger: TraceLogger) -> str:
    logger.log(
        "safety_blocked",
        "禁用工具被拦截",
        {"tool": tool_name, "arguments": arguments, "risk": "blocked"},
    )
    return f"工具 {tool_name} 已被安全策略禁用，不会执行。"


def execute_confirmed_action(action: PendingAction, logger: TraceLogger) -> str:
    registry = register_tools()
    tool_name = action["tool"]
    arguments = action["arguments"]
    logger.log(
        "tool_call",
        "执行已确认动作",
        {
            "action_id": action["id"],
            "tool": tool_name,
            "arguments": arguments,
            "risk": action["risk"],
            "side_effect": action["side_effect"],
        },
    )
    result = registry[tool_name]["function"](arguments)
    action["status"] = "executed"
    logger.log(
        "tool_result",
        "已确认动作执行成功",
        {"action_id": action["id"], "tool": tool_name, "result": result},
    )
    if tool_name == "delete_memory":
        return f"已删除记忆：{result['deleted_memory_id']}。"
    if tool_name == "send_email":
        return f"已模拟发送邮件给 {result['to']}，主题：{result['subject']}。"
    return f"已执行动作：{tool_name}。"


def handle_confirmation(
    user_input: str,
    pending_action: PendingAction,
    logger: TraceLogger,
) -> tuple[str, PendingAction | None]:
    normalized = user_input.strip().casefold()
    if normalized in {"yes", "y", "确认", "同意", "执行"}:
        logger.log(
            "confirmation_approved",
            "用户确认执行 pending action",
            {"action_id": pending_action["id"], "tool": pending_action["tool"]},
        )
        try:
            answer = execute_confirmed_action(pending_action, logger)
        except Exception as error:
            logger.log(
                "tool_error",
                "已确认动作执行失败",
                {"action_id": pending_action["id"], "error_type": type(error).__name__, "error": str(error)},
            )
            answer = f"执行失败：{error}"
        logger.log("final_answer", "确认流程完成", {"answer": answer})
        return answer, None

    if normalized in {"no", "n", "取消", "拒绝", "不"}:
        pending_action["status"] = "rejected"
        logger.log(
            "confirmation_rejected",
            "用户取消 pending action",
            {"action_id": pending_action["id"], "tool": pending_action["tool"]},
        )
        answer = f"已取消动作：{pending_action['summary']}。"
        logger.log("final_answer", "确认流程取消", {"answer": answer})
        return answer, None

    answer = "当前有待确认动作。请输入 yes 执行，或 no 取消。"
    logger.log("confirmation_requested", "等待用户明确确认或取消", {"action_id": pending_action["id"]})
    return answer, pending_action


def plan_action(user_input: str) -> tuple[str, dict[str, Any], str] | None:
    memory_text = extract_memory_text(user_input)
    if memory_text:
        return "write_memory", {"memory_text": memory_text}, f"保存记忆：{memory_text}"

    memory_id = extract_memory_id(user_input)
    if memory_id and any(word in user_input for word in ["删除", "移除", "清除"]):
        return "delete_memory", {"memory_id": memory_id}, f"删除记忆 {memory_id}"

    email_fields = extract_email_fields(user_input)
    if email_fields and any(word in user_input for word in ["发送邮件", "发邮件", "邮件"]):
        return "send_email", email_fields, f"发送邮件给 {email_fields['to']}"

    if any(word in user_input for word in ["运行", "执行命令", "shell", "rm -rf"]):
        command = user_input
        return "run_shell_command", {"command": command}, f"执行 shell 命令：{command}"

    return None


def run_agent(user_input: str) -> tuple[str, list[TraceEvent], PendingAction | None]:
    logger = TraceLogger()
    logger.log("user_input", "收到用户输入", {"input": user_input})

    planned = plan_action(user_input)
    logger.log("planner_decision", "Planner 识别用户动作", {"planned_action": planned})
    if not planned:
        answer = "我没有识别到需要执行的动作。你可以让我记住偏好、删除记忆、或模拟发送邮件。"
        logger.log("final_answer", "没有可执行动作", {"answer": answer})
        return answer, logger.get_events(), None

    tool_name, arguments, summary = planned
    risk = classify_tool_risk(tool_name)
    logger.log(
        "tool_validation",
        "校验工具风险和参数",
        {"tool": tool_name, "arguments": arguments, "risk": risk, "summary": summary},
    )

    try:
        validate_tool_call(tool_name, arguments)
    except ConfirmationAgentError as error:
        logger.log("tool_error", "工具参数校验失败", {"tool": tool_name, "error": str(error)})
        answer = f"动作参数不完整或不合法：{error}"
        logger.log("final_answer", "生成校验失败回答", {"answer": answer})
        return answer, logger.get_events(), None

    if risk == "blocked":
        answer = reject_blocked_tool(tool_name, arguments, logger)
        logger.log("final_answer", "生成禁用工具拒绝回答", {"answer": answer})
        return answer, logger.get_events(), None

    if requires_confirmation(tool_name):
        pending_action = create_pending_action(tool_name, arguments, summary)
        logger.log(
            "pending_action_created",
            "创建待确认动作",
            pending_action,
        )
        logger.log(
            "confirmation_requested",
            "请求用户确认高风险动作",
            {"action_id": pending_action["id"], "tool": tool_name},
        )
        answer = format_pending_action(pending_action)
        logger.log("final_answer", "输出确认请求", {"answer": answer})
        return answer, logger.get_events(), pending_action

    try:
        registry = register_tools()
        logger.log(
            "tool_call",
            "执行无需确认的工具",
            {
                "tool": tool_name,
                "arguments": arguments,
                "risk": risk,
                "side_effect": registry[tool_name]["side_effect"],
            },
        )
        result = registry[tool_name]["function"](arguments)
        logger.log("tool_result", "工具执行成功", {"tool": tool_name, "result": result})
        if tool_name == "write_memory":
            answer = f"已保存记忆：{result['saved']['text']}。"
        else:
            answer = f"已执行动作：{tool_name}。"
    except Exception as error:
        logger.log("tool_error", "工具执行失败", {"tool": tool_name, "error_type": type(error).__name__, "error": str(error)})
        answer = f"执行失败：{error}"

    logger.log("final_answer", "生成最终回答", {"answer": answer})
    return answer, logger.get_events(), None


def summarize_event_data(event: TraceEvent) -> str:
    data = event["data"]
    event_type = event["event_type"]
    if event_type == "planner_decision":
        return str(data.get("planned_action"))
    if event_type == "tool_validation":
        return f"{data.get('tool')} risk={data.get('risk')}"
    if event_type == "pending_action_created":
        return f"{data.get('id')} {data.get('tool')} risk={data.get('risk')}"
    if event_type == "confirmation_requested":
        return f"action_id={data.get('action_id')}"
    if event_type in {"confirmation_approved", "confirmation_rejected"}:
        return f"action_id={data.get('action_id')}"
    if event_type == "tool_call":
        return f"{data.get('tool')} risk={data.get('risk')} side_effect={data.get('side_effect')}"
    if event_type == "tool_result":
        return f"{data.get('tool')} ok"
    if event_type == "tool_error":
        return f"{data.get('error_type')}: {data.get('error')}"
    if event_type == "safety_blocked":
        return f"{data.get('tool')} blocked"
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


def print_memories() -> None:
    memories = load_memory()
    if not memories:
        print("暂无记忆。")
        return
    print("当前记忆：")
    for memory in memories:
        print(f"- {memory.get('id')}: {memory.get('text')} ({memory.get('created_at')})")


def main() -> None:
    print("Day 16 Human Confirmation Agent")
    print("命令：yes、no、trace、:traces、:memories、:clear-memory、:clear-traces、exit")
    pending_action: PendingAction | None = None
    latest_events: list[TraceEvent] = []

    while True:
        try:
            user_input = input("\n你：").strip()
        except EOFError:
            print("\n已退出。")
            break

        if user_input.lower() in {"exit", "quit", "退出"}:
            print("Agent：已退出。第 17 天可以学习任务队列与长任务状态。")
            break
        if user_input == "trace":
            print_trace_summary(latest_events)
            continue
        if user_input == ":traces":
            print_recent_traces()
            continue
        if user_input == ":memories":
            print_memories()
            continue
        if user_input == ":clear-memory":
            clear_memory()
            continue
        if user_input == ":clear-traces":
            clear_traces()
            latest_events = []
            continue
        if not user_input:
            print("Agent：请输入动作，例如：删除 memory_001，或输入 yes/no 处理待确认动作。")
            continue

        if pending_action is not None:
            logger = TraceLogger()
            logger.log("user_input", "收到确认输入", {"input": user_input})
            answer, pending_action = handle_confirmation(user_input, pending_action, logger)
            latest_events = logger.get_events()
        else:
            answer, latest_events, pending_action = run_agent(user_input)

        print("\nAgent：")
        print(answer)
        print_trace_summary(latest_events)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出。")
        sys.exit(0)
