"""
Day 17: Task Queue Agent

Run:
    python3 day17_task_queue_agent.py

Try:
    创建任务：整理第 9-10 天 RAG 学习笔记
    :tasks
    :show task_001
    :run task_001
    :run task_001
    :cancel task_001
    :resume task_001

Commands:
    :tasks           Show all saved tasks.
    :show task_001   Show one task with its steps.
    :run task_001    Advance one task by one step.
    :resume task_001 Resume a failed or paused task, then advance one step.
    :cancel task_001 Cancel a task that is not completed.
    :clear-tasks     Clear agent_tasks.json.
    trace            Show latest trace summary.
    :traces          Show recent trace summaries from agent_traces.jsonl.
    :clear-traces    Clear local trace logs.
    exit             Quit.

Today's key idea:
    A long-running Agent should save progress, expose status, resume safely,
    and allow cancellation instead of trying to finish everything in one turn.

This script stays offline so students can focus on the task queue architecture.
The rule-based planner and fake step executor are intentionally small; comments
mark the places that can later be replaced with real LLM planning and tools.
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


TASK_FILE = Path(__file__).with_name("agent_tasks.json")

TaskStatus = Literal["pending", "running", "paused", "completed", "failed", "cancelled"]
StepStatus = Literal["pending", "running", "completed", "failed", "skipped"]


class TaskQueueAgentError(Exception):
    """A readable error for the Day 17 learning script."""


class TaskStep(TypedDict):
    id: str
    description: str
    status: StepStatus
    result: str | None
    started_at: str | None
    completed_at: str | None
    error: str | None


class TaskEvent(TypedDict):
    timestamp: str
    event_type: str
    message: str
    data: dict[str, Any]


class AgentTask(TypedDict):
    id: str
    title: str
    goal: str
    status: TaskStatus
    steps: list[TaskStep]
    final_result: str | None
    created_at: str
    updated_at: str
    events: list[TaskEvent]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_task_file() -> None:
    if not TASK_FILE.exists():
        TASK_FILE.write_text("[]\n", encoding="utf-8")


def load_tasks() -> list[AgentTask]:
    ensure_task_file()
    try:
        data = json.loads(TASK_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TaskQueueAgentError(f"{TASK_FILE.name} 格式无效：{error}") from error

    if not isinstance(data, list):
        raise TaskQueueAgentError(f"{TASK_FILE.name} 必须是 JSON 数组。")

    tasks: list[AgentTask] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            tasks.append(item)  # type: ignore[arg-type]
    return tasks


def save_tasks(tasks: list[AgentTask]) -> None:
    TASK_FILE.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clear_tasks() -> None:
    save_tasks([])
    print(f"已清空 {TASK_FILE.name}。")


def next_task_id(tasks: list[AgentTask]) -> str:
    max_number = 0
    for task in tasks:
        match = re.fullmatch(r"task_(\d{3})", task["id"])
        if match:
            max_number = max(max_number, int(match.group(1)))
    return f"task_{max_number + 1:03d}"


def extract_task_goal(user_input: str) -> str | None:
    patterns = [
        r"创建任务[:：]\s*(.+)",
        r"新建任务[:：]\s*(.+)",
        r"添加任务[:：]\s*(.+)",
        r"帮我(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_input)
        if match:
            return match.group(1).strip(" 。.，,")
    return None


def extract_task_id(user_input: str) -> str | None:
    match = re.search(r"task_\d{3}", user_input)
    return match.group(0) if match else None


def create_title(goal: str) -> str:
    clean_goal = re.sub(r"\s+", " ", goal).strip()
    return clean_goal[:28] + ("..." if len(clean_goal) > 28 else "")


def plan_task_steps(goal: str) -> list[TaskStep]:
    """
    Create a small step plan for the task.

    LLM replacement point:
    In a real Agent, this function can call a model and ask it to return
    structured JSON steps based on the user's goal, constraints, tools, and
    current context. Here we keep it rule-based so the queue mechanism is easy
    to learn and test offline.
    """
    if any(word in goal for word in ["RAG", "rag", "知识库", "文档"]):
        descriptions = [
            "收集和整理相关资料范围",
            "提取核心概念和关键术语",
            "梳理流程、风险点和检查项",
            "生成简明总结和练习建议",
        ]
    elif any(word in goal for word in ["课程", "学习", "计划", "复习"]):
        descriptions = [
            "明确学习目标和适合的新手范围",
            "拆分主题和每日学习重点",
            "设计练习任务和可运行产出",
            "整理复习清单和下一步建议",
        ]
    elif any(word in goal for word in ["检查", "评估", "测试", "质量"]):
        descriptions = [
            "确定检查目标和评价标准",
            "收集需要检查的输入材料",
            "逐项执行检查并记录发现",
            "输出结论、风险和改进建议",
        ]
    else:
        descriptions = [
            "理解用户目标并确定完成标准",
            "拆分可执行步骤和所需信息",
            "执行核心处理并保存中间结果",
            "汇总结果并给出下一步建议",
        ]

    return [
        {
            "id": f"step_{index:03d}",
            "description": description,
            "status": "pending",
            "result": None,
            "started_at": None,
            "completed_at": None,
            "error": None,
        }
        for index, description in enumerate(descriptions, start=1)
    ]


def append_task_event(task: AgentTask, event_type: str, message: str, data: dict[str, Any] | None = None) -> None:
    task["events"].append(
        {
            "timestamp": now_text(),
            "event_type": event_type,
            "message": message,
            "data": data or {},
        }
    )
    task["updated_at"] = now_text()


def create_task(goal: str, logger: TraceLogger) -> AgentTask:
    tasks = load_tasks()
    task_id = next_task_id(tasks)
    task: AgentTask = {
        "id": task_id,
        "title": create_title(goal),
        "goal": goal,
        "status": "pending",
        "steps": plan_task_steps(goal),
        "final_result": None,
        "created_at": now_text(),
        "updated_at": now_text(),
        "events": [],
    }
    append_task_event(task, "task_created", "创建任务", {"goal": goal, "step_count": len(task["steps"])})
    tasks.append(task)
    save_tasks(tasks)
    logger.log("task_created", "创建长任务并保存到任务队列", {"task": summarize_task(task)})
    return task


def find_task(tasks: list[AgentTask], task_id: str) -> AgentTask:
    for task in tasks:
        if task["id"] == task_id:
            return task
    raise TaskQueueAgentError(f"没有找到任务：{task_id}")


def get_next_pending_step(task: AgentTask) -> TaskStep | None:
    for step in task["steps"]:
        if step["status"] in {"pending", "failed"}:
            return step
    return None


def run_next_step(task: AgentTask, step: TaskStep) -> str:
    """
    Execute one step and return its result.

    LLM/tool replacement point:
    Later this function can become a full Act phase:
    - ask an LLM how to execute the current step
    - call one or more tools
    - observe tool results
    - write the step result back into the task checkpoint

    This offline version returns a deterministic learning-friendly result so
    students can focus on queue state transitions first.
    """
    goal = task["goal"]
    description = step["description"]

    if "失败演示" in goal and step["id"] == "step_002":
        raise TaskQueueAgentError("这是一个故意触发的失败演示，用来观察 failed 状态和 resume。")

    if any(word in description for word in ["收集", "输入材料", "资料范围"]):
        return f"已确定资料范围：围绕“{goal}”收集必要背景、已有文件和约束。"
    if any(word in description for word in ["提取", "主题", "概念", "标准"]):
        return f"已提取关键点：目标、输入、输出、状态、失败处理和验收标准。"
    if any(word in description for word in ["执行", "梳理", "检查"]):
        return f"已完成核心处理：按步骤推进“{goal}”，并记录可复用的中间结果。"
    return f"已完成总结：形成“{goal}”的阶段性结论和下一步建议。"


def advance_task(task_id: str, logger: TraceLogger, resume: bool = False) -> AgentTask:
    tasks = load_tasks()
    task = find_task(tasks, task_id)

    if task["status"] == "completed":
        logger.log("task_noop", "任务已完成，无需重复执行", {"task_id": task_id})
        return task
    if task["status"] == "cancelled" and not resume:
        raise TaskQueueAgentError(f"{task_id} 已取消。如需继续，请使用 :resume {task_id}。")
    if task["status"] == "cancelled" and resume:
        task["status"] = "pending"
        append_task_event(task, "task_resumed", "恢复已取消任务", {})
    if task["status"] == "failed" and resume:
        task["status"] = "pending"
        append_task_event(task, "task_resumed", "恢复失败任务", {})
    if task["status"] == "paused" and resume:
        task["status"] = "pending"
        append_task_event(task, "task_resumed", "恢复暂停任务", {})

    step = get_next_pending_step(task)
    if step is None:
        task["status"] = "completed"
        task["final_result"] = create_final_result(task)
        append_task_event(task, "task_completed", "没有待执行步骤，任务标记完成", {"final_result": task["final_result"]})
        save_tasks(tasks)
        logger.log("task_completed", "任务已完成", {"task": summarize_task(task)})
        return task

    task["status"] = "running"
    step["status"] = "running"
    step["started_at"] = now_text()
    step["error"] = None
    append_task_event(task, "step_started", "开始执行步骤", {"step_id": step["id"], "description": step["description"]})
    save_tasks(tasks)
    logger.log(
        "step_started",
        "推进长任务一步",
        {"task_id": task_id, "step_id": step["id"], "description": step["description"]},
    )

    try:
        result = run_next_step(task, step)
    except Exception as error:
        step["status"] = "failed"
        step["error"] = str(error)
        step["completed_at"] = now_text()
        task["status"] = "failed"
        append_task_event(
            task,
            "step_failed",
            "步骤执行失败",
            {"step_id": step["id"], "error_type": type(error).__name__, "error": str(error)},
        )
        save_tasks(tasks)
        logger.log(
            "step_failed",
            "长任务步骤执行失败",
            {"task_id": task_id, "step_id": step["id"], "error_type": type(error).__name__, "error": str(error)},
        )
        return task

    step["status"] = "completed"
    step["result"] = result
    step["completed_at"] = now_text()
    append_task_event(task, "step_completed", "步骤执行完成", {"step_id": step["id"], "result": result})

    if get_next_pending_step(task) is None:
        task["status"] = "completed"
        task["final_result"] = create_final_result(task)
        append_task_event(task, "task_completed", "任务所有步骤已完成", {"final_result": task["final_result"]})
        logger.log("task_completed", "长任务所有步骤已完成", {"task": summarize_task(task)})
    else:
        task["status"] = "pending"
        logger.log("step_completed", "长任务步骤完成，等待下一次推进", {"task": summarize_task(task), "result": result})

    save_tasks(tasks)
    return task


def cancel_task(task_id: str, logger: TraceLogger) -> AgentTask:
    tasks = load_tasks()
    task = find_task(tasks, task_id)
    if task["status"] == "completed":
        raise TaskQueueAgentError("已完成任务不能取消。")
    if task["status"] == "cancelled":
        logger.log("task_noop", "任务已经是取消状态", {"task_id": task_id})
        return task

    task["status"] = "cancelled"
    append_task_event(task, "task_cancelled", "用户取消任务", {})
    save_tasks(tasks)
    logger.log("task_cancelled", "任务已取消", {"task": summarize_task(task)})
    return task


def create_final_result(task: AgentTask) -> str:
    completed_results = [step["result"] for step in task["steps"] if step["status"] == "completed" and step["result"]]
    if not completed_results:
        return "任务结束，但没有可汇总的步骤结果。"
    joined = "\n".join(f"- {result}" for result in completed_results)
    return f"任务“{task['title']}”已完成，结果摘要：\n{joined}"


def summarize_task(task: AgentTask) -> dict[str, Any]:
    completed_count = sum(1 for step in task["steps"] if step["status"] == "completed")
    failed_count = sum(1 for step in task["steps"] if step["status"] == "failed")
    return {
        "id": task["id"],
        "title": task["title"],
        "status": task["status"],
        "progress": f"{completed_count}/{len(task['steps'])}",
        "failed_steps": failed_count,
        "updated_at": task["updated_at"],
    }


def format_task_list(tasks: list[AgentTask]) -> str:
    if not tasks:
        return "暂无任务。你可以输入：创建任务：整理第 9-10 天 RAG 学习笔记"

    lines = ["当前任务："]
    for task in tasks:
        summary = summarize_task(task)
        lines.append(
            f"- {summary['id']} [{summary['status']}] {summary['title']} "
            f"进度 {summary['progress']}，updated_at={summary['updated_at']}"
        )
    return "\n".join(lines)


def format_task_detail(task: AgentTask) -> str:
    lines = [
        f"任务：{task['id']}",
        f"标题：{task['title']}",
        f"目标：{task['goal']}",
        f"状态：{task['status']}",
        f"创建时间：{task['created_at']}",
        f"更新时间：{task['updated_at']}",
        "步骤：",
    ]
    for step in task["steps"]:
        suffix = ""
        if step["error"]:
            suffix = f"，error={step['error']}"
        elif step["result"]:
            suffix = f" -> {step['result']}"
        lines.append(f"- {step['id']} [{step['status']}] {step['description']}{suffix}")
    if task["final_result"]:
        lines.append("\n最终结果：")
        lines.append(task["final_result"])
    return "\n".join(lines)


def format_advance_result(task: AgentTask) -> str:
    if task["status"] == "failed":
        failed_steps = [step for step in task["steps"] if step["status"] == "failed"]
        failed = failed_steps[0] if failed_steps else None
        if failed:
            return (
                f"{task['id']} 执行失败：{failed['description']}\n"
                f"错误：{failed['error']}\n"
                f"修复后可输入：:resume {task['id']}"
            )
        return f"{task['id']} 执行失败。"
    if task["status"] == "completed":
        return task["final_result"] or f"{task['id']} 已完成。"
    return f"{task['id']} 已推进一步。\n{format_task_detail(task)}"


def summarize_event_data(event: TraceEvent) -> str:
    data = event["data"]
    event_type = event["event_type"]
    if event_type in {"task_created", "task_completed", "task_cancelled"}:
        task = data.get("task", {})
        if isinstance(task, dict):
            return f"{task.get('id')} status={task.get('status')} progress={task.get('progress')}"
    if event_type in {"step_started", "step_failed"}:
        return f"{data.get('task_id')} {data.get('step_id')}"
    if event_type == "step_completed":
        task = data.get("task", {})
        if isinstance(task, dict):
            return f"{task.get('id')} progress={task.get('progress')}"
    if event_type == "task_noop":
        return str(data.get("task_id", ""))
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

    task_goal = extract_task_goal(user_input)
    if task_goal:
        task = create_task(task_goal, logger)
        answer = "已创建任务：\n" + format_task_detail(task)
        logger.log("final_answer", "返回任务创建结果", {"answer": answer})
        return answer, logger.get_events()

    task_id = extract_task_id(user_input)
    try:
        if user_input.startswith(":run") and task_id:
            task = advance_task(task_id, logger, resume=False)
            answer = format_advance_result(task)
            logger.log("final_answer", "返回任务推进结果", {"answer": answer})
            return answer, logger.get_events()

        if user_input.startswith(":resume") and task_id:
            task = advance_task(task_id, logger, resume=True)
            answer = format_advance_result(task)
            logger.log("final_answer", "返回任务恢复结果", {"answer": answer})
            return answer, logger.get_events()

        if user_input.startswith(":cancel") and task_id:
            task = cancel_task(task_id, logger)
            answer = f"已取消任务：{task['id']}。"
            logger.log("final_answer", "返回任务取消结果", {"answer": answer})
            return answer, logger.get_events()

        if user_input.startswith(":show") and task_id:
            task = find_task(load_tasks(), task_id)
            logger.log("task_inspected", "查看任务详情", {"task": summarize_task(task)})
            answer = format_task_detail(task)
            logger.log("final_answer", "返回任务详情", {"answer": answer})
            return answer, logger.get_events()
    except Exception as error:
        logger.log("error", "任务操作失败", {"error_type": type(error).__name__, "error": str(error)})
        answer = f"任务操作失败：{error}"
        logger.log("final_answer", "返回错误结果", {"answer": answer})
        return answer, logger.get_events()

    answer = (
        "我还没有识别到任务动作。\n"
        "你可以输入：创建任务：整理 RAG 学习笔记\n"
        "或使用命令：:tasks、:show task_001、:run task_001、:resume task_001、:cancel task_001"
    )
    logger.log("final_answer", "返回帮助提示", {"answer": answer})
    return answer, logger.get_events()


def main() -> None:
    print("Day 17 Task Queue Agent")
    print("命令：:tasks、:show task_001、:run task_001、:resume task_001、:cancel task_001、:clear-tasks、trace、:traces、:clear-traces、exit")
    latest_events: list[TraceEvent] = []

    while True:
        try:
            user_input = input("\n你：").strip()
        except EOFError:
            print("\n已退出。")
            break

        if user_input.lower() in {"exit", "quit", "退出"}:
            print("Agent：已退出。第 18 天可以学习如何把 Agent 封装成 API 服务。")
            break
        if user_input == "trace":
            print_trace_summary(latest_events)
            continue
        if user_input == ":traces":
            print_recent_traces()
            continue
        if user_input == ":tasks":
            print(format_task_list(load_tasks()))
            continue
        if user_input == ":clear-tasks":
            clear_tasks()
            continue
        if user_input == ":clear-traces":
            clear_traces()
            latest_events = []
            continue
        if not user_input:
            print("Agent：请输入任务，例如：创建任务：整理第 9-10 天 RAG 学习笔记")
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
