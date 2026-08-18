"""
Day 24: Auth, Permissions, and User Isolation

Run:
    python3 day24_auth_agent.py

Try:
    :whoami
    帮我准备 Agent 第三周复习材料
    :login alice
    帮我准备 Agent 第三周复习材料
    面向新手，用 Markdown，重点讲项目实践
    :run-session session_001
    :run-task task_001
    :run-task task_001
    :run-task task_001
    :run-task task_001
    :run-task task_001
    :login bob
    :sessions
    :confirm action_001 yes
    :login alice
    :confirm action_001 yes
    :outputs
    :login admin
    :sessions

Today's key idea:
    Agent permissions are not about hiding buttons. Every read, write, tool
    call, and confirmation must know who the user is and check permission before
    execution.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

from day12_observable_agent import TraceEvent, TraceLogger, clear_traces, print_recent_traces


POLICY_FILE = Path(__file__).with_name("auth_policy.json")
USER_STATE_FILE = Path(__file__).with_name("user_state.json")
REQUIRED_FIELDS = ["audience", "output_format", "focus"]

SessionStatus = Literal["clarifying", "ready", "task_created", "completed", "cancelled"]
TaskStatus = Literal["pending", "waiting_confirmation", "completed", "cancelled", "failed"]
StepStatus = Literal["pending", "completed", "waiting_confirmation", "failed"]


class AuthAgentError(Exception):
    """A readable error for the Day 24 learning script."""


class User(TypedDict):
    id: str | None
    role: str


class Session(TypedDict):
    id: str
    owner_user_id: str
    goal: str
    status: SessionStatus
    collected_context: dict[str, str]
    missing_fields: list[str]
    messages: list[dict[str, str]]
    plan: list[str]
    task_id: str | None
    created_at: str
    updated_at: str


class TaskStep(TypedDict):
    id: str
    description: str
    tool: str
    status: StepStatus
    result: str | None
    error: str | None


class Task(TypedDict):
    id: str
    owner_user_id: str
    session_id: str
    title: str
    status: TaskStatus
    steps: list[TaskStep]
    final_result: str | None
    created_at: str
    updated_at: str


class PendingAction(TypedDict):
    id: str
    owner_user_id: str
    tool: str
    risk: str
    summary: str
    task_id: str
    step_id: str
    status: str
    created_at: str


class Output(TypedDict):
    id: str
    owner_user_id: str
    title: str
    content: str
    source_task_id: str
    created_at: str


class UserState(TypedDict):
    current_user_id: str | None
    sessions: list[Session]
    tasks: list[Task]
    pending_actions: list[PendingAction]
    outputs: list[Output]
    updated_at: str


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def empty_user_state() -> UserState:
    return {
        "current_user_id": None,
        "sessions": [],
        "tasks": [],
        "pending_actions": [],
        "outputs": [],
        "updated_at": "",
    }


def load_auth_policy(path: Path = POLICY_FILE) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise AuthAgentError(f"找不到权限策略文件：{path.name}") from error
    except json.JSONDecodeError as error:
        raise AuthAgentError(f"{path.name} 不是合法 JSON：{error}") from error
    if not isinstance(data, dict):
        raise AuthAgentError("auth_policy.json 必须是 JSON 对象。")
    return data


def load_user_state(path: Path = USER_STATE_FILE) -> UserState:
    if not path.exists():
        save_user_state(empty_user_state(), path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AuthAgentError(f"{path.name} 不是合法 JSON：{error}") from error
    state = empty_user_state()
    for key in state:
        if key in data:
            state[key] = data[key]  # type: ignore[literal-required]
    return state


def save_user_state(state: UserState, path: Path = USER_STATE_FILE) -> None:
    state["updated_at"] = now_text()
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clear_user_state() -> None:
    save_user_state(empty_user_state())
    print(f"已清空 {USER_STATE_FILE.name}。")


def get_user_role(user_id: str | None, policy: dict[str, Any]) -> str:
    if not user_id:
        return policy.get("default_role", "guest")
    users = policy.get("users", {})
    return users.get(user_id, policy.get("default_role", "guest"))


def current_user(state: UserState, policy: dict[str, Any]) -> User:
    user_id = state.get("current_user_id")
    return {"id": user_id, "role": get_user_role(user_id, policy)}


def login(user_id: str, state: UserState, policy: dict[str, Any]) -> str:
    if user_id not in policy.get("users", {}):
        raise AuthAgentError(f"未知用户：{user_id}")
    state["current_user_id"] = user_id
    save_user_state(state)
    return f"已登录 {user_id}，角色 {get_user_role(user_id, policy)}。"


def has_permission(user: User, permission: str, policy: dict[str, Any]) -> bool:
    permissions = policy.get("role_permissions", {}).get(user["role"], [])
    return "*" in permissions or permission in permissions


def require_permission(user: User, permission: str, policy: dict[str, Any]) -> None:
    if not has_permission(user, permission, policy):
        name = user["id"] or "未登录用户"
        raise AuthAgentError(f"权限不足：{name}({user['role']}) 缺少 {permission}。")


def is_admin(user: User, policy: dict[str, Any]) -> bool:
    return has_permission(user, "*", policy)


def visible_to_user(item: dict[str, Any], user: User, policy: dict[str, Any]) -> bool:
    if is_admin(user, policy):
        return True
    return item.get("owner_user_id") == user["id"]


def filter_visible_items(items: list[dict[str, Any]], user: User, policy: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in items if visible_to_user(item, user, policy)]


def next_id(items: list[dict[str, Any]], prefix: str) -> str:
    max_number = 0
    pattern = re.compile(rf"{prefix}_(\d{{3}})")
    for item in items:
        value = item.get("id")
        if isinstance(value, str):
            match = pattern.fullmatch(value)
            if match:
                max_number = max(max_number, int(match.group(1)))
    return f"{prefix}_{max_number + 1:03d}"


def extract_id(text: str, prefix: str) -> str | None:
    match = re.search(rf"{prefix}_\d{{3}}", text)
    return match.group(0) if match else None


def extract_goal(text: str) -> str:
    return re.sub(r"^(帮我|请帮我|我要|我想)\s*", "", text).strip(" 。.，,")


def extract_context(text: str) -> dict[str, str]:
    context: dict[str, str] = {}
    if "新手" in text:
        context["audience"] = "新手"
    audience = re.search(r"面向([^，,。]+)", text)
    if audience:
        context["audience"] = audience.group(1).strip()
    if re.search(r"markdown|md", text, re.IGNORECASE):
        context["output_format"] = "Markdown"
    elif "大纲" in text:
        context["output_format"] = "大纲"
    if "项目" in text:
        context["focus"] = "项目实践"
    focus = re.search(r"重点(?:讲|偏)?([^，,。]+)", text)
    if focus:
        context["focus"] = focus.group(1).strip()
    return context


def update_missing_fields(session: Session) -> None:
    session["missing_fields"] = [field for field in REQUIRED_FIELDS if not session["collected_context"].get(field)]
    session["status"] = "clarifying" if session["missing_fields"] else "ready"
    session["updated_at"] = now_text()


def create_user_session(user_input: str, user: User, state: UserState, policy: dict[str, Any]) -> Session:
    require_permission(user, "create_session", policy)
    if not user["id"]:
        raise AuthAgentError("请先登录。")
    goal = extract_goal(user_input)
    session: Session = {
        "id": next_id(state["sessions"], "session"),
        "owner_user_id": user["id"],
        "goal": goal,
        "status": "clarifying",
        "collected_context": {"topic": "Agent 第三周复习材料" if "第三周" in goal else goal},
        "missing_fields": [],
        "messages": [{"role": "user", "content": user_input, "timestamp": now_text()}],
        "plan": [],
        "task_id": None,
        "created_at": now_text(),
        "updated_at": now_text(),
    }
    update_missing_fields(session)
    state["sessions"].append(session)
    save_user_state(state)
    return session


def latest_open_session(state: UserState, user: User, policy: dict[str, Any]) -> Session | None:
    visible = filter_visible_items(state["sessions"], user, policy)
    for session in reversed(visible):
        if session["status"] == "clarifying":
            return session  # type: ignore[return-value]
    return None


def update_user_session(session: Session, message: str, user: User, state: UserState, policy: dict[str, Any]) -> str:
    if not visible_to_user(session, user, policy):
        raise AuthAgentError("权限不足：不能修改别人的 session。")
    require_permission(user, "create_session", policy)
    session["messages"].append({"role": "user", "content": message, "timestamp": now_text()})
    session["collected_context"].update(extract_context(message))
    update_missing_fields(session)
    if session["status"] == "ready":
        answer = f"{session['id']} 信息已齐全，可以输入 :run-session {session['id']} 创建任务。"
    else:
        answer = f"还需要确认：{'、'.join(session['missing_fields'])}。"
    session["messages"].append({"role": "agent", "content": answer, "timestamp": now_text()})
    save_user_state(state)
    return answer


def find_session(state: UserState, session_id: str) -> Session:
    for session in state["sessions"]:
        if session["id"] == session_id:
            return session
    raise AuthAgentError(f"没有找到 session：{session_id}")


def find_task(state: UserState, task_id: str) -> Task:
    for task in state["tasks"]:
        if task["id"] == task_id:
            return task
    raise AuthAgentError(f"没有找到 task：{task_id}")


def create_task_from_session(session_id: str, user: User, state: UserState, policy: dict[str, Any]) -> Task:
    session = find_session(state, session_id)
    if not visible_to_user(session, user, policy):
        raise AuthAgentError("权限不足：不能运行别人的 session。")
    require_permission(user, "run_own_task" if not is_admin(user, policy) else "run_all_tasks", policy)
    if session["missing_fields"]:
        raise AuthAgentError(f"{session_id} 信息不完整：{', '.join(session['missing_fields'])}")
    if session["task_id"]:
        return find_task(state, session["task_id"])
    context = session["collected_context"]
    session["plan"] = [
        "收集第三周主题",
        f"面向{context.get('audience')}整理重点：{context.get('focus')}",
        f"生成{context.get('output_format')}结构",
        "保存最终输出",
    ]
    task: Task = {
        "id": next_id(state["tasks"], "task"),
        "owner_user_id": session["owner_user_id"],
        "session_id": session["id"],
        "title": context.get("topic", session["goal"]),
        "status": "pending",
        "steps": [
            create_step("step_001", session["plan"][0], "read_course_outline"),
            create_step("step_002", session["plan"][1], "summarize_week3"),
            create_step("step_003", session["plan"][2], "create_outline"),
            create_step("step_004", session["plan"][3], "write_final_report"),
        ],
        "final_result": None,
        "created_at": now_text(),
        "updated_at": now_text(),
    }
    session["task_id"] = task["id"]
    session["status"] = "task_created"
    state["tasks"].append(task)
    save_user_state(state)
    return task


def create_step(step_id: str, description: str, tool: str) -> TaskStep:
    return {"id": step_id, "description": description, "tool": tool, "status": "pending", "result": None, "error": None}


def next_pending_step(task: Task) -> TaskStep | None:
    for step in task["steps"]:
        if step["status"] == "pending":
            return step
    return None


def run_user_task(task_id: str, user: User, state: UserState, policy: dict[str, Any]) -> Task:
    task = find_task(state, task_id)
    if not visible_to_user(task, user, policy):
        raise AuthAgentError("权限不足：不能运行别人的 task。")
    require_permission(user, "run_own_task" if not is_admin(user, policy) else "run_all_tasks", policy)
    if task["status"] == "completed":
        return task
    step = next_pending_step(task)
    if not step:
        task["status"] = "completed"
        save_user_state(state)
        return task
    if step["tool"] == "write_final_report":
        action = create_user_pending_action(step, task, user, state)
        step["status"] = "waiting_confirmation"
        task["status"] = "waiting_confirmation"
        save_user_state(state)
        return task
    step["status"] = "completed"
    step["result"] = execute_read_only_tool(step)
    task["status"] = "pending" if next_pending_step(task) else "completed"
    task["updated_at"] = now_text()
    save_user_state(state)
    return task


def execute_read_only_tool(step: TaskStep) -> str:
    if step["tool"] == "read_course_outline":
        return "已收集第三周主题：多工具、人类确认、任务队列、API、前端、多轮工作流、工作台。"
    if step["tool"] == "summarize_week3":
        return "已总结第三周重点：把 Agent 做成可操作、可追踪、可恢复、可协作的小系统。"
    if step["tool"] == "create_outline":
        return "已生成复习结构：能力地图、核心边界、练习任务、常见风险。"
    return "步骤已完成。"


def create_user_pending_action(step: TaskStep, task: Task, user: User, state: UserState) -> PendingAction:
    if not user["id"]:
        raise AuthAgentError("请先登录。")
    action: PendingAction = {
        "id": next_id(state["pending_actions"], "action"),
        "owner_user_id": task["owner_user_id"],
        "tool": step["tool"],
        "risk": "high",
        "summary": f"保存 {task['title']} 的最终输出",
        "task_id": task["id"],
        "step_id": step["id"],
        "status": "pending",
        "created_at": now_text(),
    }
    state["pending_actions"].append(action)
    return action


def confirm_user_action(action_id: str, decision: str, user: User, state: UserState, policy: dict[str, Any]) -> str:
    action = find_action(state, action_id)
    owns_action = action["owner_user_id"] == user["id"]
    if owns_action:
        require_permission(user, "confirm_own_action", policy)
    else:
        require_permission(user, "confirm_all_actions", policy)
    task = find_task(state, action["task_id"])
    step = next((item for item in task["steps"] if item["id"] == action["step_id"]), None)
    if decision.lower() in {"no", "n", "取消", "拒绝"}:
        action["status"] = "rejected"
        if step:
            step["status"] = "pending"
        task["status"] = "pending"
        save_user_state(state)
        return f"已取消动作：{action['summary']}。"
    if decision.lower() not in {"yes", "y", "确认", "同意"}:
        raise AuthAgentError("请使用 yes 或 no。")
    output: Output = {
        "id": next_id(state["outputs"], "output"),
        "owner_user_id": action["owner_user_id"],
        "title": task["title"],
        "content": build_output_content(task),
        "source_task_id": task["id"],
        "created_at": now_text(),
    }
    state["outputs"].append(output)
    action["status"] = "executed"
    if step:
        step["status"] = "completed"
        step["result"] = f"已保存输出 {output['id']}。"
    task["status"] = "completed"
    task["final_result"] = f"已保存输出 {output['id']}。"
    session = find_session(state, task["session_id"])
    session["status"] = "completed"
    save_user_state(state)
    return f"已确认并执行：{action['summary']}。输出：{output['id']}。"


def build_output_content(task: Task) -> str:
    results = [step["result"] for step in task["steps"] if step["result"]]
    return "# " + task["title"] + "\n\n" + "\n".join(f"- {item}" for item in results)


def find_action(state: UserState, action_id: str) -> PendingAction:
    for action in state["pending_actions"]:
        if action["id"] == action_id and action["status"] == "pending":
            return action
    raise AuthAgentError(f"没有找到待确认动作：{action_id}")


def format_user(user: User) -> str:
    return f"当前用户：{user['id'] or '未登录'}，角色：{user['role']}"


def format_sessions(state: UserState, user: User, policy: dict[str, Any]) -> str:
    permission = "read_all_sessions" if is_admin(user, policy) else "read_own_sessions"
    require_permission(user, permission, policy)
    items = filter_visible_items(state["sessions"], user, policy)
    if not items:
        return "暂无可见 session。"
    return "\n".join(f"- {item['id']} owner={item['owner_user_id']} status={item['status']} goal={item['goal']}" for item in items)


def format_tasks(state: UserState, user: User, policy: dict[str, Any]) -> str:
    permission = "run_all_tasks" if is_admin(user, policy) else "read_own_tasks"
    require_permission(user, permission, policy)
    items = filter_visible_items(state["tasks"], user, policy)
    if not items:
        return "暂无可见 task。"
    return "\n".join(f"- {item['id']} owner={item['owner_user_id']} status={item['status']} title={item['title']}" for item in items)


def format_outputs(state: UserState, user: User, policy: dict[str, Any]) -> str:
    permission = "read_all_outputs" if is_admin(user, policy) else "read_own_outputs"
    require_permission(user, permission, policy)
    items = filter_visible_items(state["outputs"], user, policy)
    if not items:
        return "暂无可见 output。"
    return "\n".join(f"- {item['id']} owner={item['owner_user_id']} title={item['title']}" for item in items)


def format_pending_actions(state: UserState, user: User, policy: dict[str, Any]) -> str:
    items = [item for item in state["pending_actions"] if item["status"] == "pending" and visible_to_user(item, user, policy)]
    if not items:
        return "暂无可确认 action。"
    return "\n".join(f"- {item['id']} owner={item['owner_user_id']} tool={item['tool']} risk={item['risk']} summary={item['summary']}" for item in items)


def format_task_detail(task: Task) -> str:
    completed = sum(1 for step in task["steps"] if step["status"] == "completed")
    lines = [f"Task：{task['id']} owner={task['owner_user_id']} status={task['status']} progress={completed}/{len(task['steps'])}"]
    for step in task["steps"]:
        suffix = f" -> {step['result']}" if step["result"] else ""
        lines.append(f"- {step['id']} [{step['status']}] {step['description']} tool={step['tool']}{suffix}")
    return "\n".join(lines)


def handle_auth_input(user_input: str, state: UserState, policy: dict[str, Any]) -> tuple[str, list[TraceEvent]]:
    logger = TraceLogger()
    user = current_user(state, policy)
    logger.log("user_input", "收到用户输入", {"input": user_input, "user": user})
    try:
        if user_input.startswith(":login"):
            parts = user_input.split(maxsplit=1)
            if len(parts) != 2:
                raise AuthAgentError("用法：:login alice")
            answer = login(parts[1].strip(), state, policy)
        elif user_input == ":logout":
            state["current_user_id"] = None
            save_user_state(state)
            answer = "已退出登录。"
        elif user_input == ":whoami":
            require_permission(user, "whoami", policy)
            answer = format_user(user)
        elif user_input == ":sessions":
            answer = format_sessions(state, user, policy)
        elif user_input == ":tasks":
            answer = format_tasks(state, user, policy)
        elif user_input == ":outputs":
            answer = format_outputs(state, user, policy)
        elif user_input == ":actions":
            answer = format_pending_actions(state, user, policy)
        elif user_input.startswith(":run-session"):
            session_id = extract_id(user_input, "session")
            if not session_id:
                raise AuthAgentError("用法：:run-session session_001")
            answer = format_task_detail(create_task_from_session(session_id, user, state, policy))
        elif user_input.startswith(":run-task"):
            task_id = extract_id(user_input, "task")
            if not task_id:
                raise AuthAgentError("用法：:run-task task_001")
            task = run_user_task(task_id, user, state, policy)
            answer = format_task_detail(task)
            actions = format_pending_actions(state, user, policy)
            if "action_" in actions:
                answer += "\n\n待确认动作：\n" + actions
        elif user_input.startswith(":confirm"):
            action_id = extract_id(user_input, "action")
            if not action_id:
                raise AuthAgentError("用法：:confirm action_001 yes")
            decision = user_input.split(action_id, 1)[1].strip() or "yes"
            answer = confirm_user_action(action_id, decision, user, state, policy)
        elif user_input == ":clear-own":
            require_permission(user, "clear_own_state", policy)
            if not user["id"]:
                raise AuthAgentError("请先登录。")
            state["sessions"] = [item for item in state["sessions"] if item["owner_user_id"] != user["id"]]
            state["tasks"] = [item for item in state["tasks"] if item["owner_user_id"] != user["id"]]
            state["pending_actions"] = [item for item in state["pending_actions"] if item["owner_user_id"] != user["id"]]
            state["outputs"] = [item for item in state["outputs"] if item["owner_user_id"] != user["id"]]
            save_user_state(state)
            answer = f"已清空 {user['id']} 的数据。"
        elif user_input == ":clear-all":
            require_permission(user, "clear_all_state", policy)
            save_user_state(empty_user_state())
            answer = "已清空所有用户数据。"
        else:
            open_session = latest_open_session(state, user, policy)
            if open_session and not any(word in user_input for word in ["帮我", "准备", "生成"]):
                answer = update_user_session(open_session, user_input, user, state, policy)
            else:
                session = create_user_session(user_input, user, state, policy)
                if session["missing_fields"]:
                    answer = f"已创建 {session['id']}。还需要确认：{'、'.join(session['missing_fields'])}。"
                else:
                    answer = f"已创建 {session['id']}，可以输入 :run-session {session['id']}。"
        logger.log("final_answer", "返回权限 Agent 结果", {"answer": answer, "user": current_user(load_user_state(), policy)})
        return answer, logger.get_events()
    except Exception as error:
        logger.log("error", "权限 Agent 操作失败", {"error_type": type(error).__name__, "error": str(error), "user": user})
        answer = f"操作失败：{error}"
        logger.log("final_answer", "返回错误结果", {"answer": answer})
        return answer, logger.get_events()


def print_trace_summary(events: list[TraceEvent]) -> None:
    if not events:
        print("Trace：无事件。")
        return
    print(f"\nTrace Summary：{events[0]['trace_id']}")
    for event in events:
        print(f"{event['step']}. {event['event_type']} - {event['message']}")
    print("完整 JSONL 已写入：agent_traces.jsonl")


def main() -> None:
    policy = load_auth_policy()
    print("Day 24 Auth Agent")
    print("命令：:login alice、:logout、:whoami、:sessions、:tasks、:outputs、:actions、:run-session session_001、:run-task task_001、:confirm action_001 yes/no、:clear-own、:clear-all、trace、:traces、:clear-traces、exit")
    latest_events: list[TraceEvent] = []

    while True:
        try:
            user_input = input("\n你：").strip()
        except EOFError:
            print("\n已退出。")
            break
        if user_input.lower() in {"exit", "quit", "退出"}:
            print("Agent：已退出。第 25 天可以学习部署与运行。")
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
        if user_input == ":clear-user-state":
            clear_user_state()
            continue
        if not user_input:
            print("Agent：请输入命令，例如 :login alice 或 :whoami。")
            continue
        answer, latest_events = handle_auth_input(user_input, load_user_state(), policy)
        print("\nAgent：")
        print(answer)
        print_trace_summary(latest_events)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出。")
        sys.exit(0)
