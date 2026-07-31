"""
Day 18: Agent API Service

Run with the stdlib fallback server:
    python3 day18_agent_api_service.py

If FastAPI and uvicorn are installed, you can also run:
    uvicorn day18_agent_api_service:app --reload

Try:
    curl http://127.0.0.1:8000/health
    curl -X POST http://127.0.0.1:8000/tasks \
      -H 'Content-Type: application/json' \
      -d '{"goal":"整理第 9-10 天 RAG 学习笔记"}'
    curl http://127.0.0.1:8000/tasks
    curl -X POST http://127.0.0.1:8000/tasks/task_001/run

Today's key idea:
    API is not the Agent core. API is the entrance to the Agent core.

This file intentionally reuses day17_task_queue_agent.py instead of copying the
task queue logic into route handlers. That separation is the main lesson:

    HTTP API layer -> validate request -> call Agent core -> return JSON

If FastAPI is unavailable, this file still runs with Python's built-in
http.server so students can learn API design before installing dependencies.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from day12_observable_agent import TraceLogger, load_trace_events
from day17_task_queue_agent import (
    AgentTask,
    TaskQueueAgentError,
    advance_task,
    cancel_task,
    create_task,
    find_task,
    load_tasks,
    summarize_task,
)


SERVICE_NAME = "day18_agent_api_service"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_GOAL_LENGTH = 500


def task_detail(task: AgentTask) -> dict[str, Any]:
    """Return the full task object plus a compact progress summary."""
    return {
        **task,
        "summary": summarize_task(task),
    }


def ok_response(message: str, **payload: Any) -> dict[str, Any]:
    return {
        "ok": True,
        "message": message,
        **payload,
    }


def error_response(error_type: str, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "type": error_type,
            "message": message,
        },
    }


def validate_goal(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("goal 必须是字符串。")
    goal = value.strip()
    if not goal:
        raise ValueError("goal 不能为空。")
    if len(goal) > MAX_GOAL_LENGTH:
        raise ValueError(f"goal 不能超过 {MAX_GOAL_LENGTH} 个字符。")
    return goal


def api_health() -> tuple[int, dict[str, Any]]:
    return 200, ok_response(
        "服务运行中",
        service=SERVICE_NAME,
        status="running",
        endpoints=[
            "GET /health",
            "POST /tasks",
            "GET /tasks",
            "GET /tasks/{task_id}",
            "POST /tasks/{task_id}/run",
            "POST /tasks/{task_id}/resume",
            "POST /tasks/{task_id}/cancel",
            "GET /traces",
        ],
    )


def api_create_task(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    try:
        goal = validate_goal(payload.get("goal"))
    except ValueError as error:
        return 400, error_response("BadRequest", str(error))

    logger = TraceLogger()
    logger.log("api_request", "收到创建任务请求", {"endpoint": "POST /tasks", "goal": goal})
    task = create_task(goal, logger)
    logger.log("api_response", "创建任务接口返回成功", {"task": summarize_task(task)})
    return 201, ok_response("任务已创建", task=task_detail(task), trace_id=logger.trace_id)


def api_list_tasks() -> tuple[int, dict[str, Any]]:
    tasks = load_tasks()
    return 200, ok_response(
        "任务列表",
        tasks=[summarize_task(task) for task in tasks],
        count=len(tasks),
    )


def api_get_task(task_id: str) -> tuple[int, dict[str, Any]]:
    try:
        task = find_task(load_tasks(), task_id)
    except TaskQueueAgentError as error:
        return 404, error_response("TaskNotFound", str(error))
    return 200, ok_response("任务详情", task=task_detail(task))


def api_run_task(task_id: str, resume: bool = False) -> tuple[int, dict[str, Any]]:
    logger = TraceLogger()
    endpoint = f"POST /tasks/{task_id}/{'resume' if resume else 'run'}"
    logger.log("api_request", "收到任务推进请求", {"endpoint": endpoint, "task_id": task_id, "resume": resume})
    try:
        task = advance_task(task_id, logger, resume=resume)
    except TaskQueueAgentError as error:
        logger.log("api_error", "任务推进请求失败", {"task_id": task_id, "error": str(error)})
        status_code = 404 if "没有找到任务" in str(error) else 400
        return status_code, error_response(type(error).__name__, str(error))

    logger.log("api_response", "任务推进接口返回成功", {"task": summarize_task(task)})
    return 200, ok_response("任务已推进" if not resume else "任务已恢复并推进", task=task_detail(task), trace_id=logger.trace_id)


def api_cancel_task(task_id: str) -> tuple[int, dict[str, Any]]:
    logger = TraceLogger()
    logger.log("api_request", "收到任务取消请求", {"endpoint": f"POST /tasks/{task_id}/cancel", "task_id": task_id})
    try:
        task = cancel_task(task_id, logger)
    except TaskQueueAgentError as error:
        logger.log("api_error", "任务取消请求失败", {"task_id": task_id, "error": str(error)})
        status_code = 404 if "没有找到任务" in str(error) else 400
        return status_code, error_response(type(error).__name__, str(error))

    logger.log("api_response", "任务取消接口返回成功", {"task": summarize_task(task)})
    return 200, ok_response("任务已取消", task=task_detail(task), trace_id=logger.trace_id)


def api_list_traces(limit: int = 20) -> tuple[int, dict[str, Any]]:
    events = load_trace_events()
    recent_events = events[-limit:]
    return 200, ok_response("最近 Trace 事件", traces=recent_events, count=len(recent_events))


def route_request(method: str, path: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    """
    A tiny router shared by the stdlib server.

    FastAPI users will normally use the route decorators below. This router is
    here only to keep the lesson runnable without external dependencies.
    """
    payload = payload or {}
    parts = [part for part in path.split("/") if part]

    if method == "GET" and path == "/health":
        return api_health()
    if method == "POST" and path == "/tasks":
        return api_create_task(payload)
    if method == "GET" and path == "/tasks":
        return api_list_tasks()
    if method == "GET" and path == "/traces":
        return api_list_traces()

    if len(parts) == 2 and parts[0] == "tasks" and method == "GET":
        return api_get_task(parts[1])
    if len(parts) == 3 and parts[0] == "tasks" and method == "POST":
        task_id = parts[1]
        action = parts[2]
        if action == "run":
            return api_run_task(task_id, resume=False)
        if action == "resume":
            return api_run_task(task_id, resume=True)
        if action == "cancel":
            return api_cancel_task(task_id)

    return 404, error_response("NotFound", f"未知接口：{method} {path}")


class AgentAPIHandler(BaseHTTPRequestHandler):
    """Small stdlib HTTP handler used when FastAPI is not installed."""

    server_version = "Day18AgentAPI/1.0"

    def do_GET(self) -> None:  # noqa: N802 - http.server expects this name.
        self.handle_api_request()

    def do_POST(self) -> None:  # noqa: N802 - http.server expects this name.
        self.handle_api_request()

    def handle_api_request(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self.read_json_body()
        except ValueError as error:
            self.write_json(400, error_response("BadRequest", str(error)))
            return

        try:
            status_code, response = route_request(self.command, parsed.path, payload)
        except Exception as error:
            status_code = 500
            response = error_response(type(error).__name__, str(error))
        self.write_json(status_code, response)

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"请求体不是合法 JSON：{error}") from error
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象。")
        return payload

    def write_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_text: str, *args: Any) -> None:
        # Keep the teaching output calm; API responses are enough for this demo.
        del format_text, args


try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True

    class CreateTaskRequest(BaseModel):
        goal: str = Field(..., min_length=1, max_length=MAX_GOAL_LENGTH)

    app = FastAPI(title="Day 18 Agent API Service", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return api_health()[1]

    @app.post("/tasks", status_code=201)
    def create_task_endpoint(request: CreateTaskRequest) -> dict[str, Any]:
        return api_create_task({"goal": request.goal})[1]

    @app.get("/tasks")
    def list_tasks_endpoint() -> dict[str, Any]:
        return api_list_tasks()[1]

    @app.get("/tasks/{task_id}")
    def get_task_endpoint(task_id: str) -> dict[str, Any]:
        status_code, response = api_get_task(task_id)
        if status_code >= 400:
            raise HTTPException(status_code=status_code, detail=response["error"])
        return response

    @app.post("/tasks/{task_id}/run")
    def run_task_endpoint(task_id: str) -> dict[str, Any]:
        status_code, response = api_run_task(task_id, resume=False)
        if status_code >= 400:
            raise HTTPException(status_code=status_code, detail=response["error"])
        return response

    @app.post("/tasks/{task_id}/resume")
    def resume_task_endpoint(task_id: str) -> dict[str, Any]:
        status_code, response = api_run_task(task_id, resume=True)
        if status_code >= 400:
            raise HTTPException(status_code=status_code, detail=response["error"])
        return response

    @app.post("/tasks/{task_id}/cancel")
    def cancel_task_endpoint(task_id: str) -> dict[str, Any]:
        status_code, response = api_cancel_task(task_id)
        if status_code >= 400:
            raise HTTPException(status_code=status_code, detail=response["error"])
        return response

    @app.get("/traces")
    def traces_endpoint() -> dict[str, Any]:
        return api_list_traces()[1]

except ModuleNotFoundError:
    FASTAPI_AVAILABLE = False
    app = None


def run_stdlib_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    server = ThreadingHTTPServer((host, port), AgentAPIHandler)
    print(f"Day 18 Agent API Service running at http://{host}:{port}")
    print("FastAPI 未安装，当前使用 Python 标准库 HTTP 服务。")
    print("可用接口：GET /health, POST /tasks, GET /tasks, GET /tasks/task_001, POST /tasks/task_001/run")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
    finally:
        server.server_close()


def main() -> None:
    if FASTAPI_AVAILABLE:
        print("FastAPI 已安装。推荐运行：uvicorn day18_agent_api_service:app --reload")
        print("如果你直接运行本文件，也会启动标准库 fallback 服务用于学习。")
    run_stdlib_server()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出。")
        sys.exit(0)
