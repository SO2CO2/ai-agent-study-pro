"""
Day 23: Configurable Agent

Run:
    python3 day23_configurable_agent.py

Try:
    :config
    :validate-config
    帮我准备 Agent 第三周复习材料
    面向新手，用 Markdown，重点讲项目实践
    :run-session session_001
    :run-task task_001

Files:
    agent_config.json      Non-secret runtime configuration.
    .env.example           Example environment variables, no real secrets.
    workbench_state.json   State used by the Day 21 workbench.

Today's key idea:
    Config management is not moving constants to another file. It separates
    environment-specific values and secrets from stable Agent logic.

This script wraps day21_agent_workbench.py instead of rewriting it. The wrapper
loads config, validates it, masks secrets, and enforces selected runtime limits
before delegating to the Agent core.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, TypedDict

from day12_observable_agent import TRACE_FILE, TraceEvent, clear_traces, print_recent_traces
from day21_agent_workbench import (
    WorkbenchState,
    extract_id,
    find_task,
    format_outputs,
    format_sessions,
    format_tasks,
    handle_workbench_input,
    load_state,
)


CONFIG_FILE = Path(__file__).with_name(os.environ.get("AGENT_CONFIG_FILE", "agent_config.json"))
ENV_EXAMPLE_FILE = Path(__file__).with_name(".env.example")


class ConfigError(Exception):
    """A readable configuration error for the Day 23 learning script."""


class EnvInfo(TypedDict):
    app_env: str
    openai_api_key: str | None
    config_file: str


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"找不到配置文件：{path}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"{path.name} 不是合法 JSON：{error}") from error
    if not isinstance(data, dict):
        raise ConfigError("配置文件必须是 JSON 对象。")
    return data


def load_env() -> EnvInfo:
    """
    Read environment variables used by the Agent.

    Secret rule:
    Real secrets come from the environment, not from agent_config.json. The
    committed .env.example only documents names and expected shape.
    """
    return {
        "app_env": os.environ.get("APP_ENV", "development"),
        "openai_api_key": os.environ.get("OPENAI_API_KEY"),
        "config_file": str(CONFIG_FILE.name),
    }


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    require_object(config, "model", errors)
    require_object(config, "paths", errors)
    require_object(config, "features", errors)
    require_object(config, "safety", errors)
    if errors:
        return errors

    model = config["model"]
    paths = config["paths"]
    features = config["features"]
    safety = config["safety"]

    require_string(model, "chat_model", "model.chat_model", errors)
    require_string(model, "embedding_model", "model.embedding_model", errors)
    require_number_range(model, "temperature", "model.temperature", 0, 2, errors)
    require_positive_int(model, "max_output_tokens", "model.max_output_tokens", errors)

    for key in ["state_file", "trace_file", "knowledge_base_dir", "rag_index_file"]:
        require_string(paths, key, f"paths.{key}", errors)

    for key in ["enable_trace", "enable_human_confirmation", "enable_rag", "enable_memory", "enable_write_tools"]:
        require_bool(features, key, f"features.{key}", errors)

    require_string_list(safety, "blocked_tools", "safety.blocked_tools", errors)
    require_string_list(safety, "high_risk_tools", "safety.high_risk_tools", errors)
    require_positive_int(safety, "max_goal_length", "safety.max_goal_length", errors)
    require_positive_int(safety, "max_steps_per_task", "safety.max_steps_per_task", errors)
    require_bool(safety, "require_confirmation_for_write", "safety.require_confirmation_for_write", errors)

    # Safety baseline: high-risk write tools must remain high-risk even if a
    # local config disables confirmation elsewhere.
    if "write_final_report" not in safety.get("high_risk_tools", []):
        errors.append("safety.high_risk_tools 必须包含 write_final_report。")
    return errors


def require_object(config: dict[str, Any], key: str, errors: list[str]) -> None:
    if not isinstance(config.get(key), dict):
        errors.append(f"{key} 必须是对象。")


def require_string(config: dict[str, Any], key: str, label: str, errors: list[str]) -> None:
    if not isinstance(config.get(key), str) or not config[key].strip():
        errors.append(f"{label} 必须是非空字符串。")


def require_bool(config: dict[str, Any], key: str, label: str, errors: list[str]) -> None:
    if not isinstance(config.get(key), bool):
        errors.append(f"{label} 必须是布尔值。")


def require_string_list(config: dict[str, Any], key: str, label: str, errors: list[str]) -> None:
    value = config.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        errors.append(f"{label} 必须是字符串数组。")


def require_positive_int(config: dict[str, Any], key: str, label: str, errors: list[str]) -> None:
    value = config.get(key)
    if not isinstance(value, int) or value <= 0:
        errors.append(f"{label} 必须是正整数。")


def require_number_range(
    config: dict[str, Any],
    key: str,
    label: str,
    minimum: float,
    maximum: float,
    errors: list[str],
) -> None:
    value = config.get(key)
    if not isinstance(value, (int, float)) or not minimum <= value <= maximum:
        errors.append(f"{label} 必须在 {minimum} 到 {maximum} 之间。")


def mask_secret(value: str | None) -> str:
    if not value:
        return "missing"
    if len(value) <= 8:
        return "present"
    return f"{value[:3]}***{value[-4:]}"


def get_config_summary(config: dict[str, Any], env: EnvInfo) -> str:
    model = config["model"]
    features = config["features"]
    safety = config["safety"]
    paths = config["paths"]
    lines = [
        "Config Summary:",
        f"- environment: {config.get('environment', env['app_env'])}",
        f"- APP_ENV: {env['app_env']}",
        f"- config_file: {env['config_file']}",
        f"- chat_model: {model['chat_model']}",
        f"- embedding_model: {model['embedding_model']}",
        f"- temperature: {model['temperature']}",
        f"- max_output_tokens: {model['max_output_tokens']}",
        f"- trace: {'enabled' if features['enable_trace'] else 'disabled'} ({paths['trace_file']})",
        f"- rag: {'enabled' if features['enable_rag'] else 'disabled'}",
        f"- memory: {'enabled' if features['enable_memory'] else 'disabled'}",
        f"- human_confirmation: {'enabled' if features['enable_human_confirmation'] else 'enabled for high-risk baseline'}",
        f"- write_tools: {'enabled' if features['enable_write_tools'] else 'disabled'}",
        f"- max_goal_length: {safety['max_goal_length']}",
        f"- max_steps_per_task: {safety['max_steps_per_task']}",
        f"- blocked_tools: {', '.join(safety['blocked_tools']) or 'none'}",
        f"- high_risk_tools: {', '.join(safety['high_risk_tools']) or 'none'}",
        f"- OPENAI_API_KEY: {mask_secret(env['openai_api_key'])}",
    ]
    return "\n".join(lines)


def is_feature_enabled(config: dict[str, Any], feature_name: str) -> bool:
    return bool(config.get("features", {}).get(feature_name))


def is_tool_allowed(config: dict[str, Any], tool_name: str) -> tuple[bool, str]:
    safety = config["safety"]
    features = config["features"]
    if tool_name in safety["blocked_tools"]:
        return False, f"工具 {tool_name} 被 safety.blocked_tools 禁用。"
    if tool_name.startswith("write_") and not features["enable_write_tools"]:
        return False, f"写入类工具 {tool_name} 被 features.enable_write_tools=false 禁用。"
    return True, "allowed"


def classify_tool_with_config(config: dict[str, Any], tool_name: str) -> str:
    if tool_name in config["safety"]["blocked_tools"]:
        return "blocked"
    if tool_name in config["safety"]["high_risk_tools"]:
        return "high"
    return "low"


def enforce_runtime_limits(config: dict[str, Any], state: WorkbenchState, user_input: str) -> tuple[bool, str]:
    safety = config["safety"]
    if not user_input.startswith(":") and len(user_input) > safety["max_goal_length"]:
        return False, f"用户目标过长：最多 {safety['max_goal_length']} 个字符。"

    if user_input.startswith(":run-session"):
        # The Day 21 workbench creates a 5-step task. This check demonstrates
        # how a config wrapper can reject a run before creating too-large tasks.
        expected_steps = 5
        if expected_steps > safety["max_steps_per_task"]:
            return False, f"任务步骤数 {expected_steps} 超过 max_steps_per_task={safety['max_steps_per_task']}。"

    if user_input.startswith(":run-task"):
        task_id = extract_id(user_input, "task")
        if not task_id:
            return True, "ok"
        try:
            task = find_task(state, task_id)
        except Exception as error:
            return False, str(error)
        step = next((item for item in task["steps"] if item["status"] in {"pending", "failed"}), None)
        if not step:
            return True, "ok"
        allowed, reason = is_tool_allowed(config, step["tool"])
        if not allowed:
            return False, reason
        risk = classify_tool_with_config(config, step["tool"])
        if risk == "high" and not config["safety"]["require_confirmation_for_write"]:
            return False, "高风险工具不能通过配置关闭确认边界。"
    return True, "ok"


def run_configured_agent(user_input: str, config: dict[str, Any]) -> tuple[str, list[TraceEvent]]:
    state = load_state()
    ok, reason = enforce_runtime_limits(config, state, user_input)
    if not ok:
        return f"配置策略拒绝执行：{reason}", []

    answer, events = handle_workbench_input(user_input, state)
    if not is_feature_enabled(config, "enable_trace"):
        TRACE_FILE.write_text("", encoding="utf-8")
        return answer + "\n\nTrace 已按配置关闭，本次不保留 agent_traces.jsonl。", []
    return answer, events


def print_trace_summary(events: list[TraceEvent]) -> None:
    if not events:
        print("Trace：无事件，或已被配置关闭。")
        return
    print(f"\nTrace Summary：{events[0]['trace_id']}")
    for event in events:
        print(f"{event['step']}. {event['event_type']} - {event['message']}")
    print("完整 JSONL 已写入：agent_traces.jsonl")


def main() -> None:
    try:
        config = load_config()
        env = load_env()
        errors = validate_config(config)
        if errors:
            raise ConfigError("配置校验失败：\n- " + "\n- ".join(errors))
    except ConfigError as error:
        print(f"启动失败：{error}")
        sys.exit(1)

    print("Day 23 Configurable Agent")
    print("命令：:config、:validate-config、:sessions、:tasks、:outputs、:clear-workbench、trace、:traces、:clear-traces、exit")
    print(get_config_summary(config, env))
    latest_events: list[TraceEvent] = []

    while True:
        try:
            user_input = input("\n你：").strip()
        except EOFError:
            print("\n已退出。")
            break

        if user_input.lower() in {"exit", "quit", "退出"}:
            print("Agent：已退出。第 24 天可以学习权限、鉴权与用户隔离。")
            break
        if user_input == ":config":
            print(get_config_summary(config, load_env()))
            continue
        if user_input == ":validate-config":
            errors = validate_config(load_config())
            if errors:
                print("配置校验失败：")
                for error in errors:
                    print(f"- {error}")
            else:
                print("配置校验通过。")
            continue
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
        if user_input == ":sessions":
            print(format_sessions(load_state()))
            continue
        if user_input == ":tasks":
            print(format_tasks(load_state()))
            continue
        if user_input == ":outputs":
            print(format_outputs(load_state()))
            continue
        if user_input == ":clear-workbench":
            from day21_agent_workbench import clear_workbench

            clear_workbench()
            continue
        if not user_input:
            print("Agent：请输入目标，例如：帮我准备 Agent 第三周复习材料")
            continue

        answer, latest_events = run_configured_agent(user_input, config)
        print("\nAgent：")
        print(answer)
        print_trace_summary(latest_events)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出。")
        sys.exit(0)
