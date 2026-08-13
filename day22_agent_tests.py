"""
Day 22: Agent Test Runner

Run:
    python3 day22_agent_tests.py

Files:
    agent_test_cases.json     Test cases written as data.
    day21_agent_workbench.py  System under test.
    workbench_state.json      State reset before every case.
    agent_traces.jsonl        Trace reset before every case.

Today's key idea:
    Agent testing is not asking "does the answer sound right?" It verifies
    decisions, tools, state, safety, trace, and outputs.

This runner uses only the Python standard library. It imports the Day 21
workbench directly so it can inspect state after every step.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from day12_observable_agent import TRACE_FILE, load_trace_events
from day21_agent_workbench import (
    empty_state,
    handle_workbench_input,
    load_state,
    save_state,
    summarize_task,
)


TEST_CASE_FILE = Path(__file__).with_name("agent_test_cases.json")


class TestStep(TypedDict, total=False):
    input: str
    expect: dict[str, Any]


class TestCase(TypedDict):
    id: str
    name: str
    type: str
    steps: list[TestStep]


@dataclass
class TestResult:
    case_id: str
    name: str
    passed: bool
    failures: list[str]
    steps_run: int


def load_test_cases(path: Path = TEST_CASE_FILE) -> list[TestCase]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"找不到测试用例文件：{path}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{path.name} 不是合法 JSON：{error}") from error

    if not isinstance(data, list):
        raise RuntimeError(f"{path.name} 必须是 JSON 数组。")
    return data  # type: ignore[return-value]


def reset_test_environment() -> None:
    """
    Reset state before every test case.

    Teaching point:
    Agent tests must be isolated. A session, task, pending action, or output
    from a previous case should never affect the next case.
    """
    save_state(empty_state())
    TRACE_FILE.write_text("", encoding="utf-8")


def run_test_case(test_case: TestCase) -> TestResult:
    reset_test_environment()
    failures: list[str] = []
    latest_answer = ""
    latest_events: list[dict[str, Any]] = []
    steps_run = 0

    for index, step in enumerate(test_case.get("steps", []), start=1):
        steps_run += 1
        user_input = step["input"]
        state = load_state()
        answer, events = handle_workbench_input(user_input, state)
        latest_answer = answer
        latest_events = list(events)

        expect = step.get("expect")
        if expect:
            step_failures = assert_expectations(
                expect=expect,
                answer=answer,
                state=load_state(),
                traces=load_trace_events(),
                latest_events=latest_events,
            )
            failures.extend(f"step {index}: {failure}" for failure in step_failures)

    final_expect = test_case.get("expect_final")  # type: ignore[attr-defined]
    if final_expect:
        failures.extend(
            assert_expectations(
                expect=final_expect,
                answer=latest_answer,
                state=load_state(),
                traces=load_trace_events(),
                latest_events=latest_events,
            )
        )

    return TestResult(
        case_id=test_case["id"],
        name=test_case["name"],
        passed=not failures,
        failures=failures,
        steps_run=steps_run,
    )


def assert_expectations(
    expect: dict[str, Any],
    answer: str,
    state: dict[str, Any],
    traces: list[dict[str, Any]],
    latest_events: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    failures.extend(assert_output_contains(expect.get("output_contains", []), answer))
    failures.extend(assert_output_not_contains(expect.get("output_not_contains", []), answer))
    failures.extend(assert_state_expectations(expect.get("state", {}), state))
    failures.extend(assert_trace_expectations(expect, traces, latest_events))
    return failures


def assert_output_contains(expected_texts: list[str], answer: str) -> list[str]:
    failures = []
    for text in expected_texts:
        if text not in answer:
            failures.append(f"expected output to contain {text!r}, actual={answer[:180]!r}")
    return failures


def assert_output_not_contains(unexpected_texts: list[str], answer: str) -> list[str]:
    failures = []
    for text in unexpected_texts:
        if text in answer:
            failures.append(f"expected output not to contain {text!r}")
    return failures


def assert_state_expectations(expected_state: dict[str, Any], state: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key, expected in expected_state.items():
        actual = read_state_value(key, state)
        if key.endswith("_contains"):
            if not isinstance(actual, list):
                failures.append(f"expected {key} source to be list, actual={actual!r}")
                continue
            missing = [item for item in expected if item not in actual]
            if missing:
                failures.append(f"expected {key} to contain {missing!r}, actual={actual!r}")
            continue
        if isinstance(expected, dict) and isinstance(actual, dict):
            missing_pairs = {
                key: value
                for key, value in expected.items()
                if actual.get(key) != value
            }
            if missing_pairs:
                failures.append(f"expected state.{key} to include {missing_pairs!r}, actual={actual!r}")
            continue
        if actual != expected:
            failures.append(f"expected state.{key}={expected!r}, actual={actual!r}")
    return failures


def read_state_value(key: str, state: dict[str, Any]) -> Any:
    sessions = state.get("sessions", [])
    tasks = state.get("tasks", [])
    outputs = state.get("outputs", [])
    pending_action = state.get("pending_action")
    latest_session = sessions[-1] if sessions else {}
    latest_task = tasks[-1] if tasks else {}
    latest_output = outputs[-1] if outputs else {}

    values = {
        "sessions_count": len(sessions),
        "tasks_count": len(tasks),
        "outputs_count": len(outputs),
        "pending_action": pending_action,
        "pending_action_tool": pending_action.get("tool") if isinstance(pending_action, dict) else None,
        "pending_action_risk": pending_action.get("risk") if isinstance(pending_action, dict) else None,
        "latest_session_status": latest_session.get("status"),
        "latest_session_task_id": latest_session.get("task_id"),
        "latest_session_missing_fields": latest_session.get("missing_fields", []),
        "latest_session_missing_fields_contains": latest_session.get("missing_fields", []),
        "latest_task_status": latest_task.get("status"),
        "latest_task_steps_count": len(latest_task.get("steps", [])) if isinstance(latest_task, dict) else 0,
        "latest_task_progress": summarize_task(latest_task).get("progress") if latest_task else "0/0",
        "latest_output_id": latest_output.get("id"),
    }

    if key == "latest_session_context":
        return latest_session.get("collected_context", {})
    if key.startswith("latest_session_context."):
        field_name = key.split(".", 1)[1]
        return latest_session.get("collected_context", {}).get(field_name)
    return values.get(key)


def assert_trace_expectations(
    expect: dict[str, Any],
    traces: list[dict[str, Any]],
    latest_events: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    all_event_types = [event.get("event_type") for event in traces]
    latest_event_types = [event.get("event_type") for event in latest_events]

    for event_type in expect.get("trace_event_types_contains", []):
        if event_type not in all_event_types and event_type not in latest_event_types:
            failures.append(f"expected trace event {event_type!r}, actual_latest={latest_event_types!r}")

    for event_type in expect.get("latest_trace_event_types_contains", []):
        if event_type not in latest_event_types:
            failures.append(f"expected latest trace event {event_type!r}, actual_latest={latest_event_types!r}")
    return failures


def print_test_report(results: list[TestResult]) -> None:
    passed = sum(1 for result in results if result.passed)
    total = len(results)
    print(f"\nAgent Test Report: {passed}/{total} passed")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.case_id} {result.name} ({result.steps_run} steps)")
        for failure in result.failures:
            print(f"  - {failure}")


def main() -> None:
    try:
        test_cases = load_test_cases()
        results = [run_test_case(test_case) for test_case in test_cases]
    except Exception as error:
        print(f"测试运行器错误：{error}")
        sys.exit(2)

    print_test_report(results)
    reset_test_environment()
    if not all(result.passed for result in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
