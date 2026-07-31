# Agent Reliability Notes

A reliable Agent combines capability, safety, observability, and resilience.

Capability means the Agent can complete useful work, such as retrieving evidence, calling tools, or answering a user's question.

Safety means the Agent treats user input, retrieved documents, memory, and model output as untrusted until program code validates them. A model suggestion is not permission to execute a tool.

Observability means the Agent records important events with a trace_id. A useful trace includes user_input, safety_check, documents_loaded, retrieval_result, evidence_risk_check, fallback, final_answer, and error events.

Resilience means the Agent handles failure gracefully. It can retry temporary failures, fall back to a simpler path, produce a partial answer when appropriate, or safely stop when there is not enough reliable information.

Trace is important because an Agent answer depends on many intermediate steps. If the final answer is wrong, Trace helps locate whether the problem happened during safety checking, retrieval, evidence filtering, or final answer generation.
