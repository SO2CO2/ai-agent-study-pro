# Agent 开发课程大纲

## 课程定位

这是一门面向新手的 Agent 开发入门课程。课程不会一开始就进入复杂框架，而是先从底层机制开始，逐步建立 Agent 的核心心智模型。

学习路径：

```text
规则版助手
-> 意图路由
-> Tool Calling / Function Calling
-> 多步骤任务 Agent
-> 记忆、RAG、评估、部署
```

课程目标不是让学生只会调用某个 Agent 框架，而是让学生理解：Agent 框架底层到底在帮我们做什么。

## 主学习路径

```text
第 1 天：手写规则判断
第 2 天：结构化意图识别
第 3 天：Tool Calling / Function Calling
第 4 天：带任务状态的多步骤 Agent
第 5 天：真实 LLM Tool Calling
第 6 天：真实 LLM Planner 与多步骤 Agent
第 7 天：Agent 记忆与上下文管理
第 8 天：Embedding 与语义记忆检索
第 9 天：RAG 与外部文档问答
第 10 天：RAG 质量评估与优化
第 11 天：Agent 安全边界与 Prompt Injection 防护
第 12 天：Agent 日志、Trace 与可观测性
第 13 天：Agent 错误处理、重试与降级
第 14 天：第二周综合项目：可靠知识库 Agent
第 15 天：多工具组合 Agent
第 16 天：人类确认与高风险动作控制
第 17 天：任务队列与长任务状态
第 18 天：Agent API 服务封装
第 19 天：Agent 前端操作界面
第 20 天：多轮工作流 Agent
第 21 天：第三周综合项目：Agent 工作台
第 22 天：Agent 测试体系
第 23 天：配置与环境管理
第 24 天：权限、鉴权与用户隔离
```

前几天的基础模型可以总结为：

```text
Agent = 目标 + 状态 + 工具 + 决策循环 + 最终回答
```

---

## 第 1 天：Agent 最小模型

### 主题

理解最小可运行的 Agent：

```text
Agent = 判断流程 + 工具 + 执行结果
```

### 学习目标

- 理解 Agent 和普通聊天机器人的区别。
- 理解工具为什么重要。
- 使用简单关键词规则判断是否调用工具。
- 构建一个最小天气助手。

### 核心概念

- Prompt
- Tool
- 简单判断流程
- 工具执行
- 最终回答

### 练习建议

构建一个规则版天气助手：

```text
用户询问天气
-> 程序检测关键词
-> 调用 get_weather(city)
-> 返回天气结果
```

### 代码产出

建议文件：

```text
day1_rule_based_agent.py
```

第一版可以使用假天气数据和关键词匹配。

### 关键认知

```text
Agent 不只是聊天机器人，它可以根据用户输入，通过工具执行动作。
```

### 明日衔接

第 1 天的 Agent 依赖硬编码关键词，表达能力很弱。第 2 天会把“判断用户要做什么”升级为结构化意图识别。

---

## 第 2 天：意图识别与工具路由

### 主题

从硬编码关键词规则，升级到结构化意图识别。

### 学习目标

- 理解什么是 intent。
- 理解为什么结构化输出很重要。
- 使用类似 JSON 的结构表达模型决策。
- 根据不同 intent 路由到不同工具。
- 支持多个工具。

### 核心概念

- Intent
- 结构化输出
- JSON 决策结果
- 工具路由
- 多工具助手

### 练习建议

构建一个意图路由 Agent：

```text
用户输入
-> 意图判断
-> 路由到 get_weather 或 get_current_time
-> 返回结果
```

结构化决策示例：

```json
{
  "intent": "get_weather",
  "city": "北京"
}
```

### 代码产出

建议文件：

```text
day2_intent_routing_agent.py
```

这个阶段仍然可以使用 fake intent 函数模拟模型判断。

### 关键认知

```text
LLM 负责判断要做什么，程序负责真正执行。
```

### 明日衔接

第 2 天使用的是手动设计的 intent 格式。第 3 天会升级到标准的 Tool Calling / Function Calling：用 schema 描述工具，让模型选择工具和参数。

---

## 第 3 天：Tool Calling / Function Calling

### 主题

让模型根据工具 schema 决定是否调用工具。

### 学习目标

- 理解什么是 Tool Calling / Function Calling。
- 学会用 schema 描述工具。
- 理解工具名、工具说明、参数、必填字段。
- 解析模型返回的 tool call。
- 执行前校验工具参数。
- 执行匹配的 Python 函数。
- 使用工具结果生成最终回答。

### 核心概念

- Tool schema
- Tool call
- Arguments
- 工具参数校验
- 工具执行路由
- 最终回答生成

### 代码产出

已实现文件：

```text
day3_tool_calling_agent.py
```

当前支持工具：

```text
get_weather(city)
get_current_time()
```

### 核心流程

```text
用户输入
-> 模型决定是否调用工具
-> 程序校验 tool call
-> 程序执行真实函数
-> 工具结果转成最终回答
```

### 关键认知

```text
模型不会自己执行工具。模型只是提出工具调用请求，真正执行工具的是程序。
```

### 明日衔接

第 3 天一次只处理一个工具调用。第 4 天会引入 state 和 Plan -> Act -> Observe 循环，让 Agent 能完成多步骤任务。

---

## 第 4 天：多步骤 Agent 与任务状态

### 主题

从“会调用工具的助手”，升级为“能完成任务的 Agent”。

### 学习目标

- 理解单步工具调用和多步骤任务的区别。
- 理解为什么 Agent 需要 state。
- 理解 Plan -> Act -> Observe 循环。
- 学习基础停止条件。
- 理解最终回答应该解决用户目标，而不是简单粘贴工具结果。

### 核心概念

- State
- 用户目标
- 中间结果
- Plan
- Act
- Observe
- 停止条件
- 最大步骤限制
- 最终综合

### 练习建议

构建一个运动建议 Agent：

```text
用户：我今天在北京适合跑步吗？

Plan：需要查询天气
Act：调用 get_weather("北京")
Observe：晴，18-28 度

Plan：需要知道当前时间
Act：调用 get_current_time()
Observe：09:30

Plan：信息足够
Final：给出运动建议
```

### 代码产出

已实现文件：

```text
day4_multi_step_agent.py
```

建议函数结构：

```python
def extract_city(user_input):
    ...

def get_weather(city):
    ...

def get_current_time():
    ...

def plan_next_step(state):
    ...

def execute_step(step, state):
    ...

def should_stop(state):
    ...

def create_final_answer(state):
    ...

def agent(user_input):
    ...
```

### 建议状态结构

```python
state = {
    "user_goal": user_input,
    "city": None,
    "weather": None,
    "current_time": None,
    "steps": [],
    "final_answer": None,
}
```

### 重要教学点

较弱的最终回答：

```text
天气：晴，18-28 度。时间：09:30。
```

更好的最终回答：

```text
今天北京天气晴，温度 18-28 度，现在是上午 9 点半，比较适合跑步。
建议选择阴凉路线，跑 30-45 分钟，并注意补水。
```

### 关键认知

```text
真正的 Agent 会维护状态，围绕目标分步骤行动，并把工具结果转化为有用决策。
```

### 明日衔接

第 4 天引入了多步骤执行。下一步有两个方向：

```text
第 5 天：加入真实 LLM tool calling
```

或者：

```text
第 5 天：加入记忆和上下文
```

为了保持新手友好，本课程第 5 天推荐把 fake LLM 替换成真实 LLM，让学生看到同一套 Tool Calling 架构如何接入真实模型。

---

## 第 5 天：真实 LLM Tool Calling

### 主题

把第 3 天的 `fake_llm()` 替换成真实大模型，让模型根据工具 schema 决定是否调用工具。

### 学习目标

- 理解真实 LLM 在 Agent 中的职责。
- 掌握一次真实 Tool Calling 的完整流程。
- 理解工具 schema 是给模型看的，工具函数是由程序执行的。
- 学会解析模型返回的 `function_call`。
- 学会把工具执行结果作为 `function_call_output` 发回模型。
- 加入基础错误处理：缺少 API key、未知工具、参数错误、API 请求失败。

### 核心概念

- 真实 LLM 调用
- Tool schema
- Function call
- Function call output
- 工具参数校验
- 第二次模型请求
- API key
- 错误处理

### 核心流程

```text
用户输入
-> 用户问题 + 工具列表发给模型
-> 模型返回 function_call
-> 程序校验工具名和参数
-> 程序执行真实工具函数
-> 工具结果作为 function_call_output 发回模型
-> 模型生成最终自然语言回答
```

### 代码产出

已实现文件：

```text
day5_real_llm_tool_calling_agent.py
```

当前支持工具：

```text
get_weather(city)
get_current_time()
```

运行方式：

```bash
export OPENAI_API_KEY="你的 API key"
python3 day5_real_llm_tool_calling_agent.py
```

### 关键认知

```text
Agent 的智能来自模型判断，可靠性来自程序控制。
```

模型可以决定调用什么工具，但不能自己执行工具。真实工具调用、参数校验、错误处理和权限控制都必须由程序负责。

### 明日衔接

第 5 天完成了真实 LLM 的单轮 Tool Calling。第 6 天可以把第 4 天的 `state` 和多步骤循环接入真实 LLM，让模型参与 `plan_next_step()`，从“真实单步工具调用”升级到“真实多步骤 Agent”。

---

## 第 6 天：真实 LLM Planner 与多步骤 Agent

### 主题

把第 4 天的多步骤 `state` 循环和第 5 天的真实 LLM 调用结合起来，让模型根据当前 state 决定下一步动作。

### 学习目标

- 理解什么是 LLM Planner。
- 理解 Planner 和普通聊天回答的区别。
- 学会把当前 `state` 提供给模型。
- 让模型输出结构化 plan，例如 `next_step`、`arguments`、`reason`。
- 校验模型输出后再执行工具。
- 用 `MAX_STEPS`、停止条件和错误处理控制多步骤循环。

### 核心概念

- LLM Planner
- State
- Plan -> Act -> Observe
- 结构化 JSON plan
- Plan 校验
- 工具执行
- 最大步骤限制
- 执行轨迹

### 核心流程

```text
用户输入
-> 初始化 state
-> 把 state 发给 LLM Planner
-> 模型返回下一步 plan
-> 程序校验 plan
-> 程序执行工具或生成最终回答
-> 工具结果写回 state
-> 循环直到完成或达到最大步骤数
```

### 建议代码产出

建议文件：

```text
day6_llm_planner_agent.py
```

建议核心函数：

```python
def build_planner_prompt(state):
    ...

def call_llm_for_plan(prompt):
    ...

def validate_plan(plan, state):
    ...

def execute_step(plan, state):
    ...
```

### 关键认知

```text
LLM Planner 负责决定下一步，程序负责验证和执行下一步。
```

### 明日衔接

第 6 天让真实 LLM 参与多步骤规划。第 7 天可以继续学习记忆和上下文管理，让 Agent 不只完成单次任务，还能记住对话历史和用户偏好。

---

## 第 7 天：Agent 记忆与上下文管理

### 主题

为多步骤 Agent 加入短期记忆和长期记忆，让它能在连续对话或后续任务中使用已有信息。

### 学习目标

- 分清任务状态（State）、会话历史和长期记忆。
- 理解记忆不是完整日志，而是未来可能有用的信息。
- 学会根据当前问题检索相关记忆，而非把全部历史交给模型。
- 使用本地 JSON 文件实现可查看、可更新的最小长期记忆系统。
- 将近期对话和相关记忆加入 Planner 的上下文。

### 核心概念

- 当前任务状态（State）
- 短期记忆（Conversation History）
- 长期记忆（Long-term Memory）
- 记忆提炼与重要性
- 相关性检索
- 上下文长度控制
- 记忆更新、覆盖与删除

### 核心流程

```text
用户输入
-> 读取近期会话和长期记忆
-> 检索与当前问题相关的记忆
-> 组合为 Planner 上下文
-> 规划并执行任务
-> 提炼本轮值得保留的信息
-> 更新长期记忆
```

### 建议代码产出

```text
day7_memory_agent.py
memory.json
```

建议核心函数：

```python
def retrieve_relevant_memories(query, memories):
    ...

def build_agent_context(state, history, memories):
    ...

def extract_memories_from_turn(user_input, final_answer):
    ...
```

### 关键认知

```text
State 记录这次任务进展；History 记录本次会话；Memory 保存未来仍有价值的信息。
```

### 明日衔接

第 7 天先用关键词和重要性完成基础检索。后续可以引入 Embedding 和向量检索，让 Agent 按语义而不是只按字面检索长期记忆。

---

## 第 8 天：Embedding 与语义记忆检索

### 主题

将第 7 天的关键词记忆检索升级为 Embedding 语义检索，让 Agent 能理解不同表达背后的相近含义。

### 学习目标

- 理解 Embedding 是文本含义的向量表示，而不是聊天回答。
- 理解关键词匹配与语义检索的区别和适用边界。
- 学习余弦相似度、Top K 和最低相似度阈值。
- 为长期记忆生成并保存 Embedding。
- 根据用户问题检索最相关的长期记忆，并加入 Planner 上下文。
- 理解语义记忆检索与 RAG 的共同底层流程。

### 核心概念

- Embedding
- 向量（Vector）
- 余弦相似度（Cosine Similarity）
- Top K 检索
- 相似度阈值
- 语义检索
- 向量索引
- 检索增强上下文

### 核心流程

```text
新增长期记忆
-> 为记忆生成 Embedding
-> 保存文本、元数据和向量

用户提问
-> 为问题生成 Embedding
-> 计算与记忆向量的相似度
-> 取 Top K 条相关记忆
-> 加入 Planner 上下文
-> LLM 基于受控上下文规划和回答
```

### 建议代码产出

```text
day8_semantic_memory_agent.py
memory_with_embeddings.json
```

建议核心函数：

```python
def create_embedding(text):
    ...

def cosine_similarity(vector_a, vector_b):
    ...

def index_memory(memory):
    ...

def retrieve_semantic_memories(query, memories, top_k=3):
    ...
```

### 关键认知

```text
Embedding 不负责回答问题；它负责从信息中找出最值得让 LLM 看到的内容。
```

### 明日衔接

第 8 天检索的是用户记忆。第 9 天可以将同一套“切分、向量化、检索、注入上下文”的机制用于外部文档，正式进入 RAG。

---

## 第 9 天：RAG 与外部文档问答

### 主题

将 Embedding 检索用于外部 Markdown 或文本资料，让 Agent 根据可追溯的文档证据回答问题。

### 学习目标

- 区分用户记忆检索与外部知识库 RAG。
- 理解 RAG 的“索引”和“查询”两个阶段。
- 学习文档切分（Chunk）、Chunk Overlap 和元数据的作用。
- 为每个 Chunk 生成 Embedding 并构建本地索引。
- 检索 Top K 相关 Chunk，将其作为受控上下文交给 LLM。
- 要求模型仅基于检索资料回答，并标注来源。

### 核心概念

- RAG（Retrieval-Augmented Generation）
- 外部知识库
- 文档加载
- 文本切分（Chunking）
- Chunk Overlap
- 元数据（source、chunk_index）
- 向量索引
- 证据回答与来源追溯

### 核心流程

```text
外部 Markdown / 文本资料
-> 读取文档
-> 切分为带元数据的 Chunk
-> 为每个 Chunk 生成 Embedding
-> 保存本地 RAG 索引

用户提问
-> 为问题生成 Embedding
-> 检索 Top K 相关 Chunk
-> 将 Chunk 和来源注入回答 Prompt
-> LLM 基于证据回答并标注来源
```

### 建议代码产出

```text
day9_rag_document_agent.py
rag_index.json
knowledge_base/
```

建议核心函数：

```python
def load_documents(directory):
    ...

def split_text_into_chunks(text, chunk_size, overlap):
    ...

def build_rag_index(documents):
    ...

def retrieve_relevant_chunks(query, index, top_k=3):
    ...

def answer_with_rag(query, chunks):
    ...
```

### 关键认知

```text
RAG 的核心是在回答前检索最相关、可追溯的证据，而不是把整份文档塞给模型。
```

### 明日衔接

第 9 天完成最小 RAG 闭环。第 10 天可以学习 RAG 质量评估与优化，判断“是否检索正确”和“回答是否忠于资料”。

---

## 第 10 天：RAG 质量评估与优化

### 主题

从“RAG 能跑起来”升级到“RAG 能被评估、调试和优化”，判断检索结果是否正确、最终回答是否忠于资料。

### 学习目标

- 区分 RAG 的检索质量和生成质量。
- 理解 Hit@K、关键词命中、来源命中等基础评估方式。
- 学会准备一组 RAG 测试问题和期望答案要点。
- 自动检查检索结果是否命中正确文档和关键证据。
- 根据评估结果调整 `top_k`、`min_similarity`、`chunk_size`、`chunk_overlap` 等参数。
- 建立“先看检索，再看回答”的 RAG 调试习惯。

### 核心概念

- RAG Evaluation
- Retrieval Quality
- Generation Quality
- Hit@K
- Evidence Match
- Source Match
- Groundedness
- 测试集
- 参数优化

### 核心流程

```text
准备测试问题
-> 记录期望来源和关键证据词
-> 对每个问题执行 RAG 检索
-> 检查 Top K 结果是否命中正确来源
-> 检查检索内容是否包含关键证据
-> 输出评估报告
-> 根据失败原因优化 RAG 参数或文档结构
```

### 建议代码产出

```text
day10_rag_evaluation.py
rag_eval_questions.json
```

建议核心函数：

```python
def load_eval_questions(path):
    ...

def evaluate_retrieval(question, index):
    ...

def check_source_hit(retrieved_chunks, expected_sources):
    ...

def check_keyword_hit(retrieved_chunks, expected_keywords):
    ...

def print_eval_report(results):
    ...
```

### 关键认知

```text
RAG 的质量不是看回答像不像，而是看证据找得准不准、答案是否忠于证据。
```

### 明日衔接

第 10 天学会了评估和优化 RAG。第 11 天可以继续学习安全边界与防注入，让 Agent 面对不可信文档和用户输入时更稳健。

---

## 第 11 天：Agent 安全边界与 Prompt Injection 防护

### 主题

从“Agent 有能力执行任务”升级到“Agent 只能在程序允许的边界内行动”，重点学习不可信输入、RAG 文档注入和工具权限控制。

### 学习目标

- 理解 Agent 的安全边界。
- 理解 Prompt Injection 如何通过用户输入或外部文档影响模型。
- 学会区分系统指令、用户问题、检索资料和记忆内容。
- 使用工具白名单、参数校验和权限分级限制模型行动。
- 理解高风险工具需要用户确认，不能由模型自行决定执行。
- 学会检测 RAG evidence 中的可疑注入语句。

### 核心概念

- 安全边界
- Prompt Injection
- 不可信输入
- 指令与数据隔离
- 工具白名单
- 参数校验
- 权限分级
- 危险动作确认
- 记忆污染
- RAG 文档注入

### 核心流程

```text
用户输入 / 外部文档
-> 安全检查
-> 检测可疑注入语句
-> 将 evidence 明确标记为数据而不是指令
-> 校验模型建议的工具调用
-> 按风险等级允许、拒绝或要求用户确认
-> 生成安全回答和安全报告
```

### 建议代码产出

```text
day11_safe_rag_agent.py
unsafe_knowledge_base/
```

建议核心函数：

```python
def detect_prompt_injection(text):
    ...

def sanitize_evidence(chunks):
    ...

def build_safe_rag_prompt(question, evidence):
    ...

def classify_tool_risk(tool_name):
    ...

def validate_tool_call(tool_name, arguments):
    ...
```

### 关键认知

```text
Agent 的安全不是靠模型听话，而是靠程序给模型画边界。
```

### 明日衔接

第 11 天让 Agent 有了安全边界。第 12 天可以继续学习日志、Trace 与可观测性，让 Agent 的每一步决策和执行过程都能被调试、复盘和优化。

---

## 第 12 天：Agent 日志、Trace 与可观测性

### 主题

从“Agent 能执行任务”升级到“Agent 的执行过程可观察、可调试、可复盘”，为后续错误处理、重试和部署打基础。

### 学习目标

- 理解日志、Trace 和 Step 的区别。
- 理解为什么多步骤 Agent 比普通程序更需要可观测性。
- 学会为一次 Agent 任务生成唯一 `trace_id`。
- 记录用户输入、Planner 决策、工具调用、工具结果、安全检查、最终回答和错误信息。
- 使用 JSONL 保存可追加、可分析的事件日志。
- 学习基础日志脱敏，避免把 API key、token、password 等敏感信息写进日志。

### 核心概念

- Log
- Trace
- Step
- trace_id
- event_type
- JSONL
- TraceLogger
- 日志脱敏
- 错误事件
- Trace Summary

### 核心流程

```text
用户输入
-> 生成 trace_id
-> 记录 user_input
-> 记录 planner_decision
-> 记录 tool_call
-> 记录 tool_result
-> 记录 safety_check / error
-> 记录 final_answer
-> 写入 agent_traces.jsonl
-> 打印可读 Trace Summary
```

### 建议代码产出

```text
day12_observable_agent.py
agent_traces.jsonl
```

建议核心函数：

```python
class TraceLogger:
    def log(self, event_type, message, data=None):
        ...

def redact_sensitive_data(value):
    ...

def create_trace_id():
    ...

def print_trace_summary(events):
    ...
```

### 关键认知

```text
没有 Trace 的 Agent，是很难调试、评估和信任的 Agent。
```

### 明日衔接

第 12 天让 Agent 的执行过程可见。第 13 天可以继续学习错误处理、重试与降级，让 Agent 在工具失败、API 超时或模型输出异常时仍然保持稳定。

---

## 第 13 天：Agent 错误处理、重试与降级

### 主题

从“Agent 的过程可观察”升级到“Agent 面对失败时能稳定恢复”，学习错误分类、Retry、Fallback、Partial Answer 和 Safe Stop。

### 学习目标

- 理解真实 Agent 中常见的失败来源。
- 学会区分可重试错误、不可重试错误、可降级错误和需要用户补充信息的错误。
- 为工具调用、模型输出、RAG 检索和多步骤循环设计错误处理策略。
- 学会实现带最大次数限制的 retry 机制。
- 学会在主路径失败时切换到 fallback 路径。
- 将错误、重试、降级和最终兜底回答写入 Trace。
- 生成用户友好的错误说明，而不是直接暴露程序异常。

### 核心概念

- Error Handling
- Retry
- Fallback
- Partial Answer
- Safe Stop
- Transient Error
- Fatal Error
- Validation Error
- UserInputError
- MaxStepsError

### 核心流程

```text
执行工具或模型步骤
-> 如果成功，记录结果
-> 如果失败，识别错误类型
-> 可重试错误进入 retry
-> 多次失败后尝试 fallback
-> fallback 成功则生成带说明的回答
-> fallback 失败则安全停止或给出 partial answer
-> 全过程写入 Trace
```

### 建议代码产出

```text
day13_resilient_agent.py
```

建议核心函数：

```python
def retry_tool_call(tool_name, func, logger, max_attempts=3):
    ...

def get_weather_live(city):
    ...

def get_weather_cached(city):
    ...

def get_weather_with_fallback(city, logger):
    ...

def create_final_answer(state):
    ...
```

### 关键认知

```text
可靠 Agent 不是永远不失败，而是失败时知道如何重试、降级、停止，并把过程记录清楚。
```

### 明日衔接

第 13 天补齐了第二周的可靠性能力。第 14 天可以做第二周综合项目，把 RAG、评估、安全、Trace、错误处理整合成一个更完整的可靠知识库 Agent。

---

## 第 14 天：第二周综合项目：可靠知识库 Agent

### 主题

将第二周学到的 RAG、评估意识、安全边界、Trace 可观测性、错误处理与降级整合成一个小型可靠知识库 Agent。

### 学习目标

- 把知识库检索、安全检查、证据回答、Trace 记录和错误处理串成完整流程。
- 理解可靠 Agent 的最小架构：输入层、安全层、检索层、回答层、观测层、错误处理层。
- 学会为综合项目拆分清晰模块，而不是把所有逻辑写在一个函数里。
- 在资料不足时诚实拒答，不编造答案。
- 在检索结果或用户输入包含注入风险时进行拦截或标记。
- 为每次问答输出来源和 Trace Summary。

### 核心概念

- Reliable Agent
- Knowledge Agent
- RAG Pipeline
- Evidence
- Source Citation
- Safety Check
- Evidence Risk Check
- Trace Summary
- Fallback Retrieval
- Grounded Answer

### 核心流程

```text
用户输入
-> 创建 trace_id
-> 用户输入安全检查
-> 加载本地知识库
-> 文档切分与关键词检索
-> 检查 evidence 注入风险
-> 检索为空时执行 fallback 检索
-> 资料不足则诚实拒答
-> 基于 evidence 生成回答
-> 输出来源
-> 记录 Trace Summary
```

### 建议代码产出

```text
day14_reliable_knowledge_agent.py
reliable_knowledge_base/
```

建议核心函数：

```python
def load_documents(directory):
    ...

def retrieve_chunks(query, chunks, top_k=3):
    ...

def fallback_retrieve_chunks(query, chunks, top_k=3):
    ...

def detect_prompt_injection(text):
    ...

def safety_check_user_input(user_input):
    ...

def create_grounded_answer(question, evidence):
    ...
```

### 关键认知

```text
一个可靠 Agent 不是某个单点能力强，而是检索、安全、观测、错误处理这些能力能协同工作。
```

### 明日衔接

第 14 天完成第二周能力闭环。第 15 天可以进入第三周，开始学习更真实的应用形态，例如多工具组合、人类确认、任务队列、API 服务和简单界面。

---

## 第 15 天：多工具组合 Agent

### 主题

从“可靠知识库问答 Agent”升级到“能组合多个工具完成复合任务的 Agent”，学习 Tool Registry、工具调度、状态汇总和副作用记录。

### 学习目标

- 理解多工具 Agent 和单工具调用的区别。
- 学会用 Tool Registry 集中管理工具函数、参数要求、风险等级和副作用。
- 根据用户目标规划多个工具的调用顺序。
- 在每次工具调用前校验工具名和参数。
- 区分只读工具和写入工具，并记录副作用。
- 将多个工具结果写入 state，并生成综合回答。
- 使用 Trace 记录工具规划、校验、调用、结果和状态更新。

### 核心概念

- Multi-tool Agent
- Tool Registry
- Tool Orchestration
- Tool Validation
- State Aggregation
- Side Effect
- Read-only Tool
- Write Tool
- Risk Level
- Tool Trace

### 核心流程

```text
用户输入
-> 创建 trace_id
-> 安全检查
-> 初始化 state
-> Planner 生成工具调用序列
-> 校验工具名和参数
-> 按顺序执行多个工具
-> 将工具结果写入 state
-> 记录只读 / 写入工具和副作用
-> 综合生成最终回答
-> 输出 Trace Summary
```

### 建议代码产出

```text
day15_multi_tool_agent.py
multi_tool_memory.json
```

建议核心函数：

```python
def register_tools():
    ...

def plan_tools(user_input, state):
    ...

def validate_tool_call(tool_name, arguments):
    ...

def execute_tool(tool_name, arguments, state, logger):
    ...

def search_knowledge_base(query):
    ...

def read_memory():
    ...

def write_memory(memory_text):
    ...
```

### 关键认知

```text
多工具 Agent 的难点不是工具多，而是工具之间的顺序、边界、状态和副作用要清楚。
```

### 明日衔接

第 15 天让 Agent 能组合多个工具完成复合任务。第 16 天可以继续学习人类确认与高风险动作控制，让写入、删除、发送等副作用工具必须经过用户确认。

---

## 第 16 天：人类确认与高风险动作控制

### 主题

从“能组合多个工具”升级到“高风险动作必须先请求用户确认”，学习 Pending Action、人类确认、风险等级和副作用工具控制。

### 学习目标

- 理解人类确认（Human-in-the-loop）在 Agent 中的作用。
- 区分低风险、中风险、高风险和禁用工具。
- 理解模型可以建议高风险动作，但不能替用户批准动作。
- 学会把高风险工具调用先保存为 Pending Action。
- 学会处理用户的确认或取消输入。
- 对删除、发送、外部调用等副作用工具进行确认控制。
- 将确认请求、确认通过、确认拒绝和最终执行写入 Trace。

### 核心概念

- Human-in-the-loop
- Pending Action
- Confirmation
- High-risk Tool
- Blocked Tool
- Side Effect
- Approval
- Rejection
- Permission Boundary
- Confirmation Trace

### 核心流程

```text
用户输入
-> Planner 识别工具动作
-> 查询 Tool Registry 风险等级
-> low / medium 工具按策略执行
-> high 工具创建 Pending Action
-> 向用户展示动作摘要、参数、风险和副作用
-> 用户输入 yes 才执行
-> 用户输入 no 则取消
-> blocked 工具直接拒绝
-> 全过程写入 Trace
```

### 建议代码产出

```text
day16_human_confirmation_agent.py
```

可复用文件：

```text
multi_tool_memory.json
agent_traces.jsonl
```

建议核心函数：

```python
def classify_tool_risk(tool_name):
    ...

def create_pending_action(tool_name, arguments, summary):
    ...

def requires_confirmation(tool_name):
    ...

def handle_confirmation(user_input, pending_action):
    ...

def execute_confirmed_action(action):
    ...

def reject_blocked_tool(tool_name):
    ...
```

### 关键认知

```text
模型可以建议高风险动作，但只有用户确认后，程序才能执行。
```

### 明日衔接

第 16 天解决了高风险工具的确认边界。第 17 天可以继续学习任务队列与长任务状态，让 Agent 能处理需要等待、分阶段执行或跨轮继续的任务。

## 第 17 天：任务队列与长任务状态

### 主题

从“一次对话内完成任务”升级到“可创建、保存、推进、查询、取消和恢复的长任务 Agent”。

### 学习目标

- 理解为什么真实 Agent 需要任务队列。
- 学会设计任务、步骤和状态字段。
- 区分 Trace、State、Pending Action 和 Task Queue。
- 掌握 pending、running、completed、failed、cancelled 等任务状态。
- 学会每执行一步就保存 checkpoint，避免中断后丢失进度。
- 理解恢复执行和安全取消对真实 Agent 产品的重要性。

### 核心概念

- Task / Job
- Task Queue
- Task Status
- Step
- Checkpoint
- Resume
- Cancel
- Long-running Task
- Idempotency
- Task Persistence

### 核心流程

```text
用户输入长期目标
-> 创建 task_id
-> 拆分任务步骤
-> 保存到任务队列
-> 每次推进一个 step
-> 更新 step 和 task 状态
-> 保存 checkpoint
-> 用户可查询、继续、取消或恢复
-> 完成后输出最终结果
```

### 建议代码产出

```text
day17_task_queue_agent.py
agent_tasks.json
```

建议核心函数：

```python
def load_tasks():
    ...

def save_tasks(tasks):
    ...

def create_task(goal):
    ...

def plan_task_steps(goal):
    ...

def advance_task(task):
    ...

def run_next_step(task):
    ...

def cancel_task(task_id):
    ...

def print_task_summary(task):
    ...
```

### 关键认知

```text
长任务 Agent 的关键，不是一次性把事情做完，而是能保存进度、查询状态、恢复执行、安全取消。
```

### 明日衔接

第 17 天让 Agent 具备了长任务管理能力。第 18 天可以继续学习如何把 Agent 封装成 API 服务，让前端、脚本或其他系统能够调用 Agent。

## 第 18 天：Agent API 服务封装

### 主题

从“命令行运行 Agent”升级到“通过 HTTP API 调用 Agent”，学习接口层、请求校验、结构化响应和服务化边界。

### 学习目标

- 理解 API 是 Agent 的入口，而不是 Agent 的核心。
- 学会把 Agent 核心逻辑和 API 接口层分开。
- 掌握 GET、POST、Endpoint、Request、Response 和 HTTP 状态码的基础用法。
- 学会为任务队列 Agent 设计创建、查询、推进、恢复和取消接口。
- 学会返回结构化 JSON，方便前端或其他系统调用。
- 理解 API 层需要处理参数校验、错误响应和基础安全边界。

### 核心概念

- HTTP API
- Endpoint
- Request
- Response
- Status Code
- API Layer
- Agent Core
- Request Validation
- Structured JSON
- Service Boundary

### 核心流程

```text
外部系统发送 HTTP 请求
-> API 层接收和校验参数
-> 调用 Agent 核心函数
-> 读取或更新任务状态
-> 写入 Trace
-> 返回结构化 JSON 响应
```

### 建议代码产出

```text
day18_agent_api_service.py
```

可复用文件：

```text
day17_task_queue_agent.py
agent_tasks.json
agent_traces.jsonl
```

建议接口：

```text
GET  /health
POST /tasks
GET  /tasks
GET  /tasks/{task_id}
POST /tasks/{task_id}/run
POST /tasks/{task_id}/resume
POST /tasks/{task_id}/cancel
GET  /traces
```

### 关键认知

```text
API 不是 Agent 的核心，API 是 Agent 的入口；好的 Agent 服务应该把接口层和核心逻辑分开。
```

### 明日衔接

第 18 天让 Agent 可以被外部系统调用。第 19 天可以继续学习简单前端界面，让用户不通过命令行或接口文档，也能使用 Agent。

## 第 19 天：Agent 前端操作界面

### 主题

从“通过 API 调用 Agent”升级到“通过网页界面操作 Agent”，学习前端如何调用 Agent API、展示任务状态、触发动作和反馈错误。

### 学习目标

- 理解前端是 Agent 的操作台，而不是 Agent 的大脑。
- 学会用原生 HTML、CSS、JavaScript 构建最小 Agent 界面。
- 学会使用 fetch 调用第 18 天的 API 服务。
- 在页面中展示任务列表、任务详情、步骤进度和 Trace。
- 学会处理 loading、success 和 error 等基础界面状态。
- 理解前端状态只是显示缓存，后端任务状态才是事实来源。

### 核心概念

- Frontend
- fetch
- UI State
- Loading State
- Error State
- Task List
- Task Detail
- API Base URL
- User Feedback
- Agent Control Panel

### 核心流程

```text
用户打开网页
-> 前端检查 API 是否在线
-> 用户输入任务目标
-> 前端调用 POST /tasks 创建任务
-> 前端调用 GET /tasks 刷新任务列表
-> 用户点击推进、恢复或取消
-> 前端调用对应 API
-> 展示任务详情、步骤进度、错误信息和 Trace
```

### 建议代码产出

```text
day19_agent_frontend.html
```

可复用文件：

```text
day18_agent_api_service.py
day17_task_queue_agent.py
agent_tasks.json
agent_traces.jsonl
```

建议页面模块：

```text
API 状态栏
API 地址输入框
创建任务表单
任务列表
任务详情
Trace 面板
错误提示区域
```

### 关键认知

```text
前端不是 Agent 的大脑，前端是 Agent 的操作台；它让用户看见状态、发出动作、获得反馈。
```

### 明日衔接

第 19 天让 Agent 具备了可操作界面。第 20 天可以继续学习多轮工作流设计，让用户在界面中和 Agent 持续协作，而不是只执行单个按钮动作。

## 第 20 天：多轮工作流 Agent

### 主题

从“用户点击按钮触发动作”升级到“围绕一个任务持续多轮协作”，学习工作流状态、澄清问题、信息收集、阶段推进和用户反馈修改。

### 学习目标

- 理解多轮工作流和单次任务执行的区别。
- 学会为 workflow session 设计状态结构。
- 学会判断用户目标是否缺少必要信息。
- 通过多轮对话收集 audience、output_format、focus 等字段。
- 根据 collected_context 推进 clarifying、ready_to_plan、planning、drafting、reviewing、completed 等阶段。
- 保存消息历史、当前阶段、计划、草稿和每次状态变化。
- 理解哪些规则逻辑后续可以替换成 LLM。

### 核心概念

- Workflow
- Workflow Session
- Workflow State
- Clarification
- Slot Filling
- Conversation History
- Workflow Transition
- Collected Context
- Missing Fields
- User Review

### 核心流程

```text
用户提出目标
-> 创建 workflow session
-> 抽取已有信息
-> 判断缺少哪些字段
-> 如信息不足则追问
-> 用户补充信息
-> 更新 collected_context 和 messages
-> 字段齐全后生成计划
-> 执行并生成草稿
-> 用户反馈修改或确认完成
-> 保存最终结果
```

### 建议代码产出

```text
day20_workflow_agent.py
workflow_sessions.json
```

建议核心函数：

```python
def load_sessions():
    ...

def save_sessions(sessions):
    ...

def create_session(user_input):
    ...

def extract_context_from_message(message):
    ...

def find_missing_fields(session):
    ...

def update_session_with_user_message(session, message):
    ...

def maybe_advance_workflow(session):
    ...

def create_plan(session):
    ...

def run_workflow(session):
    ...

def revise_workflow(session, instruction):
    ...
```

### 关键认知

```text
多轮工作流 Agent 的核心，不是多说几轮话，而是在每一轮之后正确更新状态，并决定下一步该追问、规划、执行还是等待用户反馈。
```

### 明日衔接

第 20 天补齐了第三周的协作流程能力。第 21 天可以做第三周综合项目，把多工具、人类确认、任务队列、API、前端和多轮工作流整合成一个小型 Agent 应用。

## 第 21 天：第三周综合项目：Agent 工作台

### 主题

把第 15-20 天的能力整合成一个可操作、可追踪、可恢复、可协作的小型 Agent 应用，形成第三周闭环。

### 学习目标

- 理解第三周每个能力在完整 Agent 应用中的位置。
- 区分 Workflow、Task Queue、Tool Registry、Pending Action、Trace 和 UI/API 边界。
- 设计统一的 workbench_state.json 保存 sessions、tasks、pending_action 和 outputs。
- 实现从用户目标到多轮澄清，再到任务创建和逐步执行的完整流程。
- 对写入、删除、清空等副作用动作创建 pending action，并等待用户确认。
- 使用 Trace 记录 workflow 创建、任务推进、工具调用、确认和输出保存。
- 为第四周的测试、配置、权限、部署、成本和最终项目做准备。

### 核心概念

- Agent Workbench
- Integrated Agent Application
- Workflow Layer
- Task Queue
- Tool Registry
- Human Confirmation
- Pending Action
- Unified State
- Output Record
- Product Boundary

### 核心流程

```text
用户提出目标
-> 创建 workflow session
-> 收集缺失信息
-> 生成工作流计划
-> 从计划创建 task
-> 用户逐步推进 task
-> 工具调用前校验风险
-> 高风险动作创建 pending action
-> 用户确认后执行副作用工具
-> 保存输出和 Trace
-> 支持查看、恢复、取消和完成
```

### 建议代码产出

```text
day21_agent_workbench.py
workbench_state.json
```

建议核心函数：

```python
def load_state():
    ...

def save_state(state):
    ...

def create_session(user_input, state):
    ...

def update_session_with_message(session, message):
    ...

def create_task_from_session(session, state):
    ...

def advance_task(task, state):
    ...

def register_tools():
    ...

def create_pending_action(tool_name, arguments, summary, state):
    ...

def handle_confirmation(user_input, state):
    ...

def handle_workbench_input(user_input, state):
    ...
```

### 关键认知

```text
第三周综合项目的重点不是写更多功能，而是把 Workflow、Task Queue、Tool Registry、Human Confirmation、API/UI 思维和 Trace 串成一个能解释、能操作、能恢复的小系统。
```

### 明日衔接

第 21 天完成第三周应用形态闭环。第 22 天可以进入第四周，开始学习 Agent 测试体系，让这个小型 Agent 应用不只是能跑，还能被稳定验证。

## 第 22 天：Agent 测试体系

### 主题

从“Agent 能跑通一次”升级到“Agent 可以被稳定验证”，学习如何用测试检查输出、工具调用、状态变化、安全边界和 Trace。

### 学习目标

- 理解 Agent 测试不能只检查最终回答。
- 区分 unit test、flow test、safety test 和 regression test。
- 学会用 JSON 描述 Agent 测试用例。
- 学会在每个测试前重置 state 和 trace，保证测试隔离。
- 检查 session、task、pending_action、outputs 等关键状态。
- 验证高风险动作必须等待用户确认，确认前不能产生副作用。
- 输出清晰的 PASS / FAIL 测试报告。

### 核心概念

- Agent Test
- Unit Test
- Flow Test
- Safety Test
- Regression Test
- Test Case
- Assertion
- Test Isolation
- State Assertion
- Trace Assertion

### 核心流程

```text
读取测试用例
-> 重置 workbench_state.json 和 agent_traces.jsonl
-> 按步骤输入用户消息或命令
-> 调用 Agent 工作台核心函数
-> 获取 answer、state、trace
-> 检查输出文本、状态字段、安全动作和最终结果
-> 汇总 PASS / FAIL 报告
```

### 建议代码产出

```text
day22_agent_tests.py
agent_test_cases.json
```

测试对象：

```text
day21_agent_workbench.py
workbench_state.json
agent_traces.jsonl
```

建议核心函数：

```python
def load_test_cases():
    ...

def reset_test_environment():
    ...

def run_test_case(test_case):
    ...

def run_step(step):
    ...

def assert_expectations(expect, answer, state, traces):
    ...

def assert_state_expectations(expected_state, state):
    ...

def assert_output_contains(expected_texts, answer):
    ...

def print_test_report(results):
    ...
```

### 关键认知

```text
Agent 测试不是问“回答像不像”，而是验证“决策、工具、状态、安全和输出”是否符合预期。
```

### 明日衔接

第 22 天让 Agent 工作台具备了回归测试基础。第 23 天可以继续学习配置与环境管理，让模型、路径、开关、密钥和运行参数不再散落在代码里。

## 第 23 天：配置与环境管理

### 主题

从“配置散落在代码里”升级到“用配置文件和环境变量管理模型、路径、功能开关、安全策略和敏感信息”。

### 学习目标

- 理解代码逻辑、配置参数和环境变量的职责边界。
- 学会把模型名、温度、路径、开关、最大步数和安全策略放入 agent_config.json。
- 学会用 .env.example 描述必需环境变量，但不提交真实密钥。
- 学会读取、校验和展示配置摘要。
- 学会隐藏 OPENAI_API_KEY 等敏感信息。
- 学会用配置控制 enable_trace、enable_write_tools、blocked_tools、max_steps_per_task 等行为。
- 理解配置管理是测试、权限、部署、成本控制和最终项目的基础。

### 核心概念

- Config
- Environment Variable
- .env.example
- Feature Flag
- Runtime Limit
- Secret Masking
- Config Validation
- Config Summary
- Development / Test / Production
- Configuration Boundary

### 核心流程

```text
读取 agent_config.json
-> 读取环境变量
-> 校验配置字段和类型
-> 隐藏敏感信息
-> 输出配置摘要
-> 根据配置控制 trace、写入工具、最大步骤数和禁用工具
-> 配置错误时提前失败并给出明确提示
```

### 建议代码产出

```text
day23_configurable_agent.py
agent_config.json
.env.example
```

建议核心函数：

```python
def load_config():
    ...

def load_env():
    ...

def validate_config(config):
    ...

def get_config_summary(config, env):
    ...

def mask_secret(value):
    ...

def is_feature_enabled(config, feature_name):
    ...

def is_tool_allowed(config, tool_name):
    ...

def classify_tool_with_config(config, tool_name):
    ...

def enforce_runtime_limits(config, state):
    ...
```

### 关键认知

```text
配置管理的目的，不是把常量换个地方写，而是把“不同环境会变化的东西”和“不能公开的东西”从核心逻辑里分离出来。
```

### 明日衔接

第 23 天让 Agent 具备了配置化运行能力。第 24 天可以继续学习权限、鉴权与用户隔离，让不同用户、不同工具和不同风险动作拥有清晰的访问边界。

## 第 24 天：权限、鉴权与用户隔离

### 主题

从“单用户 Agent 工作台”升级到“知道当前用户是谁、能做什么、能访问哪些数据”的多用户安全边界。

### 学习目标

- 区分 Authentication、Authorization 和 Isolation。
- 理解权限控制不是前端按钮控制，而是执行前的强制校验。
- 设计 guest、student、admin 三种角色。
- 使用 auth_policy.json 管理用户、角色和权限。
- 让 sessions、tasks、pending_actions、outputs 带 owner_user_id。
- 查询数据时只返回当前用户可见的数据。
- 防止用户运行、确认或读取别人的任务和 pending action。
- 理解权限策略应和配置管理、测试体系一起工作。

### 核心概念

- Authentication
- Authorization
- Isolation
- User Identity
- Role
- RBAC
- Permission
- owner_user_id
- Visible Scope
- Permission Boundary

### 核心流程

```text
用户登录
-> 识别当前 user_id 和 role
-> 读取 auth_policy.json
-> 执行动作前检查 permission
-> 创建数据时写入 owner_user_id
-> 查询数据时按用户过滤
-> 高风险 pending action 只能由 owner 或 admin 确认
-> 管理命令只允许 admin 执行
-> 全过程写入 Trace
```

### 建议代码产出

```text
day24_auth_agent.py
auth_policy.json
user_state.json
```

建议核心函数：

```python
def load_auth_policy():
    ...

def load_user_state():
    ...

def save_user_state(state):
    ...

def login(user_id, state, policy):
    ...

def current_user(state, policy):
    ...

def has_permission(user, permission, policy):
    ...

def require_permission(user, permission, policy):
    ...

def visible_to_user(item, user):
    ...

def filter_visible_items(items, user):
    ...

def confirm_user_action(action_id, user, state):
    ...
```

### 关键认知

```text
Agent 的权限控制不是“按钮让不让点”，而是“每一次读、写、工具调用和确认动作，都必须在执行前知道用户是谁，并检查他是否有权这样做”。
```

### 明日衔接

第 24 天让 Agent 具备了多用户安全边界。第 25 天可以继续学习部署与运行，让 Agent 能在可启动、可检查、可恢复的运行环境中工作。

## 课程质量要求

为了让课程保持精品感和新手友好：

- 不要过早引入框架。
- 每天只聚焦一个新的核心心智模型。
- 每节课都应该产出可运行代码。
- 每节课结束时，都要有一句清晰的关键认知。
- 不要在一天里混入太多概念。
- 先展示上一节课的局限，再引出下一节课的新内容。
- 尽早教授控制、校验和停止条件。
- 反复强调：工具由代码执行，不是由模型自己执行。

## 当前项目文件

```text
day3_tool_calling_agent.py
day4_multi_step_agent.py
day5_real_llm_tool_calling_agent.py
day6_llm_planner_agent.py
day7_memory_agent.py
day8_semantic_memory_agent.py
day9_rag_document_agent.py
day10_rag_evaluation.py
day11_safe_rag_agent.py
day12_observable_agent.py
day13_resilient_agent.py
day14_reliable_knowledge_agent.py
day15_multi_tool_agent.py
day16_human_confirmation_agent.py
day17_task_queue_agent.py
day18_agent_api_service.py
day19_agent_frontend.html
day20_workflow_agent.py
day21_agent_workbench.py
day22_agent_tests.py
day23_configurable_agent.py
day24_auth_agent.py
rag_index.json
rag_eval_questions.json
agent_test_cases.json
agent_config.json
auth_policy.json
.env.example
multi_tool_memory.json
agent_tasks.json
workflow_sessions.json
workbench_state.json
user_state.json
knowledge_base/
unsafe_knowledge_base/
reliable_knowledge_base/
agent_traces.jsonl
agent_course_outline.md
```
