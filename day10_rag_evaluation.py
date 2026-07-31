"""
Day 10: RAG Retrieval Evaluation

Run after Day 9:
    export OPENAI_API_KEY="your_api_key"
    python3 day9_rag_document_agent.py
    # enter :rebuild once to create rag_index.json
    python3 day10_rag_evaluation.py

Offline learning mode, useful before you rebuild the vector index:
    python3 day10_rag_evaluation.py --mode keyword

What this script teaches:
    1. Prepare evaluation questions.
    2. Retrieve Top K chunks for each question.
    3. Check whether expected sources were retrieved.
    4. Check whether retrieved evidence contains expected keywords.
    5. Print a report that helps locate RAG quality problems.

This is intentionally a retrieval evaluator first. If retrieval is wrong,
the final LLM answer is unlikely to be reliable no matter how polished it is.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Literal, TypedDict

import day9_rag_document_agent as rag


EVAL_FILE = Path(__file__).with_name("rag_eval_questions.json")
PASS_SCORE = 0.75


class EvalError(Exception):
    """A readable error for this learning script."""


class EvalQuestion(TypedDict):
    id: str
    question: str
    expected_sources: list[str]
    expected_keywords: list[str]


class RetrievedEvidence(TypedDict):
    source: str
    chunk_index: int
    content: str
    score: float


class EvalResult(TypedDict):
    id: str
    question: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    source_hit: bool
    expected_keywords: list[str]
    matched_keywords: list[str]
    keyword_score: float
    passed: bool
    evidence: list[RetrievedEvidence]


def load_eval_questions(path: Path = EVAL_FILE) -> list[EvalQuestion]:
    """Load and validate the small golden set for RAG evaluation."""
    if not path.exists():
        raise EvalError(f"评估问题文件不存在：{path.name}")

    try:
        raw_items = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvalError(f"无法读取评估问题文件 {path.name}：{error}") from error

    if not isinstance(raw_items, list) or not raw_items:
        raise EvalError("评估问题文件必须是非空 JSON 数组。")

    questions: list[EvalQuestion] = []
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            raise EvalError(f"第 {index} 条评估问题不是对象。")

        question_id = item.get("id")
        question = item.get("question")
        expected_sources = item.get("expected_sources")
        expected_keywords = item.get("expected_keywords")

        if not isinstance(question_id, str) or not question_id.strip():
            raise EvalError(f"第 {index} 条评估问题缺少 id。")
        if not isinstance(question, str) or not question.strip():
            raise EvalError(f"第 {index} 条评估问题缺少 question。")
        if not is_string_list(expected_sources):
            raise EvalError(f"第 {index} 条评估问题的 expected_sources 必须是字符串数组。")
        if not is_string_list(expected_keywords):
            raise EvalError(f"第 {index} 条评估问题的 expected_keywords 必须是字符串数组。")

        questions.append(
            {
                "id": question_id.strip(),
                "question": question.strip(),
                "expected_sources": [source.strip() for source in expected_sources],
                "expected_keywords": [keyword.strip() for keyword in expected_keywords],
            }
        )
    return questions


def is_string_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, str) and item.strip() for item in value
    )


def normalize_text(text: str) -> str:
    """Normalize text so keyword checks are stable and easy to inspect."""
    return re.sub(r"\s+", " ", text).casefold()


def check_source_hit(
    retrieved: list[RetrievedEvidence],
    expected_sources: list[str],
) -> bool:
    """Return True when at least one expected source appears in Top K."""
    retrieved_sources = {item["source"] for item in retrieved}
    return any(source in retrieved_sources for source in expected_sources)


def check_keyword_hit(
    retrieved: list[RetrievedEvidence],
    expected_keywords: list[str],
) -> list[str]:
    """Return expected keywords that are present in the retrieved evidence."""
    evidence_text = normalize_text("\n".join(item["content"] for item in retrieved))
    return [
        keyword
        for keyword in expected_keywords
        if normalize_text(keyword) in evidence_text
    ]


def keyword_score(matched_keywords: list[str], expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0
    return len(matched_keywords) / len(expected_keywords)


def evaluate_retrieval(
    question: EvalQuestion,
    retrieved: list[RetrievedEvidence],
) -> EvalResult:
    """Evaluate one question after retrieval has already happened."""
    matched_keywords = check_keyword_hit(retrieved, question["expected_keywords"])
    score = keyword_score(matched_keywords, question["expected_keywords"])
    source_hit = check_source_hit(retrieved, question["expected_sources"])
    return {
        "id": question["id"],
        "question": question["question"],
        "expected_sources": question["expected_sources"],
        "retrieved_sources": unique_sources(retrieved),
        "source_hit": source_hit,
        "expected_keywords": question["expected_keywords"],
        "matched_keywords": matched_keywords,
        "keyword_score": score,
        "passed": source_hit and score >= PASS_SCORE,
        "evidence": retrieved,
    }


def unique_sources(retrieved: list[RetrievedEvidence]) -> list[str]:
    sources: list[str] = []
    for item in retrieved:
        if item["source"] not in sources:
            sources.append(item["source"])
    return sources


def retrieve_with_embeddings(
    question: str,
    top_k: int,
    min_similarity: float,
) -> list[RetrievedEvidence]:
    """
    Evaluate the real Day 9 vector retrieval path.

    This mode uses rag_index.json and OpenAI Embeddings for the query vector.
    It is the closest check for the actual Day 9 RAG agent.
    """
    index = rag.load_index()
    retrieved = rag.retrieve_relevant_chunks(
        question,
        index,
        top_k=top_k,
        min_similarity=min_similarity,
    )
    return [
        {
            "source": item["chunk"]["source"],
            "chunk_index": item["chunk"]["chunk_index"],
            "content": item["chunk"]["content"],
            "score": item["similarity"],
        }
        for item in retrieved
    ]


def retrieve_with_keywords(question: str, top_k: int) -> list[RetrievedEvidence]:
    """
    Offline teaching fallback.

    This does not replace vector search. It simply lets beginners run the
    evaluator before they have an API key or a built rag_index.json.
    """
    query_terms = expand_query_terms(question, tokenize(question))
    documents = rag.load_documents()
    candidates: list[RetrievedEvidence] = []

    for document in documents:
        for chunk_index, content in enumerate(rag.split_text_into_chunks(document["content"]), start=1):
            chunk_terms = tokenize(content)
            score = lexical_score(query_terms, chunk_terms, content)
            if score > 0:
                candidates.append(
                    {
                        "source": document["source"],
                        "chunk_index": chunk_index,
                        "content": content,
                        "score": score,
                    }
                )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:top_k]


def tokenize(text: str) -> set[str]:
    """A tiny tokenizer for the offline demo mode."""
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text)
        if len(token.strip()) > 1
    }


def expand_query_terms(text: str, terms: set[str]) -> set[str]:
    """
    Add a few course-specific bridge terms for Chinese questions.

    Real Embeddings solve this semantically. This bridge only keeps the offline
    keyword mode useful for students who have not built the vector index yet.
    """
    expanded = set(terms)
    bridge_terms = {
        "索引": ["index", "stage", "chunks", "embedding"],
        "查询": ["query", "time", "top", "k"],
        "切分": ["chunks", "chunk"],
        "文档": ["document", "documents"],
        "最大步骤": ["maximum", "step", "count"],
        "停止条件": ["stop", "conditions", "infinite", "loops"],
        "多步骤": ["multi", "step", "plan", "act", "observe"],
        "模型": ["model", "proposes"],
        "程序": ["program", "code", "execute"],
        "语义检索": ["semantic", "retrieval", "cosine", "similarity"],
        "负责": ["records", "stores", "converts"],
        "区别": ["records", "stores", "history", "memory"],
    }
    for chinese, english_terms in bridge_terms.items():
        if chinese in text:
            expanded.update(english_terms)
    return expanded


def lexical_score(query_terms: set[str], chunk_terms: set[str], content: str) -> float:
    if not query_terms or not chunk_terms:
        return 0.0

    overlap = len(query_terms & chunk_terms)
    # A tiny bonus for important course words that bridge Chinese questions and
    # English sample documents.
    bridge_bonus = 0.0
    lowered = content.casefold()
    for term in ["rag", "agent", "tool", "embedding", "memory", "chunk"]:
        if term in lowered and term in {item.casefold() for item in query_terms}:
            bridge_bonus += 0.2
    return overlap / len(query_terms) + bridge_bonus


def run_evaluation(
    mode: Literal["embedding", "keyword"],
    top_k: int,
    min_similarity: float,
) -> list[EvalResult]:
    questions = load_eval_questions()
    results: list[EvalResult] = []
    for question in questions:
        if mode == "embedding":
            retrieved = retrieve_with_embeddings(question["question"], top_k, min_similarity)
        else:
            retrieved = retrieve_with_keywords(question["question"], top_k)
        results.append(evaluate_retrieval(question, retrieved))
    return results


def print_eval_report(results: list[EvalResult], mode: str, top_k: int) -> None:
    """Print a human-readable report instead of hiding everything in JSON."""
    passed_count = sum(1 for result in results if result["passed"])
    total = len(results)
    print(f"Day 10 RAG Evaluation Report")
    print(f"模式：{mode}，Top K：{top_k}，通过：{passed_count}/{total}")
    print("=" * 72)

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{status}] {result['id']}")
        print(f"问题：{result['question']}")
        print(f"期望来源：{', '.join(result['expected_sources'])}")
        print(f"实际来源：{', '.join(result['retrieved_sources']) or '无'}")
        print(f"来源命中：{'是' if result['source_hit'] else '否'}")
        print(
            "关键词命中："
            f"{len(result['matched_keywords'])}/{len(result['expected_keywords'])} "
            f"({result['keyword_score']:.0%})"
        )
        print(f"已命中关键词：{', '.join(result['matched_keywords']) or '无'}")
        print("Top K 证据：")
        for evidence in result["evidence"]:
            preview = normalize_preview(evidence["content"])
            print(
                f"  - {evidence['source']} / chunk_{evidence['chunk_index']:03d} "
                f"(score {evidence['score']:.3f})：{preview}"
            )

    print("\n" + "=" * 72)
    print("如何解读：")
    print("- 来源没命中：优先检查检索参数、问题表达、文档标题和 chunk 切分。")
    print("- 来源命中了但关键词少：检查 chunk 是否太小、overlap 是否不足、top_k 是否太低。")
    print("- 检索评估稳定后，再评估最终回答是否忠于这些证据。")


def normalize_preview(text: str, max_length: int = 120) -> str:
    preview = re.sub(r"\s+", " ", text).strip()
    if len(preview) <= max_length:
        return preview
    return preview[: max_length - 3] + "..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Day 9 RAG retrieval quality.")
    parser.add_argument(
        "--mode",
        choices=["embedding", "keyword"],
        default="embedding",
        help="embedding 使用真实 rag_index.json；keyword 使用离线关键词检索教学模式。",
    )
    parser.add_argument("--top-k", type=int, default=3, help="每个问题检索多少个 chunk。")
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=rag.MIN_SIMILARITY,
        help="embedding 模式下的最低相似度阈值。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise EvalError("--top-k 必须大于 0。")
    results = run_evaluation(args.mode, args.top_k, args.min_similarity)
    print_eval_report(results, args.mode, args.top_k)


if __name__ == "__main__":
    try:
        main()
    except (EvalError, rag.RAGError) as error:
        print(f"评估失败：\n{error}")
        if "索引中还没有 Chunk" in str(error):
            print("\n你可以先运行：")
            print('export OPENAI_API_KEY="你的 API key"')
            print("python3 day9_rag_document_agent.py")
            print("然后在程序里输入 :rebuild")
            print("\n或者先用离线教学模式：")
            print("python3 day10_rag_evaluation.py --mode keyword")
        sys.exit(1)
