"""Correctness metrics computation.

Measures answer correctness against expected answers using LLM-as-judge.
"""

import asyncio
from typing import Any, Optional

from ..models import EvalUsage
from ..judges.correctness_judge import CorrectnessJudge

# Try both package and relative imports
try:
    from tavily_agent_toolkit import ModelConfig
except ImportError:
    try:
        from ...models import ModelConfig
    except ImportError:
        ModelConfig = Any  # type: ignore


async def compute_correctness_metrics(
    items: list[tuple[str, str, str]],
    judge_model_config: "ModelConfig",
    categories: Optional[list[Optional[str]]] = None,
    grader_prompt: Optional[str] = None,
    parallel: bool = True,
    max_concurrency: int = 10,
) -> dict[str, Any]:
    """Compute correctness metrics for a set of query-answer pairs.

    This function:
    1. Judges each (query, expected, predicted) tuple as CORRECT/INCORRECT
    2. Computes overall accuracy
    3. Optionally computes per-category breakdowns

    Args:
        items: List of (query, expected_answer, predicted_answer) tuples
        judge_model_config: ModelConfig for the judge LLM
        categories: Optional list of categories (same length as items) for breakdown
        grader_prompt: Custom prompt for the grader
        parallel: Whether to run judgments in parallel
        max_concurrency: Maximum concurrent judge calls when parallel=True

    Returns:
        Dict with:
            - accuracy: Fraction of correct answers (0-1)
            - correct_count: Number of correct answers
            - incorrect_count: Number of incorrect answers
            - total_count: Total items evaluated
            - per_item_results: List of individual results with grade, score, reasoning
            - by_category: Dict mapping category -> accuracy (if categories provided)
            - usage: EvalUsage with token counts

    Example:
        items = [
            ("What is 2+2?", "4", "The answer is 4"),
            ("Capital of France?", "Paris", "London"),
        ]
        result = await compute_correctness_metrics(
            items=items,
            judge_model_config=ModelConfig(model=ModelObject(model="gpt-4o-mini")),
        )
        print(f"Accuracy: {result['accuracy']:.1%}")  # 50.0%
    """
    if not items:
        return {
            "accuracy": 1.0,
            "correct_count": 0,
            "incorrect_count": 0,
            "total_count": 0,
            "per_item_results": [],
            "by_category": {},
            "usage": EvalUsage(),
        }

    # Initialize judge
    judge = CorrectnessJudge(
        model_config=judge_model_config,
        grader_prompt=grader_prompt,
    )

    # Run evaluations
    per_item_results = []

    if parallel:
        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)

        async def judge_with_semaphore(
            idx: int, query: str, expected: str, predicted: str
        ) -> tuple[int, dict]:
            async with semaphore:
                result = await judge.judge(
                    query=query,
                    expected_answer=expected,
                    predicted_answer=predicted,
                )
                return idx, result

        tasks = [
            judge_with_semaphore(i, q, e, p) for i, (q, e, p) in enumerate(items)
        ]
        indexed_results = await asyncio.gather(*tasks)

        # Sort by original index to maintain order
        indexed_results = sorted(indexed_results, key=lambda x: x[0])
        per_item_results = [r for _, r in indexed_results]
    else:
        for query, expected, predicted in items:
            result = await judge.judge(
                query=query,
                expected_answer=expected,
                predicted_answer=predicted,
            )
            per_item_results.append(result)

    # Add query info to each result
    for i, (query, expected, predicted) in enumerate(items):
        per_item_results[i]["query"] = query
        per_item_results[i]["expected_answer"] = expected
        per_item_results[i]["predicted_answer"] = predicted
        if categories and i < len(categories):
            per_item_results[i]["category"] = categories[i]

    # Compute aggregate metrics
    correct_count = sum(1 for r in per_item_results if r["grade"] == "CORRECT")
    incorrect_count = len(per_item_results) - correct_count
    total_count = len(per_item_results)
    accuracy = correct_count / total_count if total_count > 0 else 1.0

    # Compute per-category breakdown
    by_category: dict[str, dict[str, Any]] = {}
    if categories:
        category_results: dict[str, list[dict]] = {}
        for i, cat in enumerate(categories):
            if cat is not None:
                if cat not in category_results:
                    category_results[cat] = []
                category_results[cat].append(per_item_results[i])

        for cat, results in category_results.items():
            cat_correct = sum(1 for r in results if r["grade"] == "CORRECT")
            cat_total = len(results)
            by_category[cat] = {
                "accuracy": cat_correct / cat_total if cat_total > 0 else 1.0,
                "correct_count": cat_correct,
                "incorrect_count": cat_total - cat_correct,
                "total_count": cat_total,
            }

    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "total_count": total_count,
        "per_item_results": per_item_results,
        "by_category": by_category,
        "usage": judge.get_usage(),
    }
