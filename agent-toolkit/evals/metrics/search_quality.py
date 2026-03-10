"""Search quality metrics computation.

Measures the effectiveness of search queries and results.
"""

from typing import Any, Optional
from urllib.parse import urlparse
import hashlib

from ..models import SearchQualityResult, EvalUsage
from ..judges.quality_judge import QualityJudge

# Try both package and relative imports
try:
    from tavily_agent_toolkit import ModelConfig
except ImportError:
    try:
        from ...models import ModelConfig
    except ImportError:
        ModelConfig = Any  # type: ignore


async def compute_search_quality_metrics(
    research_task: str,
    queries: list[str],
    results: list[dict],
    judge_model_config: "ModelConfig",
    credits_used: int = 0,
    query_quality_prompt: Optional[str] = None,
    coverage_prompt: Optional[str] = None,
) -> SearchQualityResult:
    """Compute search quality metrics for a research task.

    This function evaluates:
    1. Quality of generated search queries
    2. Coverage of search results
    3. Deduplication efficiency
    4. Credit efficiency (quality per credit spent)

    Args:
        research_task: The original research task/question
        queries: List of search queries that were executed
        results: List of search result dicts with 'content', 'url', 'title'
        judge_model_config: ModelConfig for the judge LLM
        credits_used: Number of API credits consumed
        query_quality_prompt: Custom prompt for query evaluation
        coverage_prompt: Custom prompt for coverage evaluation

    Returns:
        SearchQualityResult with quality metrics

    Example:
        result = await compute_search_quality_metrics(
            research_task="What are the latest AI developments?",
            queries=["AI developments 2024", "machine learning trends"],
            results=search_results,
            judge_model_config=ModelConfig(model=ModelObject(model="gpt-4o-mini")),
            credits_used=10,
        )
        print(f"Query quality: {result.query_quality_score:.2f}")
        print(f"Coverage: {result.result_coverage:.1%}")
    """
    if not queries and not results:
        return SearchQualityResult(
            query_quality_score=0.0,
            result_coverage=0.0,
            deduplication_efficiency=0.0,
            credit_efficiency=0.0,
            usage=EvalUsage(),
        )

    # Initialize judge
    judge = QualityJudge(
        model_config=judge_model_config,
        query_quality_prompt=query_quality_prompt,
        coverage_prompt=coverage_prompt,
    )

    # Evaluate queries and coverage
    judge_result = await judge.judge(
        research_task=research_task,
        queries=queries if queries else None,
        results=results if results else None,
    )

    usage = judge_result["usage"]

    # Extract query quality
    query_quality_score = 0.0
    query_details = []
    if "query_evaluation" in judge_result:
        query_eval = judge_result["query_evaluation"]
        query_quality_score = query_eval.overall_quality
        for qs in query_eval.individual_scores:
            query_details.append({
                "query": qs.query,
                "specificity": qs.specificity,
                "search_effectiveness": qs.search_effectiveness,
                "overall_score": qs.overall_score,
                "reasoning": getattr(qs, "reasoning", ""),
                "suggestions": qs.suggestions,
            })

    # Extract coverage
    result_coverage = 0.0
    if "coverage_evaluation" in judge_result:
        coverage_eval = judge_result["coverage_evaluation"]
        result_coverage = coverage_eval.topic_coverage

    # Compute deduplication efficiency
    dedup_efficiency = _compute_deduplication_efficiency(results)

    # Compute credit efficiency
    credit_efficiency = 0.0
    if credits_used > 0:
        # Combine quality and coverage for efficiency calculation
        combined_quality = (query_quality_score + result_coverage) / 2
        credit_efficiency = combined_quality / credits_used

    # Count unique results
    unique_results = _count_unique_results(results)

    return SearchQualityResult(
        query_quality_score=query_quality_score,
        result_coverage=result_coverage,
        deduplication_efficiency=dedup_efficiency,
        credit_efficiency=credit_efficiency,
        queries_evaluated=len(queries),
        results_count=len(results),
        unique_results_count=unique_results,
        credits_used=credits_used,
        query_details=query_details,
        usage=usage,
    )


def _compute_deduplication_efficiency(results: list[dict]) -> float:
    """Compute the ratio of unique content in results.

    Uses content hashing to detect duplicates.

    Args:
        results: List of result dicts with 'content'

    Returns:
        Ratio of unique results (0-1)
    """
    if not results:
        return 1.0

    # Hash content to detect duplicates
    content_hashes = set()
    unique_count = 0

    for result in results:
        content = result.get("content", "")
        if content:
            # Use first 500 chars for hash to detect near-duplicates
            content_hash = hashlib.md5(content[:500].encode()).hexdigest()
            if content_hash not in content_hashes:
                content_hashes.add(content_hash)
                unique_count += 1

    return unique_count / len(results) if results else 1.0


def _count_unique_results(results: list[dict]) -> int:
    """Count unique results based on URL and content.

    Args:
        results: List of result dicts

    Returns:
        Number of unique results
    """
    seen_urls = set()
    seen_content = set()
    unique_count = 0

    for result in results:
        url = result.get("url", "")
        content = result.get("content", "")[:500]

        # Check URL uniqueness
        if url and url in seen_urls:
            continue

        # Check content uniqueness
        content_hash = hashlib.md5(content.encode()).hexdigest() if content else ""
        if content_hash and content_hash in seen_content:
            continue

        # This is unique
        if url:
            seen_urls.add(url)
        if content_hash:
            seen_content.add(content_hash)
        unique_count += 1

    return unique_count


def compute_search_quality_from_stats(
    query_quality: float,
    coverage: float,
    results_count: int,
    unique_results_count: int,
    credits_used: int = 0,
) -> SearchQualityResult:
    """Compute search quality from pre-calculated statistics.

    Use this when metrics have been computed externally.

    Args:
        query_quality: Pre-computed query quality score (0-1)
        coverage: Pre-computed result coverage (0-1)
        results_count: Total number of results
        unique_results_count: Number of unique results
        credits_used: API credits consumed

    Returns:
        SearchQualityResult with computed metrics
    """
    # Compute deduplication efficiency
    dedup_efficiency = unique_results_count / results_count if results_count > 0 else 1.0

    # Compute credit efficiency
    credit_efficiency = 0.0
    if credits_used > 0:
        combined_quality = (query_quality + coverage) / 2
        credit_efficiency = combined_quality / credits_used

    return SearchQualityResult(
        query_quality_score=query_quality,
        result_coverage=coverage,
        deduplication_efficiency=dedup_efficiency,
        credit_efficiency=credit_efficiency,
        results_count=results_count,
        unique_results_count=unique_results_count,
        credits_used=credits_used,
        query_details=[],
        usage=EvalUsage(),
    )
