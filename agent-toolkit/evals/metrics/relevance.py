"""Relevance metrics computation.

Measures how well sources align with and cover the query.
"""

from typing import Any, Optional

from ..models import RelevanceResult, EvalUsage
from ..judges.relevance_judge import RelevanceJudge

# Try both package and relative imports
try:
    from tavily_agent_toolkit import ModelConfig
except ImportError:
    try:
        from ...models import ModelConfig
    except ImportError:
        ModelConfig = Any  # type: ignore


async def compute_relevance_metrics(
    query: str,
    sources: list[dict],
    judge_model_config: "ModelConfig",
    k: int = 5,
    relevance_threshold: float = 0.5,
    query_analysis_prompt: Optional[str] = None,
    relevance_prompt: Optional[str] = None,
) -> RelevanceResult:
    """Compute relevance metrics for search sources.

    This function:
    1. Analyzes the query to identify key aspects
    2. Scores each source's relevance
    3. Computes overall metrics including top-k precision

    Args:
        query: The search query
        sources: List of source dicts with 'content', 'title', 'url'
        judge_model_config: ModelConfig for the judge LLM
        k: Number of top sources for precision calculation
        relevance_threshold: Minimum score to consider a source relevant
        query_analysis_prompt: Custom prompt for query analysis
        relevance_prompt: Custom prompt for relevance scoring

    Returns:
        RelevanceResult with metrics and per-source scores

    Example:
        result = await compute_relevance_metrics(
            query="What are the latest developments in quantum computing?",
            sources=search_results,
            judge_model_config=ModelConfig(model=ModelObject(model="gpt-4o-mini")),
        )
        print(f"Relevance: {result.source_relevance_score:.2f}")
        print(f"Coverage: {result.answer_coverage:.1%}")
    """
    if not sources:
        return RelevanceResult(
            source_relevance_score=0.0,
            answer_coverage=0.0,
            top_k_precision=0.0,
            k=k,
            per_source_scores=[],
            usage=EvalUsage(),
        )

    # Initialize judge
    judge = RelevanceJudge(
        model_config=judge_model_config,
        query_analysis_prompt=query_analysis_prompt,
        relevance_prompt=relevance_prompt,
    )

    # Perform evaluation
    judge_result = await judge.evaluate_relevance(query, sources)

    relevance_result = judge_result["relevance_result"]
    usage = judge_result["usage"]

    # Extract scores
    source_scores = relevance_result.source_scores
    scores = [s.relevance_score for s in source_scores]

    # Compute average relevance
    avg_relevance = sum(scores) / len(scores) if scores else 0.0

    # Compute top-k precision
    actual_k = min(k, len(scores))
    top_k_scores = sorted(scores, reverse=True)[:actual_k]
    relevant_in_top_k = sum(1 for s in top_k_scores if s >= relevance_threshold)
    top_k_precision = relevant_in_top_k / actual_k if actual_k > 0 else 0.0

    # Build per-source details
    per_source_scores = []
    for ss in source_scores:
        source = sources[ss.source_index] if ss.source_index < len(sources) else {}
        per_source_scores.append({
            "source_index": ss.source_index,
            "url": source.get("url", ""),
            "title": source.get("title", ""),
            "relevance_score": ss.relevance_score,
            "aspects_covered": ss.aspects_covered,
            "reasoning": ss.reasoning,
        })

    return RelevanceResult(
        source_relevance_score=avg_relevance,
        answer_coverage=relevance_result.answer_coverage,
        top_k_precision=top_k_precision,
        k=actual_k,
        per_source_scores=per_source_scores,
        usage=usage,
    )


def compute_relevance_from_scores(
    scores: list[float],
    coverage: float,
    k: int = 5,
    relevance_threshold: float = 0.5,
    source_metadata: Optional[list[dict]] = None,
) -> RelevanceResult:
    """Compute relevance metrics from pre-calculated scores.

    Use this when relevance scores have been computed externally.

    Args:
        scores: List of relevance scores (0-1) for each source
        coverage: Pre-computed answer coverage ratio (0-1)
        k: Number of top sources for precision calculation
        relevance_threshold: Minimum score to consider relevant
        source_metadata: Optional metadata dicts for each source

    Returns:
        RelevanceResult with computed metrics
    """
    if not scores:
        return RelevanceResult(
            source_relevance_score=0.0,
            answer_coverage=coverage,
            top_k_precision=0.0,
            k=k,
            per_source_scores=[],
            usage=EvalUsage(),
        )

    # Compute average relevance
    avg_relevance = sum(scores) / len(scores)

    # Compute top-k precision
    actual_k = min(k, len(scores))
    top_k_scores = sorted(scores, reverse=True)[:actual_k]
    relevant_in_top_k = sum(1 for s in top_k_scores if s >= relevance_threshold)
    top_k_precision = relevant_in_top_k / actual_k if actual_k > 0 else 0.0

    # Build per-source details
    per_source_scores = []
    for i, score in enumerate(scores):
        metadata = source_metadata[i] if source_metadata and i < len(source_metadata) else {}
        per_source_scores.append({
            "source_index": i,
            "url": metadata.get("url", ""),
            "title": metadata.get("title", ""),
            "relevance_score": score,
        })

    return RelevanceResult(
        source_relevance_score=avg_relevance,
        answer_coverage=coverage,
        top_k_precision=top_k_precision,
        k=actual_k,
        per_source_scores=per_source_scores,
        usage=EvalUsage(),
    )
