"""Retrieval evaluator for search-only evaluation."""

from typing import Any, Literal, Optional

from .base import BaseEvaluator
from ..models import EvalResult
from ..metrics.relevance import compute_relevance_metrics
from ..metrics.search_quality import compute_search_quality_metrics

# Try both package and relative imports
try:
    from tavily_agent_toolkit import ModelConfig
except ImportError:
    try:
        from ...models import ModelConfig
    except ImportError:
        ModelConfig = Any  # type: ignore


RetrievalMetricType = Literal["relevance", "search_quality"]


class RetrievalEvaluator(BaseEvaluator):
    """Evaluator for search/retrieval quality (no report evaluation).

    Use this when you want to evaluate just the search step,
    without a generated report. This is useful for:
    - Evaluating search query generation
    - Assessing retrieval relevance
    - Measuring deduplication effectiveness

    Example:
        evaluator = RetrievalEvaluator(
            judge_model_config=ModelConfig(model=ModelObject(model="gpt-4o-mini")),
        )
        result = await evaluator.evaluate(
            query="Latest AI trends",
            results=search_results,
            queries=["AI trends 2024", "machine learning developments"],
        )
    """

    def __init__(
        self,
        judge_model_config: "ModelConfig",
        metrics: Optional[list[RetrievalMetricType]] = None,
    ):
        """Initialize the retrieval evaluator.

        Args:
            judge_model_config: ModelConfig for judge LLM calls
            metrics: List of metrics. Defaults to relevance and search_quality.
        """
        super().__init__(judge_model_config, metrics or ["relevance", "search_quality"])

    async def evaluate(
        self,
        query: str,
        results: list[dict],
        queries: Optional[list[str]] = None,
        credits_used: int = 0,
        tool_usage: Optional[dict] = None,
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ) -> EvalResult:
        """Evaluate search/retrieval results.

        Args:
            query: The original search query or research task
            results: List of search result dicts with 'url', 'title', 'content'
            queries: Optional list of search queries used (for multi-query search)
            credits_used: API credits consumed
            tool_usage: Optional ToolUsageStats dict from the tool
            metadata: Optional additional metadata
            **kwargs: Additional arguments (unused)

        Returns:
            EvalResult with relevance and search quality metrics
        """
        result = EvalResult(
            query=query,
            tool_usage=tool_usage,
            metadata=metadata or {},
        )

        # Evaluate relevance
        if "relevance" in self.metrics:
            result.relevance = await compute_relevance_metrics(
                query=query,
                sources=results,
                judge_model_config=self.judge_model_config,
            )
            self.usage.merge(result.relevance.usage)

        # Evaluate search quality
        if "search_quality" in self.metrics:
            result.search_quality = await compute_search_quality_metrics(
                research_task=query,
                queries=queries or [query],
                results=results,
                judge_model_config=self.judge_model_config,
                credits_used=credits_used,
            )
            self.usage.merge(result.search_quality.usage)

        # Compute overall score (only from available metrics)
        result.compute_overall_score({
            "relevance": 0.6,
            "search_quality": 0.4,
        })

        return result


async def evaluate_retrieval(
    query: str,
    results: list[dict],
    judge_model_config: "ModelConfig",
    queries: Optional[list[str]] = None,
    metrics: Optional[list[RetrievalMetricType]] = None,
    credits_used: int = 0,
    tool_usage: Optional[dict] = None,
) -> EvalResult:
    """Convenience function to evaluate retrieval results.

    Args:
        query: The original search query
        results: List of search result dicts
        judge_model_config: ModelConfig for judge LLM calls
        queries: Optional list of queries used
        metrics: Optional list of metrics to compute
        credits_used: API credits consumed
        tool_usage: Optional ToolUsageStats dict

    Returns:
        EvalResult with retrieval metrics

    Example:
        result = await evaluate_retrieval(
            query="What is quantum computing?",
            results=search_results,
            judge_model_config=ModelConfig(model=ModelObject(model="gpt-4o-mini")),
        )
        print(f"Relevance: {result.relevance.source_relevance_score:.2f}")
    """
    evaluator = RetrievalEvaluator(
        judge_model_config=judge_model_config,
        metrics=metrics,
    )

    return await evaluator.evaluate(
        query=query,
        results=results,
        queries=queries,
        credits_used=credits_used,
        tool_usage=tool_usage,
    )
