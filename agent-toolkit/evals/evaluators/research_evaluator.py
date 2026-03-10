"""Research evaluator for comprehensive research output evaluation."""

from typing import Any, Literal, Optional, Type

from pydantic import BaseModel

from .base import BaseEvaluator, MetricType
from ..models import EvalResult
from ..metrics.grounding import compute_grounding_metrics
from ..metrics.relevance import compute_relevance_metrics
from ..metrics.content_attribution import compute_content_attribution_metrics
from ..metrics.search_quality import compute_search_quality_metrics

# Try both package and relative imports
try:
    from tavily_agent_toolkit import ModelConfig
except ImportError:
    try:
        from ...models import ModelConfig
    except ImportError:
        ModelConfig = Any  # type: ignore


class ResearchEvaluator(BaseEvaluator):
    """Evaluator for research outputs (reports with sources).

    This evaluator orchestrates multiple metrics:
    - Relevance: How well sources align with the query
    - Grounding: How well claims are supported by sources
    - Content Attribution: Breakdown of web vs internal vs prior knowledge
    - Search Quality: Quality of queries and result coverage

    Example:
        evaluator = ResearchEvaluator(
            judge_model_config=ModelConfig(model=ModelObject(model="gpt-4o-mini")),
            metrics=["grounding", "relevance"],
        )
        result = await evaluator.evaluate(
            query="What are the latest AI trends?",
            report=generated_report,
            web_sources=sources,
        )
    """

    def __init__(
        self,
        judge_model_config: "ModelConfig",
        metrics: Optional[list[MetricType]] = None,
        score_weights: Optional[dict[str, float]] = None,
    ):
        """Initialize the research evaluator.

        Args:
            judge_model_config: ModelConfig for judge LLM calls
            metrics: List of metrics to compute. Defaults to all.
            score_weights: Optional weights for overall score calculation
        """
        super().__init__(judge_model_config, metrics)
        self.score_weights = score_weights or {
            "relevance": 0.25,
            "grounding": 0.35,
            "content_attribution": 0.20,
            "search_quality": 0.20,
        }

    async def evaluate(
        self,
        query: str,
        report: str,
        web_sources: list[dict],
        internal_sources: Optional[list[dict]] = None,
        queries: Optional[list[str]] = None,
        credits_used: int = 0,
        tool_usage: Optional[dict] = None,
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ) -> EvalResult:
        """Evaluate a research output.

        Args:
            query: The original research query
            report: The generated report text
            web_sources: List of web source dicts with 'url', 'title', 'content'
            internal_sources: Optional internal RAG sources
            queries: Optional list of search queries used
            credits_used: API credits consumed
            tool_usage: Optional ToolUsageStats dict from the tool
            metadata: Optional additional metadata
            **kwargs: Additional arguments passed to metric functions

        Returns:
            EvalResult with all computed metrics
        """
        result = EvalResult(
            query=query,
            tool_usage=tool_usage,
            metadata=metadata or {},
        )

        # Run selected metrics
        if "relevance" in self.metrics:
            result.relevance = await compute_relevance_metrics(
                query=query,
                sources=web_sources,
                judge_model_config=self.judge_model_config,
            )
            self.usage.merge(result.relevance.usage)

        if "grounding" in self.metrics:
            result.grounding = await compute_grounding_metrics(
                report=report,
                sources=web_sources,
                judge_model_config=self.judge_model_config,
            )
            self.usage.merge(result.grounding.usage)

        if "content_attribution" in self.metrics:
            result.content_attribution = await compute_content_attribution_metrics(
                report=report,
                web_sources=web_sources,
                judge_model_config=self.judge_model_config,
                internal_sources=internal_sources,
            )
            self.usage.merge(result.content_attribution.usage)

        if "search_quality" in self.metrics and queries:
            result.search_quality = await compute_search_quality_metrics(
                research_task=query,
                queries=queries,
                results=web_sources,
                judge_model_config=self.judge_model_config,
                credits_used=credits_used,
            )
            self.usage.merge(result.search_quality.usage)

        # Compute overall score
        result.compute_overall_score(self.score_weights)

        return result


async def evaluate_research(
    query: str,
    report: str,
    web_sources: list[dict],
    judge_model_config: "ModelConfig",
    metrics: Optional[list[MetricType]] = None,
    internal_sources: Optional[list[dict]] = None,
    queries: Optional[list[str]] = None,
    credits_used: int = 0,
    tool_usage: Optional[dict] = None,
    metadata: Optional[dict] = None,
    score_weights: Optional[dict[str, float]] = None,
) -> EvalResult:
    """Convenience function to evaluate a research output.

    This is the primary entry point for evaluating research outputs.
    It creates a ResearchEvaluator and runs evaluation.

    Args:
        query: The original research query
        report: The generated report text
        web_sources: List of web source dicts with 'url', 'title', 'content'
        judge_model_config: ModelConfig for judge LLM calls
        metrics: List of metrics to compute. Defaults to all available.
        internal_sources: Optional internal RAG sources
        queries: Optional list of search queries used
        credits_used: API credits consumed
        tool_usage: Optional ToolUsageStats dict from the tool
        metadata: Optional additional metadata
        score_weights: Optional weights for overall score calculation

    Returns:
        EvalResult with computed metrics

    Example:
        from tavily_agent_toolkit import ModelConfig, ModelObject
        from tavily_agent_toolkit.evals import evaluate_research

        result = await evaluate_research(
            query="What are the latest AI trends?",
            report=generated_report,
            web_sources=sources,
            judge_model_config=ModelConfig(model=ModelObject(model="gpt-4o-mini")),
            metrics=["grounding", "relevance"],
        )

        print(f"Grounding: {result.grounding.grounding_ratio:.1%}")
        print(f"Relevance: {result.relevance.source_relevance_score:.2f}")
        print(f"Overall: {result.overall_score:.2f}")
    """
    evaluator = ResearchEvaluator(
        judge_model_config=judge_model_config,
        metrics=metrics,
        score_weights=score_weights,
    )

    return await evaluator.evaluate(
        query=query,
        report=report,
        web_sources=web_sources,
        internal_sources=internal_sources,
        queries=queries,
        credits_used=credits_used,
        tool_usage=tool_usage,
        metadata=metadata,
    )
