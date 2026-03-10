"""Data models for the evaluation framework.

These models follow the patterns established in agent-toolkit/models.py,
using dataclasses with to_dict() methods for JSON serialization.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class EvalUsage:
    """Tracks evaluation-specific usage metrics (LLM judge calls)."""
    judge_llm_calls: int = 0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    eval_response_time: float = 0.0

    @property
    def judge_total_tokens(self) -> int:
        """Total tokens used by judge LLM (input + output)."""
        return self.judge_input_tokens + self.judge_output_tokens

    def add_call(self, input_tokens: int, output_tokens: int, response_time: float) -> None:
        """Record a judge LLM call."""
        self.judge_llm_calls += 1
        self.judge_input_tokens += input_tokens
        self.judge_output_tokens += output_tokens
        self.eval_response_time += response_time

    def merge(self, other: "EvalUsage") -> None:
        """Merge another EvalUsage into this one."""
        self.judge_llm_calls += other.judge_llm_calls
        self.judge_input_tokens += other.judge_input_tokens
        self.judge_output_tokens += other.judge_output_tokens
        self.eval_response_time += other.eval_response_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "judge_llm_calls": self.judge_llm_calls,
            "judge_input_tokens": self.judge_input_tokens,
            "judge_output_tokens": self.judge_output_tokens,
            "judge_total_tokens": self.judge_total_tokens,
            "eval_response_time": round(self.eval_response_time, 3),
        }


# =============================================================================
# Claim and Citation Models (for Grounding)
# =============================================================================

@dataclass
class Claim:
    """A single claim extracted from a report."""
    text: str
    source_index: Optional[int] = None  # Index of supporting source, if any
    source_url: Optional[str] = None
    is_supported: bool = False
    confidence: float = 0.0  # Confidence that the claim is supported (0-1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "text": self.text,
            "is_supported": self.is_supported,
            "confidence": round(self.confidence, 3),
        }
        if self.source_index is not None:
            result["source_index"] = self.source_index
        if self.source_url is not None:
            result["source_url"] = self.source_url
        return result


@dataclass
class Citation:
    """A citation reference in the report."""
    claim_text: str
    cited_source_index: int
    cited_source_url: Optional[str] = None
    is_accurate: bool = False  # Whether the citation actually supports the claim
    reasoning: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "claim_text": self.claim_text,
            "cited_source_index": self.cited_source_index,
            "is_accurate": self.is_accurate,
        }
        if self.cited_source_url is not None:
            result["cited_source_url"] = self.cited_source_url
        if self.reasoning is not None:
            result["reasoning"] = self.reasoning
        return result


# =============================================================================
# Metric Result Models
# =============================================================================

@dataclass
class RelevanceResult:
    """Results from relevance evaluation."""
    source_relevance_score: float  # Avg relevance of sources to query (0-1)
    answer_coverage: float  # % of query aspects covered by sources (0-1)
    top_k_precision: float  # % of top-k sources that are relevant (0-1)
    k: int = 5  # The k used for top_k_precision
    per_source_scores: list[dict] = field(default_factory=list)
    usage: EvalUsage = field(default_factory=EvalUsage)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_relevance_score": round(self.source_relevance_score, 3),
            "answer_coverage": round(self.answer_coverage, 3),
            "top_k_precision": round(self.top_k_precision, 3),
            "k": self.k,
            "per_source_scores": self.per_source_scores,
            "usage": self.usage.to_dict(),
        }


@dataclass
class GroundingResult:
    """Results from grounding evaluation."""
    grounding_ratio: float  # % of claims attributed to sources (0-1)
    citation_accuracy: float  # % of citations correctly supporting claims (0-1)
    unsupported_claims_count: int  # Claims without source support
    total_claims: int
    supported_claims_count: int = 0
    claim_details: list[Claim] = field(default_factory=list)
    citation_details: list[Citation] = field(default_factory=list)
    usage: EvalUsage = field(default_factory=EvalUsage)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "grounding_ratio": round(self.grounding_ratio, 3),
            "citation_accuracy": round(self.citation_accuracy, 3),
            "unsupported_claims_count": self.unsupported_claims_count,
            "supported_claims_count": self.supported_claims_count,
            "total_claims": self.total_claims,
            "claim_details": [c.to_dict() for c in self.claim_details],
            "citation_details": [c.to_dict() for c in self.citation_details],
            "usage": self.usage.to_dict(),
        }


@dataclass
class ContentAttributionResult:
    """Results from content attribution evaluation."""
    web_content_ratio: float  # % from web sources (0-1)
    internal_content_ratio: float  # % from internal RAG (0-1)
    prior_knowledge_ratio: float  # % from model prior knowledge (0-1)
    source_diversity: float  # Unique domains / total sources (0-1)
    unique_domains: int = 0
    total_sources: int = 0
    attribution_breakdown: list[dict] = field(default_factory=list)
    usage: EvalUsage = field(default_factory=EvalUsage)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "web_content_ratio": round(self.web_content_ratio, 3),
            "internal_content_ratio": round(self.internal_content_ratio, 3),
            "prior_knowledge_ratio": round(self.prior_knowledge_ratio, 3),
            "source_diversity": round(self.source_diversity, 3),
            "unique_domains": self.unique_domains,
            "total_sources": self.total_sources,
            "attribution_breakdown": self.attribution_breakdown,
            "usage": self.usage.to_dict(),
        }


@dataclass
class SearchQualityResult:
    """Results from search quality evaluation."""
    query_quality_score: float  # Generated query effectiveness (0-1)
    result_coverage: float  # Topic coverage of results (0-1)
    deduplication_efficiency: float  # Unique content ratio (0-1)
    credit_efficiency: float  # Quality score / credits used
    queries_evaluated: int = 0
    results_count: int = 0
    unique_results_count: int = 0
    credits_used: int = 0
    query_details: list[dict] = field(default_factory=list)
    usage: EvalUsage = field(default_factory=EvalUsage)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query_quality_score": round(self.query_quality_score, 3),
            "result_coverage": round(self.result_coverage, 3),
            "deduplication_efficiency": round(self.deduplication_efficiency, 3),
            "credit_efficiency": round(self.credit_efficiency, 3),
            "queries_evaluated": self.queries_evaluated,
            "results_count": self.results_count,
            "unique_results_count": self.unique_results_count,
            "credits_used": self.credits_used,
            "query_details": self.query_details,
            "usage": self.usage.to_dict(),
        }


# =============================================================================
# Composite Evaluation Result
# =============================================================================

@dataclass
class EvalResult:
    """Complete evaluation result combining all metrics."""
    query: str
    relevance: Optional[RelevanceResult] = None
    grounding: Optional[GroundingResult] = None
    content_attribution: Optional[ContentAttributionResult] = None
    search_quality: Optional[SearchQualityResult] = None
    overall_score: float = 0.0
    tool_usage: Optional[dict] = None  # From ToolUsageStats
    metadata: dict = field(default_factory=dict)

    def compute_overall_score(
        self,
        weights: Optional[dict[str, float]] = None
    ) -> float:
        """Compute weighted overall score from available metrics.

        Args:
            weights: Optional dict of metric weights.
                     Default: equal weights for available metrics.

        Returns:
            Weighted average score (0-1).
        """
        default_weights = {
            "relevance": 0.25,
            "grounding": 0.35,
            "content_attribution": 0.20,
            "search_quality": 0.20,
        }
        weights = weights or default_weights

        scores: list[tuple[float, float]] = []  # (score, weight)

        if self.relevance is not None:
            scores.append((self.relevance.source_relevance_score, weights.get("relevance", 0.25)))
        if self.grounding is not None:
            scores.append((self.grounding.grounding_ratio, weights.get("grounding", 0.35)))
        if self.content_attribution is not None:
            # Use web_content_ratio as the primary attribution score
            scores.append((self.content_attribution.web_content_ratio, weights.get("content_attribution", 0.20)))
        if self.search_quality is not None:
            scores.append((self.search_quality.query_quality_score, weights.get("search_quality", 0.20)))

        if not scores:
            return 0.0

        # Normalize weights for available metrics
        total_weight = sum(w for _, w in scores)
        if total_weight == 0:
            return 0.0

        self.overall_score = sum(s * w for s, w in scores) / total_weight
        return self.overall_score

    def get_total_usage(self) -> EvalUsage:
        """Aggregate usage across all metrics."""
        total = EvalUsage()
        for metric_result in [self.relevance, self.grounding, self.content_attribution, self.search_quality]:
            if metric_result is not None:
                total.merge(metric_result.usage)
        return total

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "query": self.query,
            "overall_score": round(self.overall_score, 3),
        }

        if self.relevance is not None:
            result["relevance"] = self.relevance.to_dict()
        if self.grounding is not None:
            result["grounding"] = self.grounding.to_dict()
        if self.content_attribution is not None:
            result["content_attribution"] = self.content_attribution.to_dict()
        if self.search_quality is not None:
            result["search_quality"] = self.search_quality.to_dict()
        if self.tool_usage is not None:
            result["tool_usage"] = self.tool_usage
        if self.metadata:
            result["metadata"] = self.metadata

        result["total_usage"] = self.get_total_usage().to_dict()

        return result


# =============================================================================
# Batch Evaluation Results
# =============================================================================

@dataclass
class BatchEvalResult:
    """Results from evaluating a batch of queries."""
    results: list[EvalResult] = field(default_factory=list)
    aggregate_scores: dict[str, float] = field(default_factory=dict)
    total_usage: EvalUsage = field(default_factory=EvalUsage)

    def compute_aggregates(self) -> None:
        """Compute aggregate scores across all results."""
        if not self.results:
            return

        # Compute means for each metric
        relevance_scores = [r.relevance.source_relevance_score for r in self.results if r.relevance]
        grounding_scores = [r.grounding.grounding_ratio for r in self.results if r.grounding]
        attribution_scores = [r.content_attribution.web_content_ratio for r in self.results if r.content_attribution]
        search_scores = [r.search_quality.query_quality_score for r in self.results if r.search_quality]
        overall_scores = [r.overall_score for r in self.results if r.overall_score > 0]

        self.aggregate_scores = {}
        if relevance_scores:
            self.aggregate_scores["mean_relevance"] = sum(relevance_scores) / len(relevance_scores)
        if grounding_scores:
            self.aggregate_scores["mean_grounding"] = sum(grounding_scores) / len(grounding_scores)
        if attribution_scores:
            self.aggregate_scores["mean_attribution"] = sum(attribution_scores) / len(attribution_scores)
        if search_scores:
            self.aggregate_scores["mean_search_quality"] = sum(search_scores) / len(search_scores)
        if overall_scores:
            self.aggregate_scores["mean_overall"] = sum(overall_scores) / len(overall_scores)

        # Aggregate usage
        self.total_usage = EvalUsage()
        for r in self.results:
            self.total_usage.merge(r.get_total_usage())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "results": [r.to_dict() for r in self.results],
            "aggregate_scores": {k: round(v, 3) for k, v in self.aggregate_scores.items()},
            "total_usage": self.total_usage.to_dict(),
            "count": len(self.results),
        }


# =============================================================================
# Dataset Evaluation Results
# =============================================================================

@dataclass
class ItemResult:
    """Result from evaluating a single dataset item.

    Tracks the query, expected answer, predicted answer, and the
    correctness judgment.
    """
    query: str
    expected_answer: str
    predicted_answer: str
    grade: Literal["CORRECT", "INCORRECT"]
    score: float  # 1.0 for CORRECT, 0.0 for INCORRECT
    reasoning: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "query": self.query,
            "expected_answer": self.expected_answer,
            "predicted_answer": self.predicted_answer,
            "grade": self.grade,
            "score": self.score,
        }
        if self.reasoning is not None:
            result["reasoning"] = self.reasoning
        if self.category is not None:
            result["category"] = self.category
        if self.difficulty is not None:
            result["difficulty"] = self.difficulty
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class CategoryBreakdown:
    """Accuracy breakdown for a single category."""
    category: str
    accuracy: float
    correct_count: int
    incorrect_count: int
    total_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "category": self.category,
            "accuracy": round(self.accuracy, 3),
            "correct_count": self.correct_count,
            "incorrect_count": self.incorrect_count,
            "total_count": self.total_count,
        }


@dataclass
class DatasetEvalResult:
    """Results from evaluating a model against a dataset.

    Contains overall accuracy, per-item results, and optional
    breakdowns by category and difficulty.
    """
    accuracy: float  # correct_count / total_count (0-1)
    correct_count: int
    incorrect_count: int
    total_count: int
    items: list[ItemResult] = field(default_factory=list)
    by_category: dict[str, CategoryBreakdown] = field(default_factory=dict)
    by_difficulty: dict[str, CategoryBreakdown] = field(default_factory=dict)
    usage: EvalUsage = field(default_factory=EvalUsage)

    def get_incorrect_items(self) -> list[ItemResult]:
        """Get all incorrectly answered items for analysis."""
        return [item for item in self.items if item.grade == "INCORRECT"]

    def get_correct_items(self) -> list[ItemResult]:
        """Get all correctly answered items."""
        return [item for item in self.items if item.grade == "CORRECT"]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "accuracy": round(self.accuracy, 3),
            "correct_count": self.correct_count,
            "incorrect_count": self.incorrect_count,
            "total_count": self.total_count,
            "items": [item.to_dict() for item in self.items],
            "usage": self.usage.to_dict(),
        }
        if self.by_category:
            result["by_category"] = {k: v.to_dict() for k, v in self.by_category.items()}
        if self.by_difficulty:
            result["by_difficulty"] = {k: v.to_dict() for k, v in self.by_difficulty.items()}
        return result

    def to_dataframe(self) -> "Any":  # pandas.DataFrame
        """Convert item results to a pandas DataFrame for analysis.

        Returns:
            DataFrame with columns: query, expected_answer, predicted_answer,
            grade, score, reasoning, category, difficulty

        Example:
            result = await evaluator.evaluate(...)
            df = result.to_dataframe()
            incorrect_df = df[df["grade"] == "INCORRECT"]
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")

        rows = [item.to_dict() for item in self.items]
        return pd.DataFrame(rows)
