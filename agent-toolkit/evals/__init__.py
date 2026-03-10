"""Evaluation framework for measuring research quality.

This module provides tools for evaluating research outputs across four dimensions:
- Relevance: Source-query alignment scoring
- Grounding: Claim-to-source attribution
- Content Attribution: Web vs internal vs prior knowledge breakdown
- Search Quality: Query effectiveness, coverage, deduplication

Additionally, it supports custom dataset evaluation:
- Dataset Evaluation: Test any model against query-answer pair datasets

Example:
    from tavily_agent_toolkit.evals import evaluate_research, GroundingResult

    result = await evaluate_research(
        query="What are the latest AI trends?",
        report=generated_report,
        web_sources=sources,
        judge_model_config=ModelConfig(model=ModelObject(model="gpt-4o-mini")),
        metrics=["grounding", "relevance"],
    )
    print(f"Grounding: {result.grounding.grounding_ratio:.1%}")

Example (Dataset Evaluation):
    from tavily_agent_toolkit.evals import evaluate_dataset, CSVDataset

    dataset = CSVDataset("my_queries.csv")

    async def my_model(query: str) -> str:
        return "my answer"

    result = await evaluate_dataset(
        dataset=dataset,
        model=my_model,
        judge_model_config=ModelConfig(model=ModelObject(model="gpt-4o-mini")),
    )
    print(f"Accuracy: {result.accuracy:.1%}")
"""

try:
    from .models import (
        EvalUsage,
        Claim,
        Citation,
        RelevanceResult,
        GroundingResult,
        ContentAttributionResult,
        SearchQualityResult,
        EvalResult,
        BatchEvalResult,
        # Dataset evaluation models
        ItemResult,
        CategoryBreakdown,
        DatasetEvalResult,
    )

    from .judges import (
        BaseJudge,
        GroundingJudge,
        RelevanceJudge,
        QualityJudge,
        CorrectnessJudge,
    )

    from .metrics import (
        compute_grounding_metrics,
        compute_relevance_metrics,
        compute_content_attribution_metrics,
        compute_search_quality_metrics,
        compute_correctness_metrics,
    )

    from .evaluators import (
        BaseEvaluator,
        ResearchEvaluator,
        RetrievalEvaluator,
        evaluate_research,
        DatasetEvaluator,
        evaluate_dataset,
    )

    from .datasets import (
        DatasetItem,
        BaseDataset,
        FilteredDataset,
        InMemoryDataset,
        CSVDataset,
    )

except ImportError:
    # Direct import (e.g., pytest running from agent-toolkit directory)
    pass

__all__ = [
    # Models
    "EvalUsage",
    "Claim",
    "Citation",
    "RelevanceResult",
    "GroundingResult",
    "ContentAttributionResult",
    "SearchQualityResult",
    "EvalResult",
    "BatchEvalResult",
    "ItemResult",
    "CategoryBreakdown",
    "DatasetEvalResult",
    # Judges
    "BaseJudge",
    "GroundingJudge",
    "RelevanceJudge",
    "QualityJudge",
    "CorrectnessJudge",
    # Metrics
    "compute_grounding_metrics",
    "compute_relevance_metrics",
    "compute_content_attribution_metrics",
    "compute_search_quality_metrics",
    "compute_correctness_metrics",
    # Evaluators
    "BaseEvaluator",
    "ResearchEvaluator",
    "RetrievalEvaluator",
    "evaluate_research",
    "DatasetEvaluator",
    "evaluate_dataset",
    # Datasets
    "DatasetItem",
    "BaseDataset",
    "FilteredDataset",
    "InMemoryDataset",
    "CSVDataset",
]
