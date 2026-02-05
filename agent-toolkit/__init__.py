# Tavily Agent Toolkit
"""
A collection of tools and agents for building AI applications with Tavily.

Tools:
    - search_and_answer: Search the web and answer questions with LLM synthesis
    - search_and_format: Search the web and return formatted results
    - search_dedup: Async search with deduplication by URL
    - crawl_and_summarize: Crawl websites and summarize content
    - extract_and_summarize: Extract content from URLs and summarize
    - social_media_search: Search social media platforms

Agents:
    - hybrid_research: Combine internal RAG with web search

Evals:
    - evaluate_research: Evaluate research outputs for grounding, relevance, etc.
    - ResearchEvaluator: Batch evaluation orchestrator
    - Various metrics: grounding, relevance, content_attribution, search_quality

Utilities:
    - Various helper functions for LLM invocation, content cleaning, etc.

Models:
    - ModelConfig, ModelObject: Configuration for LLM models
    - OutputSchema: Base class for structured output schemas
    - Various TypedDicts for API responses
"""

# Handle both package imports (via pip install) and direct imports (pytest)
# When imported as a package, relative imports work. When imported directly, they fail.
try:
    from .models import (
        ModelConfig,
        ModelObject,
        ModelProvider,
        OutputSchema,
        SearchResult,
        ImageResult,
        SearchDedupResponse,
        WebSource,
        HybridResearchResponse,
        TavilyAPIResponse,
        TavilyUsage,
        LLMUsage,
        LLMResponse,
        ToolUsageStats,
    )

    from .tools import (
        search_and_answer,
        search_and_format,
        search_dedup,
        crawl_and_summarize,
        extract_and_summarize,
        social_media_search,
    )

    from .agents import hybrid_research

    from .utilities import (
        ainvoke_with_fallback,
        clean_raw_content,
        clean_formatted_output,
        count_tokens,
        summarize_long_content,
        generate_subqueries,
        synthesize_results,
        format_web_results,
        handle_research_stream,
    )

    # Evals module (optional - may not have all dependencies)
    try:
        from .evals import (
            # Evaluators
            evaluate_research,
            ResearchEvaluator,
            RetrievalEvaluator,
            # Metrics
            compute_grounding_metrics,
            compute_relevance_metrics,
            compute_content_attribution_metrics,
            compute_search_quality_metrics,
            # Models
            EvalResult,
            EvalUsage,
            GroundingResult,
            RelevanceResult,
            ContentAttributionResult,
            SearchQualityResult,
            BatchEvalResult,
            # Datasets
            load_eval_dataset,
            EvalTestCase,
            EvalDataset,
        )
    except ImportError:
        # Evals dependencies not available
        pass
except ImportError:
    # Direct import (e.g., pytest running from agent-toolkit directory)
    # Skip imports - tests should import from tavily_agent_toolkit package directly
    pass

__version__ = "0.1.0"

__all__ = [
    # Models
    "ModelConfig",
    "ModelObject",
    "ModelProvider",
    "OutputSchema",
    "SearchResult",
    "ImageResult",
    "SearchDedupResponse",
    "WebSource",
    "HybridResearchResponse",
    "TavilyAPIResponse",
    "TavilyUsage",
    "LLMUsage",
    "LLMResponse",
    "ToolUsageStats",
    # Tools
    "search_and_answer",
    "search_and_format",
    "search_dedup",
    "crawl_and_summarize",
    "extract_and_summarize",
    "social_media_search",
    # Agents
    "hybrid_research",
    # Utilities
    "ainvoke_with_fallback",
    "clean_raw_content",
    "clean_formatted_output",
    "count_tokens",
    "summarize_long_content",
    "generate_subqueries",
    "synthesize_results",
    "format_web_results",
    "handle_research_stream",
    # Evals (optional)
    "evaluate_research",
    "ResearchEvaluator",
    "RetrievalEvaluator",
    "compute_grounding_metrics",
    "compute_relevance_metrics",
    "compute_content_attribution_metrics",
    "compute_search_quality_metrics",
    "EvalResult",
    "EvalUsage",
    "GroundingResult",
    "RelevanceResult",
    "ContentAttributionResult",
    "SearchQualityResult",
    "BatchEvalResult",
    "load_eval_dataset",
    "EvalTestCase",
    "EvalDataset",
]
