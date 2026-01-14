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

Utilities:
    - Various helper functions for LLM invocation, content cleaning, etc.

Models:
    - ModelConfig, ModelObject: Configuration for LLM models
    - OutputSchema: Base class for structured output schemas
    - Various TypedDicts for API responses
"""

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
]
