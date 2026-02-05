"""Content attribution metrics computation.

Measures the breakdown of content sources in a report:
- Web sources (external/real-time data)
- Internal RAG sources (company knowledge base)
- Model prior knowledge (LLM's training data)

This helps quantify the value of web data in research outputs.
"""

from typing import Any, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from ..models import ContentAttributionResult, EvalUsage
from ..judges.base import BaseJudge

# Try both package and relative imports
try:
    from tavily_agent_toolkit import ModelConfig
except ImportError:
    try:
        from ...models import ModelConfig
    except ImportError:
        ModelConfig = Any  # type: ignore


class ContentSegment(BaseModel):
    """A segment of content with its attribution."""
    text: str = Field(description="The text segment from the report")
    source_type: str = Field(description="Source type: 'web', 'internal', or 'prior_knowledge'")
    source_url: Optional[str] = Field(default=None, description="URL if web source")
    confidence: float = Field(description="Confidence in the attribution (0-1)")
    reasoning: str = Field(description="Brief explanation for the attribution")


class ContentAttributionOutput(BaseModel):
    """Output schema for content attribution analysis."""
    segments: list[ContentSegment] = Field(description="List of content segments with attributions")
    overall_web_ratio: float = Field(description="Estimated percentage of content from web sources (0-1)")
    overall_internal_ratio: float = Field(description="Estimated percentage of content from internal sources (0-1)")
    overall_prior_ratio: float = Field(description="Estimated percentage of content from model prior knowledge (0-1)")


ATTRIBUTION_PROMPT = """You are an expert at analyzing the provenance of content in research reports.

Your task is to determine what percentage of a report's content comes from:
1. **Web sources**: Information that appears to come from the provided external web sources
2. **Internal sources**: Information that would come from internal company/organization knowledge bases (if any internal sources are provided)
3. **Prior knowledge**: Information that comes from the LLM's training data, general knowledge, or synthesis that goes beyond what sources provide

Guidelines:
- Content that directly quotes or closely paraphrases a source = attributed to that source
- Content that synthesizes multiple sources in a way clearly supported by those sources = web/internal
- Content that provides context, definitions, or background not in sources = prior knowledge
- Content that makes logical inferences not explicitly stated in sources = prior knowledge
- When a claim could come from sources OR prior knowledge, favor source attribution if plausible

Be precise in your analysis. The goal is to quantify how much value web/internal data adds to the output.
"""


class ContentAttributionJudge(BaseJudge):
    """Judge for analyzing content attribution in reports."""

    def __init__(
        self,
        model_config: "ModelConfig",
        attribution_prompt: Optional[str] = None,
    ):
        """Initialize the content attribution judge.

        Args:
            model_config: ModelConfig for the judge LLM
            attribution_prompt: Custom prompt for attribution analysis
        """
        super().__init__(model_config, system_prompt=attribution_prompt or ATTRIBUTION_PROMPT)

    async def analyze_attribution(
        self,
        report: str,
        web_sources: list[dict],
        internal_sources: Optional[list[dict]] = None,
    ) -> ContentAttributionOutput:
        """Analyze content attribution in a report.

        Args:
            report: The report text to analyze
            web_sources: List of web source dicts with 'url', 'title', 'content'
            internal_sources: Optional list of internal source dicts

        Returns:
            ContentAttributionOutput with segment breakdowns
        """
        # Format sources
        sources_text = "\n--- WEB SOURCES ---\n"
        for i, source in enumerate(web_sources):
            title = source.get("title", f"Web Source {i+1}")
            url = source.get("url", "")
            content = source.get("content", "")[:2000]  # Truncate for context limits
            sources_text += f"\n[Web {i+1}] {title}\nURL: {url}\n{content}\n"

        if internal_sources:
            sources_text += "\n--- INTERNAL SOURCES ---\n"
            for i, source in enumerate(internal_sources):
                title = source.get("title", f"Internal Source {i+1}")
                content = source.get("content", "")[:2000]
                sources_text += f"\n[Internal {i+1}] {title}\n{content}\n"

        messages = [
            {"role": "user", "content": f"""Analyze the content attribution of this report:

REPORT:
{report}

AVAILABLE SOURCES:
{sources_text}

For the report above, determine what percentage of the content comes from:
1. Web sources (the external sources listed above)
2. Internal sources (if any internal sources were provided)
3. Prior knowledge (LLM training data, general knowledge, or synthesis beyond sources)

Identify key segments and their likely sources."""},
        ]

        result = await self.invoke_llm(messages, output_schema=ContentAttributionOutput)
        return result

    async def judge(
        self,
        report: str,
        web_sources: list[dict],
        internal_sources: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> dict:
        """Perform content attribution analysis.

        Args:
            report: The report text
            web_sources: List of web source dicts
            internal_sources: Optional internal source dicts
            **kwargs: Additional arguments (unused)

        Returns:
            Dict with attribution results and usage
        """
        result = await self.analyze_attribution(report, web_sources, internal_sources)
        return {
            "attribution": result,
            "usage": self.get_usage(),
        }


async def compute_content_attribution_metrics(
    report: str,
    web_sources: list[dict],
    judge_model_config: "ModelConfig",
    internal_sources: Optional[list[dict]] = None,
    attribution_prompt: Optional[str] = None,
) -> ContentAttributionResult:
    """Compute content attribution metrics for a report.

    This function analyzes what percentage of a report's content comes from:
    - Web sources (external real-time data)
    - Internal sources (RAG/knowledge base)
    - Model prior knowledge (training data)

    Args:
        report: The report text to analyze
        web_sources: List of web source dicts with 'url', 'title', 'content'
        judge_model_config: ModelConfig for the judge LLM
        internal_sources: Optional list of internal RAG source dicts
        attribution_prompt: Custom prompt for attribution analysis

    Returns:
        ContentAttributionResult with breakdown metrics

    Example:
        result = await compute_content_attribution_metrics(
            report="According to recent data, AI spending increased 40%...",
            web_sources=[{"url": "...", "content": "..."}],
            judge_model_config=ModelConfig(model=ModelObject(model="gpt-4o-mini")),
        )
        print(f"Web content: {result.web_content_ratio:.1%}")
        print(f"Prior knowledge: {result.prior_knowledge_ratio:.1%}")
    """
    judge = ContentAttributionJudge(
        model_config=judge_model_config,
        attribution_prompt=attribution_prompt,
    )

    judge_result = await judge.judge(
        report=report,
        web_sources=web_sources,
        internal_sources=internal_sources,
    )

    attribution = judge_result["attribution"]
    usage = judge_result["usage"]

    # Compute source diversity
    unique_domains = _count_unique_domains(web_sources)
    total_sources = len(web_sources) + (len(internal_sources) if internal_sources else 0)
    source_diversity = unique_domains / total_sources if total_sources > 0 else 0.0

    # Build attribution breakdown
    attribution_breakdown = []
    for segment in attribution.segments:
        attribution_breakdown.append({
            "text_preview": segment.text[:100] + "..." if len(segment.text) > 100 else segment.text,
            "source_type": segment.source_type,
            "source_url": segment.source_url,
            "confidence": segment.confidence,
        })

    return ContentAttributionResult(
        web_content_ratio=attribution.overall_web_ratio,
        internal_content_ratio=attribution.overall_internal_ratio,
        prior_knowledge_ratio=attribution.overall_prior_ratio,
        source_diversity=source_diversity,
        unique_domains=unique_domains,
        total_sources=total_sources,
        attribution_breakdown=attribution_breakdown,
        usage=usage,
    )


def _count_unique_domains(sources: list[dict]) -> int:
    """Count unique domains from source URLs."""
    domains = set()
    for source in sources:
        url = source.get("url", "")
        if url:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                # Remove www. prefix for deduplication
                if domain.startswith("www."):
                    domain = domain[4:]
                domains.add(domain)
            except Exception:
                pass
    return len(domains)


def compute_attribution_from_ratios(
    web_ratio: float,
    internal_ratio: float,
    prior_ratio: float,
    web_sources: Optional[list[dict]] = None,
    internal_sources: Optional[list[dict]] = None,
) -> ContentAttributionResult:
    """Compute attribution result from pre-calculated ratios.

    Use this when ratios have been computed externally or manually.

    Args:
        web_ratio: Ratio of content from web sources (0-1)
        internal_ratio: Ratio of content from internal sources (0-1)
        prior_ratio: Ratio of content from prior knowledge (0-1)
        web_sources: Optional web sources for diversity calculation
        internal_sources: Optional internal sources for count

    Returns:
        ContentAttributionResult with the provided ratios
    """
    # Normalize ratios to sum to 1.0
    total = web_ratio + internal_ratio + prior_ratio
    if total > 0:
        web_ratio = web_ratio / total
        internal_ratio = internal_ratio / total
        prior_ratio = prior_ratio / total

    # Compute source diversity
    unique_domains = _count_unique_domains(web_sources) if web_sources else 0
    total_sources = (len(web_sources) if web_sources else 0) + (len(internal_sources) if internal_sources else 0)
    source_diversity = unique_domains / total_sources if total_sources > 0 else 0.0

    return ContentAttributionResult(
        web_content_ratio=web_ratio,
        internal_content_ratio=internal_ratio,
        prior_knowledge_ratio=prior_ratio,
        source_diversity=source_diversity,
        unique_domains=unique_domains,
        total_sources=total_sources,
        attribution_breakdown=[],
        usage=EvalUsage(),
    )
