"""Content attribution metrics computation.

Measures the breakdown of content sources in a report:
- Web sources (external/real-time data)
- Internal RAG sources (company knowledge base)
- Model prior knowledge (LLM's training data)

The report is split into sentences, then an LLM judge classifies each
sentence as originating from a web source, internal source, or prior
knowledge.
"""

import re
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


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences (keeps only non-trivial ones)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip() and len(s.strip()) > 15]


# ---------------------------------------------------------------------------
# Pydantic output schemas for structured LLM response
# ---------------------------------------------------------------------------

class ChunkAttribution(BaseModel):
    """Attribution judgment for a single report sentence."""
    chunk_index: int = Field(description="0-based index of the sentence being classified")
    source_type: str = Field(description="One of: 'web', 'internal', 'prior_knowledge'")
    source_label: Optional[str] = Field(
        default=None,
        description="Label of the matched source (e.g. 'Web 1', 'Internal 2') or null for prior_knowledge",
    )
    confidence: float = Field(description="Confidence in the attribution (0.0 to 1.0)")
    reasoning: str = Field(description="Brief explanation for why this attribution was chosen")


class AttributionOutput(BaseModel):
    """Structured output from the attribution judge."""
    chunks: list[ChunkAttribution] = Field(
        description="One entry per sentence in the report, in order"
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

ATTRIBUTION_SYSTEM_PROMPT = """\
You are an expert at tracing the provenance of sentences in research reports.

You will receive:
1. A list of **numbered sentences** extracted from a report.
2. A list of **numbered sources** (web and/or internal).

For EACH sentence, determine which source (if any) it most likely came from:
- **web** — the sentence's information can be traced to one of the web sources.
- **internal** — the sentence's information can be traced to one of the internal sources.
- **prior_knowledge** — the sentence is general knowledge, definitions, or synthesis
  that goes beyond what the provided sources contain.

Guidelines:
- Direct quotes or close paraphrases of a source → attribute to that source.
- Specific facts, numbers, or claims that appear in a source → attribute to that source.
- General context, definitions, or background not present in any source → prior_knowledge.
- When a claim could plausibly come from a source OR prior knowledge, favour source attribution.
- Set confidence between 0.0 and 1.0 reflecting how certain you are.
"""


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

class ContentAttributionJudge(BaseJudge):
    """LLM judge that classifies each report sentence by source."""

    def __init__(self, model_config: "ModelConfig") -> None:
        super().__init__(model_config, system_prompt=ATTRIBUTION_SYSTEM_PROMPT)

    async def classify_chunks(
        self,
        chunks: list[str],
        web_sources: list[dict],
        internal_sources: list[dict],
    ) -> AttributionOutput:
        """Send all chunks + sources to the LLM in one call.

        Returns structured ``AttributionOutput``.
        """
        # ---- Build the user message ----
        lines: list[str] = ["REPORT SENTENCES:"]
        for i, chunk in enumerate(chunks):
            lines.append(f"  [{i}] {chunk}")

        lines.append("")
        lines.append("WEB SOURCES:")
        for i, src in enumerate(web_sources):
            title = src.get("title", f"Web Source {i + 1}")
            url = src.get("url", "")
            content = src.get("content", "")[:2000]
            lines.append(f"  [Web {i}] {title}  ({url})")
            lines.append(f"    {content}")

        if internal_sources:
            lines.append("")
            lines.append("INTERNAL SOURCES:")
            for i, src in enumerate(internal_sources):
                title = src.get("title", f"Internal Source {i + 1}")
                content = src.get("content", "")[:2000]
                lines.append(f"  [Internal {i}] {title}")
                lines.append(f"    {content}")

        lines.append("")
        lines.append(
            "Classify each sentence. Return one entry per sentence index."
        )

        messages = [{"role": "user", "content": "\n".join(lines)}]
        result: AttributionOutput = await self.invoke_llm(
            messages, output_schema=AttributionOutput
        )
        return result

    async def judge(self, **kwargs: Any) -> dict:
        """Required by BaseJudge ABC — delegates to classify_chunks."""
        chunks = kwargs["chunks"]
        web_sources = kwargs["web_sources"]
        internal_sources = kwargs.get("internal_sources", [])
        result = await self.classify_chunks(chunks, web_sources, internal_sources)
        return {"attribution": result, "usage": self.get_usage()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def compute_content_attribution_metrics(
    report: str,
    web_sources: list[dict],
    judge_model_config: "ModelConfig",
    internal_sources: Optional[list[dict]] = None,
    **kwargs: Any,
) -> ContentAttributionResult:
    """Compute content attribution by chunking the report then using an LLM judge.

    1. The report is split into sentences.
    2. All sentences + sources are sent to the LLM in a single call.
    3. The LLM classifies each sentence as *web*, *internal*, or
       *prior_knowledge*.
    4. Ratios are computed by character count of attributed sentences.

    Args:
        report: The report text to analyse.
        web_sources: Web source dicts (``url``, ``title``, ``content``).
        judge_model_config: :class:`ModelConfig` for the judge LLM.
        internal_sources: Optional internal/RAG source dicts
            (``title``, ``content``).
        **kwargs: Forwarded to the judge (unused today).

    Returns:
        :class:`ContentAttributionResult` with ratio breakdowns and
        per-sentence attribution details.

    Example::

        result = await compute_content_attribution_metrics(
            report="According to recent data, AI spending increased 40%...",
            web_sources=[{"url": "...", "content": "..."}],
            judge_model_config=ModelConfig(
                model=ModelObject(model="gpt-4o-mini", api_key="..."),
            ),
        )
        print(f"Web content: {result.web_content_ratio:.1%}")
    """
    internal_sources = internal_sources or []

    # ---- 1. Chunk the report ----
    chunks = _split_into_sentences(report)
    if not chunks:
        return ContentAttributionResult(
            web_content_ratio=0.0,
            internal_content_ratio=0.0,
            prior_knowledge_ratio=1.0,
            source_diversity=0.0,
            unique_domains=0,
            total_sources=len(web_sources) + len(internal_sources),
            attribution_breakdown=[],
            usage=EvalUsage(),
        )

    # ---- 2. LLM judge classifies each chunk ----
    judge = ContentAttributionJudge(model_config=judge_model_config)
    llm_result = await judge.classify_chunks(chunks, web_sources, internal_sources)
    usage = judge.get_usage()

    # ---- 3. Aggregate results ----
    # Build a lookup from chunk index -> LLM judgment
    judgments: dict[int, ChunkAttribution] = {}
    for entry in llm_result.chunks:
        judgments[entry.chunk_index] = entry

    attribution_breakdown: list[dict] = []
    web_chars = 0
    internal_chars = 0
    prior_chars = 0

    for i, chunk in enumerate(chunks):
        chunk_len = len(chunk)
        entry = judgments.get(i)

        if entry is None:
            # LLM didn't return a judgment for this chunk — treat as prior
            source_type = "prior_knowledge"
            confidence = 0.0
            source_ref = None
        else:
            source_type = entry.source_type
            confidence = entry.confidence
            source_ref = entry.source_label

        if source_type == "web":
            web_chars += chunk_len
        elif source_type == "internal":
            internal_chars += chunk_len
        else:
            prior_chars += chunk_len

        attribution_breakdown.append({
            "text_preview": (chunk[:100] + "...") if len(chunk) > 100 else chunk,
            "source_type": source_type,
            "source_url": source_ref,
            "confidence": round(confidence, 2),
        })

    # ---- 4. Compute ratios ----
    total_chars = web_chars + internal_chars + prior_chars or 1
    web_ratio = web_chars / total_chars
    internal_ratio = internal_chars / total_chars
    prior_ratio = prior_chars / total_chars

    unique_domains = _count_unique_domains(web_sources)
    total_sources = len(web_sources) + len(internal_sources)
    source_diversity = unique_domains / total_sources if total_sources > 0 else 0.0

    return ContentAttributionResult(
        web_content_ratio=web_ratio,
        internal_content_ratio=internal_ratio,
        prior_knowledge_ratio=prior_ratio,
        source_diversity=source_diversity,
        unique_domains=unique_domains,
        total_sources=total_sources,
        attribution_breakdown=attribution_breakdown,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_unique_domains(sources: list[dict]) -> int:
    """Count unique domains from source URLs."""
    domains: set[str] = set()
    for source in sources:
        url = source.get("url", "")
        if url:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
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
    """Build a :class:`ContentAttributionResult` from pre-calculated ratios.

    Use when ratios have been computed externally or manually.

    Args:
        web_ratio: Ratio of content from web sources (0-1).
        internal_ratio: Ratio of content from internal sources (0-1).
        prior_ratio: Ratio of content from prior knowledge (0-1).
        web_sources: Optional web sources for diversity calculation.
        internal_sources: Optional internal sources for count.

    Returns:
        ContentAttributionResult with the provided ratios.
    """
    total = web_ratio + internal_ratio + prior_ratio
    if total > 0:
        web_ratio /= total
        internal_ratio /= total
        prior_ratio /= total

    unique_domains = _count_unique_domains(web_sources) if web_sources else 0
    total_sources = (len(web_sources) if web_sources else 0) + (
        len(internal_sources) if internal_sources else 0
    )
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
