"""Grounding metrics computation.

Measures how well a report's claims are supported by sources.
"""

from typing import Any, Optional

from ..models import GroundingResult, EvalUsage
from ..judges.grounding_judge import GroundingJudge

# Try both package and relative imports
try:
    from tavily_agent_toolkit import ModelConfig
except ImportError:
    try:
        from ...models import ModelConfig
    except ImportError:
        ModelConfig = Any  # type: ignore


async def compute_grounding_metrics(
    report: str,
    sources: list[dict],
    judge_model_config: "ModelConfig",
    verify_citations: bool = True,
    extraction_prompt: Optional[str] = None,
    verification_prompt: Optional[str] = None,
) -> GroundingResult:
    """Compute grounding metrics for a report.

    This function:
    1. Extracts factual claims from the report
    2. Verifies each claim against provided sources
    3. Optionally verifies existing citations
    4. Computes grounding ratio and citation accuracy

    Args:
        report: The report text to evaluate
        sources: List of source dicts with 'content' (and optionally 'url', 'title')
        judge_model_config: ModelConfig for the judge LLM
        verify_citations: Whether to verify existing citations in the report
        extraction_prompt: Custom prompt for claim extraction
        verification_prompt: Custom prompt for claim verification

    Returns:
        GroundingResult with metrics and claim details

    Example:
        result = await compute_grounding_metrics(
            report="NVIDIA reported $26B revenue in Q3 2024 [1]...",
            sources=[{"url": "...", "title": "...", "content": "..."}],
            judge_model_config=ModelConfig(model=ModelObject(model="gpt-4o-mini")),
        )
        print(f"Grounding ratio: {result.grounding_ratio:.1%}")
        print(f"Unsupported claims: {result.unsupported_claims_count}")
    """
    # Initialize judge
    judge = GroundingJudge(
        model_config=judge_model_config,
        extraction_prompt=extraction_prompt,
        verification_prompt=verification_prompt,
    )

    # Perform evaluation
    judge_result = await judge.judge(
        report=report,
        sources=sources,
        verify_citations=verify_citations,
    )

    claims = judge_result["claims"]
    citations = judge_result.get("citations", [])
    usage = judge_result["usage"]

    # Compute metrics
    total_claims = len(claims)
    supported_claims = [c for c in claims if c.is_supported]
    unsupported_claims = [c for c in claims if not c.is_supported]

    grounding_ratio = len(supported_claims) / total_claims if total_claims > 0 else 1.0

    # Citation accuracy: of the citations that exist, how many are accurate?
    total_citations = len(citations)
    accurate_citations = [c for c in citations if c.is_accurate]
    citation_accuracy = len(accurate_citations) / total_citations if total_citations > 0 else 1.0

    return GroundingResult(
        grounding_ratio=grounding_ratio,
        citation_accuracy=citation_accuracy,
        unsupported_claims_count=len(unsupported_claims),
        supported_claims_count=len(supported_claims),
        total_claims=total_claims,
        claim_details=claims,
        citation_details=citations,
        usage=usage,
    )


def compute_grounding_from_claims(
    claims: list[dict],
    citations: Optional[list[dict]] = None,
) -> GroundingResult:
    """Compute grounding metrics from pre-extracted claims.

    Use this when claims have already been extracted and verified externally.

    Args:
        claims: List of claim dicts with 'text', 'is_supported', 'confidence'
        citations: Optional list of citation dicts with 'claim_text', 'is_accurate'

    Returns:
        GroundingResult with computed metrics
    """
    from ..models import Claim, Citation

    # Convert dicts to Claim objects
    claim_objects = []
    for c in claims:
        claim = Claim(
            text=c.get("text", ""),
            is_supported=c.get("is_supported", False),
            confidence=c.get("confidence", 0.0),
            source_index=c.get("source_index"),
            source_url=c.get("source_url"),
        )
        claim_objects.append(claim)

    # Convert citation dicts
    citation_objects = []
    if citations:
        for c in citations:
            citation = Citation(
                claim_text=c.get("claim_text", ""),
                cited_source_index=c.get("cited_source_index", 0),
                cited_source_url=c.get("cited_source_url"),
                is_accurate=c.get("is_accurate", False),
                reasoning=c.get("reasoning"),
            )
            citation_objects.append(citation)

    # Compute metrics
    total_claims = len(claim_objects)
    supported = [c for c in claim_objects if c.is_supported]

    grounding_ratio = len(supported) / total_claims if total_claims > 0 else 1.0

    total_citations = len(citation_objects)
    accurate = [c for c in citation_objects if c.is_accurate]
    citation_accuracy = len(accurate) / total_citations if total_citations > 0 else 1.0

    return GroundingResult(
        grounding_ratio=grounding_ratio,
        citation_accuracy=citation_accuracy,
        unsupported_claims_count=total_claims - len(supported),
        supported_claims_count=len(supported),
        total_claims=total_claims,
        claim_details=claim_objects,
        citation_details=citation_objects,
        usage=EvalUsage(),  # No LLM usage for pre-computed claims
    )
