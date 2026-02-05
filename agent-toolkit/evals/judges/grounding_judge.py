"""Grounding judge for evaluating claim-to-source attribution."""

from typing import Any, Optional

from pydantic import BaseModel, Field

from .base import BaseJudge
from ..models import Claim, Citation, EvalUsage

# Try both package and relative imports
try:
    from tavily_agent_toolkit import ModelConfig
except ImportError:
    try:
        from ...models import ModelConfig
    except ImportError:
        ModelConfig = Any  # type: ignore


class ExtractedClaim(BaseModel):
    """Schema for a single extracted claim."""
    claim_text: str = Field(description="The factual claim extracted from the text")
    requires_source: bool = Field(description="Whether this claim requires external source support (vs. being common knowledge or definitional)")


class ClaimsExtractionOutput(BaseModel):
    """Schema for claim extraction output."""
    claims: list[ExtractedClaim] = Field(description="List of factual claims extracted from the report")


class ClaimVerification(BaseModel):
    """Schema for verifying a single claim against sources."""
    is_supported: bool = Field(description="Whether the claim is supported by the provided sources")
    supporting_source_index: Optional[int] = Field(default=None, description="Index of the source that supports this claim (0-indexed), or null if not supported")
    confidence: float = Field(description="Confidence score (0-1) that the claim is supported")
    reasoning: str = Field(description="Brief explanation of why the claim is or is not supported")


class CitationVerification(BaseModel):
    """Schema for verifying a citation's accuracy."""
    is_accurate: bool = Field(description="Whether the cited source actually supports the claim")
    reasoning: str = Field(description="Brief explanation of the verification result")


CLAIM_EXTRACTION_PROMPT = """You are an expert at extracting factual claims from research reports.

Extract all factual claims from the following report that could be verified against external sources.
Focus on claims about:
- Statistics and numbers
- Events and dates
- Technical specifications
- Company or product information
- Research findings
- Quotes or attributed statements

Do NOT include:
- Common knowledge (e.g., "water is wet")
- Definitions (e.g., "AI stands for Artificial Intelligence")
- Opinions or subjective statements
- Meta-statements about the report itself

For each claim, indicate whether it requires external source verification.
"""

CLAIM_VERIFICATION_PROMPT = """You are an expert fact-checker. Your task is to verify whether a claim is supported by the provided sources.

A claim is "supported" if:
1. The source explicitly states the same fact, OR
2. The source strongly implies the fact with clear evidence

A claim is "not supported" if:
1. No source mentions this information, OR
2. Sources contradict the claim, OR
3. The claim goes beyond what sources actually state

Be strict: partial support or vague references do not count as full support.
"""


class GroundingJudge(BaseJudge):
    """Judge for evaluating grounding of claims in sources.

    This judge:
    1. Extracts factual claims from a report
    2. Verifies each claim against provided sources
    3. Optionally verifies existing citations for accuracy

    Example:
        judge = GroundingJudge(model_config=ModelConfig(...))
        claims = await judge.extract_claims(report_text)
        verified = await judge.verify_claims(claims, sources)
    """

    def __init__(
        self,
        model_config: "ModelConfig",
        extraction_prompt: Optional[str] = None,
        verification_prompt: Optional[str] = None,
    ):
        """Initialize the grounding judge.

        Args:
            model_config: ModelConfig for the judge LLM
            extraction_prompt: Custom prompt for claim extraction
            verification_prompt: Custom prompt for claim verification
        """
        super().__init__(model_config)
        self.extraction_prompt = extraction_prompt or CLAIM_EXTRACTION_PROMPT
        self.verification_prompt = verification_prompt or CLAIM_VERIFICATION_PROMPT

    async def extract_claims(self, report: str) -> list[Claim]:
        """Extract factual claims from a report.

        Args:
            report: The report text to extract claims from

        Returns:
            List of Claim objects
        """
        messages = [
            {"role": "system", "content": self.extraction_prompt},
            {"role": "user", "content": f"Extract factual claims from this report:\n\n{report}"},
        ]

        result = await self.invoke_llm(messages, output_schema=ClaimsExtractionOutput)

        claims = []
        for extracted in result.claims:
            if extracted.requires_source:
                claims.append(Claim(text=extracted.claim_text))

        return claims

    async def verify_claim(
        self,
        claim: Claim,
        sources: list[dict],
    ) -> Claim:
        """Verify a single claim against sources.

        Args:
            claim: The claim to verify
            sources: List of source dicts with 'content' and optionally 'url', 'title'

        Returns:
            Updated Claim with verification results
        """
        # Format sources for the prompt
        sources_text = ""
        for i, source in enumerate(sources):
            title = source.get("title", f"Source {i}")
            url = source.get("url", "")
            content = source.get("content", "")
            sources_text += f"\n--- Source {i}: {title} ---\n"
            if url:
                sources_text += f"URL: {url}\n"
            sources_text += f"{content}\n"

        messages = [
            {"role": "system", "content": self.verification_prompt},
            {"role": "user", "content": f"""Verify this claim against the sources:

CLAIM: {claim.text}

SOURCES:
{sources_text}

Determine if this claim is supported by any of the sources."""},
        ]

        result = await self.invoke_llm(messages, output_schema=ClaimVerification)

        # Update claim with verification results
        claim.is_supported = result.is_supported
        claim.confidence = result.confidence
        if result.supporting_source_index is not None:
            claim.source_index = result.supporting_source_index
            if result.supporting_source_index < len(sources):
                claim.source_url = sources[result.supporting_source_index].get("url")

        return claim

    async def verify_claims(
        self,
        claims: list[Claim],
        sources: list[dict],
    ) -> list[Claim]:
        """Verify multiple claims against sources.

        Args:
            claims: List of claims to verify
            sources: List of source dicts

        Returns:
            List of verified Claims
        """
        verified_claims = []
        for claim in claims:
            verified = await self.verify_claim(claim, sources)
            verified_claims.append(verified)
        return verified_claims

    async def verify_citation(
        self,
        claim_text: str,
        source: dict,
        source_index: int,
    ) -> Citation:
        """Verify whether a citation accurately supports its claim.

        Args:
            claim_text: The claim being cited
            source: The source dict being cited
            source_index: Index of the source

        Returns:
            Citation object with verification results
        """
        source_content = source.get("content", "")
        source_title = source.get("title", f"Source {source_index}")

        messages = [
            {"role": "system", "content": "You are verifying citation accuracy. Determine if the cited source actually supports the claim made."},
            {"role": "user", "content": f"""Does this source support the following claim?

CLAIM: {claim_text}

SOURCE ({source_title}):
{source_content}

Verify whether the source actually supports this claim."""},
        ]

        result = await self.invoke_llm(messages, output_schema=CitationVerification)

        return Citation(
            claim_text=claim_text,
            cited_source_index=source_index,
            cited_source_url=source.get("url"),
            is_accurate=result.is_accurate,
            reasoning=result.reasoning,
        )

    async def judge(
        self,
        report: str,
        sources: list[dict],
        verify_citations: bool = True,
        **kwargs: Any,
    ) -> dict:
        """Perform full grounding evaluation.

        Args:
            report: The report text to evaluate
            sources: List of source dicts
            verify_citations: Whether to also verify existing citations
            **kwargs: Additional arguments (unused)

        Returns:
            Dict with 'claims', 'citations' (if verify_citations), and 'usage'
        """
        # Extract and verify claims
        claims = await self.extract_claims(report)
        verified_claims = await self.verify_claims(claims, sources)

        result = {
            "claims": verified_claims,
            "usage": self.get_usage(),
        }

        # Optionally verify existing citations in the report
        if verify_citations:
            citations = self._extract_citations_from_report(report, sources)
            verified_citations = []
            for citation in citations:
                if citation.cited_source_index < len(sources):
                    verified = await self.verify_citation(
                        citation.claim_text,
                        sources[citation.cited_source_index],
                        citation.cited_source_index,
                    )
                    verified_citations.append(verified)
            result["citations"] = verified_citations
            result["usage"] = self.get_usage()

        return result

    def _extract_citations_from_report(
        self,
        report: str,
        sources: list[dict],
    ) -> list[Citation]:
        """Extract citation references from the report.

        This method looks for common citation patterns like [1], [Source 1], etc.

        Args:
            report: The report text
            sources: List of sources (for URL matching)

        Returns:
            List of Citation objects (unverified)
        """
        import re

        citations = []

        # Pattern for numbered citations like [1], [2], [Source 1], etc.
        # Also captures the preceding sentence as the claim
        pattern = r'([^.!?]*[.!?]?)\s*\[(?:Source\s*)?(\d+)\]'

        for match in re.finditer(pattern, report, re.IGNORECASE):
            claim_text = match.group(1).strip()
            source_index = int(match.group(2)) - 1  # Convert to 0-indexed

            if claim_text and source_index >= 0:
                citation = Citation(
                    claim_text=claim_text,
                    cited_source_index=source_index,
                    cited_source_url=sources[source_index].get("url") if source_index < len(sources) else None,
                )
                citations.append(citation)

        return citations
