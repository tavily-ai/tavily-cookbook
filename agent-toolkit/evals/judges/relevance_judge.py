"""Relevance judge for evaluating source-query alignment."""

from typing import Any, Optional

from pydantic import BaseModel, Field

from .base import BaseJudge

# Try both package and relative imports
try:
    from tavily_agent_toolkit import ModelConfig
except ImportError:
    try:
        from ...models import ModelConfig
    except ImportError:
        ModelConfig = Any  # type: ignore


class QueryAspect(BaseModel):
    """An aspect or subtopic of the query."""
    aspect: str = Field(description="A specific aspect or subtopic of the query")
    importance: float = Field(description="How important this aspect is to answering the query (0-1)")


class QueryAnalysisOutput(BaseModel):
    """Output schema for query analysis."""
    aspects: list[QueryAspect] = Field(description="Key aspects/subtopics of the query")
    query_type: str = Field(description="Type of query: 'factual', 'comparative', 'explanatory', 'exploratory'")


class SourceRelevanceScore(BaseModel):
    """Relevance score for a single source."""
    source_index: int = Field(description="Index of the source being scored (0-indexed)")
    relevance_score: float = Field(description="Overall relevance to the query (0-1)")
    aspects_covered: list[str] = Field(description="Which query aspects this source covers")
    reasoning: str = Field(description="Brief explanation of the relevance score")


class CoverageDetail(BaseModel):
    """Detail on whether a specific query aspect is covered."""
    aspect: str = Field(description="The query aspect")
    covered: bool = Field(description="Whether this aspect is covered by the sources")
    source_indices: list[int] = Field(description="Indices of sources that cover this aspect")


class RelevanceEvaluationOutput(BaseModel):
    """Output schema for relevance evaluation."""
    source_scores: list[SourceRelevanceScore] = Field(description="Relevance scores for each source")
    answer_coverage: float = Field(description="Percentage of query aspects covered by sources (0-1)")
    coverage_details: list[CoverageDetail] = Field(description="Details on which aspects are covered")


QUERY_ANALYSIS_PROMPT = """You are an expert at analyzing search queries.

Break down the query into its key aspects/subtopics that need to be addressed.
Consider:
- What specific information is the user seeking?
- What are the implicit sub-questions?
- What aspects would a complete answer need to cover?

Assign importance weights (0-1) based on how central each aspect is to the query.
"""

RELEVANCE_PROMPT = """You are an expert at evaluating search result relevance.

Score each source's relevance to the query based on:
1. **Direct relevance**: Does the source directly address the query?
2. **Information quality**: Is the information useful and substantive?
3. **Aspect coverage**: Which parts of the query does it help answer?

Scoring guidelines:
- 0.9-1.0: Directly and comprehensively addresses the query
- 0.7-0.8: Highly relevant, addresses core aspects
- 0.5-0.6: Moderately relevant, addresses some aspects
- 0.3-0.4: Tangentially relevant, limited usefulness
- 0.0-0.2: Not relevant or marginally relevant

Be strict: sources that mention keywords but don't actually help answer the query should score low.
"""


class RelevanceJudge(BaseJudge):
    """Judge for evaluating source-query relevance.

    This judge:
    1. Analyzes the query to identify key aspects
    2. Scores each source's relevance to the query
    3. Computes coverage of query aspects

    Example:
        judge = RelevanceJudge(model_config=ModelConfig(...))
        result = await judge.evaluate_relevance(query, sources)
    """

    def __init__(
        self,
        model_config: "ModelConfig",
        query_analysis_prompt: Optional[str] = None,
        relevance_prompt: Optional[str] = None,
    ):
        """Initialize the relevance judge.

        Args:
            model_config: ModelConfig for the judge LLM
            query_analysis_prompt: Custom prompt for query analysis
            relevance_prompt: Custom prompt for relevance scoring
        """
        super().__init__(model_config)
        self.query_analysis_prompt = query_analysis_prompt or QUERY_ANALYSIS_PROMPT
        self.relevance_prompt = relevance_prompt or RELEVANCE_PROMPT

    async def analyze_query(self, query: str) -> QueryAnalysisOutput:
        """Analyze a query to identify its key aspects.

        Args:
            query: The search query

        Returns:
            QueryAnalysisOutput with aspects and query type
        """
        messages = [
            {"role": "system", "content": self.query_analysis_prompt},
            {"role": "user", "content": f"Analyze this query: {query}"},
        ]

        result = await self.invoke_llm(messages, output_schema=QueryAnalysisOutput)
        return result

    async def score_sources(
        self,
        query: str,
        sources: list[dict],
        query_aspects: list[QueryAspect],
    ) -> RelevanceEvaluationOutput:
        """Score source relevance to the query.

        Args:
            query: The search query
            sources: List of source dicts with 'content', 'title', 'url'
            query_aspects: Aspects from query analysis

        Returns:
            RelevanceEvaluationOutput with scores and coverage
        """
        # Format sources
        sources_text = ""
        for i, source in enumerate(sources):
            title = source.get("title", f"Source {i}")
            url = source.get("url", "")
            content = source.get("content", "")[:1500]  # Truncate
            sources_text += f"\n--- Source {i}: {title} ---\n"
            if url:
                sources_text += f"URL: {url}\n"
            sources_text += f"{content}\n"

        # Format aspects
        aspects_text = "\n".join([f"- {a.aspect} (importance: {a.importance:.1f})" for a in query_aspects])

        messages = [
            {"role": "system", "content": self.relevance_prompt},
            {"role": "user", "content": f"""Evaluate the relevance of these sources to the query.

QUERY: {query}

KEY ASPECTS TO COVER:
{aspects_text}

SOURCES:
{sources_text}

Score each source's relevance and determine how well the sources collectively cover the query aspects."""},
        ]

        result = await self.invoke_llm(messages, output_schema=RelevanceEvaluationOutput)
        return result

    async def evaluate_relevance(
        self,
        query: str,
        sources: list[dict],
    ) -> dict:
        """Perform full relevance evaluation.

        Args:
            query: The search query
            sources: List of source dicts

        Returns:
            Dict with query_analysis, source_scores, coverage, and usage
        """
        # Analyze query
        query_analysis = await self.analyze_query(query)

        # Score sources
        relevance_result = await self.score_sources(query, sources, query_analysis.aspects)

        return {
            "query_analysis": query_analysis,
            "relevance_result": relevance_result,
            "usage": self.get_usage(),
        }

    async def judge(
        self,
        query: str,
        sources: list[dict],
        **kwargs: Any,
    ) -> dict:
        """Perform relevance judgment.

        Args:
            query: The search query
            sources: List of source dicts
            **kwargs: Additional arguments (unused)

        Returns:
            Dict with evaluation results
        """
        return await self.evaluate_relevance(query, sources)
