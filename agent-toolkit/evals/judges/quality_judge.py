"""Quality judge for evaluating generated queries and search effectiveness."""

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


class QueryQualityScore(BaseModel):
    """Quality assessment for a generated search query."""
    query: str = Field(description="The query being evaluated")
    specificity: float = Field(description="How specific and targeted the query is (0-1)")
    search_effectiveness: float = Field(description="How effective this query would be for search (0-1)")
    overall_score: float = Field(description="Overall quality score (0-1)")
    suggestions: list[str] = Field(description="Suggestions for improving the query")


class QuerySetEvaluation(BaseModel):
    """Evaluation of a set of queries for a research task."""
    individual_scores: list[QueryQualityScore] = Field(description="Scores for each query")
    coverage_score: float = Field(description="How well the queries cover the research topic (0-1)")
    diversity_score: float = Field(description="How diverse/non-redundant the queries are (0-1)")
    overall_quality: float = Field(description="Overall quality of the query set (0-1)")


class ResultCoverageEvaluation(BaseModel):
    """Evaluation of search result coverage."""
    topic_coverage: float = Field(description="How well results cover the topic (0-1)")
    key_aspects_found: list[str] = Field(description="Key aspects of the topic found in results")
    key_aspects_missing: list[str] = Field(description="Key aspects not adequately covered")
    redundancy_ratio: float = Field(description="Ratio of redundant/duplicate content (0-1)")


QUERY_QUALITY_PROMPT = """You are an expert at evaluating search query quality.

A good search query should be:
1. **Specific**: Targets exactly what information is needed
2. **Search-optimized**: Uses terms likely to appear in relevant documents
3. **Unambiguous**: Clear meaning, not vague or overly broad
4. **Appropriately scoped**: Neither too narrow nor too broad

Score queries on:
- Specificity (0-1): How targeted is the query?
- Search effectiveness (0-1): How likely to return good results?
- Overall quality (0-1): Combined assessment

Provide actionable suggestions for improvement.
"""

COVERAGE_PROMPT = """You are an expert at evaluating search result coverage.

Assess how well the search results cover the research topic:
1. **Topic coverage**: Do results address the core questions?
2. **Comprehensiveness**: Are different aspects/angles covered?
3. **Redundancy**: How much duplicate information is there?

Identify what's well-covered and what's missing.
"""


class QualityJudge(BaseJudge):
    """Judge for evaluating query quality and search effectiveness.

    This judge:
    1. Evaluates generated search queries for quality
    2. Assesses coverage of search results
    3. Identifies redundancy in results

    Example:
        judge = QualityJudge(model_config=ModelConfig(...))
        quality = await judge.evaluate_queries(task, queries)
    """

    def __init__(
        self,
        model_config: "ModelConfig",
        query_quality_prompt: Optional[str] = None,
        coverage_prompt: Optional[str] = None,
    ):
        """Initialize the quality judge.

        Args:
            model_config: ModelConfig for the judge LLM
            query_quality_prompt: Custom prompt for query evaluation
            coverage_prompt: Custom prompt for coverage evaluation
        """
        super().__init__(model_config)
        self.query_quality_prompt = query_quality_prompt or QUERY_QUALITY_PROMPT
        self.coverage_prompt = coverage_prompt or COVERAGE_PROMPT

    async def evaluate_queries(
        self,
        research_task: str,
        queries: list[str],
    ) -> QuerySetEvaluation:
        """Evaluate the quality of generated search queries.

        Args:
            research_task: The original research task/question
            queries: List of generated search queries

        Returns:
            QuerySetEvaluation with scores for each query and set
        """
        queries_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(queries)])

        messages = [
            {"role": "system", "content": self.query_quality_prompt},
            {"role": "user", "content": f"""Evaluate these search queries generated for a research task.

RESEARCH TASK: {research_task}

GENERATED QUERIES:
{queries_text}

Score each query individually, then evaluate the set as a whole for coverage and diversity."""},
        ]

        result = await self.invoke_llm(messages, output_schema=QuerySetEvaluation)
        return result

    async def evaluate_coverage(
        self,
        research_task: str,
        results: list[dict],
    ) -> ResultCoverageEvaluation:
        """Evaluate how well search results cover the research topic.

        Args:
            research_task: The original research task/question
            results: List of search result dicts with 'content', 'title', 'url'

        Returns:
            ResultCoverageEvaluation with coverage and redundancy metrics
        """
        # Format results
        results_text = ""
        for i, result in enumerate(results[:15]):  # Limit for context
            title = result.get("title", f"Result {i+1}")
            content = result.get("content", "")[:500]
            results_text += f"\n[{i+1}] {title}\n{content}\n"

        messages = [
            {"role": "system", "content": self.coverage_prompt},
            {"role": "user", "content": f"""Evaluate how well these search results cover the research topic.

RESEARCH TASK: {research_task}

SEARCH RESULTS:
{results_text}

Assess topic coverage, identify what's covered well and what's missing, and estimate redundancy."""},
        ]

        result = await self.invoke_llm(messages, output_schema=ResultCoverageEvaluation)
        return result

    async def judge(
        self,
        research_task: str,
        queries: Optional[list[str]] = None,
        results: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> dict:
        """Perform quality evaluation.

        Args:
            research_task: The original research task
            queries: Optional list of generated queries to evaluate
            results: Optional list of search results to evaluate
            **kwargs: Additional arguments (unused)

        Returns:
            Dict with query_evaluation, coverage_evaluation, and usage
        """
        output = {"usage": self.get_usage()}

        if queries:
            query_eval = await self.evaluate_queries(research_task, queries)
            output["query_evaluation"] = query_eval

        if results:
            coverage_eval = await self.evaluate_coverage(research_task, results)
            output["coverage_evaluation"] = coverage_eval

        output["usage"] = self.get_usage()
        return output
