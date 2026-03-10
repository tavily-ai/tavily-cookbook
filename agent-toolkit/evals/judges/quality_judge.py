"""Quality judge for evaluating generated queries and search effectiveness."""

import asyncio
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
    specificity: float = Field(description="How specific and targeted the query is (0-1). Use rubric: 0.9-1.0 highly specific, 0.7-0.8 reasonably specific, 0.5-0.6 moderate, 0.3-0.4 vague, 0.0-0.2 extremely vague")
    search_effectiveness: float = Field(description="How well this works as a search query (0-1). Use rubric: 0.9-1.0 excellent keywords, 0.7-0.8 good, 0.5-0.6 fair/natural language, 0.3-0.4 poor, 0.0-0.2 very poor")
    overall_score: float = Field(description="Combined quality assessment (0-1). Use rubric: 0.9-1.0 excellent, 0.7-0.8 good, 0.5-0.6 fair, 0.3-0.4 poor, 0.0-0.2 very poor")
    reasoning: str = Field(default="", description="Brief explanation of the score — what works and what doesn't")
    suggestions: list[str] = Field(description="Actionable suggestions: specific terms to add, remove, or change")


class QuerySetEvaluation(BaseModel):
    """Evaluation of a set of queries for a research task."""
    individual_scores: list[QueryQualityScore] = Field(description="Scores for each query")
    coverage_score: float = Field(description="How well the queries cover the research topic (0-1)")
    diversity_score: float = Field(description="How diverse/non-redundant the queries are (0-1)")
    overall_quality: float = Field(description="Overall quality of the query set (0-1)")


class ResultCoverageEvaluation(BaseModel):
    """Evaluation of search result coverage."""
    topic_coverage: float = Field(description="How well results cover the topic (0-1). Use rubric: 0.9-1.0 comprehensive, 0.7-0.8 good with minor gaps, 0.5-0.6 partial with significant gaps, 0.3-0.4 sparse, 0.0-0.2 very sparse")
    key_aspects_found: list[str] = Field(description="Key aspects of the topic found in results")
    key_aspects_missing: list[str] = Field(description="Key aspects not adequately covered")
    redundancy_ratio: float = Field(description="Ratio of redundant/duplicate content (0-1). Use rubric: 0.0-0.1 minimal, 0.2-0.3 low, 0.4-0.6 moderate, 0.7-0.8 high, 0.9-1.0 very high")
    reasoning: str = Field(default="", description="Brief explanation of the coverage assessment")


QUERY_QUALITY_PROMPT = """You are an expert at evaluating search query quality for web search engines.

A good search query should be:
1. **Specific**: Targets exactly what information is needed
2. **Search-optimized**: Uses terms likely to appear in relevant documents (not full prose questions)
3. **Unambiguous**: Clear meaning, not vague or overly broad
4. **Appropriately scoped**: Neither too narrow (zero results) nor too broad (noise)
5. **Concise**: Under 400 characters — a web search query, not a long-form prompt

Scoring guidelines for SPECIFICITY (how targeted is the query):
- 0.9-1.0: Highly specific — targets a single, well-defined information need with precise terms
- 0.7-0.8: Reasonably specific — clear focus but could be more targeted (e.g., missing region, time period, or metric)
- 0.5-0.6: Moderate — somewhat broad or ambiguous, multiple interpretations possible
- 0.3-0.4: Vague — very broad topic with no targeting, would return generic results
- 0.0-0.2: Extremely vague — a single word or meaningless phrase

Scoring guidelines for SEARCH EFFECTIVENESS (how well this works as a search query):
- 0.9-1.0: Excellent — uses precise, search-friendly keywords; appropriate scope; would return highly relevant results
- 0.7-0.8: Good — solid keyword choice, will return mostly relevant results with some noise
- 0.5-0.6: Fair — uses natural language or overly complex phrasing; mixed results likely
- 0.3-0.4: Poor — too long, conversational, or keyword choices unlikely to match relevant pages
- 0.0-0.2: Very poor — would not work as a search query (e.g., full paragraph, nonsensical)

Scoring guidelines for OVERALL QUALITY (combined assessment):
- 0.9-1.0: Excellent — a search expert would use this query as-is
- 0.7-0.8: Good — solid query with minor room for improvement
- 0.5-0.6: Fair — usable but needs refinement to get quality results
- 0.3-0.4: Poor — significant issues that would hurt result quality
- 0.0-0.2: Very poor — needs to be completely rewritten

Be strict: use the full range of scores. A generic query like "AI" should score 0.1-0.2, not 0.5.
Provide actionable suggestions for improvement — specifically what terms to add, remove, or change.
Do not suggest boolean operators (AND/OR/NOT) or boolean-style syntax as improvements.
"""

COVERAGE_PROMPT = """You are an expert at evaluating search result coverage for research tasks.

Assess how well the search results cover the research topic:
1. **Topic coverage**: Do results address the core questions and sub-topics?
2. **Comprehensiveness**: Are different aspects, perspectives, and angles covered?
3. **Redundancy**: How much duplicate or near-duplicate information is there?

Scoring guidelines for TOPIC COVERAGE (0-1):
- 0.9-1.0: Comprehensive — all major aspects of the topic are well-represented with substantive content
- 0.7-0.8: Good — most important aspects covered, only minor gaps remain
- 0.5-0.6: Partial — significant aspects are missing or only superficially covered
- 0.3-0.4: Sparse — only covers a narrow slice of the topic, major gaps
- 0.0-0.2: Very sparse — barely addresses the research topic

Scoring guidelines for REDUNDANCY RATIO (0-1, where 0 = no redundancy, 1 = all redundant):
- 0.0-0.1: Minimal — almost all results provide unique information
- 0.2-0.3: Low — some overlap but mostly distinct content
- 0.4-0.6: Moderate — noticeable duplication across results
- 0.7-0.8: High — many results cover the same ground
- 0.9-1.0: Very high — almost all results are redundant

Be strict: use the full range of scores. Specifically list what aspects are found and what is missing.
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

        # Run query evaluation and coverage evaluation in parallel
        tasks = {}
        if queries:
            tasks["query_evaluation"] = self.evaluate_queries(research_task, queries)
        if results:
            tasks["coverage_evaluation"] = self.evaluate_coverage(research_task, results)

        if tasks:
            results_list = await asyncio.gather(*tasks.values())
            for key, result in zip(tasks.keys(), results_list):
                output[key] = result

        output["usage"] = self.get_usage()
        return output
