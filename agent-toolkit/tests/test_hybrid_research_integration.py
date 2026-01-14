"""Integration tests for hybrid_researcher module - calls real APIs."""

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from tavily_agent_toolkit import ModelConfig, ModelObject, OutputSchema, hybrid_research
from pydantic import Field


class FinancialAnalysis(OutputSchema):
    """Structured output schema for financial performance analysis."""
    company_name: str = Field(..., description="The name of the company being analyzed")
    year: int = Field(..., description="The fiscal year of the analysis")
    revenue_summary: str = Field(..., description="Summary of the company's revenue performance")
    key_highlights: list[str] = Field(..., description="Key highlights from the financial analysis")
    outlook: str = Field(..., description="Future outlook and projections for the company")

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# Set your API keys here or use environment variables
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Shared query for both tests
QUERY = "How was Apple's financial performance in 2024?"


def mock_internal_rag(query: str) -> str:
    """Mock internal RAG that returns relevant internal company data."""
    return """
INTERNAL DOCUMENT: Q3 2024 Financial Summary

Revenue: $2.4B (up 12% YoY)
Operating Income: $480M
Net Income: $320M
EPS: $1.25

Key Highlights:
- Cloud services segment grew 28% to $890M
- Enterprise software licenses increased 8%
- Customer retention rate: 94%

Internal Notes:
- New product launch scheduled for Q4
- Expansion into APAC market on track
- 3 strategic acquisitions pending board approval
"""


@pytest.mark.skipif(not TAVILY_API_KEY or not OPENAI_API_KEY, reason="API keys not set")
class TestHybridResearchIntegration:
    """Integration tests that call real APIs."""

    @pytest.mark.asyncio
    async def test_fast_mode(self):
        """Test hybrid_research with fast mode - real API calls."""
        result = await hybrid_research(
            api_key=TAVILY_API_KEY,
            query=QUERY,
            model_config=ModelConfig(model=ModelObject(model="gpt-5.1", model_provider="openai")),
            internal_rag_function=mock_internal_rag,
            mode="fast",
            research_synthesis_prompt="Structure as: Executive Summary, Revenue & Earnings, Product Performance, and Outlook with bullet points under each section. Focus on YoY comparisons and keep it under 300 words."
        )

        print("\n" + "=" * 50)
        print("FAST MODE RESULTS")
        print("=" * 50)
        print("\nReport preview (first 500 chars):")
        print(result["report"][:500] + "..." if len(result["report"]) > 500 else result["report"])
        print(f"\nWeb Sources ({len(result['web_sources'])} total):")
        for source in result["web_sources"][:5]:
            print(f"  - {source['title']}")
        
        # New usage assertions
        print("\nUsage Metrics:")
        print(json.dumps(result["usage"], indent=2))

        assert "report" in result
        assert "web_sources" in result
        assert len(result["web_sources"]) > 0
        
        # Usage field assertions
        assert "usage" in result
        usage = result["usage"]
        assert "response_time" in usage
        assert usage["response_time"] > 0
        
        # Tavily usage (fast mode uses search_dedup)
        assert "tavily" in usage
        assert usage["tavily"]["total_credits"] > 0
        assert usage["tavily"]["search_count"] > 0
        assert usage["tavily"]["search_response_time"] > 0
        
        # LLM usage (subqueries + synthesis = 2 calls)
        assert "llm" in usage
        assert usage["llm"]["llm_call_count"] >= 2
        assert usage["llm"]["total_tokens"] > 0
        assert usage["llm"]["llm_response_time"] > 0
        
        # Internal function time (may be very fast for mock functions)
        assert "internal_function_response_time" in usage
        assert usage["internal_function_response_time"] >= 0

    @pytest.mark.asyncio
    async def test_multi_agent_mode(self):
        """Test hybrid_research with multi_agent mode - real API calls."""
        result = await hybrid_research(
            api_key=TAVILY_API_KEY,
            query=QUERY,
            model_config=ModelConfig(model=ModelObject(model="gpt-5.1", model_provider="openai")),
            internal_rag_function=mock_internal_rag,
            mode="multi_agent"
        )

        print("\n" + "=" * 50)
        print("MULTI-AGENT MODE RESULTS")
        print("=" * 50)
        print("\nReport preview (first 500 chars):")
        print(result["report"][:500] + "..." if len(result["report"]) > 500 else result["report"])
        print(f"\nWeb Sources ({len(result['web_sources'])} total):")
        for source in result["web_sources"][:5]:
            print(f"  - {source.get('title', source.get('url', 'N/A'))}")
        
        # New usage assertions
        print("\nUsage Metrics:")
        print(json.dumps(result["usage"], indent=2))

        assert "report" in result
        assert "web_sources" in result
        
        # Usage field assertions
        assert "usage" in result
        usage = result["usage"]
        assert "response_time" in usage
        assert usage["response_time"] > 0
        
        # Tavily research endpoint time (credits not tracked for research)
        assert "tavily_research_response_time" in usage
        assert usage["tavily_research_response_time"] > 0
        
        # LLM usage (brief generation + synthesis = 2 calls)
        assert "llm" in usage
        assert usage["llm"]["llm_call_count"] >= 2
        assert usage["llm"]["total_tokens"] > 0
        
        # Internal function time (may be very fast for mock functions)
        assert "internal_function_response_time" in usage
        assert usage["internal_function_response_time"] >= 0

    @pytest.mark.asyncio
    async def test_fast_mode_with_output_schema(self):
        """Test hybrid_research with fast mode and output schema - real API calls."""
        result = await hybrid_research(
            api_key=TAVILY_API_KEY,
            query=QUERY,
            model_config=ModelConfig(model=ModelObject(model="gpt-5.1", model_provider="openai")),
            internal_rag_function=mock_internal_rag,
            mode="fast",
            output_schema=FinancialAnalysis
        )

        print("\n" + "=" * 50)
        print("FAST MODE WITH OUTPUT SCHEMA RESULTS")
        print("=" * 50)
        print("\nReport (structured):")
        print(result["report"][:500] + "..." if len(result["report"]) > 500 else result["report"])
        print(f"\nWeb Sources ({len(result['web_sources'])} total)")
        
        print("\nUsage Metrics:")
        print(json.dumps(result["usage"], indent=2))

        assert "report" in result
        assert "web_sources" in result
        assert len(result["web_sources"]) > 0
        
        # Usage assertions
        assert "usage" in result
        assert result["usage"]["response_time"] > 0
        assert "tavily" in result["usage"]
        assert "llm" in result["usage"]

    @pytest.mark.asyncio
    async def test_multi_agent_mode_with_output_schema(self):
        """Test hybrid_research with multi_agent mode and output schema - real API calls."""
        result = await hybrid_research(
            api_key=TAVILY_API_KEY,
            query=QUERY,
            model_config=ModelConfig(model=ModelObject(model="gpt-5.1", model_provider="openai")),
            internal_rag_function=mock_internal_rag,
            mode="multi_agent",
            output_schema=FinancialAnalysis
        )

        print("\n" + "=" * 50)
        print("MULTI-AGENT MODE WITH OUTPUT SCHEMA RESULTS")
        print("=" * 50)
        print("\nReport (structured):")
        print(result["report"][:500] + "..." if len(result["report"]) > 500 else result["report"])
        print(f"\nWeb Sources ({len(result['web_sources'])} total)")
        
        print("\nUsage Metrics:")
        print(json.dumps(result["usage"], indent=2))

        assert "report" in result
        assert "web_sources" in result
        
        # Usage assertions
        assert "usage" in result
        assert result["usage"]["response_time"] > 0
        assert "tavily_research_response_time" in result["usage"]
        assert "llm" in result["usage"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
