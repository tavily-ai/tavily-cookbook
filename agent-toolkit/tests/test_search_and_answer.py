"""Tests for search_and_answer function."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from models import ModelConfig, ModelObject
from tools.search_and_answer import search_and_answer

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


class NvidiaStockInfo(BaseModel):
    """Schema for NVIDIA stock information output."""
    company_name: str = Field(description="The full official name of the company")
    ticker_symbol: str = Field(description="The stock ticker symbol (e.g., NVDA)")
    key_points: list[str] = Field(description="List of key facts about the company's business and market position")
    summary: str = Field(description="A brief overview summarizing NVIDIA as a company")
    primary_products: list[str] = Field(description="List of NVIDIA's main product lines (e.g., GPUs, data center chips, AI accelerators)")
    market_cap_range: str = Field(description="Approximate market capitalization range (e.g., 'over $1 trillion', '$500B-$1T')")
    ceo_name: str = Field(description="Name of NVIDIA's current Chief Executive Officer")


@pytest.mark.skipif(not TAVILY_API_KEY or not OPENAI_API_KEY, reason="API keys not set")
class TestSearchAndAnswerBasic:
    """Test basic search_and_answer functionality."""

    @pytest.mark.asyncio
    async def test_regular_search_and_answer(self):
        """Test 1: Regular search_and_answer with only required params."""
        result = await search_and_answer(
            query="What is NVIDIA's main business and what products do they make?",
            api_key=TAVILY_API_KEY,
            model_config=ModelConfig(
                model=ModelObject(
                    model="gpt-5.1",
                    model_provider="openai",
                    api_key=OPENAI_API_KEY,
                ),
            ),
        )
        print("\nanswer: \n", result["answer"])

        assert "answer" in result
        assert result["answer"] is not None
        assert len(result["answer"]) > 0
        assert "results" in result
        assert isinstance(result["results"], list)
        
        # Verify usage metrics (search + LLM)
        assert "usage" in result
        usage = result["usage"]
        assert "response_time" in usage
        assert usage["response_time"] > 0
        
        # Should have tavily search usage
        assert "tavily" in usage
        assert usage["tavily"]["total_credits"] > 0
        assert "search_count" in usage["tavily"]
        assert usage["tavily"]["search_count"] >= 1
        
        # Should have llm usage
        assert "llm" in usage
        assert usage["llm"]["llm_call_count"] >= 1
        assert usage["llm"]["total_tokens"] > 0
        
        print("\nUsage metrics:", usage)


@pytest.mark.skipif(not TAVILY_API_KEY or not OPENAI_API_KEY, reason="API keys not set")
class TestSearchAndAnswerWithMostParams:
    """Test search_and_answer with most parameters set."""

    @pytest.mark.asyncio
    async def test_with_most_params(self):
        """Test search_and_answer with most available parameters."""
        result = await search_and_answer(
            query="What are NVIDIA's latest GPU announcements and releases?",
            api_key=TAVILY_API_KEY,
            model_config=ModelConfig(
                model=ModelObject(
                    model="gpt-5.1",
                    model_provider="openai",
                    api_key=OPENAI_API_KEY,
                ),
            ),
            token_limit=50000,
            max_number_of_subqueries=3,
            threshold=0.3,
            search_depth="advanced",
            topic="news",
            time_range="month",
            max_results=15,
            include_domains=["nvidia.com", "techcrunch.com", "theverge.com"],
            include_raw_content="markdown",
            include_images=True,
            include_image_descriptions=True,
            timeout=90,
            country="united states",
            include_favicon=True,
        )
        print("\nanswer:\n", result["answer"])

        assert "answer" in result
        assert result["answer"] is not None
        assert len(result["answer"]) > 0
        assert "results" in result
        assert isinstance(result["results"], list)


@pytest.mark.skipif(not TAVILY_API_KEY or not OPENAI_API_KEY, reason="API keys not set")
class TestSearchAndAnswerWithSchema:
    """Test search_and_answer with structured output schema."""

    @pytest.mark.asyncio
    async def test_with_output_schema(self):
        """Test 3: Search with an output schema for structured response."""
        result = await search_and_answer(
            query="Tell me about NVIDIA as a company including their stock ticker, CEO, main products, and market capitalization",
            max_number_of_subqueries=3,
            api_key=TAVILY_API_KEY,
            model_config=ModelConfig(
                model=ModelObject(
                    model="gpt-5.1",
                    model_provider="openai",
                    api_key=OPENAI_API_KEY,
                ),
            ),
            output_schema=NvidiaStockInfo,
        )
        print(result["answer"])

        assert "answer" in result
        answer = result["answer"]
        
        # Verify the answer is an instance of our schema
        assert isinstance(answer, NvidiaStockInfo)
        assert answer.company_name is not None
        assert answer.ticker_symbol is not None
        assert isinstance(answer.key_points, list)
        assert answer.summary is not None
        # Verify the new specific fields
        assert isinstance(answer.primary_products, list)
        assert len(answer.primary_products) > 0
        assert answer.market_cap_range is not None
        assert answer.ceo_name is not None


@pytest.mark.skipif(not TAVILY_API_KEY or not OPENAI_API_KEY, reason="API keys not set")
class TestSearchAndAnswerWithSubqueries:
    """Test search_and_answer with subquery generation."""

    @pytest.mark.asyncio
    async def test_with_subqueries(self):
        """Test 4: Regular search with subquery generation enabled."""
        result = await search_and_answer(
            query="What is NVIDIA's position in the AI chip market and how does it compare to competitors?",
            api_key=TAVILY_API_KEY,
            model_config=ModelConfig(
                model=ModelObject(
                    model="gpt-5.1",
                    model_provider="openai",
                    api_key=OPENAI_API_KEY,
                ),
            ),
            max_number_of_subqueries=3,
        )

        assert "answer" in result
        assert result["answer"] is not None
        assert len(result["answer"]) > 0
        assert "results" in result
        assert isinstance(result["results"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
