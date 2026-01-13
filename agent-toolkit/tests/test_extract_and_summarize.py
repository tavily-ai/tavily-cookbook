"""Tests for extract_and_summarize function."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from models import ModelConfig, ModelObject
from tools.extract_and_summarize import extract_and_summarize


class PageSummary(BaseModel):
    """Schema for structured page summary output."""
    main_topic: str = Field(description="The main topic of the page")
    key_points: list[str] = Field(description="Key points from the content")
    summary: str = Field(description="A concise summary of the page")


class StockSummary(BaseModel):
    """Schema for structured stock information output."""
    company_name: str = Field(description="The name of the company")
    ticker_symbol: str = Field(description="The stock ticker symbol")
    sector: str = Field(description="The sector the company operates in")
    industry: str = Field(description="The specific industry of the company")
    stock_price: str = Field(description="The current stock price")
    market_cap: str = Field(description="The market capitalization")
    pe_ratio: str = Field(description="The price-to-earnings ratio")
    key_highlights: list[str] = Field(description="Other things to mention you think are relevant")


# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

@pytest.mark.skipif(not TAVILY_API_KEY or not OPENAI_API_KEY, reason="API keys not set")
class TestExtractAndSummarizeWithQuery:
    """Test extract_and_summarize with query parameter (chunk-focused summary)."""

    @pytest.mark.asyncio
    async def test_extract_with_query(self):
        """Test extract_and_summarize with query for chunk-based summarization."""
        result = await extract_and_summarize(
            urls=["https://finance.yahoo.com/quote/NVDA/"],
            api_key=TAVILY_API_KEY,
            model_config=ModelConfig(
                model=ModelObject(
                    model="gpt-5.1",
                    model_provider="openai",
                    api_key=OPENAI_API_KEY,
                ),
            ),
            query="Nvidia financial news",
            chunks_per_source=5,
        )
        print("\nResult (with query):\n", result)
        # print("\nResult (with query):\n", result)

        # Verify the result structure
        assert result is not None
        assert "results" in result
        assert len(result["results"]) > 0
        
        # Each result should have a summary
        for item in result["results"]:
            assert "summary" in item
            assert item["summary"] is not None
        
        # Verify usage metrics (extract + LLM)
        assert "usage" in result
        usage = result["usage"]
        assert "response_time" in usage
        assert usage["response_time"] > 0
        
        # Should have tavily extract usage (not search)
        assert "tavily" in usage
        assert "total_credits" in usage["tavily"]  # Credits may be 0 for extract
        assert "extract_count" in usage["tavily"]
        assert usage["tavily"]["extract_count"] == 1
        assert "search_count" not in usage["tavily"]  # No search
        
        # Should have llm usage
        assert "llm" in usage
        assert usage["llm"]["llm_call_count"] >= 1
        assert usage["llm"]["total_tokens"] > 0
        
        print("\nUsage metrics:", usage)


@pytest.mark.skipif(not TAVILY_API_KEY or not OPENAI_API_KEY, reason="API keys not set")
class TestExtractAndSummarizeWithOutputSchema:
    """Test extract_and_summarize with output_schema parameter for NVIDIA stock."""

    @pytest.mark.asyncio
    async def test_extract_with_output_schema(self):
        """Test extract_and_summarize with structured output schema for NVIDIA stock."""
        result = await extract_and_summarize(
            urls=["https://finance.yahoo.com/quote/NVDA/"],
            api_key=TAVILY_API_KEY,
            model_config=ModelConfig(
                model=ModelObject(
                    model="gpt-5.1",
                    model_provider="openai",
                    api_key=OPENAI_API_KEY,
                ),
            ),
            output_schema=StockSummary,
            query="Nvidia financial news",
            chunks_per_source=5,
        )
        # print("\nResult (with output schema):\n", result)

        # Verify the result structure
        assert result is not None
        assert "results" in result
        assert len(result["results"]) > 0
        
        # Each result should have a structured summary
        for item in result["results"]:
            assert "summary" in item
            summary = item["summary"]
            assert isinstance(summary, StockSummary)
            assert summary.company_name is not None
            assert summary.ticker_symbol is not None
            assert summary.sector is not None
            assert summary.industry is not None
            assert summary.stock_price is not None
            assert summary.market_cap is not None
            assert isinstance(summary.key_highlights, list)