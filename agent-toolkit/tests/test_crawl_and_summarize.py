"""Tests for crawl_and_summarize function."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from models import ModelConfig, ModelObject
from tools.crawl_and_summarize import crawl_and_summarize


class NvidiaNews(BaseModel):
    """Schema for structured Nvidia news output."""
    stock_price: str = Field(description="Current or mentioned stock price of Nvidia")
    recent_announcements: list[str] = Field(description="List of recent Nvidia announcements or news")
    market_sentiment: str = Field(description="Overall market sentiment about Nvidia (bullish, bearish, neutral)")
    summary: str = Field(description="A detailed summary of the latest Nvidia news")

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


@pytest.mark.skipif(not TAVILY_API_KEY or not OPENAI_API_KEY, reason="API keys not set")
class TestCrawlAndSummarizeWithoutInstructions:
    """Test crawl_and_summarize without instructions parameter."""

    @pytest.mark.asyncio
    async def test_crawl_without_instructions(self):
        """Test crawl_and_summarize with only required params (no instructions)."""
        result = await crawl_and_summarize(
            url="https://www.yahoo.com/",
            api_key=TAVILY_API_KEY,
            model_config=ModelConfig(
                model=ModelObject(
                    model="gpt-5.1",
                    model_provider="openai",
                    api_key=OPENAI_API_KEY,
                ),
            ),
            max_depth=1,
            limit=3,
        )
        print("\nResult (no instructions):\n", result)

        # Verify the result has a summary
        assert result is not None
        assert "summary" in result
        assert result["summary"] is not None
        
        # Verify usage metrics are present
        assert "usage" in result
        usage = result["usage"]
        assert "response_time" in usage
        assert usage["response_time"] > 0
        
        # Should have tavily crawl usage
        assert "tavily" in usage
        assert usage["tavily"]["total_credits"] >= 0  # Some APIs may return 0
        assert "crawl_count" in usage["tavily"]
        assert usage["tavily"]["crawl_count"] == 1
        
        # Should have llm usage
        assert "llm" in usage
        assert usage["llm"]["llm_call_count"] >= 1
        assert usage["llm"]["total_tokens"] > 0
        
        print("\nUsage metrics:", usage)


@pytest.mark.skipif(not TAVILY_API_KEY or not OPENAI_API_KEY, reason="API keys not set")
class TestCrawlAndSummarizeWithInstructions:
    """Test crawl_and_summarize with instructions parameter."""

    @pytest.mark.asyncio
    async def test_crawl_with_instructions(self):
        """Test crawl_and_summarize with specific instructions for summarization."""
        result = await crawl_and_summarize(
            url="https://www.yahoo.com/",
            api_key=TAVILY_API_KEY,
            model_config=ModelConfig(
                model=ModelObject(
                    model="gpt-5.1",
                    model_provider="openai",
                    api_key=OPENAI_API_KEY,
                ),
            ),
            instructions="Find me financial news about nvidia",
            chunks_per_source=3,
            max_depth=1,
            limit=5,
        )
        print("\nResult (with instructions):\n", result)

        # Verify the result has a summary
        assert result is not None
        assert "summary" in result
        assert result["summary"] is not None


@pytest.mark.skipif(not TAVILY_API_KEY or not OPENAI_API_KEY, reason="API keys not set")
class TestCrawlAndSummarizeWithOutputSchema:
    """Test crawl_and_summarize with output_schema parameter."""

    @pytest.mark.asyncio
    async def test_crawl_with_output_schema(self):
        """Test crawl_and_summarize with structured output schema."""
        result = await crawl_and_summarize(
            url="https://www.yahoo.com/",
            api_key=TAVILY_API_KEY,
            model_config=ModelConfig(
                model=ModelObject(
                    model="gpt-5.1",
                    model_provider="openai",
                    api_key=OPENAI_API_KEY,
                ),
            ),
            output_schema=NvidiaNews,
            instructions="Find me latest news about nvidia",
            max_depth=3,
            limit=5,
        )
        print("\nResult (with output schema):\n", result)

        # Verify the result structure
        assert result is not None
        assert "summary" in result
        assert "results" in result
        
        # Verify the summary is an instance of our schema
        summary = result["summary"]
        assert isinstance(summary, NvidiaNews)
        assert summary.stock_price is not None
        assert isinstance(summary.recent_announcements, list)
        assert len(summary.recent_announcements) > 0
        assert summary.market_sentiment is not None
        assert summary.summary is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
