"""Tests for search_and_format function."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from tavily_agent_toolkit import search_and_format

# Load .env from agent-toolkit folder
load_dotenv(Path(__file__).parent.parent / ".env")

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")


@pytest.mark.skipif(not TAVILY_API_KEY, reason="TAVILY_API_KEY not set")
class TestSearchAndFormat:
    """Test search_and_format functionality."""

    @pytest.mark.asyncio
    async def test_single_query(self):
        """Test 1: Single query search and format."""
        result = await search_and_format(
            queries=["What is Python programming language?"],
            api_key=TAVILY_API_KEY,
        )
        
        print("\n" + "="*60)
        print("TEST 1: Single Query")
        print("="*60)
        print("INPUT:")
        print("  queries: ['What is Python programming language?']")
        print("\nOUTPUT:")
        print(result[:2000] + "..." if len(result) > 2000 else result)
        print("="*60)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "SOURCE" in result

    @pytest.mark.asyncio
    async def test_multiple_queries(self):
        """Test 2: Multiple queries with deduplication."""
        result = await search_and_format(
            queries=[
                "Python programming basics",
                "Python web development frameworks",
                "Python data science libraries",
            ],
            api_key=TAVILY_API_KEY,
        )
        
        print("\n" + "="*60)
        print("TEST 2: Multiple Queries")
        print("="*60)
        print("INPUT:")
        print("  queries: [")
        print("    'Python programming basics',")
        print("    'Python web development frameworks',")
        print("    'Python data science libraries'")
        print("  ]")
        print("  threshold: 0.3 (default)")
        print("\nOUTPUT:")
        print(result[:2000] + "..." if len(result) > 2000 else result)
        print("="*60)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "SOURCE" in result

    @pytest.mark.asyncio
    async def test_multiple_queries_with_threshold(self):
        """Test 3: Multiple queries with custom threshold filtering."""
        result = await search_and_format(
            queries=[
                "NVIDIA GPU latest releases",
                "NVIDIA AI chip technology",
            ],
            api_key=TAVILY_API_KEY,
            threshold=0.5,  # Higher threshold for stricter filtering
            max_results=5,
        )
        
        print("\n" + "="*60)
        print("TEST 3: Multiple Queries with Threshold")
        print("="*60)
        print("INPUT:")
        print("  queries: [")
        print("    'NVIDIA GPU latest releases',")
        print("    'NVIDIA AI chip technology'")
        print("  ]")
        print("  threshold: 0.5")
        print("  max_results: 5")
        print("\nOUTPUT:")
        print(result[:2000] + "..." if len(result) > 2000 else result)
        print("="*60)
        
        assert isinstance(result, str)
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
