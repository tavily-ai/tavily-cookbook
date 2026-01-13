"""Tests for social_media_search function."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from tools.social_media import PLATFORM_DOMAINS, social_media_search

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

API_KEY = os.environ.get("TAVILY_API_KEY", "")


@pytest.fixture
def api_key():
    """Fixture providing the Tavily API key."""
    if not API_KEY:
        pytest.skip("TAVILY_API_KEY not found in .env file")
    return API_KEY


class TestSocialMediaSearch:
    """Test cases for social media search across all platforms."""

    def test_search_tiktok(self, api_key):
        """Test searching TikTok."""
        result = social_media_search(
            query="trending dance",
            api_key=api_key,
            platform="tiktok",
            max_results=3,
        )
        
        assert "results" in result
        assert isinstance(result["results"], list)
        # Check that results are from tiktok.com if any exist
        for r in result["results"]:
            assert "tiktok.com" in r["url"]

    def test_search_facebook(self, api_key):
        """Test searching Facebook."""
        result = social_media_search(
            query="community groups",
            api_key=api_key,
            platform="facebook",
            max_results=3,
        )
        
        assert "results" in result
        assert isinstance(result["results"], list)
        for r in result["results"]:
            assert "facebook.com" in r["url"]

    def test_search_instagram(self, api_key):
        """Test searching Instagram."""
        result = social_media_search(
            query="photography",
            api_key=api_key,
            platform="instagram",
            max_results=3,
        )
        
        assert "results" in result
        assert isinstance(result["results"], list)
        for r in result["results"]:
            assert "instagram.com" in r["url"]

    def test_search_reddit(self, api_key):
        """Test searching Reddit."""
        result = social_media_search(
            query="programming tips",
            api_key=api_key,
            platform="reddit",
            max_results=3,
        )
        
        assert "results" in result
        assert isinstance(result["results"], list)
        # Reddit typically returns results
        assert len(result["results"]) > 0
        for r in result["results"]:
            assert "reddit.com" in r["url"]
        
        # Verify usage metrics (search only, no extract, no LLM)
        assert "usage" in result
        usage = result["usage"]
        assert "response_time" in usage
        assert usage["response_time"] > 0
        
        # Should have tavily search usage
        assert "tavily" in usage
        assert usage["tavily"]["total_credits"] > 0
        assert "search_count" in usage["tavily"]
        assert usage["tavily"]["search_count"] == 1
        
        # Should NOT have extract_count (not used)
        assert "extract_count" not in usage["tavily"]
        
        # Should NOT have llm usage (not used)
        assert "llm" not in usage
        
        print("\nUsage metrics (search only):", usage)

    def test_search_linkedin(self, api_key):
        """Test searching LinkedIn."""
        result = social_media_search(
            query="job opportunities",
            api_key=api_key,
            platform="linkedin",
            max_results=3,
        )
        
        assert "results" in result
        assert isinstance(result["results"], list)
        for r in result["results"]:
            assert "linkedin.com" in r["url"]

    def test_search_x(self, api_key):
        """Test searching X (Twitter)."""
        result = social_media_search(
            query="tech news",
            api_key=api_key,
            platform="x",
            max_results=3,
        )
        
        assert "results" in result
        assert isinstance(result["results"], list)
        for r in result["results"]:
            assert "x.com" in r["url"]

    def test_search_combined(self, api_key):
        """Test searching all platforms combined."""
        result = social_media_search(
            query="artificial intelligence",
            api_key=api_key,
            platform="combined",
            max_results=10,
        )
        
        assert "results" in result
        assert isinstance(result["results"], list)
        # Combined search should return results
        assert len(result["results"]) > 0
        
        # Verify results are from one of the allowed domains
        allowed_domains = set(PLATFORM_DOMAINS.values())
        for r in result["results"]:
            url = r["url"]
            assert any(domain in url for domain in allowed_domains), \
                f"URL {url} not from any social media platform"

    def test_invalid_platform_raises_error(self, api_key):
        """Test that invalid platform raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            social_media_search(
                query="test",
                api_key=api_key,
                platform="invalid_platform",
            )
        
        assert "Invalid platform" in str(exc_info.value)

    def test_result_structure(self, api_key):
        """Test that results have expected fields."""
        result = social_media_search(
            query="technology",
            api_key=api_key,
            platform="reddit",
            max_results=3,
        )
        
        assert "results" in result
        if result["results"]:
            first_result = result["results"][0]
            # Check expected fields exist
            assert "url" in first_result
            assert "title" in first_result or "content" in first_result


class TestSocialMediaSearchWithRawContent:
    """Test cases for social media search with raw content extraction."""

    def test_tiktok_with_raw_content(self, api_key):
        """Test TikTok search with raw content extraction."""
        result = social_media_search(
            query="viral video",
            api_key=api_key,
            platform="tiktok",
            include_raw_content=True,
            max_results=2,
        )
        
        # Assert search results
        assert "results" in result
        assert isinstance(result["results"], list)
        for r in result["results"]:
            assert "tiktok.com" in r["url"]
        
        # Assert raw_content field exists
        if result["results"]:
            for r in result["results"]:
                assert "raw_content" in r

    def test_facebook_with_raw_content(self, api_key):
        """Test Facebook search with raw content extraction."""
        result = social_media_search(
            query="community",
            api_key=api_key,
            platform="facebook",
            include_raw_content=True,
            max_results=2,
        )
        
        assert "results" in result
        assert isinstance(result["results"], list)
        for r in result["results"]:
            assert "facebook.com" in r["url"]
        
        if result["results"]:
            for r in result["results"]:
                assert "raw_content" in r

    def test_instagram_with_raw_content(self, api_key):
        """Test Instagram search with raw content extraction."""
        result = social_media_search(
            query="travel photos",
            api_key=api_key,
            platform="instagram",
            include_raw_content=True,
            max_results=2,
        )
        
        assert "results" in result
        assert isinstance(result["results"], list)
        for r in result["results"]:
            assert "instagram.com" in r["url"]
        
        if result["results"]:
            for r in result["results"]:
                assert "raw_content" in r

    def test_reddit_with_raw_content(self, api_key):
        """Test Reddit search with raw content extraction."""
        result = social_media_search(
            query="python tips",
            api_key=api_key,
            platform="reddit",
            include_raw_content=True,
            max_results=2,
        )
        
        assert "results" in result
        assert isinstance(result["results"], list)
        assert len(result["results"]) > 0
        for r in result["results"]:
            assert "reddit.com" in r["url"]
        
        # Reddit usually has good raw_content
        for r in result["results"]:
            assert "raw_content" in r
        
        # Verify usage metrics (search + extract, no LLM)
        assert "usage" in result
        usage = result["usage"]
        assert "response_time" in usage
        
        # Should have tavily search AND extract usage
        assert "tavily" in usage
        assert usage["tavily"]["total_credits"] > 0
        assert "search_count" in usage["tavily"]
        assert usage["tavily"]["search_count"] == 1
        assert "extract_count" in usage["tavily"]
        assert usage["tavily"]["extract_count"] == 1
        
        # Should NOT have llm usage (not used)
        assert "llm" not in usage
        
        print("\nUsage metrics (search + extract):", usage)

    def test_linkedin_with_raw_content(self, api_key):
        """Test LinkedIn search with raw content extraction."""
        result = social_media_search(
            query="career advice",
            api_key=api_key,
            platform="linkedin",
            include_raw_content=True,
            max_results=2,
        )
        
        assert "results" in result
        assert isinstance(result["results"], list)
        for r in result["results"]:
            assert "linkedin.com" in r["url"]
        
        if result["results"]:
            for r in result["results"]:
                assert "raw_content" in r

    def test_x_with_raw_content(self, api_key):
        """Test X search with raw content extraction."""
        result = social_media_search(
            query="breaking news",
            api_key=api_key,
            platform="x",
            include_raw_content=True,
            max_results=2,
        )
        
        assert "results" in result
        assert isinstance(result["results"], list)
        for r in result["results"]:
            assert "x.com" in r["url"]
        
        if result["results"]:
            for r in result["results"]:
                assert "raw_content" in r

    def test_combined_with_raw_content(self, api_key):
        """Test combined search with raw content extraction."""
        result = social_media_search(
            query="machine learning",
            api_key=api_key,
            platform="combined",
            include_raw_content=True,
            max_results=5,
        )
        usage = result["usage"]
        
        # Assert search results
        assert "results" in result
        assert isinstance(result["results"], list)
        assert len(result["results"]) > 0
        
        # Verify results are from allowed domains
        allowed_domains = set(PLATFORM_DOMAINS.values())
        for r in result["results"]:
            assert any(domain in r["url"] for domain in allowed_domains)
        
        # Assert raw_content field exists on all results
        for r in result["results"]:
            assert "raw_content" in r
        
        print("\nUsage metrics (search + extract):", usage)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

