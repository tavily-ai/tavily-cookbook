"""Tests for _deduplicate_by_url function and search_dedup."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from tools.async_search_and_dedup import _deduplicate_by_url, search_dedup

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

API_KEY = os.environ.get("TAVILY_API_KEY", "")


@pytest.mark.skipif(not API_KEY, reason="TAVILY_API_KEY not set")
class TestSearchDedupUsage:
    """Test search_dedup returns proper usage metrics."""

    @pytest.mark.asyncio
    async def test_search_dedup_usage_metrics(self):
        """Test that search_dedup returns tavily usage metrics."""
        result = await search_dedup(
            api_key=API_KEY,
            queries=["python programming", "javascript frameworks"],
            max_results=3,
            search_depth="basic",
        )
        
        assert "results" in result
        assert "queries" in result
        
        # Verify tavily usage metrics
        assert "tavily_usage" in result
        tavily_usage = result["tavily_usage"]
        assert tavily_usage["total_credits"] >= 2  # At least 2 searches
        assert "search_count" in tavily_usage
        assert tavily_usage["search_count"] == 2  # 2 queries
        assert "search_response_time" in tavily_usage
        assert tavily_usage["search_response_time"] > 0
        
        # Should NOT have extract or crawl counts
        assert "extract_count" not in tavily_usage
        assert "crawl_count" not in tavily_usage
        
        # Overall timing
        assert "response_time" in result
        assert result["response_time"] > 0
        
        print("\nTavily usage:", tavily_usage)
        print("Total time:", result["response_time"])


class TestDeduplicateByUrl:
    """Test cases for URL-based deduplication logic."""

    def test_single_response_passthrough(self):
        """Single response with unique URLs should pass through unchanged."""
        responses = [
            {
                "query": "Who is Messi?",
                "results": [
                    {
                        "title": "Messi Bio",
                        "url": "https://example.com/messi",
                        "content": "Messi is a footballer",
                        "score": 0.9,
                    }
                ],
                "response_time": 1.5,
            }
        ]
        
        result = _deduplicate_by_url(responses)
        
        assert len(result["results"]) == 1
        assert result["results"][0]["url"] == "https://example.com/messi"
        assert result["results"][0]["content"] == "Messi is a footballer"
        assert result["response_time"] == 1.5

    def test_duplicate_urls_merged(self):
        """Results with same URL from different queries should be merged."""
        responses = [
            {
                "query": "Messi biography",
                "results": [
                    {
                        "title": "Messi Bio",
                        "url": "https://example.com/messi",
                        "content": "Born in Argentina",
                        "score": 0.8,
                    }
                ],
                "response_time": 1.0,
            },
            {
                "query": "Messi career",
                "results": [
                    {
                        "title": "Messi Bio",
                        "url": "https://example.com/messi",
                        "content": "Plays for Inter Miami",
                        "score": 0.85,
                    }
                ],
                "response_time": 1.2,
            },
        ]
        
        result = _deduplicate_by_url(responses)
        
        # Should have only one result (merged)
        assert len(result["results"]) == 1
        # Both chunks should be present
        content = result["results"][0]["content"]
        assert "Born in Argentina" in content
        assert "Plays for Inter Miami" in content

    def test_chunk_deduplication(self):
        """Identical chunks from same URL should not be duplicated."""
        responses = [
            {
                "query": "query1",
                "results": [
                    {
                        "url": "https://example.com/page",
                        "content": "Chunk A [...] Chunk B",
                        "score": 0.7,
                    }
                ],
                "response_time": 1.0,
            },
            {
                "query": "query2",
                "results": [
                    {
                        "url": "https://example.com/page",
                        "content": "Chunk B [...] Chunk C",
                        "score": 0.75,
                    }
                ],
                "response_time": 1.1,
            },
        ]
        
        result = _deduplicate_by_url(responses)
        
        content = result["results"][0]["content"]
        chunks = [c.strip() for c in content.split(" [...] ")]
        
        # Should have 3 unique chunks, not 4
        assert len(chunks) == 3
        assert set(chunks) == {"Chunk A", "Chunk B", "Chunk C"}

    def test_higher_score_preserved(self):
        """When merging duplicates, higher score should be kept."""
        responses = [
            {
                "query": "query1",
                "results": [
                    {
                        "url": "https://example.com/page",
                        "content": "Content 1",
                        "score": 0.5,
                    }
                ],
                "response_time": 1.0,
            },
            {
                "query": "query2",
                "results": [
                    {
                        "url": "https://example.com/page",
                        "content": "Content 2",
                        "score": 0.9,
                    }
                ],
                "response_time": 1.0,
            },
        ]
        
        result = _deduplicate_by_url(responses)
        
        assert result["results"][0]["score"] == 0.9

    def test_results_sorted_by_score(self):
        """Results should be sorted by score in descending order."""
        responses = [
            {
                "query": "test",
                "results": [
                    {"url": "https://low.com", "content": "Low", "score": 0.3},
                    {"url": "https://high.com", "content": "High", "score": 0.95},
                    {"url": "https://mid.com", "content": "Mid", "score": 0.6},
                ],
                "response_time": 1.0,
            }
        ]
        
        result = _deduplicate_by_url(responses)
        
        scores = [r["score"] for r in result["results"]]
        assert scores == [0.95, 0.6, 0.3]

    def test_image_deduplication(self):
        """Images with same URL should be deduplicated."""
        responses = [
            {
                "query": "query1",
                "results": [],
                "images": [
                    {"url": "https://img.com/1.jpg", "description": "Image 1"},
                    {"url": "https://img.com/2.jpg", "description": "Image 2"},
                ],
                "response_time": 1.0,
            },
            {
                "query": "query2",
                "results": [],
                "images": [
                    {"url": "https://img.com/2.jpg", "description": "Image 2 again"},
                    {"url": "https://img.com/3.jpg", "description": "Image 3"},
                ],
                "response_time": 1.0,
            },
        ]
        
        result = _deduplicate_by_url(responses)
        
        # Should have 3 unique images
        assert len(result["images"]) == 3
        image_urls = [img["url"] for img in result["images"]]
        assert set(image_urls) == {
            "https://img.com/1.jpg",
            "https://img.com/2.jpg",
            "https://img.com/3.jpg",
        }

    def test_image_string_format(self):
        """Images can be plain strings (URLs) instead of dicts."""
        responses = [
            {
                "query": "query1",
                "results": [],
                "images": ["https://img.com/1.jpg", "https://img.com/2.jpg"],
                "response_time": 1.0,
            },
            {
                "query": "query2",
                "results": [],
                "images": ["https://img.com/2.jpg"],
                "response_time": 1.0,
            },
        ]
        
        result = _deduplicate_by_url(responses)
        
        assert len(result["images"]) == 2

    def test_answers_concatenated(self):
        """Answers from multiple responses should be joined."""
        responses = [
            {
                "query": "query1",
                "results": [],
                "answer": "First answer.",
                "response_time": 1.0,
            },
            {
                "query": "query2",
                "results": [],
                "answer": "Second answer.",
                "response_time": 1.0,
            },
        ]
        
        result = _deduplicate_by_url(responses)
        
        assert result["answer"] == "First answer.\n\nSecond answer."

    def test_max_response_time(self):
        """Response time should be max of all responses (parallel execution)."""
        responses = [
            {"query": "q1", "results": [], "response_time": 1.5},
            {"query": "q2", "results": [], "response_time": 2.3},
            {"query": "q3", "results": [], "response_time": 1.8},
        ]
        
        result = _deduplicate_by_url(responses)
        
        assert result["response_time"] == 2.3

    def test_queries_collected(self):
        """All original queries should be collected."""
        responses = [
            {"query": "Who is Messi?", "results": [], "response_time": 1.0},
            {"query": "Messi awards", "results": [], "response_time": 1.0},
            {"query": "Messi goals", "results": [], "response_time": 1.0},
        ]
        
        result = _deduplicate_by_url(responses)
        
        assert result["queries"] == ["Who is Messi?", "Messi awards", "Messi goals"]

    def test_empty_url_skipped(self):
        """Results with empty or missing URL should be skipped."""
        responses = [
            {
                "query": "test",
                "results": [
                    {"url": "", "content": "No URL", "score": 0.9},
                    {"url": "https://valid.com", "content": "Valid", "score": 0.5},
                    {"content": "Missing URL key", "score": 0.8},
                ],
                "response_time": 1.0,
            }
        ]
        
        result = _deduplicate_by_url(responses)
        
        assert len(result["results"]) == 1
        assert result["results"][0]["url"] == "https://valid.com"

    def test_empty_content_handled(self):
        """Results with empty content should not cause errors."""
        responses = [
            {
                "query": "test",
                "results": [
                    {"url": "https://example.com", "content": "", "score": 0.5},
                ],
                "response_time": 1.0,
            }
        ]
        
        result = _deduplicate_by_url(responses)
        
        assert len(result["results"]) == 1
        assert result["results"][0]["content"] == ""

    def test_no_images_returns_none(self):
        """When no images present, images field should be None."""
        responses = [
            {"query": "test", "results": [], "response_time": 1.0},
        ]
        
        result = _deduplicate_by_url(responses)
        
        assert result["images"] is None

    def test_no_answer_returns_none(self):
        """When no answers present, answer field should be None."""
        responses = [
            {"query": "test", "results": [], "response_time": 1.0},
        ]
        
        result = _deduplicate_by_url(responses)
        
        assert result["answer"] is None

    def test_preserves_other_result_fields(self):
        """Fields like title, favicon, raw_content should be preserved."""
        responses = [
            {
                "query": "test",
                "results": [
                    {
                        "url": "https://example.com",
                        "title": "Example Page",
                        "content": "Content here",
                        "score": 0.8,
                        "favicon": "https://example.com/favicon.ico",
                        "raw_content": "<html>...</html>",
                    }
                ],
                "response_time": 1.0,
            }
        ]
        
        result = _deduplicate_by_url(responses)
        
        r = result["results"][0]
        assert r["title"] == "Example Page"
        assert r["favicon"] == "https://example.com/favicon.ico"
        assert r["raw_content"] == "<html>...</html>"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

