# Tests

Integration tests for all tools and agents. These tests serve as living documentation—see exactly what inputs each tool expects and what outputs to expect. With AI coding assistants like Claude Code and Cursor, test-driven development is faster than ever. Use these tests as a starting point when building your own research workflows.

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_search_and_answer.py

# Run with verbose output
pytest tests/ -v
```

## Test Files

| Test File | What It Covers |
|-----------|----------------|
| `test_search_and_answer.py` | Web search + LLM synthesis |
| `test_async_search_and_dedup.py` | Parallel search with deduplication |
| `test_crawl_and_summarize.py` | Website crawling + summarization |
| `test_extract_and_summarize.py` | URL extraction + summarization |
| `test_social_media.py` | Platform-specific social search |
| `test_hybrid_research_integration.py` | Full hybrid research agent |
