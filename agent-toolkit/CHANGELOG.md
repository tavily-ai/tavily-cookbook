# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-27

### Added

- Initial release of `tavily-agent-toolkit`
- **Tools:**
  - `search_and_answer` - Answer questions with web research + LLM synthesis
  - `search_and_format` - Search and format results for downstream use
  - `search_dedup` - Run multiple queries in parallel with deduplication
  - `crawl_and_summarize` - Extract and summarize entire websites
  - `extract_and_summarize` - Get focused summaries from specific URLs
  - `social_media_search` - Search Reddit, X, LinkedIn, TikTok, etc.
- **Agents:**
  - `hybrid_research` - Combine internal RAG with web research (fast + multi-agent modes)
- **Utilities:**
  - `handle_research_stream` - Stream research results
  - `format_web_results` - Format search results for LLM context
  - `clean_raw_content` - Clean and normalize web content
  - `ainvoke_with_fallback` - Model fallback chains
- **Models:**
  - `ModelConfig`, `ModelObject`, `OutputSchema` and related TypedDicts
- 20+ LLM provider support via LangChain integration

[Unreleased]: https://github.com/tavily-ai/tavily-cookbook/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tavily-ai/tavily-cookbook/releases/tag/v0.1.0
