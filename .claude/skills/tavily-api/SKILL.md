---
name: tavily-api
description: Implement web search, url content extraction, website crawling, and deep research with the tavily API. Use when building agentic workflows, rag systems, or autonomous agents that require real-time context from the web.
---

Tavily is a specialized search API designed specifically for LLMs, enabling developers to build AI applications that can access real-time, accurate web data. Let's use the Python SDK to build with tavily.

# Tavily Python SDK

## Installation

```bash
pip install tavily-python
```

## Client Initialization

```python
from tavily import TavilyClient

client = TavilyClient(api_key="tvly-YOUR_API_KEY")

# Or use environment variable TAVILY_API_KEY
client = TavilyClient()
```

**Async client:**

The async client enables parallel query execution, ideal for agentic workflows that need to gather information quickly before passing it to a model for analysis.

```python
from tavily import AsyncTavilyClient

async_client = AsyncTavilyClient(api_key="tvly-YOUR_API_KEY")
```

## Available Endpoints

| Endpoint | Purpose | Use Case |
|--------|---------|----------|
| `search()` | Web search | real time data retrieval from the web |
| `extract()` | Scrape content from URLs | Page content extraction |
| `crawl() and map()` | Traverse website structures and simultaneously scrape pages | Documentation, site-wide extraction |
| `research` | Out of the box research agent | ready-to-use iterative research |




## Choosing the Right Method

**If you are building a custom agent or agentic workflow:**

| Need | Method |
|------|--------|
| Web search results | `search()` |
| Content from specific URLs | `extract()` |
| Content from an entire site | `crawl()` |
| URL discovery from a site | `map()` |

These methods give you full control but require additional work: data processing, LLM integration, and workflow orchestration.

**If you want an out-of-the-box solution:**

| Need | Method |
|------|--------|
| End-to-end research with AI synthesis and built-in context engineering | `research()` |

The research endpoint provides faster time-to-value with AI-synthesized insights, but offers less flexibility than building custom workflows.

## Detailed Guides

For detailed usage instructions, parameters, patterns, and best practices:

- **[references/search.md](references/search.md)** — Query optimization, filtering, async patterns, post-filtering strategies (regex + LLM verification)
- **[references/extract.md](references/extract.md)** — One-step vs two-step extraction, advanced mode, research pipelines
- **[references/crawl.md](references/crawl.md)** — Crawl vs Map, depth/breadth control, path patterns, performance optimization
- **[references/research.md](references/research.md)** — Usage, Streaming, structured output, polling