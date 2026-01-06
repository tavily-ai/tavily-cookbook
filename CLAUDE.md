# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is the Tavily Hub - a resource for building AI applications with Tavily's API, maintained by Claude Code and the Tavily FDE team. Contains Jupyter notebook tutorials, solution patterns, and Claude Code skills/automations that use Tavily for real-time web data.

**Philosophy**: Automation-driven, always current. The automations that maintain this repo serve as live examples of what you can build.

## Focus Areas

This repository centers on three emerging AI engineering trends identified through ongoing research:

| Trend | Description |
|-------|-------------|
| **Agent Engineering** | Building, evaluating, and deploying LLM-based agents with emphasis on harnesses, observability, composability, and pragmatic evaluation loops |
| **Context Engineering** | Techniques for selecting, maintaining, pruning, and presenting context to foundation models—treating memory as a living system (PCE, RAG 2.0, MCP) |
| **AI-Assisted Programming** | LLM-powered code generation workflows, "vibe coding" patterns, and tooling for validating/maintaining AI-generated code |

Cookbooks, solution patterns, and automations in this repo demonstrate practical implementations across these areas using Tavily's real-time web data capabilities.

## Repository Structure

| Directory | Purpose |
|-----------|---------|
| `cookbooks/` | Jupyter notebooks organized by API: `getting-started/`, `search/`, `research/`, `crawl/`, `integrations/`, `mcp/` |
| `solution-patterns/` | Production agent architecture guides and enablement material |
| `use-cases/` | Real automations using Claude Code, Tavily, and others |
| `research/` | AI trends research outputs (dated directories with `report.md` + `sources.json`) |
| `.claude/skills/` | Claude Code skills including `tavily-api` reference and `technical-trends-discovery` |


## Tavily Python SDK Quick Reference

```bash
pip install tavily-python
```

```python
from tavily import TavilyClient, AsyncTavilyClient

client = TavilyClient()  # Uses TAVILY_API_KEY env var
async_client = AsyncTavilyClient()  # For parallel queries
```

### API Methods

| Method | Purpose |
|--------|---------|
| `search()` | Web search with filtering, scoring, domain control |
| `extract()` | Scrape content from specific URLs (max 20 per call) |
| `crawl()` | Traverse and extract from entire websites |
| `map()` | Discover URLs without content extraction (faster) |
| `research()` | End-to-end research agent with AI synthesis |

### Key Patterns

**Search**: Use `search_depth="advanced"` with `chunks_per_source=5` for best quality (2s latency tradeoff). Use `AsyncTavilyClient` for parallel queries in agentic workflows.

**Extract**: Two-step pattern recommended - search first, filter by `score > 0.5`, then extract filtered URLs.

**Crawl vs Map**: Use `map()` for URL discovery (faster, no content), `crawl()` when you need actual content.

**Research**: Two-step async - call `research()` to get `request_id`, poll with `get_research()` until `status == "completed"`.

### Detailed API Documentation

See `.claude/skills/tavily-api/references/` for comprehensive guides:
- `search.md` - Query optimization, domain filtering, async patterns, post-filtering (regex + LLM verification)
- `extract.md` - One-step vs two-step extraction, advanced mode
- `crawl.md` - Depth/breadth control, path patterns, map-then-extract pattern
- `research.md` - Streaming, structured output, polling

## Running Notebooks

Notebooks are standalone Jupyter files in the `cookbooks` directory. Each requires:
1. `TAVILY_API_KEY` environment variable
2. Some LLM API key, `OPENAI_API_KEY` environment variable by default
2. Dependencies installed (typically `tavily-python`, framework-specific packages like `langchain`, `chromadb`, etc.)

## Automations

The `.claude/skills/` directory contains Claude Code skills that power repo maintenance:
- `tavily-api/` - Reference documentation for Tavily API usage
- `technical-trends-discovery/` - Research emerging AI/tech trends using Tavily Research API

### Planned Automation Agents

1. **Technical Trends Agent** - Identifies tech trends from thought leaders, passes to Tavily Research API for deep research
2. **Integration Cookbook Update Agent** - Uses Tavily Crawl to find integration docs and ensure cookbooks are current
3. **API Best Practices Update Agent** - Crawls docs to keep skills updated with latest patterns

## Platform Integrations

MCP servers connect this repo to the FDE automation stack:

| Platform | Purpose |
|----------|---------|
| **Linear** | FDE team workspace for task management |
| **Notion** | Central FDE documentation |
| **Slack** | #fde channel notifications |
| **GitHub** | This repo |
