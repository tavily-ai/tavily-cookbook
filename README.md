# Tavily Hub

Learn to build AI applications with [Tavily](https://tavily.com) — a search API designed for LLMs and AI agents.

## Contribute!

Built something cool with Tavily? We'd love to see it. **Open a PR** to add your tools, agents, or use-cases.

Have a request? Open an issue.

## Structure

| Directory | Purpose |
|-----------|---------|
| [**Cookbooks**](./cookbooks) | Click-through Jupyter notebooks for getting started with the Tavily API |
| [**Agent Toolkit**](./agent-toolkit) | Production-ready tools, agents, and utilities to build research agents faster.<br>Install with `pip install git+https://github.com/tavily-ai/tavily-cookbook.git#subdirectory=agent-toolkit` |
| [**Agent skills**](./.claude/skills) | Folders of instructions, scripts, and resources that agents can discover and use to do things more accurately and efficiently |

## Agent Skills

This repo includes [Agent skills](https://docs.anthropic.com/en/docs/claude-code/skills) for two use cases:

### For Agent Builders

Skills that teach coding assistants (Claude Code, Cursor, etc.) Tavily best practices. Install via Claude Code and your AI assistant will know how to build with Tavily.

| Skill | Description |
|-------|-------------|
| [`tavily-api`](./.claude/skills/tavily-api/SKILL.md) | Comprehensive Tavily API reference with best practices baked in |

### For Terminal Workflows

Real-time web data as context directly in your terminal and IDE. Tavily is a gateway to web data in your coding workflow through agent skills.

| Skill | What it does |
|-------|--------------|
| [`research`](./.claude/skills/research/SKILL.md) | Deep research directly in your terminal with AI-synthesized insights and citations |
| [`crawl-url`](./.claude/skills/crawl-url/SKILL.md) | Crawl documentation at scale - discover and scrape up to 100 pages in seconds |
| [`technical-trends-discovery`](./.claude/skills/technical-trends-discovery/SKILL.md) | Discover emerging AI/tech trends from thought leaders |

## Automation-First

This repo is maintained by Tavily engineers, in collaboration with Claude Code. The skills above double as working examples of what you can build with Tavily.

## Prerequisites

**Sign up** for Tavily at [app.tavily.com](https://app.tavily.com/home/) to get your API key.

## Resources

- [Tavily developer documentation](https://docs.tavily.com/welcome)
- [Tavily support](https://www.tavily.com/contact)
- [Tavily community](https://community.tavily.com/)
- [Tavily github](https://github.com/tavily-ai)