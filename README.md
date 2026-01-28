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

## Agent Skills

**For AI coding agents:** Tavily provides official [agent skills](https://docs.tavily.com/documentation/agent-skills) that integrate web intelligence directly into your development workflow.

Available skills:

| Skill | Command | Purpose |
|-------|---------|---------|
| **Search** | `/search` | Web search with content snippets, scores, and metadata |
| **Research** | `/research` | AI-synthesized research with citations and structured JSON output |
| **Extract** | `/extract` | Clean content extraction from specific URLs |
| **Crawl** | `/crawl` | Download websites as local markdown files |
| **Best Practices** | `/tavily-best-practices` | Production-ready Tavily integration patterns |

### Installation

```bash
npx skills add tavily-ai/skills
```

Then add your API key to your agent's environment settings (`TAVILY_API_KEY`).

See the [skills repo](https://github.com/tavily-ai/skills) for full documentation, or the [`.claude-plugin`](https://github.com/tavily-ai/skills/tree/main/.claude-plugin) directory for Claude Code-specific configuration.

## Prerequisites

**Sign up** for Tavily at [app.tavily.com](https://app.tavily.com/home/) to get your API key.

## Resources

- [Tavily developer documentation](https://docs.tavily.com/welcome)
- [Tavily support](https://www.tavily.com/contact)
- [Tavily community](https://community.tavily.com/)
- [Tavily github](https://github.com/tavily-ai)