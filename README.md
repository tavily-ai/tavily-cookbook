# Tavily Hub

Learn to build AI applications with [Tavily](https://tavily.com) — a search API designed for LLMs and AI agents.

## Contribute!

Built something cool with Tavily? We'd love to see it. **Open a PR** to add your tools, agents, or use-cases.

Have a request? Open an issue.

## Structure

| Directory | Purpose |
|-----------|---------|
| [**Cookbooks**](./cookbooks) | Click-through Jupyter notebooks for getting started with the Tavily API |
| [**Agent Toolkit**](./agent-toolkit) | Production-ready tools, agents, and utilities to build research agents faster. Current version: **v0.1.0**<br><br>**Stable:** `pip install "tavily-agent-toolkit @ git+https://github.com/tavily-ai/tavily-cookbook.git@v0.1.0#subdirectory=agent-toolkit"`<br>**Latest:** `pip install "tavily-agent-toolkit @ git+https://github.com/tavily-ai/tavily-cookbook.git@main#subdirectory=agent-toolkit"` |

## Claude Code Plugin

**For Claude Code developers:** Tavily skills are now available as an official Claude Code plugin at [**tavily-ai/tavily-plugins**](https://github.com/tavily-ai/tavily-plugins).

The plugin serves two purposes:

- **Building with Tavily:** Includes comprehensive API reference documentation. Claude automatically uses this when you ask it to integrate Tavily into your code, making Claude a Tavily API expert.
- **Web research in your terminal:** Access real-time web data, run deep research, and crawl documentation directly in your terminal.

Install directly from the Claude Code marketplace:

```
/plugin marketplace add tavily-ai/tavily-plugins
/plugin install tavily@tavily-plugins
```

See the [plugin repo](https://github.com/tavily-ai/tavily-plugins) for full installation instructions and usage examples.

## Prerequisites

**Sign up** for Tavily at [app.tavily.com](https://app.tavily.com/home/) to get your API key.

## Resources

- [Tavily developer documentation](https://docs.tavily.com/welcome)
- [Tavily support](https://www.tavily.com/contact)
- [Tavily community](https://community.tavily.com/)
- [Tavily github](https://github.com/tavily-ai)