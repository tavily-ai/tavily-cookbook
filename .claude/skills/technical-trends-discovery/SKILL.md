---
name: technical-trends-discovery
description: "Discover emerging AI/tech trends using a two-step workflow: (1) Search X for what thought leaders are discussing, (2) Deep research on the indentified tech trend with structured JSON output containing docs URLs, packages with versions, key concepts, and insights. Use when you need to find what's trending in AI engineering, research emerging patterns, or get insights from industry voices."
---

# Technical Trends Discovery

X/Twitter is where thought leaders share real-time opinions and insights. The xAI API excels at searching and analyzing X posts to identify what's actually important right now. Once we identify THE trend, Tavily Research extracts structured metadata including **latest package versions**, key concepts, documentation URLs, and insights.

Two-step automated pipeline for discovering and deeply researching the **most important** AI/tech trend:

1. **X Search** (xAI API) → Find what thought leaders are discussing, identify #1 trend
2. **Deep Research** (Tavily API) → Comprehensive research on the topic



## Quick Start

```bash
# Run full pipeline: X → Research → Structured JSON
python .claude/skills/technical-trends-discovery/scripts/discover_trends.py

# X discovery only (skip Tavily research)
python .claude/skills/technical-trends-discovery/scripts/discover_trends.py --x-only

# Custom handles and date range
python .claude/skills/technical-trends-discovery/scripts/discover_trends.py \
  --handles karpathy simonw swyx \
  --days 7
```

## Output Format

Results are saved to `trends-reports/` at the repo root:

```
trends-reports/
└── trends_2025-01-06_143022/
    ├── data.json           # Structured trend metadata (main output)
    ├── x_discovery.md      # X findings with trend ranking
    └── sources.json        # Research citations
```

### data.json Structure

```json
{
  "trend": {
    "name": "Model Context Protocol",
    "summary": "An open standard for sharing context between AI tools...",
    "why_important": "MCP is emerging as the standard for tool interoperability...",
    "insights": "Adoption is accelerating as major AI companies integrate MCP...",
    "docs_url": "https://modelcontextprotocol.io",
    "github_repo": "https://github.com/modelcontextprotocol",
    "additional_resources": [
      "https://modelcontextprotocol.io/quickstart",
      "https://github.com/modelcontextprotocol/servers"
    ],
    "key_packages": [
      {
        "name": "mcp",
        "latest_version": "1.25.0",
        "package_manager": "pip"
      }
    ],
    "getting_started_link": "https://modelcontextprotocol.io/quickstart",
    "category": "context_engineering",
    "key_concepts": [
      "Resources",
      "Tools",
      "Prompts",
      "Transports",
      "Server-Client Architecture"
    ],
    "related_projects": [
      "Claude Code",
      "Cursor",
      "Continue.dev"
    ]
  },
  "meta": {
    "research_date": "2025-01-06"
  }
}
```

## Default Thought Leaders

| Handle | Person |
|--------|--------|
| hwchase17 | Harrison Chase (LangChain) |
| rlancemartin | Lance Martin (LangChain) |
| simonw | Simon Willison |
| karpathy | Andrej Karpathy |
| cherny | Boris Cherny |
| swyx | Swyx |
| alexalbert__ | Alex Albert (Anthropic) |

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--handles`, `-H` | 7 AI leaders | X handles to search |
| `--days`, `-d` | 20 | Days back to search |

## Environment Variables

```bash
export XAI_API_KEY="your-xai-key"      # Required for X search
export TAVILY_API_KEY="your-tavily-key" # Required for research
```

## Python Usage

```python
from discover_trends import discover_trends

# Run full pipeline
results = discover_trends(
    handles=["karpathy", "simonw", "swyx"],
    days_back=14,
)

# Access results
print(results["x_trends"]["content"])       # X discovery markdown
print(results["research"]["content"])       # Structured JSON with trend data
print(results["output_dir"])                # Where files were saved
```

## Categories

Trends are automatically categorized as:
- `agent_engineering` - Building/deploying LLM agents, frameworks, orchestration
- `context_engineering` - RAG, memory, context management, MCP
- `ai_programming` - Code generation, AI-assisted development
- `other` - Everything else

