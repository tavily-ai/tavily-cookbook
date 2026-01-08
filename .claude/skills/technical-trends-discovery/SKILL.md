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

Results are saved to `trends-reports/` at the repo root as a single consolidated JSON file:

```
trends-reports/
└── trends_2025-01-06_143022/
    └── report.json         # All results in one file
```

### report.json Structure

```json
{
  "meta": {
    "generated_at": "2025-01-06T14:30:22.123456",
    "pipeline": "x_discovery → tavily_research",
    "sources_count": 15
  },
  "x_discovery": {
    "content": "# X Trends Analysis\n\nThe #1 trend identified...",
    "citations": ["https://x.com/..."]
  },
  "research": {
    "trend": {
      "name": "Model Context Protocol",
      "summary": "An open standard for sharing context between AI tools...",
      "why_important": "MCP is emerging as the standard...",
      "docs_url": "https://modelcontextprotocol.io",
      "github_repo": "https://github.com/modelcontextprotocol",
      "quickstart": {
        "prerequisites": ["Python 3.10+"],
        "install_commands": "pip install mcp==1.25.0",
        "hello_world_code": "from mcp import Client...",
        "expected_output": "Connected to MCP server"
      },
      "use_cases": [...],
      "common_pitfalls": [...],
      "key_packages": [
        {"name": "mcp", "latest_version": "1.25.0", "package_manager": "pip"}
      ],
      "key_concepts": ["Resources", "Tools", "Prompts"],
      "additional_resources": [...]
    },
    "meta": {"research_date": "2025-01-06"}
  },
  "sources": [
    {"url": "https://...", "title": "MCP Documentation"}
  ]
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

