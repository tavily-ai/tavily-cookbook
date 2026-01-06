---
name: technical-trends
description: "Discover emerging AI/tech trends using a two-step workflow: (1) Search X for what thought leaders are discussing, (2) Deep research on identified trends via Tavily. Use when you need to find what's trending in AI engineering, research emerging patterns, or get insights from industry voices."
---

# Technical Trends Discovery

Two-step workflow for discovering and researching AI/tech trends:

1. **X Search** (xAI API) → Find what thought leaders are discussing
2. **Deep Research** (Tavily API) → Comprehensive research on identified trends

## Why Two Steps?

X/Twitter is where thought leaders share real-time opinions and insights. The xAI API excels at searching and analyzing X posts. Once we identify the trends, Tavily Research does comprehensive web research to provide deep, actionable insights.

## Quick Start

```bash
# Run full workflow: X discovery → Tavily research
python .claude/skills/technical-trends-discovery/scripts/discover_trends.py

# X discovery only (faster, no Tavily)
python .claude/skills/technical-trends-discovery/scripts/discover_trends.py --x-only

# Custom handles and date range
python .claude/skills/technical-trends-discovery/scripts/discover_trends.py \
  --handles karpathy simonw swyx \
  --days 7
```

## Output Format

Results are saved to `research/` at the repo root:

```
research/
└── trends_2025-01-04_143022/
    ├── x_discovery.md   # What thought leaders are discussing
    ├── report.md        # Deep research report
    └── sources.json     # Research citations
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
| `--handles` | 7 AI leaders | X handles to search (max 10) |
| `--days` | 20 | Days back to search |
| `--min-favorites` | 100 | Filter by engagement |
| `--x-only` | false | Skip Tavily research |
| `--no-save` | false | Print only, don't save |
| `--output` | auto | Custom output directory |

## Individual Scripts

### x_trends.py - X Search Only
```bash
python .claude/skills/technical-trends-discovery/scripts/x_trends.py
```

### research_trends.py - Tavily Research Only
```bash
python .claude/skills/technical-trends-discovery/scripts/research_trends.py
```

## Environment Variables

```bash
export XAI_API_KEY="your-xai-key"      # Required for X search
export TAVILY_API_KEY="your-tavily-key" # Required for deep research
```

## Python Usage

```python
from discover_trends import discover_trends

# Run full workflow
results = discover_trends(
    handles=["karpathy", "simonw", "swyx"],
    days_back=14,
    min_favorites=200,
)

# Access results
print(results["x_trends"]["content"])      # X discovery
print(results["research"]["content"])       # Deep research
print(results["research"]["sources"])       # Citations
```
