---
name: technical-trends
description: "Research emerging AI/tech trends using Tavily Research API. Use when you need to: (1) discover what topics are trending in AI engineering, (2) research emerging patterns and frameworks, (3) get insights from industry thought leaders. Triggers on requests like 'find tech trends', 'what is trending in AI', 'research AI agent trends', or 'what are thought leaders talking about'."
---

# Technical Trends Agent

Research emerging technical trends from industry thought leaders using Tavily Research API.

## Usage

```bash
# Run research (saves to research/trends_<timestamp>/ directory at repo root)
python .claude/skills/technical-trends-discovery/scripts/research_trends.py

# Custom output directory
python .claude/skills/technical-trends-discovery/scripts/research_trends.py -o research/custom_name

# Print only, don't save
python .claude/skills/technical-trends-discovery/scripts/research_trends.py --no-save
```

## Output Format

Results are saved to `research/` at the repo root in timestamped directories:

```
research/
└── trends_2025-01-04_143022/
    ├── report.md      # Full research report in markdown
    └── sources.json   # Source citations (url + title)
```

This format makes reports easy to visualize in markdown viewers while keeping sources organized alongside.

## How It Works

The script uses Tavily Research API with a prompt that:
1. Identifies trending topics from AI/tech thought leaders
2. Analyzes which topics have the most momentum
3. Performs deep research on the most relevant trends

**Default research prompt:**
> Identify the most important emerging trends in AI engineering by analyzing what industry thought leaders are discussing. Look for topics from voices like Harrison Chase, Lance Martin, Simon Willison, Andrej Karpathy, Chip Huyen, and other prominent AI engineers. Focus on: agent architectures, RAG patterns, evaluation frameworks, tool use, and production patterns. For the top 3-5 trending topics, provide deep research on current developments, key implementations, and practical applications.

## Python Usage

```python
from tavily import TavilyClient
import time

client = TavilyClient()

# Initiate research
result = client.research(
    input="""Identify the most important emerging trends in AI engineering by analyzing
    what industry thought leaders are discussing. Look for topics from voices like
    Harrison Chase, Lance Martin, Simon Willison, Andrej Karpathy, Chip Huyen, and
    other prominent AI engineers. Focus on: agent architectures, RAG patterns,
    evaluation frameworks, tool use, and production patterns. For the top 3-5
    trending topics, provide deep research on current developments, key
    implementations, and practical applications.""",
    model="pro"  # Use pro for comprehensive multi-angle analysis
)
request_id = result["request_id"]

# Poll until completed
while True:
    response = client.get_research(request_id)
    if response["status"] == "completed":
        print(response["content"])  # Full research report
        print(response["sources"])  # Citations
        break
    elif response["status"] == "failed":
        raise RuntimeError(response.get("error"))
    time.sleep(5)
```

## Resources

### scripts/
- `research_trends.py` - CLI tool for Tavily trend research
- `x_trends.py` - CLI tool for X/Twitter trend search via xAI API
