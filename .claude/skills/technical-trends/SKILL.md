---
name: technical-trends
description: "Research emerging AI/tech trends using Tavily Research API. Use when you need to: (1) discover what topics are trending in AI engineering, (2) research emerging patterns and frameworks, (3) get insights from industry thought leaders. Triggers on requests like 'find tech trends', 'what is trending in AI', 'research AI agent trends', or 'what are thought leaders talking about'."
---

# Technical Trends Agent

Research emerging technical trends from industry thought leaders using Tavily Research API.

## Usage

```bash
# Run with default prompt
python scripts/research_trends.py

# Custom topic focus
python scripts/research_trends.py --topic "AI agents"

# Save to file
python scripts/research_trends.py -o trends.json
```

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

## Customization

Modify the research prompt to focus on specific areas:

```python
# Focus on a specific domain
result = client.research(
    input="""Research the latest trends in LLM evaluation and testing.
    What frameworks are gaining traction? What patterns are thought leaders
    recommending for production agent evaluation?""",
    model="pro"
)
```

## Resources

### scripts/
- `research_trends.py` - CLI tool for trend research
