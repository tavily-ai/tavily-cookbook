The AI space shifts rapidly, and our goal is to keep this repo up-to-date and relevant for AI builders.

In order to achieve this, we're creating automations using some of our favorite dev tools: Claude Code, Notion, Tavily, Linear, etc.

These automations will serve as our “use cases”, and they can be found in the [.claude](../.claude) directory


### [technical-trends-discovery](../.claude/skills/technical-trends-discovery/SKILL.md)

Two-step workflow for discovering emerging AI/tech trends:

1. **X Search** (xAI API) - Find what thought leaders are discussing in real-time
2. **Deep Research** (Tavily API) - Comprehensive research on identified trends
3. **Crawl Docs** (Tavily Crawl) - Investigate specific libraries or tools discovered

**Why this approach?** X/Twitter is where thought leaders share real-time opinions. Once trends are identified, Tavily Research provides deep, actionable insights with citations.

**Scripts:**
| Script | Purpose |
|--------|---------|
| `discover_trends.py` | Full workflow: X discovery → Tavily research |
| `x_trends.py` | X search only |
| `research_trends.py` | Tavily research only |
| `crawl_docs.py` | Crawl documentation sites for libraries |

**Quick Start:**
```bash
# Full workflow
python .claude/skills/technical-trends-discovery/scripts/discover_trends.py

# X discovery only
python .claude/skills/technical-trends-discovery/scripts/x_trends.py --days 7

# Crawl docs for a library you discovered
python .claude/skills/technical-trends-discovery/scripts/crawl_docs.py "https://docs.example.com"
```

**Required env vars:** `XAI_API_KEY`, `TAVILY_API_KEY`