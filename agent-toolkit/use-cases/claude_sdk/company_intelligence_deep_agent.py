"""
Company Intelligence Research Agent

Uses the Claude Agent SDK with custom Tavily-powered MCP tools to research
companies by crawling websites, extracting content, and searching the web.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions,
                              ClaudeSDKClient, ResultMessage, TextBlock,
                              ToolUseBlock, create_sdk_mcp_server, tool)
from dotenv import load_dotenv
from tavily_agent_toolkit import (ModelConfig, ModelObject,
                                  crawl_and_summarize, extract_and_summarize,
                                  format_web_results, search_dedup)

load_dotenv(Path(__file__).parent / ".env")

TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "")
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY not set (env or .env).")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY not set (env or .env).")

SUMMARIZER_CONFIG = ModelConfig(
    model=ModelObject(
        model="claude-haiku-4-5",
        api_key=ANTHROPIC_API_KEY,
    ),
)

SYSTEM_PROMPT = """\
You are a business intelligence analyst researching companies.

You have three tools available:
- `crawl_company_website` – Crawl a company's website to discover and summarize pages
- `extract_from_urls` – Extract detailed content from specific URLs
- `tavily_search` – Search the web for news, funding, reviews, and other external info

Use these tools as you see fit to gather comprehensive information. \
Combine website insights with external sources for a complete picture.

When you're done researching, write up your findings in a clear report. \
Include citations [1], [2], etc. linking to your sources, and list all \
sources at the end.
"""

TOOL_LABELS: dict[str, str] = {
    "mcp__company-intel-tools__crawl_company_website": "Crawling website",
    "mcp__company-intel-tools__extract_from_urls": "Extracting URLs",
    "mcp__company-intel-tools__tavily_search": "Searching the web",
}

# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@tool(
    "crawl_company_website",
    "Crawl a company's website to extract and summarize information.",
    {
        "url": str,
        "instructions": Optional[str],
        "max_depth": Optional[int],
        "max_breadth": Optional[int],
        "limit": Optional[int],
    },
)
async def crawl_company_website(args: dict[str, Any]) -> dict[str, Any]:
    result = await crawl_and_summarize(
        url=args.get("url", ""),
        model_config=SUMMARIZER_CONFIG,
        instructions=args.get("instructions"),
        max_depth=args.get("max_depth", 2),
        max_breadth=args.get("max_breadth", 10),
        limit=args.get("limit", 20),
        api_key=TAVILY_API_KEY,
    )
    return {"content": [{"type": "text", "text": result.get("summary", "")}]}


@tool(
    "extract_from_urls",
    "Extract and summarize content from specific URLs.",
    {
        "urls": list,
        "query": Optional[str],
    },
)
async def extract_from_urls(args: dict[str, Any]) -> dict[str, Any]:
    result = await extract_and_summarize(
        urls=args.get("urls") or [],
        model_config=SUMMARIZER_CONFIG,
        query=args.get("query"),
        extract_depth="advanced",
        api_key=TAVILY_API_KEY,
    )
    return {
        "content": [
            {"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}
        ]
    }


@tool(
    "tavily_search",
    "Search the web with multiple queries in parallel and get deduplicated results.",
    {
        "queries": list,
        "max_results": Optional[int],
        "topic": Optional[str],
        "time_range": Optional[str],
    },
)
async def tavily_search(args: dict[str, Any]) -> dict[str, Any]:
    # The model sometimes passes queries as a comma-separated string instead
    # of a list, and numeric params as strings — coerce to expected types.
    queries = args.get("queries") or []
    if isinstance(queries, str):
        queries = [q.strip() for q in queries.split(",") if q.strip()]
    max_results = int(args.get("max_results", 5))

    result = await search_dedup(
        api_key=TAVILY_API_KEY,
        queries=queries,
        max_results=max_results,
        topic=args.get("topic", "general"),
        time_range=args.get("time_range"),
        search_depth="advanced",
        include_answer=True,
    )
    return {
        "content": [{"type": "text", "text": format_web_results(result.get("results", []))}]
    }


# ---------------------------------------------------------------------------
# MCP Server & Agent Options
# ---------------------------------------------------------------------------

mcp_server = create_sdk_mcp_server(
    name="company-intel-tools",
    version="1.0.0",
    tools=[crawl_company_website, extract_from_urls, tavily_search],
)

agent_options = ClaudeAgentOptions(
    model="haiku",
    system_prompt=SYSTEM_PROMPT,
    mcp_servers={"company-intel-tools": mcp_server},
    allowed_tools=[
        "mcp__company-intel-tools__crawl_company_website",
        "mcp__company-intel-tools__extract_from_urls",
        "mcp__company-intel-tools__tavily_search",
    ],
)

# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------


def _build_prompt(company_name: str, website_url: str, focus: str) -> str:
    return (
        f"Research this company:\n\n"
        f"**Company Name:** {company_name}\n"
        f"**Website:** {website_url}\n"
        f"**Focus:** {focus}\n"
    )


def _tool_detail(inp: dict[str, Any]) -> str:
    """Extract a short detail string from tool input for progress display."""
    if "url" in inp:
        return f" -> {inp['url']}"
    if "urls" in inp:
        return f" -> {len(inp['urls'])} URL(s)"
    if "queries" in inp:
        q = inp["queries"]
        return f" -> {len(q)} query/queries" if isinstance(q, list) else f" -> {q[:60]}"
    return ""


async def run_research(company_name: str, website_url: str, focus: str) -> str:
    """Run the research agent and return the final report."""
    prompt = _build_prompt(company_name, website_url, focus)
    report = ""
    step = 0

    async with ClaudeSDKClient(options=agent_options) as client:
        await client.query(prompt)

        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, ToolUseBlock):
                        step += 1
                        label = TOOL_LABELS.get(block.name, block.name)
                        detail = _tool_detail(block.input or {})
                        print(f"  [{step}] {label}{detail}")
                    elif isinstance(block, TextBlock) and block.text:
                        report = block.text

            elif isinstance(msg, ResultMessage):
                if msg.result:
                    report = msg.result
                secs = msg.duration_ms / 1000
                cost_str = f" | ${msg.total_cost_usd:.4f}" if msg.total_cost_usd else ""
                print(f"\n  Completed in {secs:.1f}s | {msg.num_turns} turns{cost_str}")

    return report

async def main():
    print()
    print("=" * 60)
    print("  Company Intelligence Research Agent")
    print("  Powered by Claude + Tavily MCP Tools")
    print("=" * 60)

    company_name = input("\n  Company name: ").strip()
    website_url = input("  Website URL:  ").strip()
    focus = input("  Research focus (Enter for general): ").strip() or (
        "Provide a comprehensive overview of the company"
    )

    print()
    print("-" * 60)
    print(f"  Researching {company_name} ({website_url})")
    print(f"  Focus: {focus}")
    print("-" * 60)
    print()

    report = await run_research(company_name, website_url, focus)

    print()
    print("=" * 60)
    print("  RESEARCH REPORT")
    print("=" * 60)
    print()
    print(report)
    print()
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
