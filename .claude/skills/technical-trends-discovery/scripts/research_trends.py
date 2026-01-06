#!/usr/bin/env python3
"""
Technical Trends Research Script

Uses Tavily Research API to identify and research emerging AI/tech trends
from industry thought leaders.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if env vars set directly

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

# Output directory at repo root
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parents[3]  # Go up from scripts/ -> skill/ -> skills/ -> .claude/ -> root
TRENDS_REPORTS_DIR = REPO_ROOT / "trends-reports"


def get_default_output_dir() -> Path:
    """Generate timestamped output directory in the trends-reports directory."""
    TRENDS_REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = TRENDS_REPORTS_DIR / f"trends_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    return output_dir


DEFAULT_PROMPT = """Identify the most important emerging trends in AI engineering by analyzing what industry thought leaders are discussing. Look for topics from voices like Harrison Chase, Lance Martin, Simon Willison, Andrej Karpathy, Chip Huyen, Jerry Liu, Boris Cherny, Swyx.

Discover the top 2-3 AI trends from the last 30 days, especially common patterns and frameworks across the different voices.

For the top trending topics, provide research on:
1. Major developments and announcements
2. Key implementations and patterns, applications and use cases
3. Unique, novel, or interesting ideas or concepts
4. Relevant libraries and tools"""


def research_trends(client: TavilyClient, poll_interval: int, max_wait: int) -> dict:
    """Perform trend research using Tavily Research API."""

    print(f"Prompt: {DEFAULT_PROMPT[:100]}...\n")

    # Initiate research
    result = client.research(input=DEFAULT_PROMPT, model="pro")
    request_id = result["request_id"]
    print(f"Research initiated (request_id: {request_id})")

    # Poll until completed
    elapsed = 0
    while elapsed < max_wait:
        response = client.get_research(request_id)
        status = response.get("status", "unknown")

        if status == "completed":
            print(f"Research completed in {elapsed}s\n")
            return {
                "status": "completed",
                "content": response.get("content"),
                "sources": response.get("sources", []),
                "response_time": response.get("response_time"),
                "request_id": request_id,
                "generated_at": datetime.now().isoformat(),
            }

        elif status == "failed":
            error = response.get("error", "Unknown error")
            print(f"Research failed: {error}")
            return {"status": "failed", "error": error}

        else:
            print(f"Status: {status}... waiting {poll_interval}s")
            time.sleep(poll_interval)
            elapsed += poll_interval

    print(f"Research timed out after {max_wait}s")
    return {"status": "timeout", "error": f"Timed out after {max_wait}s"}


def main():
    parser = argparse.ArgumentParser(
        description="Research AI/tech trends using Tavily Research API"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory path (default: trends-reports/trends_<timestamp>/)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save to file, only print to stdout"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Seconds between polling (default: 5)"
    )
    parser.add_argument(
        "--max-wait",
        type=int,
        default=300,
        help="Max seconds to wait (default: 300)"
    )

    args = parser.parse_args()

    # Check for API key
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        print("Error: TAVILY_API_KEY environment variable not set")
        return 1

    if TavilyClient is None:
        print("Error: tavily-python not installed. Run: pip install tavily-python")
        return 1

    # Initialize client and run research
    client = TavilyClient(api_key=api_key)

    result = research_trends(
        client,
        args.poll_interval,
        args.max_wait
    )

    # Determine output directory
    if args.no_save:
        output_dir = None
    else:
        output_dir = Path(args.output) if args.output else get_default_output_dir()
        if args.output:
            output_dir.mkdir(parents=True, exist_ok=True)

    # Save to files (report.md + sources.json)
    if output_dir and result.get("status") == "completed":
        # Save markdown report
        report_path = output_dir / "report.md"
        with open(report_path, "w") as f:
            f.write(result.get("content", ""))
        print(f"Report saved to: {report_path}")

        # Save sources as simplified JSON (url + title only)
        sources_path = output_dir / "sources.json"
        simplified_sources = [
            {"url": s.get("url", ""), "title": s.get("title", "Untitled")}
            for s in result.get("sources", [])
        ]
        with open(sources_path, "w") as f:
            json.dump(simplified_sources, f, indent=2)
        print(f"Sources saved to: {sources_path}")

    # Print report to stdout
    if result.get("status") == "completed":
        print("=" * 60)
        print("RESEARCH REPORT")
        print("=" * 60)
        print(result.get("content", ""))
        print("\n" + "=" * 60)
        print("SOURCES")
        print("=" * 60)
        for source in result.get("sources", []):
            print(f"- {source.get('title', 'Untitled')}: {source.get('url', '')}")
    else:
        print(json.dumps(result, indent=2))

    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    exit(main())
