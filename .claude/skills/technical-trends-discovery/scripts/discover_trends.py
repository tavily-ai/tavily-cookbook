#!/usr/bin/env python3
"""
Technical Trends Discovery - Two-Step Workflow

Step 1: Search X for what thought leaders are discussing (xAI API)
Step 2: Deep research on identified trends (Tavily Research API)

This workflow leverages X as the source for real-time opinions from top voices,
then uses Tavily to do comprehensive research on the trends discovered.
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Output directory at repo root
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parents[3]
TRENDS_REPORTS_DIR = REPO_ROOT / "trends-reports"


def get_output_dir() -> Path:
    """Generate timestamped output directory."""
    TRENDS_REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = TRENDS_REPORTS_DIR / f"trends_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# =============================================================================
# Step 1: X Search
# =============================================================================

DEFAULT_HANDLES = [
    "hwchase17",      # Harrison Chase (LangChain)
    "rlancemartin",   # Lance Martin (LangChain)
    "simonw",         # Simon Willison
    "karpathy",       # Andrej Karpathy
    "cherny",         # Boris Cherny
    "swyx",           # Swyx
    "alexalbert__",   # Alex Albert (Anthropic)
]

X_DISCOVERY_PROMPT = """Analyze the recent posts from these AI thought leaders and identify:

1. **Top 3 Emerging Trends** - What topics are multiple people discussing? Be specific about the trend name.
2. **Key Insights** - Novel ideas, patterns, or frameworks mentioned
3. **Who Said What** - Cite which thought leader mentioned each topic

Format the trends as a clear numbered list that can be used for follow-up research."""


def search_x_trends(handles: list[str], days_back: int, min_favorites: int) -> dict:
    """Step 1: Search X for trends from thought leaders."""
    try:
        from xai_sdk import Client
        from xai_sdk.chat import user
        from xai_sdk.search import SearchParameters, x_source
    except ImportError:
        raise ImportError("xai-sdk not installed. Run: pip install xai-sdk")

    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY environment variable not set")

    to_date = datetime.now()
    from_date = to_date - timedelta(days=days_back)

    print("=" * 60)
    print("STEP 1: X SEARCH")
    print("=" * 60)
    print(f"Searching posts from {len(handles)} thought leaders...")
    print(f"Date range: {from_date.date()} to {to_date.date()}")
    print(f"Handles: {', '.join(handles)}\n")

    client = Client(api_key=api_key)

    chat = client.chat.create(
        model="grok-3",
        search_parameters=SearchParameters(
            mode="on",
            from_date=from_date,
            to_date=to_date,
            return_citations=True,
            sources=[
                x_source(
                    included_x_handles=handles,
                    post_favorite_count=min_favorites,
                )
            ],
        ),
    )

    chat.append(user(X_DISCOVERY_PROMPT))
    response = chat.sample()

    print("X search complete.\n")
    return {
        "content": response.content,
        "citations": getattr(response, "citations", []),
    }


# =============================================================================
# Step 2: Tavily Deep Research
# =============================================================================

def build_research_prompt(x_trends: str) -> str:
    """Build a Tavily research prompt based on trends discovered on X."""
    return f"""Based on recent discussions from AI thought leaders on X, the following trends have been identified:

{x_trends}

For each of these trends, provide comprehensive research including:

1. **Technical Deep Dive** - What exactly is this trend? How does it work?
2. **Major Developments** - Recent announcements, releases, or breakthroughs
3. **Practical Implementations** - Real-world applications and use cases
4. **Key Libraries & Tools** - Specific tools, frameworks, or projects to explore
5. **Getting Started** - How can developers start experimenting with this?

Focus on actionable insights and specific resources."""


def research_trends(prompt: str, poll_interval: int = 5) -> dict:
    """Step 2: Deep research on trends using Tavily."""
    try:
        from tavily import TavilyClient
    except ImportError:
        raise ImportError("tavily-python not installed. Run: pip install tavily-python")

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set")

    print("=" * 60)
    print("STEP 2: TAVILY DEEP RESEARCH")
    print("=" * 60)
    print("Initiating deep research on identified trends...\n")

    client = TavilyClient(api_key=api_key)
    result = client.research(input=prompt, model="pro")
    request_id = result["request_id"]
    print(f"Research initiated (request_id: {request_id})")

    elapsed = 0
    while True:
        response = client.get_research(request_id)
        status = response.get("status", "unknown")

        if status == "completed":
            print(f"Research completed in {elapsed}s\n")
            return {
                "status": "completed",
                "content": response.get("content"),
                "sources": response.get("sources", []),
            }
        elif status == "failed":
            return {"status": "failed", "error": response.get("error")}
        else:
            print(f"Status: {status}... waiting {poll_interval}s")
            time.sleep(poll_interval)
            elapsed += poll_interval


# =============================================================================
# Main Orchestration
# =============================================================================

def discover_trends(
    handles: list[str] = None,
    days_back: int = 20,
    min_favorites: int = 100,
    skip_research: bool = False,
) -> dict:
    """
    Two-step trend discovery:
    1. Search X for what thought leaders are discussing
    2. Deep research on the identified trends via Tavily
    """
    handles = handles or DEFAULT_HANDLES

    # Step 1: X Search
    x_result = search_x_trends(handles, days_back, min_favorites)

    if skip_research:
        return {"x_trends": x_result, "research": None}

    # Step 2: Build research prompt and run Tavily
    research_prompt = build_research_prompt(x_result["content"])
    research_result = research_trends(research_prompt)

    return {
        "x_trends": x_result,
        "research": research_result,
    }


def save_results(output_dir: Path, results: dict):
    """Save results to output directory."""
    # Save X trends discovery
    x_path = output_dir / "x_discovery.md"
    with open(x_path, "w") as f:
        f.write("# X Trends Discovery\n\n")
        f.write(results["x_trends"]["content"])
        if results["x_trends"]["citations"]:
            f.write("\n\n## Sources\n\n")
            for url in results["x_trends"]["citations"]:
                f.write(f"- {url}\n")
    print(f"X discovery saved to: {x_path}")

    # Save deep research if available
    if results["research"] and results["research"].get("status") == "completed":
        report_path = output_dir / "report.md"
        with open(report_path, "w") as f:
            f.write(results["research"]["content"])
        print(f"Research report saved to: {report_path}")

        sources_path = output_dir / "sources.json"
        simplified_sources = [
            {"url": s.get("url", ""), "title": s.get("title", "Untitled")}
            for s in results["research"].get("sources", [])
        ]
        with open(sources_path, "w") as f:
            json.dump(simplified_sources, f, indent=2)
        print(f"Sources saved to: {sources_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Discover AI trends: X search → Tavily deep research"
    )
    parser.add_argument(
        "--handles", "-H",
        nargs="+",
        help="X handles to search (default: AI thought leaders)"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=20,
        help="Days back to search (default: 20)"
    )
    parser.add_argument(
        "--min-favorites", "-f",
        type=int,
        default=100,
        help="Minimum favorites filter (default: 100)"
    )
    parser.add_argument(
        "--x-only",
        action="store_true",
        help="Only run X search, skip Tavily research"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save to files, only print to stdout"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory (default: trends-reports/trends_<timestamp>/)"
    )

    args = parser.parse_args()

    # Run discovery
    results = discover_trends(
        handles=args.handles,
        days_back=args.days,
        min_favorites=args.min_favorites,
        skip_research=args.x_only,
    )

    # Save results
    if not args.no_save:
        output_dir = Path(args.output) if args.output else get_output_dir()
        if args.output:
            output_dir.mkdir(parents=True, exist_ok=True)
        save_results(output_dir, results)

    # Print summary
    print("\n" + "=" * 60)
    print("X TRENDS DISCOVERED")
    print("=" * 60)
    print(results["x_trends"]["content"])

    if results["research"] and results["research"].get("status") == "completed":
        print("\n" + "=" * 60)
        print("DEEP RESEARCH REPORT")
        print("=" * 60)
        print(results["research"]["content"])


if __name__ == "__main__":
    main()
