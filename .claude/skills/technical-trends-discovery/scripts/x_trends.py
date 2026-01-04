#!/usr/bin/env python3
"""
X Trends Search Script

Uses xAI Live Search API to find trends from AI thought leaders on X.
"""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

# Load .env file from repo root
try:
    from dotenv import load_dotenv
    repo_root = Path(__file__).resolve().parents[4]
    load_dotenv(repo_root / ".env")
except ImportError:
    pass  # dotenv not installed, rely on environment variables

try:
    from xai_sdk import Client
    from xai_sdk.chat import user
    from xai_sdk.search import SearchParameters, x_source
except ImportError:
    print("Error: xai-sdk not installed. Run: pip install xai-sdk")
    exit(1)

# Default thought leaders to track
DEFAULT_HANDLES = [
    "hwchase17",      # Harrison Chase (LangChain)
    "rlancemartin",   # Lance Martin (LangChain)
    "simonw",         # Simon Willison
    "karpathy",       # Andrej Karpathy
    "bcherny",      # Boris Cherny
    "swyx",
    "alexalbert__"
]

DEFAULT_PROMPT = """Analyze the recent posts from these AI thought leaders. Focus on ACTIONABLE information for AI developers - briefly mention high-level trends, but focus on specific things they can use today.

Extract and organize:

## Libraries & Tools
List specific libraries, frameworks, SDKs, or tools mentioned with:
- Name and link (links are very valuable all relevant links should be included!!)
- Who mentioned it

## Code Patterns & Techniques
Specific implementation patterns, prompts, or techniques shared:
- The pattern/technique
- Code snippets or examples if shared
- Who shared it

## Docs & Resources
Links to documentation, tutorials, papers, or repos mentioned:
- Resource name and URL
- Why it's useful
- Who recommended it

## Hot Takes & Insights
Brief, opinionated takes worth noting (1-2 sentences each):
- The insight
- Who said it

Prioritize specificity over comprehensiveness. A single useful library link is worth more than a paragraph about "the rise of agents."
"""


def search_x_trends(
    handles: list[str] = None,
    days_back: int = 20,
    prompt: str = None,
    min_favorites: int = 100,
) -> dict:
    """Search X for trends from specified handles."""

    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY environment variable not set")

    handles = handles or DEFAULT_HANDLES
    prompt = prompt or DEFAULT_PROMPT

    # Date range
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days_back)

    print(f"Searching X posts from {len(handles)} handles...")
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

    chat.append(user(prompt))
    response = chat.sample()

    return {
        "content": response.content,
        "citations": getattr(response, "citations", []),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Search X for AI trends from thought leaders"
    )
    parser.add_argument(
        "--handles", "-H",
        nargs="+",
        help="X handles to search (default: AI thought leaders)"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=30,
        help="Days back to search (default: 30)"
    )
    parser.add_argument(
        "--min-favorites", "-f",
        type=int,
        default=100,
        help="Minimum favorites filter (default: 100)"
    )
    parser.add_argument(
        "--prompt", "-p",
        help="Custom prompt (default: trend analysis)"
    )

    args = parser.parse_args()

    result = search_x_trends(
        handles=args.handles,
        days_back=args.days,
        prompt=args.prompt,
        min_favorites=args.min_favorites,
    )

    print("=" * 60)
    print("TRENDS ANALYSIS")
    print("=" * 60)
    print(result["content"])

    if result["citations"]:
        print("\n" + "=" * 60)
        print("SOURCES")
        print("=" * 60)
        for url in result["citations"]:
            print(f"- {url}")


if __name__ == "__main__":
    main()
