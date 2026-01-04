#!/usr/bin/env python3
"""
Documentation Crawler Script

Uses Tavily Crawl API to gather documentation from a given site.
Useful for investigating libraries discovered through X trends.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

# Load .env file from repo root
try:
    from dotenv import load_dotenv
    repo_root = Path(__file__).resolve().parents[4]
    load_dotenv(repo_root / ".env")
except ImportError:
    pass  # dotenv not installed, rely on environment variables

try:
    from tavily import TavilyClient
except ImportError:
    print("Error: tavily-python not installed. Run: pip install tavily-python")
    exit(1)


def crawl_site(
    url: str,
    max_depth: int = 2,
    max_breadth: int = 50,
    limit: int = 100,
    select_paths: list[str] = None,
    exclude_paths: list[str] = None,
    instructions: str = None,
) -> dict:
    """Crawl a site and extract documentation content."""

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set")

    client = TavilyClient(api_key=api_key)

    print(f"Crawling: {url}")
    print(f"Settings: depth={max_depth}, breadth={max_breadth}, limit={limit}")
    if select_paths:
        print(f"Include paths: {select_paths}")
    if exclude_paths:
        print(f"Exclude paths: {exclude_paths}")
    if instructions:
        print(f"Instructions: {instructions}")
    print()

    # Build crawl parameters
    crawl_params = {
        "url": url,
        "max_depth": max_depth,
        "max_breadth": max_breadth,
        "limit": limit,
        "format": "markdown",
    }

    if select_paths:
        crawl_params["select_paths"] = select_paths
    if exclude_paths:
        crawl_params["exclude_paths"] = exclude_paths
    if instructions:
        crawl_params["instructions"] = instructions

    response = client.crawl(**crawl_params)

    results = response.get("results", [])
    print(f"Crawled {len(results)} pages\n")

    return {
        "url": url,
        "crawled_at": datetime.now().isoformat(),
        "pages_count": len(results),
        "results": results,
    }


def save_results(data: dict, output_dir: Path, name: str) -> Path:
    """Save crawl results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.json"
    output_path = output_dir / filename

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Crawl documentation sites using Tavily API"
    )
    parser.add_argument(
        "url",
        help="URL to crawl (e.g., https://docs.anthropic.com)"
    )
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=2,
        help="Max crawl depth (default: 2)"
    )
    parser.add_argument(
        "--breadth", "-b",
        type=int,
        default=50,
        help="Max links per level (default: 50)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=100,
        help="Max total pages (default: 100)"
    )
    parser.add_argument(
        "--include", "-i",
        nargs="+",
        help="Path patterns to include (regex)"
    )
    parser.add_argument(
        "--exclude", "-e",
        nargs="+",
        help="Path patterns to exclude (regex)"
    )
    parser.add_argument(
        "--instructions",
        help="Natural language guidance for crawler"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory (default: research/crawls/)"
    )
    parser.add_argument(
        "--name", "-n",
        help="Output filename prefix (default: derived from URL)"
    )
    parser.add_argument(
        "--print", "-p",
        action="store_true",
        dest="print_results",
        help="Print results to stdout instead of saving"
    )

    args = parser.parse_args()

    result = crawl_site(
        url=args.url,
        max_depth=args.depth,
        max_breadth=args.breadth,
        limit=args.limit,
        select_paths=args.include,
        exclude_paths=args.exclude,
        instructions=args.instructions,
    )

    if args.print_results:
        print("=" * 60)
        print("CRAWL RESULTS")
        print("=" * 60)
        for page in result["results"]:
            print(f"\n### {page['url']}")
            print("-" * 40)
            content = page.get("raw_content", "")
            # Print first 500 chars of each page
            print(content[:500] + "..." if len(content) > 500 else content)
    else:
        # Derive name from URL if not provided
        if args.name:
            name = args.name
        else:
            from urllib.parse import urlparse
            parsed = urlparse(args.url)
            name = parsed.netloc.replace(".", "_")

        output_dir = args.output or (repo_root / "research" / "crawls")
        output_path = save_results(result, output_dir, name)

        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Pages crawled: {result['pages_count']}")
        print(f"Saved to: {output_path}")

        # Print URL list
        print(f"\nPages:")
        for page in result["results"]:
            content_len = len(page.get("raw_content", ""))
            print(f"  - {page['url']} ({content_len:,} chars)")


if __name__ == "__main__":
    main()
