#!/usr/bin/env python3
"""
URL Crawler Script

Crawls a website using Tavily Crawl API and saves each page
as a separate markdown file in a flat directory structure.
"""

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

try:
    from tavily import TavilyClient
except ImportError:
    print("Error: tavily-python not installed. Run: pip install tavily-python")
    exit(1)

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Output directory at repo root
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parents[3]
CRAWLED_CONTEXT_DIR = REPO_ROOT / "crawled_context"


def url_to_filename(url: str) -> str:
    """
    Convert a URL to a safe filename.

    Example:
        https://docs.example.com/api/users -> docs_example_com_api_users.md
    """
    # Remove protocol
    url_clean = re.sub(r'^https?://', '', url)

    # Remove trailing slash
    url_clean = url_clean.rstrip('/')

    # Replace special characters with underscores
    url_clean = re.sub(r'[^\w\-]', '_', url_clean)

    # Remove duplicate underscores
    url_clean = re.sub(r'_+', '_', url_clean)

    # Limit length to avoid filesystem issues
    if len(url_clean) > 200:
        url_clean = url_clean[:200]

    return f"{url_clean}.md"


def crawl_and_save(
    url: str,
    output_dir: Path,
    instruction: str = None,
    max_depth: int = 2,
    max_breadth: int = 50,
    limit: int = 50,
) -> dict:
    """
    Crawl a site and save each page as a markdown file.

    Returns:
        dict with 'pages_saved', 'output_dir', and 'crawled_at'
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set")

    client = TavilyClient(api_key=api_key)

    print(f"🔍 Crawling: {url}")
    if instruction:
        print(f"📋 Instruction: {instruction}")
    print(f"⚙️  Settings: depth={max_depth}, breadth={max_breadth}, limit={limit}")
    print()

    # Build crawl parameters
    crawl_params = {
        "url": url,
        "max_depth": max_depth,
        "max_breadth": max_breadth,
        "limit": limit,
        "format": "markdown",
        "extract_depth": "advanced",
    }

    if instruction:
        crawl_params["instructions"] = instruction

    # Execute crawl
    print("⏳ Crawling (this may take a minute)...")
    response = client.crawl(**crawl_params)

    results = response.get("results", [])
    print(f"✅ Found {len(results)} pages\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each page as a markdown file
    print(f"💾 Saving to: {output_dir}\n")
    saved_count = 0

    for idx, page in enumerate(results, 1):
        page_url = page.get("url", "")
        content = page.get("raw_content", "")

        if not content:
            print(f"⚠️  Skipping {page_url} (no content)")
            continue

        # Generate filename from URL
        filename = url_to_filename(page_url)
        filepath = output_dir / filename

        # Add metadata header to markdown file
        markdown_output = f"""---
source_url: {page_url}
crawled_at: {datetime.now().isoformat()}
---

{content}
"""

        # Save file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_output)

        print(f"  [{idx}/{len(results)}] {filename}")
        saved_count += 1

    print(f"\n✅ Saved {saved_count} markdown files to {output_dir}")

    return {
        "pages_saved": saved_count,
        "output_dir": str(output_dir),
        "crawled_at": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Crawl websites and save as markdown files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic crawl
  python crawl_url.py https://docs.stripe.com/api

  # With instruction
  python crawl_url.py https://react.dev --instruction "Focus on API reference pages"

  # Custom output directory
  python crawl_url.py https://docs.anthropic.com -o ./anthropic-docs

  # Control crawl depth and breadth
  python crawl_url.py https://nextjs.org/docs --depth 3 --limit 100
        """
    )

    parser.add_argument(
        "url",
        help="URL to crawl (e.g., https://docs.example.com)"
    )

    parser.add_argument(
        "--instruction", "-i",
        help="Natural language guidance for the crawler (e.g., 'Focus on API endpoints')"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory (default: <repo_root>/crawled_context/<domain>)"
    )

    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=2,
        help="Max crawl depth (default: 2, range: 1-5)"
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
        default=50,
        help="Max total pages (default: 50)"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        # Default: <repo_root>/crawled_context/<domain>
        parsed = urlparse(args.url)
        domain = parsed.netloc.replace(".", "_")
        output_dir = CRAWLED_CONTEXT_DIR / domain

    # Execute crawl
    try:
        result = crawl_and_save(
            url=args.url,
            output_dir=output_dir,
            instruction=args.instruction,
            max_depth=args.depth,
            max_breadth=args.breadth,
            limit=args.limit,
        )

        print(f"\n{'='*60}")
        print("✨ Crawl complete!")
        print(f"{'='*60}")
        print(f"📁 Output: {result['output_dir']}")
        print(f"📄 Files: {result['pages_saved']}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
