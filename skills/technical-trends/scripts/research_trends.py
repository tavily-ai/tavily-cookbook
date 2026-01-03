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

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None


DEFAULT_PROMPT = """Identify the most important emerging trends in AI engineering by analyzing what industry thought leaders are discussing. Look for topics from voices like Harrison Chase, Lance Martin, Simon Willison, Andrej Karpathy, Chip Huyen, Jerry Liu, Boris Cherny, and Swyx.


For the top 3-5 trending topics, provide deep research on:
1. Current developments and announcements
2. Key implementations and patterns
3. Practical applications and use cases
4. Relevant libraries and tools"""


def research_trends(client: TavilyClient, prompt: str, model: str, poll_interval: int, max_wait: int) -> dict:
    """Perform trend research using Tavily Research API."""

    print(f"Starting research with model: {model}")
    print(f"Prompt: {prompt[:100]}...\n")

    # Initiate research
    result = client.research(input=prompt, model=model)
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
        "--topic",
        help="Specific topic to focus on (modifies the default prompt)"
    )
    parser.add_argument(
        "--prompt",
        help="Custom research prompt (overrides default)"
    )
    parser.add_argument(
        "--model",
        choices=["mini", "pro", "auto"],
        default="pro",
        help="Research model: mini (fast), pro (comprehensive), auto (default: pro)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: stdout)"
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

    # Build prompt
    if args.prompt:
        prompt = args.prompt
    elif args.topic:
        prompt = f"""Research the latest trends in {args.topic}.

Analyze what industry thought leaders are discussing about this topic. Look for insights from prominent AI engineers and researchers.

Provide deep research on:
1. Current developments and announcements
2. Key implementations and patterns
3. Practical applications and use cases
4. Relevant libraries and tools
5. What thought leaders are recommending"""
    else:
        prompt = DEFAULT_PROMPT

    # Initialize client and run research
    client = TavilyClient(api_key=api_key)

    result = research_trends(
        client,
        prompt,
        args.model,
        args.poll_interval,
        args.max_wait
    )

    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {args.output}")
    else:
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
