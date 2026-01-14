"""
Social Media Deep Research Agent

Uses Claude with Anthropic SDK to conduct comprehensive research across social media
platforms. Searches TikTok, Reddit, Instagram, X, Facebook, and LinkedIn to gather
real-world opinions, discussions, and insights on any topic.

Prerequisites:
    pip install anthropic tavily-python python-dotenv

Usage:
    # Set ANTHROPIC_API_KEY and TAVILY_API_KEY in .env file
    python social_media_research.py
"""

import asyncio
import json
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from tavily_agent_toolkit import social_media_search

# Load .env from the same folder as this script
load_dotenv(Path(__file__).parent / ".env")

# Get API keys from environment
TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "")
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

# Define tools for Claude
TOOLS = [
    {
        "name": "search_social_media",
        "description": """Search social media platforms for discussions, opinions, and content.

Use this tool to search across TikTok, Reddit, Instagram, X, Facebook, and LinkedIn.
You can search a specific platform or all platforms at once with "combined".""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query - be specific with relevant keywords"
                },
                "platform": {
                    "type": "string",
                    "enum": ["tiktok", "facebook", "instagram", "reddit", "linkedin", "x", "combined"],
                    "description": "Which platform to search. Options: tiktok, instagram, reddit, x, facebook, linkedin, or combined (all platforms). Default: combined"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10)"
                },
                "include_raw_content": {
                    "type": "boolean",
                    "description": "Whether to extract full content from posts (default: true)"
                },
                "time_range": {
                    "type": "string",
                    "enum": ["day", "week", "month", "year"],
                    "description": "Time range for results - day, week, month, or year (default: month)"
                }
            },
            "required": ["query"]
        }
    }
]

SYSTEM_PROMPT = """You research topics by searching social media platforms (TikTok, Reddit, Instagram, X, Facebook, LinkedIn).

Search multiple times with different queries or platforms to get a complete picture. Reddit is great for honest opinions, TikTok for trends, X for real-time reactions, LinkedIn for professional takes.

Synthesize what you find into a clear answer. Include inline citations [1], [2] and list sources with URLs at the end."""


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return the result."""
    if tool_name == "search_social_media":
        query = tool_input.get("query", "")
        platform = tool_input.get("platform", "combined")
        max_results = tool_input.get("max_results", 10)
        include_raw_content = tool_input.get("include_raw_content", True)
        time_range = tool_input.get("time_range", "month")
        
        result = social_media_search(
            query=query,
            api_key=TAVILY_API_KEY,
            platform=platform,
            include_raw_content=include_raw_content,
            max_results=max_results,
            search_depth="advanced",
            include_answer=True,
            time_range=time_range,
        )
        return json.dumps(result)
    
    return f"Unknown tool: {tool_name}"


async def run_research(query: str) -> str:
    """
    Run social media research for a given query.
    
    Args:
        query: The research question or topic.
        
    Returns:
        The research report as a string.
    """
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    messages = [{"role": "user", "content": query}]
    
    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )
        
        if response.stop_reason == "tool_use":
            # Process all tool calls
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})
            
            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    print(f"  [Searching social media on {block.input.get('platform', 'all social media platforms')}...] ", flush=True)
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            messages.append({"role": "user", "content": tool_results})
        
        else:
            # Final response - extract and return text
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""


async def main():
    """Social media research CLI."""
    print("=" * 60)
    print("Social Media Research Agent (Claude)")
    print("=" * 60)
    print("\nThis agent searches across TikTok, Reddit, Instagram, X,")
    print("Facebook, and LinkedIn to research any topic.\n")
    
    # Check for API keys
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not set.")
        print("  Add it to .env or run: export ANTHROPIC_API_KEY=your-api-key")
        return
    if not TAVILY_API_KEY:
        print("Error: TAVILY_API_KEY not set.")
        print("  Add it to .env or run: export TAVILY_API_KEY=your-api-key")
        return
    
    query = input("What would you like to research?\n> ").strip()
    
    if not query:
        print("No query provided.")
        return
    
    print("\n" + "-" * 60)
    print("Researching...\n")
    
    report = await run_research(query)
    
    print("\n" + "=" * 60)
    print("REPORT")
    print("=" * 60 + "\n")
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
