"""
Social Media Deep Research Agent

Uses LangGraph's ReAct agent to conduct comprehensive research across social media
platforms. Searches TikTok, Reddit, Instagram, X, Facebook, and LinkedIn to gather
real-world opinions, discussions, and insights on any topic.
"""

import asyncio
import os
import sys
from typing import Literal, Optional

from dotenv import load_dotenv

# Add solution-patterns directory to path for imports
parent_dir = os.path.join(os.path.dirname(__file__), "..")
solution_patterns_dir = os.path.join(parent_dir, "solution-patterns")
sys.path.insert(0, solution_patterns_dir)

# Load environment variables from .env file
load_dotenv(os.path.join(parent_dir, ".env"))

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from tools.social_media import social_media_search
from utils import stream_agent_response

# Get API keys from environment
TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY environment variable is required")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")


@tool
def search_social_media(
    query: str,
    platform: Literal["tiktok", "facebook", "instagram", "reddit", "linkedin", "x", "combined"] = "combined",
    max_results: int = 10,
    include_raw_content: bool = True,
    time_range: Optional[Literal["day", "week", "month", "year"]] = "month",
) -> dict:
    """
    Search social media platforms for discussions, opinions, and content.
    
    Args:
        query: The search query - be specific with relevant keywords
        platform: Which platform to search. Options:
            - "tiktok": TikTok videos and comments
            - "instagram": Instagram posts and reels
            - "reddit": Reddit discussions and threads
            - "x": X/Twitter posts
            - "facebook": Facebook posts
            - "linkedin": LinkedIn posts
            - "combined": Search all platforms at once (default)
        max_results: Maximum number of results to return (default: 5)
        include_raw_content: Whether to extract full content from posts (default: False)
        time_range: Time range for results - "day", "week", "month", or "year" (default: "month")
    
    Returns:
        Dictionary containing search results with titles, URLs, content snippets,
        and optionally full raw content from each social media post.
    """
    return social_media_search(
        query=query,
        api_key=TAVILY_API_KEY,
        platform=platform,
        include_raw_content=include_raw_content,
        max_results=max_results,
        search_depth="advanced",
        include_answer=True,
        time_range=time_range,
    )


SYSTEM_PROMPT = """You research topics by searching social media platforms (TikTok, Reddit, Instagram, X, Facebook, LinkedIn).

Search multiple times with different queries or platforms to get a complete picture. Reddit is great for honest opinions, TikTok for trends, X for real-time reactions, LinkedIn for professional takes.

Synthesize what you find into a clear answer. Include inline citations [1], [2] and list sources with URLs at the end."""


model = ChatOpenAI(model="gpt-5.1", api_key=OPENAI_API_KEY)
agent = create_agent(
    model=model,
    tools=[search_social_media],
    system_prompt=SYSTEM_PROMPT,
)

async def main():
    """Social media research CLI."""
    print("=" * 60)
    print("Social Media Research Agent")
    print("=" * 60)
    print("\nThis agent searches across TikTok, Reddit, Instagram, X,")
    print("Facebook, and LinkedIn to research any topic.\n")
    
    query = input("What would you like to research?\n> ").strip()
    
    if not query:
        print("No query provided.")
        return
    
    print("\n" + "-" * 60)
    print("Researching...\n")
    
    inputs = {"messages": [{"role": "user", "content": query}]}
    report = await stream_agent_response(agent, inputs)
    
    print("\n" + "=" * 60)
    print("REPORT")
    print("=" * 60 + "\n")
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
