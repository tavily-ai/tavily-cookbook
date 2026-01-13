"""
Deep Agent for Company Intelligence Research

Uses LangGraph's ReAct agent to conduct comprehensive research on companies
by crawling their website and searching for relevant information across the web.
"""

# Agent(tools=crawl_company_website, extract_from_urls, tavily_search)

import asyncio
import os
import sys
from typing import List, Optional

from dotenv import load_dotenv

# Add agent-toolkit directory to path for imports
parent_dir = os.path.join(os.path.dirname(__file__), "..")
root_dir = os.path.join(parent_dir, "..")
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(root_dir, "agent-toolkit"))

# Load environment variables from .env file
load_dotenv(os.path.join(parent_dir, ".env"))

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from models import ModelConfig, ModelObject
from tools.async_search_and_dedup import search_dedup
from tools.crawl_and_summarize import crawl_and_summarize
from tools.extract_and_summarize import extract_and_summarize
from utilities.utils import format_web_results
from utils import stream_agent_response

# Get API keys from environment
TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY environment variable is required")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Model config for summarization
SUMMARIZER_CONFIG = ModelConfig(
    model=ModelObject(
        model="gpt-5.1",
        api_key=OPENAI_API_KEY,
    ),
)

async def crawl_company_website(
    url: str,
    instructions: Optional[str] = None,
    max_depth: int = 2,
    max_breadth: int = 10,
    limit: int = 20,
) -> dict:
    """
    Crawl a company's website to extract and summarize information.
    
    Args:
        url: The company website URL to crawl (e.g., "https://example.com")
        instructions: Optional specific instructions for what to extract 
            (e.g., "Find information about leadership team and company history")
        max_depth: How deep to crawl from the starting URL (default: 2)
        max_breadth: Maximum pages to crawl per level (default: 10)
        limit: Total maximum pages to crawl (default: 20)
    
    Returns:
        Dictionary containing crawled results and a summary of the website content.
    """
    # Run the async function synchronously
    result = await crawl_and_summarize(
            url=url,
            model_config=SUMMARIZER_CONFIG,
            instructions=instructions,
            max_depth=max_depth,
            max_breadth=max_breadth,
            limit=limit,
            api_key=TAVILY_API_KEY,
        )
    return result["summary"]


async def extract_from_urls(
    urls: List[str],
    query: Optional[str] = None,
) -> dict:
    """
    Extract and summarize content from specific URLs.
    
    Args:
        urls: List of URLs to extract content from (e.g., ["https://example.com/about", "https://example.com/team"])
        query: Optional query to focus the extraction on specific information
            (e.g., "What are the company's main products and pricing?")
    
    Returns:
        Dictionary containing extracted results with summaries for each URL.
    """
    # Run the async function synchronously
    return await extract_and_summarize(
            urls=urls,
            model_config=SUMMARIZER_CONFIG,
            query=query,
            extract_depth="advanced",
            api_key=TAVILY_API_KEY,
        )

async def tavily_search(
    queries: List[str],
    max_results: int = 5,
    topic: str = "general",
    time_range: Optional[str] = None,
):
    """
    Search the web with multiple queries in parallel and get deduplicated results.
    
    Args:
        queries: List of search queries to run (e.g., ["Company X funding", "Company X CEO"])
        max_results: Maximum results per query (default: 5)
        topic: Search topic - "general", "news", or "finance" (default: "general")
        time_range: Filter by time - "day", "week", "month", or "year" (optional)
    
    Returns:
        Dictionary containing deduplicated search results with URLs, titles, and content.
    """
    result = await search_dedup(api_key=TAVILY_API_KEY, queries=queries, max_results=max_results, topic=topic, time_range=time_range, search_depth="advanced", include_answer=True)
    return format_web_results(result["results"])


# System prompt for the company intelligence research agent
COMPANY_INTELLIGENCE_INSTRUCTIONS = """You are a business intelligence analyst researching companies.

You have three tools available:
- `crawl_company_website` - Crawl a company's website to discover and summarize pages
- `extract_from_urls` - Extract detailed content from specific URLs  
- `tavily_search` - Search the web for news, funding, reviews, and other external info

Use these tools as you see fit to gather comprehensive information. Combine website insights with external sources for a complete picture.

When you're done researching, write up your findings in a clear report. Include citations [1], [2], etc. linking to your sources, and list all sources at the end.
"""

# Create the ReAct agent with OpenAI
model = ChatOpenAI(model="gpt-5-mini")
agent = create_agent(
    model=model,
    tools=[crawl_company_website, extract_from_urls, tavily_search],
    system_prompt=COMPANY_INTELLIGENCE_INSTRUCTIONS,
)

async def main():
    print("=" * 60)
    print("Company Intelligence Research Agent")
    print("=" * 60)
    
    # Get user input
    company_name = input("\nEnter company name: ").strip()
    website_url = input("Enter website URL: ").strip()
    research_prompt = input("Enter research focus (or press Enter for general research): ").strip()
    
    if not research_prompt:
        research_prompt = "Provide a comprehensive overview of the company"
    
    print(f"\nResearching: {company_name}")
    print(f"Website: {website_url}")
    print(f"Focus: {research_prompt}\n")
    print("-" * 60 + "\n")
    
    # Build the user message
    user_message = f"""Research this company:

**Company Name:** {company_name}
**Website:** {website_url}
**Focus:** {research_prompt}
"""
    
    # Stream the agent with progress updates
    inputs = {"messages": [{"role": "user", "content": user_message}]}
    final_response = await stream_agent_response(agent, inputs)
    
    print("\n" + "=" * 60)
    print("RESEARCH REPORT")
    print("=" * 60 + "\n")
    print(final_response)


if __name__ == "__main__":
    asyncio.run(main())

