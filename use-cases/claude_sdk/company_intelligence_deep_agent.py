"""
Deep Agent for Company Intelligence Research

Uses Claude with Anthropic SDK to conduct comprehensive research on companies
by crawling their website and searching for relevant information across the web.

Prerequisites:
    pip install anthropic tavily-python python-dotenv

Usage:
    # Set ANTHROPIC_API_KEY and TAVILY_API_KEY in .env file
    python company_intelligence_deep_agent.py
"""

import asyncio
import json
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from tavily_agent_toolkit import (ModelConfig, ModelObject,
                                  crawl_and_summarize, extract_and_summarize,
                                  format_web_results, search_dedup)

# Load .env from the same folder as this script
load_dotenv(Path(__file__).parent / ".env")

# Get API keys from environment
TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "")
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

# Model config for summarization (uses Claude)
SUMMARIZER_CONFIG = ModelConfig(
    model=ModelObject(
        model="claude-haiku-4-5-20251001",
        api_key=ANTHROPIC_API_KEY,
    ),
)

# Define tools for Claude
TOOLS = [
    {
        "name": "crawl_company_website",
        "description": """Crawl a company's website to extract and summarize information.

Use this tool to discover and analyze pages on a company's website. Great for understanding 
what the company does, their products/services, team, and company information.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The company website URL to crawl (e.g., 'https://example.com')"
                },
                "instructions": {
                    "type": "string",
                    "description": "Optional specific instructions for what to extract (e.g., 'Find information about leadership team and company history')"
                },
                "max_depth": {
                    "type": "integer",
                    "description": "How deep to crawl from the starting URL (default: 2)"
                },
                "max_breadth": {
                    "type": "integer",
                    "description": "Maximum pages to crawl per level (default: 10)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Total maximum pages to crawl (default: 20)"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "extract_from_urls",
        "description": """Extract and summarize content from specific URLs.

Use this tool when you have specific URLs you want to analyze in detail.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of URLs to extract content from (e.g., ['https://example.com/about', 'https://example.com/team'])"
                },
                "query": {
                    "type": "string",
                    "description": "Optional query to focus the extraction on specific information (e.g., 'What are the company's main products and pricing?')"
                }
            },
            "required": ["urls"]
        }
    },
    {
        "name": "tavily_search",
        "description": """Search the web with multiple queries in parallel and get deduplicated results.

Use this tool to find external information about a company - news, funding, reviews, etc.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search queries to run (e.g., ['Company X funding', 'Company X CEO'])"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results per query (default: 5)"
                },
                "topic": {
                    "type": "string",
                    "enum": ["general", "news", "finance"],
                    "description": "Search topic - 'general', 'news', or 'finance' (default: 'general')"
                },
                "time_range": {
                    "type": "string",
                    "enum": ["day", "week", "month", "year"],
                    "description": "Filter by time - 'day', 'week', 'month', or 'year' (optional)"
                }
            },
            "required": ["queries"]
        }
    }
]

# System prompt for the company intelligence research agent
SYSTEM_PROMPT = """You are a business intelligence analyst researching companies.

You have three tools available:
- `crawl_company_website` - Crawl a company's website to discover and summarize pages
- `extract_from_urls` - Extract detailed content from specific URLs  
- `tavily_search` - Search the web for news, funding, reviews, and other external info

Use these tools as you see fit to gather comprehensive information. Combine website insights with external sources for a complete picture.

When you're done researching, write up your findings in a clear report. Include citations [1], [2], etc. linking to your sources, and list all sources at the end."""


async def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return the result."""
    if tool_name == "crawl_company_website":
        url = tool_input.get("url", "")
        instructions = tool_input.get("instructions")
        max_depth = tool_input.get("max_depth", 2)
        max_breadth = tool_input.get("max_breadth", 10)
        limit = tool_input.get("limit", 20)
        
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
    
    elif tool_name == "extract_from_urls":
        urls = tool_input.get("urls", [])
        query = tool_input.get("query")
        
        result = await extract_and_summarize(
            urls=urls,
            model_config=SUMMARIZER_CONFIG,
            query=query,
            extract_depth="advanced",
            api_key=TAVILY_API_KEY,
        )
        return json.dumps(result)
    
    elif tool_name == "tavily_search":
        queries = tool_input.get("queries", [])
        max_results = tool_input.get("max_results", 5)
        topic = tool_input.get("topic", "general")
        time_range = tool_input.get("time_range")
        
        result = await search_dedup(
            api_key=TAVILY_API_KEY,
            queries=queries,
            max_results=max_results,
            topic=topic,
            time_range=time_range,
            search_depth="advanced",
            include_answer=True
        )
        return format_web_results(result["results"])
    
    return f"Unknown tool: {tool_name}"


async def run_research(user_message: str) -> str:
    """
    Run company intelligence research for a given message.
    
    Args:
        user_message: The research request with company details.
        
    Returns:
        The research report as a string.
    """
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
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
                    print(f"  [Using {block.name}...] ", flush=True)
                    result = await execute_tool(block.name, block.input)
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
    """Company intelligence research CLI."""
    print("=" * 60)
    print("Company Intelligence Research Agent (Claude)")
    print("=" * 60)
    
    # Check for API keys
    if not ANTHROPIC_API_KEY:
        print("\nError: ANTHROPIC_API_KEY not set.")
        print("  Add it to .env or run: export ANTHROPIC_API_KEY=your-api-key")
        return
    if not TAVILY_API_KEY:
        print("\nError: TAVILY_API_KEY not set.")
        print("  Add it to .env or run: export TAVILY_API_KEY=your-api-key")
        return
    
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
    
    report = await run_research(user_message)
    
    print("\n" + "=" * 60)
    print("RESEARCH REPORT")
    print("=" * 60 + "\n")
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
