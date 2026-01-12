"""
Chatbot implementation using the Anthropic SDK with Claude.

This chatbot uses Tavily tools for web search and research capabilities,
powered by Claude as the underlying model.

Prerequisites:
    pip install anthropic tavily-python python-dotenv

Usage:
    # Set ANTHROPIC_API_KEY and TAVILY_API_KEY in .env file
    python chatbot_claude_sdk.py
"""

import asyncio
import json
import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv
from tavily import TavilyClient

parent_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, parent_dir)

# Load .env from use-cases directory first, then parent directory as fallback
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(parent_dir, ".env"))

sys.path.insert(0, os.path.join(parent_dir, "..", "research-toolkit"))

from tools.search_and_format import search_and_format
from utilities.research_stream import handle_research_stream

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Define tools for Claude
TOOLS = [
    {
        "name": "search_and_format",
        "description": """Search the web and return formatted results from multiple queries.
Use this tool for very simple, straightforward questions that need minimal data (e.g., "What is the capital of France?", "Who is the CEO of Apple?", single-fact lookups)""",
        "input_schema": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search queries. Use concise, Google-style queries. Can be 1 or more queries."
                },
                "time_range": {
                    "type": "string",
                    "enum": ["day", "week", "month", "year"],
                    "description": "Optional time filter: 'day', 'week', 'month', or 'year'"
                }
            },
            "required": ["queries"]
        }
    },
    {
        "name": "stream_research",
        "description": """Research the web and answer questions using Tavily search with AI-powered synthesis.
Use this tool for anything requiring multiple sources, detailed analysis, comparisons, comprehensive answers, or any query that isn't trivially simple.

Parameters:
- input: The research prompt. Define a clear goal with all details and direction.
    Be specific when you can. If you already know important details, include them.
    (E.g. Target market or industry, key competitors, customer segments, geography, or constraints)
    Only stay open-ended if you don't know details and want discovery.
    Avoid contradictions. Don't include conflicting information, constraints, or goals.
    Share what's already known to avoid repeating existing knowledge.
    Keep the prompt clean and directed.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The research prompt with clear goal and context."
                }
            },
            "required": ["input"]
        }
    }
]

SYSTEM_PROMPT = """You are a helpful chatbot assistant with access to a research tool and a search tool.

For simple questions, use the search_and_format tool to find information.
For complex questions, use the stream_research tool to find information.

CRITICAL: YOU CAN ONLY USE THE RESEARCH TOOL ONCE. DO NOT USE IT TWICE.
You can call search_and_format multiple times until you have enough information to give a complete answer.

CITATIONS: Use numbered in-text citations [1], [2], etc. to back up your claims. 
At the end, include a "Sources:" section with only the sources you actually cited (format: [number] Title - URL).
"""


async def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return the result."""
    if tool_name == "search_and_format":
        queries = tool_input.get("queries", [])
        time_range = tool_input.get("time_range")
        result = await search_and_format(
            queries=queries, 
            api_key=TAVILY_API_KEY, 
            time_range=time_range
        )
        return result
    
    elif tool_name == "stream_research":
        input_prompt = tool_input.get("input", "")
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.research(
            input=input_prompt, 
            model="mini", 
            max_results=10, 
            stream=True
        )
        report = handle_research_stream(response, stream_content_generation=False)
        return json.dumps({"route": "research", "response": report})
    
    return f"Unknown tool: {tool_name}"


async def run_chatbot():
    """Run an interactive chatbot using Claude with Tavily tools."""
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    messages = []
    
    print("Claude + Tavily Chatbot ready! Type 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
            
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        
        messages.append({"role": "user", "content": user_input})
        
        print("\nAssistant: ", end="", flush=True)
        
        try:
            # Agent loop - keep calling Claude until we get a final response
            while True:
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages
                )
                
                # Check if Claude wants to use tools
                if response.stop_reason == "tool_use":
                    # Process all tool calls
                    assistant_content = response.content
                    messages.append({"role": "assistant", "content": assistant_content})
                    
                    tool_results = []
                    for block in assistant_content:
                        if block.type == "tool_use":
                            print(f"[Using {block.name}...] ", end="", flush=True)
                            result = await execute_tool(block.name, block.input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result
                            })
                    
                    messages.append({"role": "user", "content": tool_results})
                
                else:
                    # Final response - extract and print text
                    assistant_message = ""
                    for block in response.content:
                        if hasattr(block, "text"):
                            assistant_message += block.text
                    
                    print(assistant_message)
                    messages.append({"role": "assistant", "content": response.content})
                    break
            
            print()  # Extra newline for spacing
            
        except Exception as e:
            print(f"\nError: {e}\n")


async def single_query(prompt: str) -> str:
    """
    Run a single query and return the response.
    
    Args:
        prompt: The question or prompt to send to the chatbot.
        
    Returns:
        The assistant's response text.
    """
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    messages = [{"role": "user", "content": prompt}]
    
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )
        
        if response.stop_reason == "tool_use":
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})
            
            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    result = await execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            messages.append({"role": "user", "content": tool_results})
        else:
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""


if __name__ == "__main__":
    # Check for API keys
    if not ANTHROPIC_API_KEY:
        print("Warning: ANTHROPIC_API_KEY not set.")
        print("  Add it to .env or run: export ANTHROPIC_API_KEY=your-api-key")
        print()
    if not TAVILY_API_KEY:
        print("Warning: TAVILY_API_KEY not set.")
        print("  Add it to .env or run: export TAVILY_API_KEY=your-api-key")
        print()
    
    asyncio.run(run_chatbot())
