import asyncio
import os
import sys

from dotenv import load_dotenv
from tavily import TavilyClient

parent_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, "src"))

load_dotenv(os.path.join(parent_dir, ".env"))

from langchain.agents import create_agent
from langchain_core.tools import tool

from helpers.research_stream import handle_research_stream
from models.types import ModelConfig, ModelObject
from tools.search_and_answer import search_and_answer

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")


def create_chatbot_agent(model_config: ModelConfig):
    @tool
    async def search_and_answer_tool(query: str, time_range=None) -> dict:
        """Search the web and answer questions using Tavily search with AI-powered synthesis.
        Use this tool for very simple, straightforward questions that need minimal data (e.g., "What is the capital of France?", "Who is the CEO of Apple?", single-fact lookups)
        parameters:
        - query: The search query to answer. These should be concise and clear google style queries.
        - time_range: Time range to filter search results. Use this based on how recent the information is needed
        """
        response = await search_and_answer(query, api_key=TAVILY_API_KEY, model_config=model_config, search_depth="advanced", time_range=time_range, max_number_of_subqueries=5, include_citations=True)
        return response.get("answer", "No answer found.")

    @tool
    async def stream_research_tool(input: str) -> dict:
        """Research the web and answer questions using Tavily search with AI-powered synthesis.
        Use this tool for anything requiring multiple sources, detailed analysis, comparisons, comprehensive answers, or any query that isn't trivially simple
        parameters:
        - input: The research prompt. Define a clear goal with all details and direction.
            Be specific when you can. If you already know important details, include them.
            (E.g. Target market or industry, key competitors, customer segments, geography, or constraints)
            Only stay open-ended if you don't know details and want discovery. If you're exploring broadly, make that explicit (e.g., "tell me about the most impactful AI innovations in healthcare in 2025").
            Avoid contradictions. Don't include conflicting information, constraints, or goals in your prompt.
            Share what's already known. Include prior assumptions, existing decisions, or baseline knowledge—so the research doesn't repeat what you already have.
            Keep the prompt clean and directed. Use a clear task statement + essential context + desired output format. Avoid messy background dumps.
        """
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.research(input=input, model="mini", max_results=10, stream=True)
        report = handle_research_stream(response, stream_content_generation=False)
        return {"route": "research", "response": report}
    
    # Add other tools here (e.g. internal research tool, etc.)

    return create_agent(
        model=model_config.model.model,
        tools=[search_and_answer_tool, stream_research_tool],
        system_prompt="""You are a helpful chatbot assistant with access to a research tool and a search tool. For simple questions, use the search_and_answer_tool to find information.
        For complex questions, use the stream_research_tool to find information.
        CRITICAL: YOU CAN ONLY USE THE RESEARCH TOOL ONCE. DO NOT USE IT TWICE.
        You can call search_and_answer until you have enough information to give a complete answer.
        CITATIONS: Use numbered in-text citations [1], [2], etc. to back up your claims. At the end, include a "Sources:" section with only the sources you actually cited (format: [number] Title - URL).
        """
    )


async def run_chatbot():
    config = ModelConfig(
        model=ModelObject(model="gpt-5.1")
    )
    agent = create_chatbot_agent(config)
    messages = []
    print("Chatbot ready! Type 'quit' to exit.\n")
    
    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            print("Goodbye!")
            break
        
        messages.append({"role": "user", "content": query})
        response = await agent.ainvoke({"messages": messages})
        assistant_message = response["messages"][-1].content
        messages.append({"role": "assistant", "content": assistant_message})
        print(f"\nAssistant: {assistant_message}\n")


if __name__ == "__main__":
    asyncio.run(run_chatbot())