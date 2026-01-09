import time
from typing import Any, Callable, Literal, Optional, Type, cast

import httpx
from pydantic import BaseModel
from tavily import TavilyClient

from utilities.utils import (ainvoke_with_fallback, clean_formatted_output,
                           format_web_results, generate_subqueries,
                           synthesize_results)
from models import (HybridResearchResponse, LLMUsage, ModelConfig,
                          OutputSchema, ToolUsageStats)
from tools.async_search_and_dedup import search_dedup


async def hybrid_research(
    api_key: str,
    query: str,
    model_config: ModelConfig,
    internal_rag_function: Callable[[str], str],
    mode: Literal["fast", "multi_agent"] = "fast",
    output_schema: Optional[Type[OutputSchema]] = None,
    research_synthesis_prompt: Optional[str] = None,
) -> dict[str, Any]:
    """Hybrid research combining internal RAG with web search.
    
    Returns:
        Dictionary containing report, web_sources, and usage metrics
    """
    if mode == "fast":
        return await fast_mode(api_key, query, model_config, internal_rag_function, output_schema, research_synthesis_prompt)
    elif mode == "multi_agent":
        return await multi_agent_mode(api_key, query, model_config, internal_rag_function, output_schema, research_synthesis_prompt)
    else:
        raise ValueError(f"Invalid mode: {mode}")


async def fast_mode(
    api_key: str,
    query: str,
    model_config: ModelConfig,
    internal_rag_function: Callable[[str], str],
    output_schema: Optional[Type[OutputSchema]] = None,
    research_synthesis_prompt: Optional[str] = None,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    usage = ToolUsageStats()
    
    # Run internal RAG function with timing
    internal_start = time.perf_counter()
    internal_results = internal_rag_function(query)
    usage.internal_function_response_time = time.perf_counter() - internal_start
    
    # Generate subqueries with LLM usage tracking
    subquery_result = await generate_subqueries(
        query, model_config,
        max_number_of_subqueries=5,
        context=str(internal_results),
        output_schema=output_schema,
        return_usage=True,
    )
    subqueries, subquery_usage = cast(tuple[list[str], LLMUsage], subquery_result)
    usage.llm.merge(subquery_usage)
    
    # Web search with Tavily usage tracking
    web_results = await search_dedup(api_key, subqueries)
    
    # Extract tavily usage from search_dedup result
    if "tavily_usage" in web_results:
        tavily_usage_dict = web_results.pop("tavily_usage")
        usage.tavily.total_credits += tavily_usage_dict.get("total_credits", 0)
        usage.tavily.search_count += tavily_usage_dict.get("search_count", 0)
        usage.tavily.search_response_time += tavily_usage_dict.get("search_response_time", 0)
    if "response_time" in web_results:
        web_results.pop("response_time")
    
    formatted_web_results = format_web_results(web_results["results"])
    cleaned_formatted_output = clean_formatted_output(formatted_web_results)
    research_results = f"{internal_results}\n\n{cleaned_formatted_output}"
    
    # Synthesize results with LLM usage tracking
    synthesis_result = await synthesize_results(
        query, model_config, research_results,
        output_schema, research_synthesis_prompt,
        return_usage=True,
    )
    report, synthesis_usage = cast(tuple[str | BaseModel, LLMUsage], synthesis_result)
    usage.llm.merge(synthesis_usage)
    
    sources = []
    search_results = web_results["results"]
    for result in search_results:
        sources.append({"title": result["title"], "url": result["url"]})
    
    if isinstance(report, BaseModel):
        report = report.model_dump_json()
    
    usage.response_time = time.perf_counter() - start_time
    
    return {
        "report": report,
        "web_sources": sources,
        "usage": usage.to_dict(),
    }


async def multi_agent_mode(
    api_key: str,
    query: str,
    model_config: ModelConfig,
    internal_rag_function: Callable[[str], str],
    output_schema: Optional[Type[OutputSchema]] = None,
    research_synthesis_prompt: Optional[str] = None,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    usage = ToolUsageStats()
    
    # Internal Research Agent with timing
    internal_start = time.perf_counter()
    internal_results = internal_rag_function(query)
    usage.internal_function_response_time = time.perf_counter() - internal_start

    # Generate Brief for web agent to fill in gaps
    refine_brief_prompt = f"""Context (internal research): {internal_results}
        Identify what information is missing or incomplete above. Output a concise research prompt for a web search to fill those gaps. The prompt should not include details from the internal research just a new research prompt.
        Do not reference or include any of the internal research content—just output the research prompt."""

    brief_response = await ainvoke_with_fallback(model_config, refine_brief_prompt, return_usage=True)
    usage.llm.merge(brief_response.usage)
    refined_brief = cast(str, brief_response.result.content)
    
    if output_schema:
        refined_brief = f"Fill out this schema: {output_schema.model_json_schema()}"
   
    # Web Agent - Tavily Research endpoint (track time but not credits per user request)
    tavily_client = TavilyClient(api_key=api_key)
    schema_dict = None
    if output_schema:
        schema_dict = output_schema.to_tavily_schema()
    
    research_start = time.perf_counter()
    response = cast(dict[str, Any], tavily_client.research(refined_brief, stream=False, output_schema=cast(dict[str, Any], schema_dict)))
    request_id = response["request_id"]

    url = "https://api.tavily.com/research/"
    headers = {"Authorization": f"Bearer {api_key}"}
    research_report = ""
    sources = []

    # Poll every 10 seconds
    while True:
        response = httpx.get(url + request_id, headers=headers)
        response_json = response.json()

        status = response_json["status"]
        if status == "completed":
            research_report = response_json["content"]
            sources = response_json["sources"]
            break

        if status == "failed":
            raise RuntimeError("Research failed to complete")

        time.sleep(10)
    
    # Track research endpoint time (credits not available for research endpoint)
    research_time = time.perf_counter() - research_start
    # Store in a custom field since research is different from search/extract/crawl
    # We'll add it to the tavily dict manually in to_dict output

    research_results = f"{str(internal_results)}\n\n{research_report}"
    
    # Synthesis into a report with sources
    synthesis_result = await synthesize_results(
        query, model_config, research_results,
        output_schema, research_synthesis_prompt,
        return_usage=True,
    )
    report, synthesis_usage = cast(tuple[str | BaseModel, LLMUsage], synthesis_result)
    usage.llm.merge(synthesis_usage)
    
    if isinstance(report, BaseModel):
        report = report.model_dump_json()
    
    usage.response_time = time.perf_counter() - start_time
    
    # Build usage dict with research time added
    usage_dict = usage.to_dict()
    usage_dict["tavily_research_response_time"] = round(research_time, 3)
    
    return {
        "report": report,
        "web_sources": sources,
        "usage": usage_dict,
    }