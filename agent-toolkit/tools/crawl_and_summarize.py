import os
import time
from typing import Optional, Type

from pydantic import BaseModel
from tavily import TavilyClient

from ..models import ModelConfig, ToolUsageStats
from ..utilities.utils import (ainvoke_with_fallback, clean_raw_content,
                               crawl_with_retry)


async def crawl_and_summarize(
    url: str,
    model_config: ModelConfig,
    instructions: Optional[str] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    chunks_per_source: int = 3,
    max_depth: int = 1,
    max_breadth: int = 20,
    limit: int = 50,
    select_paths: Optional[list[str]] = None,
    select_domains: Optional[list[str]] = None,
    exclude_paths: Optional[list[str]] = None,
    exclude_domains: Optional[list[str]] = None,
    allow_external: bool = True,
    include_images: bool = False,
    extract_depth: str = "basic",
    format: str = "markdown",
    include_favicon: bool = False,
    timeout: float = 150,
    api_key: Optional[str] = None,
    max_retries: int = 1,
) -> dict:
    start_time = time.perf_counter()
    usage = ToolUsageStats()
    
    api_key = api_key or os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("API key must be provided or set in TAVILY_API_KEY environment variable")

    client = TavilyClient(api_key)

    kwargs = {
        "url": url,
        "max_depth": max_depth,
        "max_breadth": max_breadth,
        "limit": limit,
        "allow_external": allow_external,
        "include_images": include_images,
        "extract_depth": extract_depth,
        "format": format,
        "include_favicon": include_favicon,
        "timeout": timeout,
    }

    if instructions is not None:
        kwargs["instructions"] = instructions
        kwargs["chunks_per_source"] = chunks_per_source

    if select_paths is not None:
        kwargs["select_paths"] = select_paths

    if select_domains is not None:
        kwargs["select_domains"] = select_domains

    if exclude_paths is not None:
        kwargs["exclude_paths"] = exclude_paths

    if exclude_domains is not None:
        kwargs["exclude_domains"] = exclude_domains

    crawl_response = crawl_with_retry(client, max_retries, **kwargs)
    usage.tavily.add_crawl(crawl_response.credits, crawl_response.response_time)

    results = crawl_response.data.get("results", [])
    formatted_output = ""
    for i, item in enumerate(results):
        formatted_output += f"URL: {item['url']}\n\n"
        if instructions is not None:
            formatted_output += f"SUMMARY OF WEBPAGE:\n{item['raw_content']}\n\n"
        else:
            cleaned_raw_content = clean_raw_content(item['raw_content'])
            formatted_output += f"SUMMARY OF WEBPAGE:\n{cleaned_raw_content}\n\n"
        formatted_output += "\n"
    
    # Grounding instructions - keep factual but don't restrict format
    grounding = "Only include information explicitly present in the content. Do not fabricate or infer missing information or information that is not directly supported in the source."
    
    if output_schema:
        grounding += " For fields without supporting information, set it to null."

    # Create prompt based on whether instructions were provided
    if instructions is not None:
        summary_prompt = f"""Summarize the following content based on these instructions: {instructions}

{grounding}

Content:
{formatted_output}"""
    else:
        summary_prompt = f"""Summarize the following content.

{grounding}

Content:
{formatted_output}"""
    
    llm_response = await ainvoke_with_fallback(
        model_config, summary_prompt, output_schema=output_schema, return_usage=True
    )
    usage.llm.merge(llm_response.usage)
    
    summary = llm_response.result
    if not output_schema:
        summary = llm_response.result.content
    
    usage.response_time = time.perf_counter() - start_time
    
    return {
        "results": results,
        "summary": summary,
        "usage": usage.to_dict(),
    }
