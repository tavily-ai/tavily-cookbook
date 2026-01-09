import os
import time
from typing import Literal, Optional, Type

from pydantic import BaseModel
from tavily import TavilyClient

from utilities.utils import (ainvoke_with_fallback, clean_raw_content,
                           extract_with_retry)
from models import ModelConfig, ToolUsageStats


async def extract_and_summarize(
    urls: list[str],
    model_config: ModelConfig,
    query: Optional[str] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    chunks_per_source: int = 5,
    extract_depth: Literal["basic", "advanced"] = "basic",
    include_images: bool = False,
    include_favicon: bool = False,
    format: Literal["markdown", "text"] = "markdown",
    timeout: Optional[float] = None,
    api_key: Optional[str] = None,
    max_retries: int = 1,
) -> dict:
    """
    Extract content from URLs and summarize using an LLM.
    
    When a query is provided, extracts relevant chunks and summarizes those chunks.
    Without a query, extracts full content and summarizes the entire page.
    
    Args:
        urls: List of URLs to extract content from.
        model_config: Configuration for the LLM used for summarization.
        query: User intent for reranking extracted content chunks. When provided,
            chunks are reranked based on relevance and the summary focuses on
            those relevant chunks.
        output_schema: Optional Pydantic model for structured output.
        chunks_per_source: Max chunks per source (1-5). Only used when query is provided.
        extract_depth: Extraction depth - "basic" or "advanced". Advanced retrieves
            more data including tables and embedded content.
        include_images: Include extracted images in the response.
        include_favicon: Include favicon URL for each result.
        format: Output format - "markdown" or "text".
        timeout: Request timeout in seconds (1-60). Defaults based on extract_depth.
        api_key: Tavily API key. Falls back to TAVILY_API_KEY env var.
    
    Returns:
        Dictionary containing:
            - results: List of extraction results, each with an added "summary" field
            - observability: Timing and usage metrics
    
    Example:
        >>> from assets.tools.extract_and_summarize import extract_and_summarize
        >>> from models import ModelConfig, ModelObject
        >>> 
        >>> # Basic extraction with summary
        >>> result = await extract_and_summarize(
        ...     urls=["https://en.wikipedia.org/wiki/Artificial_intelligence"],
        ...     model_config=ModelConfig(model=ModelObject(model="gpt-5-mini"))
        ... )
        >>> 
        >>> # Query-focused extraction (summarizes relevant chunks)
        >>> result = await extract_and_summarize(
        ...     urls=["https://en.wikipedia.org/wiki/Artificial_intelligence"],
        ...     model_config=ModelConfig(model=ModelObject(model="gpt-5-mini")),
        ...     query="What are the main applications of AI?",
        ...     chunks_per_source=5
        ... )
    """
    start_time = time.perf_counter()
    usage = ToolUsageStats()
    
    api_key = api_key or os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("API key must be provided or set in TAVILY_API_KEY environment variable")

    client = TavilyClient(api_key)

    # Build extract parameters
    kwargs = {
        "urls": urls,
        "extract_depth": extract_depth,
        "include_images": include_images,
        "include_favicon": include_favicon,
        "format": format,
    }

    # Add query-related parameters if query is provided
    if query is not None:
        kwargs["query"] = query
        kwargs["chunks_per_source"] = chunks_per_source

    if timeout is not None:
        kwargs["timeout"] = timeout

    # Execute extraction with retry logic
    extract_response = extract_with_retry(client, max_retries, **kwargs)
    usage.tavily.add_extract(extract_response.credits, extract_response.response_time)

    results = extract_response.data.get("results", [])

    # Generate a summary for each page
    for item in results:
        # Get content to summarize
        if query is not None:
            # When query is provided, raw_content contains relevant chunks
            content = item['raw_content']
        else:
            # Full content extraction - clean up boilerplate
            content = clean_raw_content(item['raw_content'])

        # Grounding instructions - keep factual but don't restrict format
        grounding = "Only include information explicitly present in the content. Do not fabricate or infer missing information or give any opinions, interpretations, or information not directly supported in the source."
        
        if output_schema:
            grounding += " For fields without supporting information, use null or empty values. Do not fabricate or infer missing information or give any opinions, interpretations, or information not directly supported in the source."

        # Create prompt based on whether query was provided
        if query is not None:
            summary_prompt = f"""Summarize the following content based on this query: {query}

{grounding}

Content:
{content}"""
        else:
            summary_prompt = f"""Summarize the following content.

{grounding}

Content:
{content}"""
        
        llm_response = await ainvoke_with_fallback(
            model_config, summary_prompt, output_schema=output_schema, return_usage=True
        )
        usage.llm.merge(llm_response.usage)
        
        if output_schema:
            item["summary"] = llm_response.result
        else:
            item["summary"] = llm_response.result.content
        
        # Remove raw content from response
        del item["raw_content"]

    usage.response_time = time.perf_counter() - start_time
    
    return {
        "results": results,
        "usage": usage.to_dict(),
    }
