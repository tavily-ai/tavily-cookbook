"""
Tavily async search with deduplication by URL.
Aggregates unique content chunks from results sharing the same URL.
"""

import asyncio
import time
from typing import Any, Optional, Union

from tavily import AsyncTavilyClient

from utilities.utils import async_retry
from models import SearchDedupResponse, TavilyAPIResponse, TavilyUsage


async def search_dedup(
    api_key: str,
    queries: list[str],
    # Core parameters
    search_depth: str = "advanced",
    topic: str = "general",
    max_results: int = 5,
    chunks_per_source: int = 3,
    # Time filtering
    time_range: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    # Content options
    include_images: bool = False,
    include_image_descriptions: bool = False,
    include_answer: Union[bool, str] = False,
    include_raw_content: Union[bool, str] = False,
    # Domain filtering
    include_domains: Optional[list[str]] = None,
    exclude_domains: Optional[list[str]] = None,
    # Other options
    country: Optional[str] = None,
    auto_parameters: bool = False,
    include_favicon: bool = False,
    timeout: float = 60,
    max_retries: int = 1,
    **kwargs: Any,
) -> SearchDedupResponse:
    """
    Perform async Tavily searches and deduplicate results by URL.
    
    Content chunks (separated by ' [...] ') are split and only unique chunks
    are kept when multiple results share the same URL.
    
    Args:
        api_key: Tavily API key
        queries: List of search queries to run in parallel
        search_depth: "basic" or "advanced" (advanced required for chunks)
        topic: "general", "news", or "finance"
        max_results: Max results per query (0-20)
        chunks_per_source: Max chunks per source (only with advanced search)
        time_range: "day"/"d", "week"/"w", "month"/"m", "year"/"y"
        start_date: Filter results after date (YYYY-MM-DD)
        end_date: Filter results before date (YYYY-MM-DD)
        include_images: Include query-related images
        include_image_descriptions: Include image descriptions
        include_answer: Include LLM-generated answer (bool or "basic"/"advanced")
        include_raw_content: Include raw HTML content (bool or "markdown"/"text")
        include_domains: Domains to include (max 300)
        exclude_domains: Domains to exclude (max 150)
        country: Boost results from specific country (general topic only)
        auto_parameters: Let Tavily auto-configure based on query intent
        include_favicon: Include favicon URL for each result
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts per query (default: 3)
        
    Returns:
        SearchDedupResponse with deduplicated results, metadata, and observability
        
    Example:
        results = await search_dedup(
            api_key="your-api-key",
            queries=["query 1", "query 2", "query 3"],
            max_results=5,
            topic="general",
            search_depth="advanced",
        )
    """
    start_time = time.perf_counter()
    
    client = AsyncTavilyClient(api_key=api_key)
    
    # Build search kwargs
    search_kwargs = {
        "search_depth": search_depth,
        "topic": topic,
        "max_results": max_results,
        "include_images": include_images,
        "include_image_descriptions": include_image_descriptions,
        "include_answer": include_answer,
        "include_raw_content": include_raw_content,
        "auto_parameters": auto_parameters,
        "include_favicon": include_favicon,
        "timeout": timeout,
    }
    
    if search_depth == "advanced":
        search_kwargs["chunks_per_source"] = chunks_per_source
    
    # Add optional parameters in one pass
    optional_params = {
        "time_range": time_range,
        "start_date": start_date,
        "end_date": end_date,
        "include_domains": include_domains,
        "exclude_domains": exclude_domains,
        "country": country,
    }
    search_kwargs.update({k: v for k, v in optional_params.items() if v is not None})
    
    # Add any additional kwargs
    search_kwargs.update(kwargs)
    
    # Execute searches in parallel with retry logic
    api_responses: list[TavilyAPIResponse] = await asyncio.gather(
        *(async_retry(client.search, max_retries, query, **search_kwargs) for query in queries)
    )
    
    # Calculate timing and usage from all parallel requests
    total_time = time.perf_counter() - start_time
    tavily_usage = TavilyUsage()
    for resp in api_responses:
        tavily_usage.add_search(resp.credits, resp.response_time)
    
    # Extract raw data dicts for deduplication
    search_responses = [resp.data for resp in api_responses]
    
    result = _deduplicate_by_url(search_responses)
    result["tavily_usage"] = tavily_usage.to_dict()
    result["response_time"] = round(total_time, 3)
    
    return result


def _deduplicate_by_url(search_responses: list[dict]) -> SearchDedupResponse:
    """
    Deduplicate search results by URL, merging unique content chunks.
    
    Args:
        search_responses: List of Tavily search response dicts
        
    Returns:
        SearchDedupResponse with deduplicated results
    """
    # Single dict tracks both result data and chunks per URL
    url_data: dict[str, tuple[dict, set[str]]] = {}
    
    # Deduplicate images inline during collection
    seen_image_urls: set[str] = set()
    unique_images: list = []
    answers: list[str] = []
    max_response_time = 0.0
    
    for response in search_responses:
        # Track max response time (parallel requests complete at slowest)
        if (rt := response.get("response_time", 0)) > max_response_time:
            max_response_time = rt
        
        # Collect and deduplicate images inline
        if images := response.get("images"):
            for img in images:
                img_url = img if isinstance(img, str) else img.get("url", "")
                if img_url and img_url not in seen_image_urls:
                    seen_image_urls.add(img_url)
                    unique_images.append(img)
        
        # Collect answers
        if answer := response.get("answer"):
            answers.append(answer)
        
        # Process results
        for result in response.get("results") or ():
            if not (url := result.get("url")):
                continue
            
            # Parse chunks inline (avoid separate function call overhead)
            content = result.get("content", "")
            chunks = {c for c in (s.strip() for s in content.split(" [...] ")) if c}
            score = result.get("score", 0)
            
            if url in url_data:
                existing_result, existing_chunks = url_data[url]
                existing_chunks.update(chunks)
                # Keep higher score
                if score > existing_result.get("score", 0):
                    existing_result["score"] = score
            else:
                # Store result copy with its chunks
                url_data[url] = (result.copy(), chunks)
    
    # Build final results with merged chunks, sorted by score
    results = []
    for result, chunks in url_data.values():
        result["content"] = " [...] ".join(chunks)
        results.append(result)
    
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    return {
        "results": results,
        "images": unique_images or None,
        "answer": "\n\n".join(answers) if answers else None,
        "response_time": max_response_time,
        "queries": [r.get("query", "") for r in search_responses],
    }
