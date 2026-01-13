from typing import Any, Dict, List, Literal, Optional, Sequence, Union, cast

from tavily import TavilyClient
from tools.async_search_and_dedup import search_dedup
from utilities.utils import (clean_formatted_output, format_web_results,
                             search_with_retry)


async def search_and_format(
    queries: list[str],
    api_key: str,
    threshold: Optional[float] = 0.3,
    search_depth: Optional[Literal["basic", "advanced"]] = "advanced",
    topic: Optional[Literal["general", "news", "finance"]] = None,
    time_range: Optional[Literal["day", "week", "month", "year"]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: Optional[int] = None,
    max_results: Optional[int] = 10,
    include_domains: Optional[Sequence[str]] = None,
    exclude_domains: Optional[Sequence[str]] = None,
    include_raw_content: Optional[Union[bool, Literal["markdown", "text"]]] = None,
    include_images: Optional[bool] = None,
    include_image_descriptions: Optional[bool] = None,
    timeout: int = 60,
    country: Optional[str] = None,
    auto_parameters: Optional[bool] = None,
    include_favicon: Optional[bool] = None,
    include_citations: Optional[bool] = False,
    max_retries: int = 1,
    **kwargs: Any,
) -> str:
    """
    Search the web and return formatted results as a string.
    
    Uses single search for one query, or deduplicated parallel search for multiple queries.
    
    Args:
        queries: List of search queries (1 or more)
        api_key: Tavily API key
        threshold: Minimum score threshold for results (default: 0.3)
        search_depth: "basic" or "advanced" search
        topic: "general", "news", or "finance"
        time_range: "day", "week", "month", or "year"
        start_date: Filter results after date (YYYY-MM-DD)
        end_date: Filter results before date (YYYY-MM-DD)
        days: Number of days to filter results
        max_results: Maximum results per query
        include_domains: Domains to include
        exclude_domains: Domains to exclude
        include_raw_content: Include raw HTML content
        include_images: Include query-related images
        include_image_descriptions: Include image descriptions
        timeout: Request timeout in seconds
        country: Boost results from specific country
        auto_parameters: Let Tavily auto-configure
        include_favicon: Include favicon URL for each result
        include_citations: Reserved for future use
        max_retries: Maximum retry attempts
        
    Returns:
        Formatted search results as a string
    """
    search_params = {
        "max_results": max_results,
        "topic": topic,
        "time_range": time_range,
        "start_date": start_date,
        "end_date": end_date,
        "days": days,
        "search_depth": search_depth,
        "include_domains": include_domains,
        "exclude_domains": exclude_domains,
        "include_raw_content": include_raw_content,
        "include_images": include_images,
        "include_image_descriptions": True if include_images or include_image_descriptions else False,
        "include_favicon": include_favicon,
        "timeout": timeout,
        "country": country,
        "auto_parameters": auto_parameters,
    }
    
    result: Dict[str, Any] = {}
    
    if len(queries) == 1:
        # Single query: use TavilyClient directly
        tavily_client = TavilyClient(api_key=api_key)
        search_response = search_with_retry(
            tavily_client, max_retries, query=queries[0], **search_params
        )
        result = search_response.data
        if "results" not in result:
            raise ValueError("No results found")
        results_list = result.get("results", [])
    else:
        # Multiple queries: use search_dedup for parallel search with deduplication
        dedup_result = await search_dedup(api_key=api_key, queries=queries, max_retries=max_retries, **search_params)
        results_list = dedup_result["results"]
    
    # Apply threshold filtering and sort by score
    if threshold is not None:
        results_list = [r for r in results_list if r.get("score", 0) >= threshold]
    results_list = sorted(results_list, key=lambda x: x.get("score", 0), reverse=True)[:20]
    
    # Format the search results
    formatted_output = format_web_results(results_list)
    
    # Clean and return
    return formatted_output