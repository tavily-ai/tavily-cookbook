import time
from typing import Any, Dict, Literal, Optional

from tavily import TavilyClient

from ..models import ToolUsageStats
from ..utilities.utils import extract_with_retry, search_with_retry

# Platform domain mapping
PLATFORM_DOMAINS = {
    "tiktok": "tiktok.com",
    "facebook": "facebook.com",
    "instagram": "instagram.com",
    "reddit": "reddit.com",
    "linkedin": "linkedin.com",
    "x": "x.com",
}

"""
Social Media Search Tool using Tavily API

Provides search functionality across major social media platforms with optional
raw content extraction for detailed analysis.
"""
def social_media_search(
    query: str,
    api_key: str,
    platform: Literal["tiktok", "facebook", "instagram", "reddit", "linkedin", "x", "combined"] = "combined",
    include_raw_content: bool = False,
    max_results: Optional[int] = 5,
    search_depth: Literal["basic", "advanced"] = "basic",
    include_answer: bool = False,
    include_images: bool = False,
    time_range: Optional[Literal["day", "week", "month", "year"]] = None,
    max_retries: int = 1,
) -> Dict[str, Any]:
    """
    Search social media platforms using Tavily API.
    
    Args:
        query: The search query string
        api_key: Your Tavily API key
        platform: Social media platform to search. Options are:
            - "tiktok": Search TikTok only
            - "facebook": Search Facebook only
            - "instagram": Search Instagram only
            - "reddit": Search Reddit only
            - "linkedin": Search LinkedIn only
            - "x": Search X (formerly Twitter) only
            - "combined": Search all platforms (default)
        include_raw_content: If True, extracts full content from result URLs.
            This uses advanced extraction and populates the raw_content field.
            Default is False.
        max_results: Maximum number of results to return (default: 5)
        include_answer: Include AI-generated answer in response
        include_images: Include images in the response
        time_range: Time range to search (day, week, month, year)
        max_retries: Maximum number of retry attempts per request (default: 3)
    
    Returns:
        Dictionary containing search results with optional raw content and observability
    
    Raises:
        ValueError: If an invalid platform is specified
    
    Example:
        >>> from tools.social_media import social_media_search
        >>> 
        >>> # Search Reddit for AI discussions
        >>> results = social_media_search(
        ...     query="artificial intelligence trends",
        ...     api_key="tvly-YOUR_API_KEY",
        ...     platform="reddit",
        ...     include_raw_content=True,
        ...     max_results=10
        ... )
        >>> 
        >>> # Search all platforms
        >>> results = social_media_search(
        ...     query="climate change",
        ...     api_key="tvly-YOUR_API_KEY",
        ...     platform="combined",
        ...     time_range="week"
        ... )
    """
    start_time = time.perf_counter()
    usage = ToolUsageStats()
    
    # Initialize Tavily client
    tavily_client = TavilyClient(api_key=api_key)
    
    # Determine which domains to include
    if platform == "combined":
        include_domains = list(PLATFORM_DOMAINS.values())
    elif platform in PLATFORM_DOMAINS:
        include_domains = [PLATFORM_DOMAINS[platform]]
    else:
        raise ValueError(
            f"Invalid platform '{platform}'. Must be one of: "
            f"{', '.join(list(PLATFORM_DOMAINS.keys()) + ['combined'])}"
        )
    
    # Prepare search parameters
    search_params = {
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
        "include_domains": include_domains,
        "include_raw_content": False,  # Always False; we handle this manually
        "include_answer": include_answer,
    }
    

    if time_range:
        search_params["time_range"] = time_range
        
    # Execute the search with retry logic
    search_response = search_with_retry(tavily_client, max_retries, **search_params)
    usage.tavily.add_search(search_response.credits, search_response.response_time)
    response = search_response.data
    
    # Early return if no raw content needed or no results
    results = response.get("results")
    if not include_raw_content or not results:
        usage.response_time = time.perf_counter() - start_time
        response["usage"] = usage.to_dict()
        return response
    
    # Extract URLs in single pass
    urls = [r["url"] for r in results if "url" in r]
    if not urls:
        usage.response_time = time.perf_counter() - start_time
        response["usage"] = usage.to_dict()
        return response
    
    # Extract content from all URLs with retry logic
    try:
        extract_response = extract_with_retry(
            tavily_client, max_retries,
            urls=urls,
            extract_depth="advanced",
            include_images=include_images
        )
        usage.tavily.add_extract(extract_response.credits, extract_response.response_time)
        extracted_data = extract_response.data
        
        # Build combined mapping in single pass using dict comprehension
        url_data = {
            item["url"]: (item.get("raw_content"), item.get("images", []))
            for item in extracted_data.get("results", [])
            if "url" in item
        }
        
        # Populate fields with tuple unpacking
        for result in results:
            content, images = url_data.get(result.get("url"), (None, []))
            result["raw_content"] = content
            result["images"] = images
                
    except Exception as e:
        # If extraction fails, add error info but still return search results
        response["extraction_error"] = str(e)
        for result in results:
            result["raw_content"] = None
    
    usage.response_time = time.perf_counter() - start_time
    response["usage"] = usage.to_dict()
    
    return response

