import time
from typing import (Any, Dict, List, Literal, Optional, Sequence, Type, Union,
                    cast)

from pydantic import BaseModel
from tavily import TavilyClient

from ..models import ModelConfig, ToolUsageStats
from ..utilities.utils import (ainvoke_with_fallback, clean_formatted_output,
                               count_tokens, format_web_results,
                               search_with_retry, summarize_long_content)
from .async_search_and_dedup import search_dedup


class SubqueriesOutput(BaseModel):
    subqueries: List[str]

async def search_and_answer(
    query: str,
    api_key: str,
    model_config: ModelConfig,
    token_limit: int = 50000,
    output_schema: Optional[Type[BaseModel]] = None,
    max_number_of_subqueries: Optional[int] = None,
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
) -> Dict[str, Any]:
    start_time = time.perf_counter()
    usage = ToolUsageStats()
    
    tavily_client = TavilyClient(api_key=api_key)

    search_params = {
        "query": query,
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

    # Execute the search (multiple if subqueries are generated)
    result: Dict[str, Any] = {}
    if max_number_of_subqueries is not None and max_number_of_subqueries > 1:
        subquery_prompt = f"""Generate up to {max_number_of_subqueries} short Google-style search queries covering different subtopics to answer: {query}
Only generate multiple queries if needed to cover the topic comprehensively.
Do not include dates or years in the queries unless explicitly specified in the original query."""
        if output_schema:
            schema_str = str(output_schema.model_json_schema())
            subquery_prompt = f"""Generate up to {max_number_of_subqueries} short Google-style search queries covering different subtopics to fill out this schema: {schema_str}

Query: {query}
Generate multiple queries specifically for the fields in the schema.
Do not include dates or years in the queries unless explicitly specified in the original query or schema."""
        
        subquery_response = await ainvoke_with_fallback(
            model_config, subquery_prompt, output_schema=SubqueriesOutput, return_usage=True
        )
        usage.llm.merge(subquery_response.usage)
        subqueries: list[str] = cast(SubqueriesOutput, subquery_response.result).subqueries
        
        if len(subqueries) == 1:
            search_response = search_with_retry(tavily_client, max_retries, **search_params)
            usage.tavily.add_search(search_response.credits, search_response.response_time)
            result = search_response.data
        else:
            # Remove 'query' from params since search_dedup uses 'queries' instead
            dedup_params = {k: v for k, v in search_params.items() if k != "query"}
            result = cast(Dict[str, Any], await search_dedup(api_key=api_key, queries=subqueries, max_retries=max_retries, **dedup_params))
            # Extract tavily_usage from search_dedup result
            if "tavily_usage" in result:
                tavily_usage_dict = result.pop("tavily_usage")
                usage.tavily.total_credits += tavily_usage_dict.get("total_credits", 0)
                usage.tavily.search_count += tavily_usage_dict.get("search_count", 0)
                usage.tavily.search_response_time += tavily_usage_dict.get("search_response_time", 0)
            if "response_time" in result:
                result.pop("response_time")  # Remove since we track overall time
    else:
        search_response = search_with_retry(tavily_client, max_retries, **search_params)
        usage.tavily.add_search(search_response.credits, search_response.response_time)
        result = search_response.data

    # Apply threshold filtering to results list
    results_list = result.get("results", [])
    results_list = [r for r in results_list if threshold is not None and r.get('score', 0) >= threshold]
    # Filter for 20 highest scoring results
    results_list = sorted(results_list, key=lambda x: x.get('score', 0), reverse=True)[:20]

    # Format the search results
    formatted_output = format_web_results(results_list)

    # Format the image results
    image_list = cast(List[Dict[str, Any]], result.get("images") or [])
    for i, image in enumerate(image_list):
        formatted_output += f"\n\n--- IMAGE {i+1} ---\n"
        formatted_output += f"URL: {image.get('url')}\n\n"
        formatted_output += f"DESCRIPTION: {image.get('description')}\n\n"
        formatted_output += "\n"
    
    cleaned_formatted_output = clean_formatted_output(formatted_output)

    # Summarize if content exceeds token limit
    # Note: summarize_long_content uses LLM internally but doesn't track usage
    # for simplicity - could be enhanced in future if needed
    if count_tokens(cleaned_formatted_output, model_config.model.model) > token_limit * 0.8:
        cleaned_formatted_output = await summarize_long_content(model_config, cleaned_formatted_output, query, token_limit)

    # Pass into the LLM to answer the question --> Slot it into the search results (structured output or unstructured)
    answer_prompt = f"""Answer/respond the following query: {query}. You have access to web search results below. Make sure everything you say is supported by the search results.
If insufficient data, say "I don't have enough information to answer that question." """
    if include_citations:
        answer_prompt += "\n\nCITATIONS: Use numbered in-text citations [1], [2], etc. to back up your claims. At the end, include a \"Sources:\" section with only the sources you actually cited (format: [number] Title - URL)."
    messages: list[dict] = [
        {"role": "system", "content": answer_prompt},
        {"role": "user", "content": cleaned_formatted_output},
    ]
    if image_list and image_list != []:
        messages.append({"role": "user", "content": "Here are the image descriptions that are related to the question: " + str(image_list)})

    messages.append({"role": "user", "content": "Output the answer with in-text citations and relevant sources at the bottom. No other text or formatting."})
    answer_response = await ainvoke_with_fallback(
        model_config, messages, output_schema=output_schema, return_usage=True
    )
    usage.llm.merge(answer_response.usage)
    
    answer = answer_response.result
    if not output_schema:
        answer = answer_response.result.content

    usage.response_time = time.perf_counter() - start_time

    return {
        **result,
        "answer": answer,
        "usage": usage.to_dict(),
    }