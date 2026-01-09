import asyncio
import re
import time
from typing import (Any, Callable, Optional, Sequence, Type, TypeVar, Union,
                    cast)

import tiktoken
from langchain.chat_models import init_chat_model
from models import (LLMResponse, LLMUsage, ModelConfig, SearchResult,
                    TavilyAPIResponse)
from pydantic import BaseModel

T = TypeVar("T")


async def async_retry(
    func: Callable[..., Any],
    max_retries: int = 1,
    *args: Any,
    **kwargs: Any,
) -> TavilyAPIResponse:
    """Execute an async function with retry logic and exponential backoff.
    
    Args:
        func: Async callable to execute
        max_retries: Maximum number of retry attempts (default: 3)
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func
        
    Returns:
        TavilyAPIResponse with data, timing, and credits
    """
    # Ensure include_usage is always True for observability
    kwargs["include_usage"] = True
    
    retry_count = 0
    while True:
        try:
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            credits = result.get("usage", {}).get("credits", 0) if isinstance(result, dict) else 0
            return TavilyAPIResponse(data=result, response_time=elapsed, credits=credits)
        except asyncio.TimeoutError:
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                await asyncio.sleep(wait_time)
                retry_count += 1
            else:
                return TavilyAPIResponse(
                    data={"results": [], "error": "Request timed out after multiple retries"},
                    response_time=0.0,
                    credits=0
                )
        except Exception as e:
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                await asyncio.sleep(wait_time)
                retry_count += 1
            else:
                return TavilyAPIResponse(
                    data={"results": [], "error": str(e)},
                    response_time=0.0,
                    credits=0
                )


def search_with_retry(
    client: Any,
    max_retries: int = 1,
    **kwargs: Any,
) -> TavilyAPIResponse:
    """Execute client.search() with retry logic and exponential backoff.
    
    Args:
        client: TavilyClient instance
        max_retries: Maximum number of retry attempts (default: 1)
        **kwargs: Keyword arguments to pass to client.search()
        
    Returns:
        TavilyAPIResponse with data, timing, and credits
    """
    # Ensure include_usage is always True for observability
    kwargs["include_usage"] = True
    
    retry_count = 0
    while True:
        try:
            start_time = time.perf_counter()
            result = client.search(**kwargs)
            elapsed = time.perf_counter() - start_time
            credits = result.get("usage", {}).get("credits", 0) if isinstance(result, dict) else 0
            return TavilyAPIResponse(data=result, response_time=elapsed, credits=credits)
        except TimeoutError:
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                time.sleep(wait_time)
                retry_count += 1
            else:
                return TavilyAPIResponse(
                    data={"results": [], "error": "Search timed out after multiple retries"},
                    response_time=0.0,
                    credits=0
                )
        except Exception as e:
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                time.sleep(wait_time)
                retry_count += 1
            else:
                return TavilyAPIResponse(
                    data={"results": [], "error": str(e)},
                    response_time=0.0,
                    credits=0
                )


def extract_with_retry(
    client: Any,
    max_retries: int = 1,
    **kwargs: Any,
) -> TavilyAPIResponse:
    """Execute client.extract() with retry logic and exponential backoff.
    
    Args:
        client: TavilyClient instance
        max_retries: Maximum number of retry attempts (default: 1)
        **kwargs: Keyword arguments to pass to client.extract()
        
    Returns:
        TavilyAPIResponse with data, timing, and credits
    """
    # Ensure include_usage is always True for observability
    kwargs["include_usage"] = True
    
    retry_count = 0
    while True:
        try:
            start_time = time.perf_counter()
            result = client.extract(**kwargs)
            elapsed = time.perf_counter() - start_time
            credits = result.get("usage", {}).get("credits", 0) if isinstance(result, dict) else 0
            return TavilyAPIResponse(data=result, response_time=elapsed, credits=credits)
        except TimeoutError:
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                time.sleep(wait_time)
                retry_count += 1
            else:
                return TavilyAPIResponse(
                    data={"results": [], "error": "Extract timed out after multiple retries"},
                    response_time=0.0,
                    credits=0
                )
        except Exception as e:
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                time.sleep(wait_time)
                retry_count += 1
            else:
                return TavilyAPIResponse(
                    data={"results": [], "error": str(e)},
                    response_time=0.0,
                    credits=0
                )


def crawl_with_retry(
    client: Any,
    max_retries: int = 1,
    **kwargs: Any,
) -> TavilyAPIResponse:
    """Execute client.crawl() with retry logic and exponential backoff.
    
    Args:
        client: TavilyClient instance
        max_retries: Maximum number of retry attempts (default: 1)
        **kwargs: Keyword arguments to pass to client.crawl()
        
    Returns:
        TavilyAPIResponse with data, timing, and credits
    """
    # Ensure include_usage is always True for observability
    kwargs["include_usage"] = True
    
    retry_count = 0
    while True:
        try:
            start_time = time.perf_counter()
            result = client.crawl(**kwargs)
            elapsed = time.perf_counter() - start_time
            credits = result.get("usage", {}).get("credits", 0) if isinstance(result, dict) else 0
            return TavilyAPIResponse(data=result, response_time=elapsed, credits=credits)
        except TimeoutError:
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                time.sleep(wait_time)
                retry_count += 1
            else:
                return TavilyAPIResponse(
                    data={"results": [], "error": "Crawl timed out after multiple retries"},
                    response_time=0.0,
                    credits=0
                )
        except Exception as e:
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                time.sleep(wait_time)
                retry_count += 1
            else:
                return TavilyAPIResponse(
                    data={"results": [], "error": str(e)},
                    response_time=0.0,
                    credits=0
                )


async def ainvoke_with_fallback(
    model_config: ModelConfig,
    messages: Union[list[dict], str],
    output_schema: Optional[Type[BaseModel]] = None,
    return_usage: bool = False,
    **invoke_kwargs: Any
) -> Any:
    """Invoke a model with fallback cascade support.
    
    Tries the primary model, and if it fails, tries each fallback model in order.
    
    Retry behavior:
    - If fallback_models is provided: each model gets 1 attempt before moving to next
    - If no fallback_models: primary model gets 1 retry (2 attempts total)
    
    Args:
        model_config: ModelConfig with primary model and optional fallback_models
        messages: Messages to pass to ainvoke (list of dicts or string prompt)
        output_schema: Optional Pydantic model for structured output
        return_usage: If True, returns LLMResponse with usage metadata
        **invoke_kwargs: Additional kwargs to pass to ainvoke (e.g., max_tokens)
        
    Returns:
        If return_usage=False: The model response (AIMessage or structured output)
        If return_usage=True: LLMResponse with result and usage metadata
        
    Raises:
        Exception: If all models fail, raises the last exception encountered
    """
    all_models = model_config.get_all_models()
    has_fallbacks = len(all_models) > 1
    last_error: Optional[Exception] = None
    
    for i, model_obj in enumerate(all_models):
        # Determine retry count: only retry if no fallbacks and this is primary model
        max_attempts = 1 if has_fallbacks else 2
        
        for attempt in range(max_attempts):
            try:
                # Initialize model (use fallback ModelObject if not primary)
                model_to_use = model_obj if i > 0 else None
                llm = init_chat_model(**model_config.to_init_kwargs(model_to_use))
                
                start_time = time.perf_counter()
                
                # Apply structured output if requested
                if output_schema:
                    # Use include_raw=True to get both parsed result and raw message for usage
                    llm_structured = llm.with_structured_output(output_schema, include_raw=True)
                    raw_result = await llm_structured.ainvoke(messages, **invoke_kwargs)
                    elapsed = time.perf_counter() - start_time
                    
                    # Extract parsed result and usage from raw response
                    parsed = raw_result.get("parsed") if isinstance(raw_result, dict) else raw_result
                    raw_message = raw_result.get("raw") if isinstance(raw_result, dict) else None
                    
                    if return_usage:
                        usage = _extract_llm_usage(raw_message, elapsed)
                        return LLMResponse(result=parsed, usage=usage)
                    return parsed
                else:
                    # Regular invocation
                    result = await llm.ainvoke(messages, **invoke_kwargs)
                    elapsed = time.perf_counter() - start_time
                    
                    if return_usage:
                        usage = _extract_llm_usage(result, elapsed)
                        return LLMResponse(result=result, usage=usage)
                    return result
                
            except Exception as e:
                last_error = e
                # If we have more attempts, wait before retry
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
                # Otherwise, we'll try next model in the cascade (if any)
    
    # All models failed - raise the last error
    if last_error:
        raise last_error
    raise RuntimeError("No models available to invoke")


def _extract_llm_usage(message: Any, response_time: float) -> LLMUsage:
    """Extract token usage from an AIMessage or similar response."""
    usage = LLMUsage()
    usage.llm_call_count = 1
    usage.llm_response_time = response_time
    
    if message is None:
        return usage
    
    # Try to get usage_metadata (standard LangChain format)
    usage_metadata = getattr(message, "usage_metadata", None)
    if usage_metadata:
        usage.total_input_tokens = usage_metadata.get("input_tokens", 0)
        usage.total_output_tokens = usage_metadata.get("output_tokens", 0)
        return usage
    
    # Try response_metadata as fallback (some providers use this)
    response_metadata = getattr(message, "response_metadata", {})
    if response_metadata:
        # OpenAI format
        if "token_usage" in response_metadata:
            token_usage = response_metadata["token_usage"]
            usage.total_input_tokens = token_usage.get("prompt_tokens", 0)
            usage.total_output_tokens = token_usage.get("completion_tokens", 0)
        # Anthropic format
        elif "usage" in response_metadata:
            resp_usage = response_metadata["usage"]
            usage.total_input_tokens = resp_usage.get("input_tokens", 0)
            usage.total_output_tokens = resp_usage.get("output_tokens", 0)
    
    return usage


def clean_raw_content(content: str) -> str:
    """
    Clean raw web content by removing common web noise patterns including
    markdown artifacts, navigation elements, and boilerplate content.
    """
    if not content:
        return content
    
    cleaned = content
    
    # === MARKDOWN CLEANUP ===
    
    # Remove markdown image references: ![alt text](url) or ![Image N: description](url)
    cleaned = re.sub(r'!\[(?:Image\s*\d*:?\s*)?[^\]]*\]\([^)]+\)', '', cleaned)
    
    # Convert markdown links to just text: [text](url) -> text
    # But remove navigation-style links entirely (short text that's just menu items)
    def replace_link(match):
        text = match.group(1)
        # Remove if it looks like navigation (very short or common nav terms)
        nav_terms = ['home', 'menu', 'search', 'sign in', 'sign out', 'subscribe', 
                     'newsletter', 'view', 'more', 'skip', 'rss', 'premium', 'forums',
                     'contact', 'about', 'privacy', 'terms', 'cookies', 'advertise',
                     'careers', 'us edition', 'uk edition', 'au edition', 'ca edition']
        if len(text) < 4 or text.lower().strip() in nav_terms:
            return ''
        return text
    cleaned = re.sub(r'\[([^\]]*)\]\([^)]+\)', replace_link, cleaned)
    
    # Remove bare URLs (http/https links not in markdown format)
    cleaned = re.sub(r'https?://[^\s\)\]]+', '', cleaned)
    
    # Remove HTML comments
    cleaned = re.sub(r'<!--.*?-->', '', cleaned, flags=re.DOTALL)
    
    # Remove checkbox markers
    cleaned = re.sub(r'- \[[ x]\]\s*', '', cleaned)
    
    # === NAVIGATION AND BOILERPLATE PATTERNS ===
    
    # Remove common navigation/boilerplate lines
    boilerplate_patterns = [
        r'^Open menu\s*$',
        r'^Close\s*$',
        r'^Search\s+Search\s+.*$',
        r'^Sign in\s*$',
        r'^Sign out\s*$',
        r'^View Profile\s*$',
        r'^Subscribe\s*$',
        r'^Newsletter\s*$',
        r'^RSS\s*$',
        r'^Premium\s*$',
        r'^Forums?\s*$',
        r'^Advertisement\s*$',
        r'^Sponsored\s*$',
        r'^Trending\s*$',
        r'^Popular\s*$',
        r'^Related\s*$',
        r'^Share\s*$',
        r'^Comments?\s*\(\d*\)\s*$',
        r'^See all comments.*$',
        r'^Show more comments\s*$',
        r'^View All \d+ Comments\s*$',
        r'^\d+ Comments?\s*$',
        r'^Comment from the forums\s*$',
        r'^Reply\s*$',
        r'^Read more\s*$',
        r'^Load more\s*$',
        r'^Show more\s*$',
        r'^View\s+\w+\s*$',  # "View CPUs", "View News", etc.
        r'^Skip to .*content.*$',
        r'^Jump to.*$',
        r'^Back to top\s*$',
        r'^Table of contents\s*$',
        r'^On this page\s*$',
        r'^In this article\s*$',
        r'^TOPICS?\s*$',
        r'^TAGS?\s*$',
        r'^Latest Videos?.*$',
        r'^Latest in .*$',
        r'^Latest News\s*$',
        r'^Don\'t miss.*$',
        r'^You may (?:also )?like\s*$',
        r'^Recommended\s*$',
        r'^More from.*$',
        r'^See also\s*$',
        r'^.*Edition.*flag of.*$',
        r'^Follow.*on Google News.*$',
        r'^Get .* Newsletter\s*$',
        r'^Stay On the Cutting Edge.*$',
        r'^By submitting your information.*$',
        r'^Contact me with news.*$',
        r'^Receive email from us.*$',
        r'^Terms and conditions\s*$',
        r'^Privacy policy\s*$',
        r'^Cookies? policy\s*$',
        r'^Accessibility Statement\s*$',
        r'^Advertise with us\s*$',
        r'^About us\s*$',
        r'^Careers\s*$',
        r'^Do not sell.*personal information.*$',
        r'^©.*Full.*Floor.*$',  # Copyright with address
        r'^©\s*\d{4}.*$',
        r'^All rights reserved.*$',
        r'^When you purchase through links.*$',
        r'^We may earn.*affiliate.*$',
        r'^Here\'s how it works.*$',
        r'^Keyboard Shortcuts.*$',
        r'^Press shift question mark.*$',
        r'^Shortcuts Open/Close.*$',
        r'^\s*Play/Pause\s+SPACE\s*$',
        r'^\s*Increase Volume.*$',
        r'^\s*Decrease Volume.*$',
        r'^\s*Seek Forward.*$',
        r'^\s*Seek Backward.*$',
        r'^\s*Captions On/Off.*$',
        r'^\s*Fullscreen.*$',
        r'^\s*Mute/Unmute.*$',
        r'^\s*Next Up\s*$',
        r'^\s*More Videos\s*$',
        r'^\s*PLAY SOUND\s*$',
        r'^\s*Live\s*$',
        r'^\s*\d{2}:\d{2}\s*$',  # Timestamps like 01:35
        r'^Add as a preferred source.*$',
        r'^.*part of Future.*Inc.*$',
        r'^Visit our corporate site.*$',
        r'^Contact Future\'s experts.*$',
    ]
    
    for pattern in boilerplate_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove lines that are just social media platform names or share buttons
    social_patterns = [
        r'^\s*facebook\s*$',
        r'^\s*twitter\s*$',
        r'^\s*x\s*$',
        r'^\s*instagram\s*$',
        r'^\s*youtube\s*$',
        r'^\s*linkedin\s*$',
        r'^\s*reddit\s*$',
        r'^\s*pinterest\s*$',
        r'^\s*whatsapp\s*$',
        r'^\s*flipboard\s*$',
        r'^\s*email\s*$',
        r'^\s*link\s*$',
        r'^\s*copied\s*$',
        r'^\s*share\s*$',
    ]
    for pattern in social_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
    
    # === LIST AND MENU CLEANUP ===
    
    # Remove empty list items (just bullets with no content)
    cleaned = re.sub(r'^\s*[\*\-\+]\s*$', '', cleaned, flags=re.MULTILINE)
    
    # Remove lines that are just numbers (often from lists/navigation)
    cleaned = re.sub(r'^\s*\d+\.?\s*$', '', cleaned, flags=re.MULTILINE)
    
    # Remove breadcrumb-style navigation (numbered items 1. 2. at start)
    cleaned = re.sub(r'^\d+\.\s+$', '', cleaned, flags=re.MULTILINE)
    
    # === REPEATED SEPARATOR CLEANUP ===
    
    # Collapse multiple dashes, equals, underscores
    cleaned = re.sub(r'-{3,}', '--', cleaned)
    cleaned = re.sub(r'={3,}', '==', cleaned)
    cleaned = re.sub(r'_{3,}', '__', cleaned)
    cleaned = re.sub(r'\*{3,}', '**', cleaned)
    
    # Collapse multiple hash marks
    cleaned = re.sub(r'#{3,}', '##', cleaned)
    
    # === WHITESPACE CLEANUP ===
    
    # Remove lines with only whitespace and punctuation
    cleaned = re.sub(r'^\s*[\|\-\*\+\>\<\#\=\_]+\s*$', '', cleaned, flags=re.MULTILINE)
    
    # Remove empty table patterns
    cleaned = re.sub(r'\|\s*\|', '|', cleaned)
    cleaned = re.sub(r'\n\s*\|\s*\n', '\n', cleaned)
    
    # Remove standalone ellipsis markers
    cleaned = re.sub(r'\s*\[\.\.\.\]\s*', ' ', cleaned)
    cleaned = re.sub(r'\s*\[…\]\s*', ' ', cleaned)
    cleaned = re.sub(r'\s*\.\.\.\s*', ' ', cleaned)
    
    # Remove excessive whitespace
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    cleaned = re.sub(r'\t+', ' ', cleaned)
    
    # Collapse multiple newlines (more than 2) to 2
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    # Remove lines that are just whitespace
    cleaned = re.sub(r'^\s+$', '', cleaned, flags=re.MULTILINE)
    
    # Strip leading/trailing whitespace from each line
    cleaned = '\n'.join(line.strip() for line in cleaned.split('\n'))
    
    # Final collapse of multiple newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned.strip()


def clean_formatted_output(formatted_output: str) -> str:
    cleaned = re.sub(r'-{3,}', '--', formatted_output)
    cleaned = re.sub(r'={3,}', '==', cleaned)
    cleaned = re.sub(r'_{3,}', '__', cleaned)
            
    # Replace multiple newlines with 1
    cleaned = re.sub(r'\n{2,}', '\n', cleaned)
            
    # Remove standalone [...] ellipsis markers (but keep content)
    cleaned = re.sub(r'\s*\[\.\.\.\]\s*', ' ', cleaned)
    cleaned = re.sub(r'\s*\[…\]\s*', ' ', cleaned)
            
    # Remove excessive whitespace
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    cleaned = re.sub(r'\t+', ' ', cleaned)

    cleaned = re.sub(r'#{2,}', '#', cleaned)
            
    # Remove common noise patterns
    cleaned = re.sub(r'\|\s*\|', '|', cleaned)  # Empty table cells
    cleaned = re.sub(r'\n\s*\|\s*\n', '\n', cleaned)  # Empty table rows
    return cleaned

def get_tiktoken_encoding(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str, model: str = "cl100k_base") -> int:
    encoding = get_tiktoken_encoding(model)
    return len(encoding.encode(text))

async def summarize_long_content(model_config: ModelConfig, content: str, query: str, token_limit: int = 40000) -> str:
    """Group sources into minimal chunks, summarize each with capped output tokens.
    
    Args:
        model_config: ModelConfig for the model to use (with optional fallbacks)
        content: The content to summarize
        query: The original query for context
        token_limit: Maximum tokens for the combined output
    """
    # Parse sources with metadata
    source_pattern = r'--- SOURCE \d+: (.*?) ---\nURL: (.*?)\n(.*?)(?=--- SOURCE \d+:|--- IMAGE \d+|$)'
    matches = re.findall(source_pattern, content, re.DOTALL)
    
    if not matches:
        return content[:token_limit * 3]
    
    # Calculate tokens per source
    model_name = model_config.model.model
    sources = [(title.strip(), url.strip(), body.strip(), count_tokens(body, model_name)) for title, url, body in matches]
    total_tokens = sum(s[3] for s in sources)
    
    # Determine number of chunks based on model context limits (~40k input per chunk)
    max_input_per_chunk = 40000
    num_chunks = max(1, (total_tokens + max_input_per_chunk - 1) // max_input_per_chunk)
    target_per_chunk = total_tokens // num_chunks
    
    # Group sources into chunks, keeping sources whole
    chunks = []
    current_chunk, current_tokens, current_sources = [], 0, []
    for title, url, body, tokens in sources:
        if current_tokens + tokens > target_per_chunk and current_chunk:
            chunks.append((current_chunk, current_sources))
            current_chunk, current_tokens, current_sources = [], 0, []
        current_chunk.append(body)
        current_tokens += tokens
        current_sources.append((title, url))
    if current_chunk:
        chunks.append((current_chunk, current_sources))
    
    # Calculate max tokens per summary so combined output fits under token_limit
    num_chunks = len(chunks)
    max_tokens_per_summary = token_limit // num_chunks
    
    # Summarize each chunk with capped output
    summaries = []
    for bodies, source_list in chunks:
        chunk_content = "\n\n---\n\n".join(bodies)
        prompt = f"""Summarize these search results for the query: {query}
Preserve all key facts. Be concise but thorough.

{chunk_content}"""
        summary = await ainvoke_with_fallback(
            model_config,
            [{"role": "user", "content": prompt}],
            max_tokens=int(max_tokens_per_summary)
        )
        # Append actual sources
        sources_text = "\n".join(f"- [{t}]({u})" for t, u in source_list)
        summaries.append(f"{summary.content}\n\nSources:\n{sources_text}")
    
    return "\n\n---\n\n".join(summaries)

class SubqueriesOutput(BaseModel):
    """Output schema for subqueries generation."""
    subqueries: list[str]


async def generate_subqueries(
    query: str,
    model_config: ModelConfig,
    max_number_of_subqueries: int = 4,
    context: str | None = None,
    output_schema: Optional[Type[BaseModel]] = None,
    return_usage: bool = False,
) -> list[str] | tuple[list[str], LLMUsage]:
    """Generate subqueries to cover different subtopics of the main query.
    
    Args:
        query: The main query to generate subqueries for
        model_config: ModelConfig for the model to use (with optional fallbacks)
        max_number_of_subqueries: Maximum number of subqueries to generate
        context: Optional context to help generate better subqueries
        output_schema: Optional output schema for structured output
        return_usage: If True, returns tuple of (subqueries, LLMUsage)
    """
    prompt = f"""Generate up to {max_number_of_subqueries} short and directed Google-style search queries covering different subtopics to answer: {query}
Only generate as many queries as needed to cover the topic comprehensively.
Do not include dates or years in the queries unless explicitly specified in the original query."""
    messages: list[dict] = [{"role": "system", "content": prompt}]
    if context:
        messages.append({"role": "user", "content": f"Here is the context that you can use to generate the subqueries: {context}\n We want to generate subqueries that fill the gaps in the context for the query: {query}"})
    if output_schema:
        messages.append({"role": "user", "content": f"Your research goal is to fill out this schema: {output_schema.model_json_schema()}"})
    
    if return_usage:
        response = await ainvoke_with_fallback(model_config, messages, output_schema=SubqueriesOutput, return_usage=True)
        subqueries = cast(SubqueriesOutput, response.result).subqueries
        return subqueries, response.usage
    else:
        result = await ainvoke_with_fallback(model_config, messages, output_schema=SubqueriesOutput)
        return cast(SubqueriesOutput, result).subqueries


async def synthesize_results(
    query: str,
    model_config: ModelConfig,
    research_results: str,
    output_schema: Optional[Type[BaseModel]] = None,
    research_synthesis_prompt: Optional[str] = None,
    return_usage: bool = False,
) -> str | BaseModel | tuple[str | BaseModel, LLMUsage]:
    """Synthesize research results into a report or structured output.
    
    Args:
        query: The original query/task
        model_config: ModelConfig for the model to use (with optional fallbacks)
        research_results: The research results to synthesize
        output_schema: Optional Pydantic model for structured output
        research_synthesis_prompt: Optional instructions for synthesis
        return_usage: If True, returns tuple of (result, LLMUsage)
    """
    messages: list[dict] = []
    if research_synthesis_prompt:
            messages.append({"role": "user", "content": f"Here are some instructions for how to structure the synthesis of the research results: {research_synthesis_prompt}"})
    
    if output_schema:
        prompt = f"""You are a research assistant tasked with extracting and synthesizing information from research results.

            Task: {query}

            Research Results:
            {research_results}

            Instructions:
            - Carefully analyze the research results above
            - Extract relevant information to populate each field in the output schema
            - Only include information that is directly supported by the research results
            - If information for a field is not available in the results, use null or an empty value as appropriate
            - Be accurate and precise - do not fabricate or infer information not present in the sources
            - For any text fields, include source attribution where relevant"""
        messages.insert(0, {"role": "user", "content": prompt})
        
        if return_usage:
            response = await ainvoke_with_fallback(model_config, prompt, output_schema=output_schema, return_usage=True)
            return cast(BaseModel, response.result), response.usage
        else:
            result = await ainvoke_with_fallback(model_config, prompt, output_schema=output_schema)
            return cast(BaseModel, result)
    else:
        prompt = f"""Here is your task: {query}. Synthesize the following research results into a report: {research_results}.
    Output only the report, no other text or formatting. It should be in markdown format with a sources section at the end and in text citations throughout the report.
    Make sure all claims are supported by the sources."""
        messages.insert(0, {"role": "user", "content": prompt})
        
        if return_usage:
            response = await ainvoke_with_fallback(model_config, messages, return_usage=True)
            return cast(str, response.result.content), response.usage
        else:
            result = await ainvoke_with_fallback(model_config, messages)
            return cast(str, result.content)

def format_web_results(web_results: Sequence[SearchResult]) -> str:
    formatted_output = "Search results: \n\n"
    for i, item in enumerate(web_results):
        formatted_output += f"\n\n--- SOURCE {i+1}: {item['title']} ---\n"
        formatted_output += f"URL: {item['url']}\n\n"
        formatted_output += f"SUMMARY OF WEBPAGE:\n{item['content']}\n\n"
        formatted_output += "\n"
    
    return formatted_output