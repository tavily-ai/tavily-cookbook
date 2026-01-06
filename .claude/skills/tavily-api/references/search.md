# Search API Reference

## Table of Contents

- [Query Optimization](#query-optimization)
- [Key Parameters](#key-parameters)
- [Basic Usage](#basic-usage)
- [Domain Filtering](#domain-filtering)
- [Image Search](#image-search)
- [Async Patterns](#async-patterns)
- [Response Fields](#response-fields)
- [Post-Filtering Strategies](#post-filtering-strategies)
  - [Score-Based Filtering](#score-based-filtering)
  - [Regex Filtering](#regex-filtering)
  - [LLM Verification](#llm-verification)

---

## Query Optimization

Keep queries concise (just a few words). Break complex searches into smaller, focused sub-queries for better relevance.

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | Required | The search query to execute |
| `search_depth` | enum | `"basic"` | `"basic"` (1 snippet/URL) or `"advanced"` (multiple semantic snippets/URL, 2 credits) |
| `chunks_per_source` | integer | 3 | Max relevant chunks per source (1-3, advanced depth only) |
| `max_results` | integer | 5 | Maximum results to return (0-20) |
| `time_range` | enum | null | `"day"`, `"week"`, `"month"`, `"year"` |
| `start_date` | string | null | Results after this date (YYYY-MM-DD) |
| `end_date` | string | null | Results before this date (YYYY-MM-DD) |
| `include_domains` | array | [] | Domains to include (max 300) |
| `exclude_domains` | array | [] | Domains to exclude (max 150) |
| `include_answer` | bool/enum | false | `true`/`"basic"` for quick answer, `"advanced"` for detailed |
| `include_raw_content` | bool/enum | false | `true`/`"markdown"` or `"text"` for full page content |
| `include_images` | boolean | false | Include image search results |
| `include_image_descriptions` | boolean | false | Add AI-generated descriptions for images |
| `include_favicon` | boolean | false | Include favicon URL for each result |
| `country` | enum | null | Boost results from specific country (general topic only) |
| `auto_parameters` | boolean | false | Auto-configure parameters based on query intent |

**Notes:**

- **`include_answer`**: Only use this if the user explicitly does not want to bring their own LLM for inference. Most users bring their own model to customize prompts and model selection.

- **`search_depth="advanced"`**: Provides the best quality results at the cost of ~2s additional latency. For use cases that can tolerate added latency, using `search_depth="advanced"` with `chunks_per_source=5` yields significant performance gains in downstream tasks.


## Basic Usage

```python
from tavily import TavilyClient

client = TavilyClient()

response = client.search(
    query="latest developments in quantum computing",
    max_results=10,
    search_depth="advanced",
    chunks_per_source=5
)

for result in response["results"]:
    print(f"{result['title']}: {result['url']}")
    print(f"Score: {result['score']}")
```


## Domain Filtering

```python
# Only search trusted sources
response = client.search(
    query="machine learning best practices",
    include_domains=["arxiv.org", "github.com", "pytorch.org"],
)

# Exclude specific domains
response = client.search(
    query="openai product reviews",
    exclude_domains=["reddit.com", "quora.com"]
)
```



## Async Patterns

Leveraging the async client enables scaled search with higher breadth and reach by running multiple queries in parallel. This is the best practice for agentic systems where you need to gather comprehensive information quickly before passing it to a model for analysis.

```python
import asyncio
from tavily import AsyncTavilyClient

# Initialize Tavily client
tavily_client = AsyncTavilyClient("tvly-YOUR_API_KEY")

async def fetch_and_gather():
    queries = ["latest AI trends", "future of quantum computing"]

    # Perform search and continue even if one query fails (using return_exceptions=True)
    try:
        responses = await asyncio.gather(*(tavily_client.search(q) for q in queries), return_exceptions=True)

        # Handle responses and print
        for response in responses:
            if isinstance(response, Exception):
                print(f"Search query failed: {response}")
            else:
                print(response)

    except Exception as e:
        print(f"Error during search queries: {e}")

# Run the function
asyncio.run(fetch_and_gather())
```


## Response Fields

**Top-level response:**

| Field | Description |
|-------|-------------|
| `query` | The original search query |
| `answer` | AI-generated answer (if `include_answer` enabled) |
| `results` | Array of search result objects |
| `images` | Array of image results (if `include_images=True`) |

**Each result object:**

| Field | Description |
|-------|-------------|
| `title` | Page title |
| `url` | Source URL |
| `content` | Extracted text snippet(s) |
| `score` | Semantic relevance score (0-1) |
| `raw_content` | Full page content (if `include_raw_content` enabled) |
| `favicon` | Favicon URL (if `include_favicon=True`) |

**Each image object (if `include_images=True`):**

| Field | Description |
|-------|-------------|
| `url` | Image URL |
| `description` | AI-generated description (if `include_image_descriptions=True`) |

---

## Post-Filtering Strategies

Since Tavily provides raw web data, you have full configurability to implement filtering and post-processing to meet your specific requirements.

The `score` field measures query relevance, but doesn't guarantee the result matches specific criteria (e.g., correct person, exact product, specific company). Use post-filtering to validate results against strict requirements.

### Score-Based Filtering

Simple threshold filtering based on relevance score:

```python
results = response["results"]

# Filter by score threshold
high_quality = [r for r in results if r["score"] > 0.7]

# Sort by score
sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

# Top N above threshold
top_relevant = sorted(
    [r for r in results if r["score"] > 0.5],
    key=lambda x: x["score"],
    reverse=True
)[:3]
```

**Limitation:** Score indicates relevance to query, not accuracy of match to specific criteria.

### Regex Filtering

Fast, deterministic filtering using pattern matching. Use for:
- URL pattern validation
- Required keywords/phrases
- Structural requirements

```python
import re

def regex_filter(result, criteria: dict) -> dict:
    """
    Filter a search result using regex checks.

    Args:
        result: Search result dict with url, content, title, raw_content
        criteria: Dict with patterns to match:
            - url_pattern: Regex for URL validation
            - required_terms: List of terms that must appear in content
            - excluded_terms: List of terms that must NOT appear

    Returns:
        dict with check results and validity
    """
    url = result.get("url", "")
    content = result.get("content", "") or ""
    title = result.get("title", "") or ""
    raw_content = result.get("raw_content", "") or ""

    full_text = f"{content} {title} {raw_content}".lower()

    checks = {}

    # URL pattern check
    if "url_pattern" in criteria:
        checks["url_valid"] = bool(re.search(criteria["url_pattern"], url.lower()))

    # Required terms check
    if "required_terms" in criteria:
        checks["required_found"] = all(
            re.search(re.escape(term.lower()), full_text)
            for term in criteria["required_terms"]
        )

    # Excluded terms check
    if "excluded_terms" in criteria:
        checks["excluded_absent"] = not any(
            re.search(re.escape(term.lower()), full_text)
            for term in criteria["excluded_terms"]
        )

    # Valid if all checks pass
    is_valid = all(checks.values()) if checks else True

    return {"checks": checks, "is_valid": is_valid, "url": url}
```

**Example: LinkedIn Profile Search**

```python
criteria = {
    "url_pattern": r"linkedin\.com/in/",  # Profile URL, not company page
    "required_terms": ["Jane Smith", "Acme Corp"],
    "excluded_terms": ["job posting", "careers"]
}

for result in response["results"]:
    validation = regex_filter(result, criteria)
    if validation["is_valid"]:
        print(f"Valid: {validation['url']}")
```

**Example: GitHub Repository Search**

```python
criteria = {
    "url_pattern": r"github\.com/[\w-]+/[\w-]+$",  # Repo URL, not file
    "required_terms": ["MIT License"],
    "excluded_terms": ["archived", "deprecated"]
}
```

### LLM Verification

Semantic validation using an LLM. Use for:
- Synonym/abbreviation matching ("FDE" = "Forward Deployed Engineer")
- Context-aware validation
- Confidence scoring with reasoning

```python
from openai import OpenAI
import json

def llm_verify(result, target_description: str, validation_criteria: list[str]) -> dict:
    """
    Use LLM to verify if a search result matches target criteria.

    Args:
        result: Search result dict
        target_description: What you're looking for
        validation_criteria: List of criteria to check

    Returns:
        dict with is_match, confidence (high/medium/low), reasoning
    """
    content = result.get("content", "") or ""
    title = result.get("title", "") or ""
    url = result.get("url", "")

    criteria_text = "\n".join(f"- {c}" for c in validation_criteria)

    prompt = f"""Verify if this search result matches the target.

Target: {target_description}

Validation Criteria:
{criteria_text}

Search Result:
URL: {url}
Title: {title}
Content: {content}

Does this result match ALL criteria?

Respond with JSON only:
{{"is_match": true/false, "confidence": "high/medium/low", "reasoning": "brief explanation"}}"""

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)
```

**Example: Profile Verification**

```python
result = llm_verify(
    result=search_result,
    target_description="Jane Smith, Software Engineer at Acme Corp",
    validation_criteria=[
        "Name matches Jane Smith",
        "Currently works at Acme Corp (or recently)",
        "Role is software engineering related",
        "Professional customer-facing experience"
    ]
)

if result["is_match"] and result["confidence"] in ["high", "medium"]:
    print(f"Verified: {result['reasoning']}")
```

For more details, please read the [full API reference](https://docs.tavily.com/documentation/api-reference/endpoint/search)