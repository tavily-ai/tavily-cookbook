# Extract API Reference

## Table of Contents

- [Extraction Strategies](#extraction-strategies)
  - [One-Step Extraction](#one-step-extraction)
  - [Two-Step Extraction](#two-step-extraction-recommended)
- [Key Parameters](#key-parameters)
- [Notes](#notes)


---

## Extraction Strategies

### One-Step Extraction

Enable `include_raw_content=True` during search to get content immediately. Simple but may extract irrelevant content.

```python
from tavily import TavilyClient

client = TavilyClient()

response = client.search(
    query="Python async best practices",
    include_raw_content=True,
    max_results=10
)

for result in response["results"]:
    print(f"Content length: {len(result.get('raw_content', ''))}")
```

### Two-Step Extraction (Recommended)

Search first, filter by relevance, then extract from selected URLs. More control and accuracy.
Can extract up to 20 URLs in one extract call.

```python
# Step 1: Search
search_results = client.search(
    query="Python async best practices",
    max_results=10
)

# Step 2: Filter by relevance score
relevant_urls = [
    r["url"] for r in search_results["results"]
    if r["score"] > 0.5
]

# Step 3: Extract from filtered URLs
extracted = client.extract(urls=relevant_urls)

for item in extracted["results"]:
    print(f"URL: {item['url']}")
    print(f"Content: {item['raw_content'][:500]}...")
```

---

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `urls` | string/array | Required | Single URL or list of URLs to extract (max 20) |
| `extract_depth` | enum | `"basic"` | `"basic"` or `"advanced"` (for complex/JS-rendered pages) |
| `query` | string | null | Reranks extracted chunks based on relevance to this query |
| `chunks_per_source` | integer | 3 | Number of content chunks per source (1-5, max 500 chars each) |
| `format` | enum | `"markdown"` | Output format: `"markdown"` or `"text"` |
| `include_images` | boolean | false | Include image URLs from extracted pages |
| `include_favicon` | boolean | false | Include favicon URL for each result |
| `timeout` | float | varies | Max wait time in seconds (1.0-60.0) |


## Notes

- **Token optimization**: Use `query` and `chunks_per_source` to reduce the number of tokens returned. This is ideal for LLM agents and context engineering where you want only the most relevant content.

- **Timeout tuning**: If latency is not a concern, set `timeout` to the maximum (60.0) to increase the chances of a successful scrape on slower or complex pages.


- **Advanced Mode** Use `extract_depth="advanced"` for complex pages:

**Best for:**
- Dynamic content with JavaScript
- Embedded media and tables
- Complex page structures
- When precision is critical

- **Fallback strategy**: If `extract_depth="basic"` fails for a URL, retry with `extract_depth="advanced"` which handles more complex page structures and dynamically rendered content.

---

## Response Fields

**Top-level response:**

| Field | Description |
|-------|-------------|
| `results` | Array of successfully extracted content |
| `failed_results` | Array of URLs that failed extraction |
| `response_time` | Time in seconds to complete the request |

**Each result object:**

| Field | Description |
|-------|-------------|
| `url` | The URL from which content was extracted |
| `raw_content` | Full extracted content. When `query` is provided, contains top-ranked chunks joined by `[...]` |
| `images` | Array of image URLs (if `include_images=true`) |
| `favicon` | Favicon URL (if `include_favicon=true`) |

**Each failed_results object:**

| Field | Description |
|-------|-------------|
| `url` | The URL that failed extraction |
| `error` | Error message describing why extraction failed |

For more details, please read the [full API reference](https://docs.tavily.com/documentation/api-reference/endpoint/extract)