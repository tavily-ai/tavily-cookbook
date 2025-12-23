# Crawl & Map API Reference

## Table of Contents

- [Crawl vs Map](#crawl-vs-map)
- [Basic Usage](#basic-usage)
  - [Crawl](#basic-crawl)
  - [Map](#basic-map)
- [Key Parameters](#key-parameters)
- [Path Filtering](#path-filtering)
- [Map then Extract Pattern](#map-then-extract-pattern)
- [Performance Optimization](#performance-optimization)
- [Response Fields](#response-fields)

---

## Crawl vs Map

| Use Case | API | Why |
|----------|-----|-----|
| Full content extraction | **Crawl** | Returns page content |
| Site structure discovery | **Map** | Returns URLs only, faster, less tokens for LLM |
| URL collection | **Map** | No content overhead |
| RAG integration | **Crawl** | Need actual content |

---

## Basic Usage

### Basic Crawl

```python
from tavily import TavilyClient

client = TavilyClient()

response = client.crawl(
    url="https://docs.example.com"
)

for page in response["results"]:
    print(f"{page['url']}: {len(page['raw_content'])} chars")
```

### Basic Map

```python
response = client.map(
    url="https://docs.example.com"
)

for url in response["urls"]:
    print(url)
```

---

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | string | Required | The root URL to begin the crawl |
| `max_depth` | integer | 1 | How deep to follow links (1-5). Start with 1-2, increase gradually |
| `max_breadth` | integer | 20 | Links followed per level. 50-100 for focused crawls |
| `limit` | integer | 50 | Total pages to crawl before stopping |
| `chunks_per_source` | integer | 3 | Content chunks per page (1-5, max 500 chars each). Reduces tokens for LLM context |
| `extract_depth` | enum | `"basic"` | `"basic"` (1 credit/5 URLs) or `"advanced"` (2 credits/5 URLs) |
| `format` | enum | `"markdown"` | Output format: `"markdown"` or `"text"` |
| `select_paths` | array | null | Regex patterns to include specific URL paths |
| `exclude_paths` | array | null | Regex patterns to exclude specific URL paths |
| `select_domains` | array | null | Regex patterns to include specific domains |
| `exclude_domains` | array | null | Regex patterns to exclude specific domains |
| `allow_external` | boolean | true | Include or exclude external domain links |
| `include_images` | boolean | false | Include images in results |
| `include_favicon` | boolean | false | Include favicon URL for each result |
| `instructions` | string | null | Natural language guidance for crawler (2 credits/10 pages) |
| `timeout` | float | 150 | Max wait time in seconds (10-150) |

**Notes:**

- **Token efficiency**: Use `chunks_per_source` and `instructions` to optimize token usage for LLM agents. `chunks_per_source` limits content per page to only the most relevant snippets, while `instructions` guides the crawler to focus on specific content types, reducing irrelevant data in your context window.

---

## Path Filtering

Focus crawling on specific sections using regex patterns:

```python
# Only crawl documentation and API reference
response = client.crawl(
    url="https://docs.example.com",
    max_depth=2,
    select_paths=["/docs/.*", "/api/.*", "/reference/.*"],
    exclude_paths=["/blog/.*", "/changelog/.*"]
)
```

**Domain control:**

```python
response = client.crawl(
    url="https://example.com",
    select_domains=["example.com", "docs.example.com"],
    exclude_domains=["ads.example.com"]
)
```


---

## Map then Extract Pattern

Discover structure first, then extract strategically:

```python
# Step 1: Map to discover structure
map_result = client.map(
    url="https://docs.example.com",
    instructions="find all api docs and guides"
    max_depth=2
)

# Step 2: Filter discovered URLs
api_docs = [url for url in map_result["urls"] if "/api/" in url]
guides = [url for url in map_result["urls"] if "/guides/" in url]
print(f"Found {len(api_docs)} API docs, {len(guides)} guides")

# Step 3: Extract content from filtered URLs
target_urls = api_docs + guides
response = client.extract(
    urls=target_urls[:20],  # Max 20 URLs per extract call
    extract_depth="advanced"
)

for item in response["results"]:
    print(f"{item['url']}: {len(item['raw_content'])} chars")
```

---

## Performance Optimization

### Depth Impact

Each depth level increases crawl time exponentially:

| Depth | Typical Pages | Time |
|-------|---------------|------|
| 1 | 10-50 | Seconds |
| 2 | 50-500 | Minutes |
| 3 | 500-5000 | Many minutes |

## Response Fields

### Crawl Response

| Field | Description |
|-------|-------------|
| `results` | List of crawled pages |
| `results[].url` | Page URL |
| `results[].raw_content` | Extracted content |

### Map Response

| Field | Description |
|-------|-------------|
| `urls` | List of discovered URLs |

For more details, please read the [full API reference](https://docs.tavily.com/documentation/api-reference/endpoint/crawl)