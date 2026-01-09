# Tools

Your agent is only as good as the actions it can take. In research, that means retrieving the right information at the right time. These tools handle the common retrieval patterns—searching, crawling, extracting—so your agent can focus on reasoning while we handle the complexity.

Each tool combines Tavily endpoints with context engineering: formatting results for LLMs, managing token limits, deduplicating sources, and cleaning web noise.

| Tool | When to Use |
|------|-------------|
| `search_and_answer` | Answer questions with web research + LLM synthesis |
| `search_dedup` | Run multiple queries in parallel, deduplicate results |
| `crawl_and_summarize` | Extract and summarize entire websites |
| `extract_and_summarize` | Get focused summaries from specific URLs |
| `social_media_search` | Search Reddit, X, LinkedIn, TikTok, etc. |

---

## `search_and_answer`

Answer a question using web research. Optionally generates subqueries for comprehensive coverage, handles token limits, and synthesizes an answer. Bring your own model—we handle the prompts and context engineering.

**Key parameters:** `query`, `model_config`, `max_number_of_subqueries` (2-4), `output_schema`, `token_limit` (default 50k), `threshold` (default 0.3), `topic` ("general"/"news"/"finance"), `time_range`, `include_domains`, `exclude_domains`

```python
from tools.search_and_answer import search_and_answer
from models import ModelConfig, ModelObject

result = await search_and_answer(
    query="What are the pros and cons of Rust vs Go?",
    api_key="tvly-xxx",
    model_config=ModelConfig(model=ModelObject(model="anthropic:claude-sonnet-4-5")),
    max_number_of_subqueries=3,
)
print(result["answer"])
```

---

## `search_dedup`

Run multiple search queries in parallel and consolidate results. Deduplicates by URL and merges content chunks from the same source.

**Key parameters:** `queries`, `search_depth` ("advanced"), `topic`, `max_results` (5), `chunks_per_source` (3), `time_range`, `include_domains`, `exclude_domains`

```python
from tools.async_search_and_dedup import search_dedup

results = await search_dedup(
    api_key="tvly-xxx",
    queries=[
        "transformer architecture explained",
        "attention mechanism neural networks",
        "BERT GPT comparison",
    ],
    search_depth="advanced",
    max_results=5,
)

for r in results["results"]:
    print(f"{r['title']}: {r['score']}")
```

---

## `crawl_and_summarize`

Crawl an entire website and summarize the content. Bring your own model for the summarization—useful for documentation sites, knowledge bases, or product catalogs.

**Key parameters:** `url`, `model_config`, `instructions`, `output_schema`, `max_depth` (1-5), `max_breadth` (20), `limit` (50), `select_paths`, `exclude_paths`

```python
from tools.crawl_and_summarize import crawl_and_summarize
from models import ModelConfig, ModelObject

result = await crawl_and_summarize(
    url="https://docs.example.com",
    model_config=ModelConfig(model=ModelObject(model="anthropic:claude-sonnet-4-20250514")),
    instructions="Extract all API endpoints and their parameters",
    max_depth=2,
    select_paths=["/docs/.*", "/api/.*"],
)
print(result["summary"])
```

---

## `extract_and_summarize`

Extract content from specific URLs and summarize with your model. Use when you already know which pages have the information.

**Key parameters:** `urls` (max 20), `model_config`, `query` (focuses extraction), `output_schema`, `chunks_per_source` (5), `extract_depth` ("basic"/"advanced")

```python
from tools.extract_and_summarize import extract_and_summarize
from models import ModelConfig, ModelObject

result = await extract_and_summarize(
    urls=["https://en.wikipedia.org/wiki/Artificial_intelligence"],
    model_config=ModelConfig(model=ModelObject(model="groq:llama-3.3-70b-versatile")),
    query="What are the main ethical concerns with AI?",
    chunks_per_source=5,
)
print(result["results"][0]["summary"])
```

---

## `social_media_search`

Search specific social platforms for discussions and content.

**Key parameters:** `query`, `platform` ("reddit"/"x"/"linkedin"/"tiktok"/"instagram"/"facebook"/"combined"), `include_raw_content`, `max_results` (5), `time_range`

```python
from tools.social_media import social_media_search

results = social_media_search(
    query="best practices for LLM fine-tuning",
    api_key="tvly-xxx",
    platform="reddit",
    max_results=10,
    time_range="month",
)
```

---

## Quick Reference

| Scenario | Tool |
|----------|------|
| "Answer this question with web research" | `search_and_answer` |
| "Research this topic from multiple angles" | `search_dedup` |
| "What does this website say?" | `crawl_and_summarize` |
| "Summarize these specific pages" | `extract_and_summarize` |
| "What are people saying on Reddit/Twitter?" | `social_media_search` |

## Model Configuration

See the [main README](../README.MD#model-configuration) for `ModelConfig` details and [supported providers](https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model).
