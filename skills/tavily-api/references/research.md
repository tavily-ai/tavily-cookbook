# Research API Reference

## Table of Contents

- [Overview](#overview)
- [Basic Usage](#basic-usage)
- [Key Parameters](#key-parameters)
- [Response Fields](#response-fields)
- [Structured Output](#structured-output)
- [Best Practices](#best-practices)

---

## Overview

The Research API conducts comprehensive research on any topic with automatic source gathering, analysis, and response generation with citations.


---

## Basic Usage

Research tasks are two-step: initiate with `research()`, retrieve with `get_research()`.

```python
from tavily import TavilyClient

client = TavilyClient()

# Step 1: Start research task
result = client.research(
    input="Latest developments in quantum computing and their practical applications",
)
request_id = result["request_id"]

response = tavily_client.get_research(request_id)

# Poll until the research is completed
while response["status"] not in ["completed", "failed"]:
    print(f"Status: {response['status']}... polling again in 10 seconds")
    time.sleep(10)
    response = tavily_client.get_research(request_id)

# Check if the research completed successfully
if response["status"] == "failed":
    raise RuntimeError(f"Research failed: {response.get('error', 'Unknown error')}")

report = response["content"]

```

---

## Key Parameters

### research()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | Required | The research topic or question |
| `model` | enum | `"auto"` | `"mini"` (targeted research), `"pro"` (comprehensive multi-angle analysis), `"auto"` |
| `stream` | boolean | false | Enable streaming responses |
| `output_schema` | object | null | JSON Schema defining structured response format |
| `citation_format` | enum | `"numbered"` | `"numbered"`, `"mla"`, `"apa"`, `"chicago"` |

### get_research()

| Parameter | Type | Description |
|-----------|------|-------------|
| `request_id` | string | Task ID from `research()` response |



---

## Response Fields

### research() Response

| Field | Description |
|-------|-------------|
| `request_id` | Unique identifier for tracking the task |
| `created_at` | Timestamp when the research task was created |
| `status` | Initial status of the research task |
| `input` | The research topic or question submitted |
| `model` | The model used by the research agent |

### get_research() Response

| Field | Description |
|-------|-------------|
| `status` | Task status: `"pending"`, `"processing"`, `"completed"`, `"failed"` |
| `content` | The generated research report (when completed) |
| `sources` | Array of source citations used in the report |
| `response_time` | Time in seconds to complete the request |

### Source Object

| Field | Description |
|-------|-------------|
| `url` | Source URL |
| `title` | Source title |
| `citation` | Formatted citation string |

---


## Structured Output

Use `output_schema` to receive research results in a predefined JSON structure:

```python
schema = {
    "properties": {
        "summary": {
            "type": "string",
            "description": "Executive summary of findings"
        },
        "key_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Main takeaways from the research"
        },
        "metrics": {
            "type": "object",
            "properties": {
                "market_size": {"type": "string", "description": "Total market size"},
                "growth_rate": {"type": "number", "description": "Annual growth percentage"}
            }
        }
    },
    "required": ["summary", "key_points"]
}

response = client.research(
    input="Electric vehicle market analysis 2024",
    output_schema=schema
)
```

**Schema requirements:**
- Every property needs both `type` and `description`
- Supported types: `object`, `string`, `integer`, `number`, `array`
- Use `required` arrays to enforce mandatory fields at any nesting level

---

## Best Practices

**Use streaming for UX** — Display progress to users during long research tasks to reduce perceived latency
**Be specific in topics** — More focused research queries yield more relevant results
**Use structured output** — Define schemas for consistent, parseable responses in production applications


For more examples, see the [Tavily Cookbook](https://github.com/tavily-ai/tavily-cookbook/tree/main/research).
