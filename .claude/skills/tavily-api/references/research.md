# Research API Reference

## Table of Contents

- [Overview](#overview)
- [Basic Usage](#basic-usage)
- [Key Parameters](#key-parameters)
- [Response Fields](#response-fields)
- [Streaming](#streaming)
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

## Streaming

Enable real-time monitoring of research progress using Server-Sent Events (SSE) by setting `stream=True`.

### Basic Streaming Usage

```python
from tavily import TavilyClient

client = TavilyClient()

stream = client.research(
    input="Latest developments in quantum computing",
    model="pro",
    stream=True
)

for chunk in stream:
    print(chunk.decode('utf-8'))
```

### Event Structure

All streaming events follow this format:

```json
{
  "id": "unique-identifier",
  "object": "chat.completion.chunk",
  "model": "mini|pro",
  "created": 1705329000,
  "choices": [{"delta": {...}}]
}
```

### Streaming Event Types

| Event Type | Description |
|------------|-------------|
| **Tool Call** | Emitted when agent initiates actions (Planning, WebSearch, etc.) |
| **Tool Response** | Returned after tool execution with results and sources |
| **Content** | Research report streamed as markdown chunks (or structured objects with `output_schema`) |
| **Sources** | Complete list of all sources used, emitted after content |
| **Done** | Signals completion (`event: done`) |

### Tool Types

| Tool | Description | Models |
|------|-------------|--------|
| `Planning` | Initializes research strategy | mini, pro |
| `WebSearch` | Executes web searches | mini, pro |
| `Generating` | Creates final report | mini, pro |
| `ResearchSubtopic` | Deep research on subtopics | pro only |

### Typical Research Flow

1. `Planning` tool_call → tool_response
2. `WebSearch` tool_call (with queries array) → tool_response (with sources)
3. `ResearchSubtopic` cycles (Pro mode only)
4. `Generating` tool_call → tool_response
5. `Content` event chunks (markdown or structured JSON)
6. `Sources` event (complete source list)
7. `Done` event

### Streaming with Structured Output

When `output_schema` is provided, content arrives as structured objects:

```python
schema = {
    "properties": {
        "summary": {"type": "string", "description": "Executive summary"},
        "key_points": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["summary", "key_points"]
}

stream = client.research(
    input="AI agent frameworks comparison",
    model="mini",
    stream=True,
    output_schema=schema
)

for chunk in stream:
    data = chunk.decode('utf-8')
    # Content chunks will be structured JSON matching schema
    print(data)
```



## Best Practices

**Use streaming for UX** — Display progress to users during long research tasks to reduce perceived latency
**Be specific in topics** — More focused research queries yield more relevant results
**Use structured output** — Define schemas for consistent, parseable responses in production applications


For more examples, see the [Tavily Cookbook](https://github.com/tavily-ai/tavily-cookbook/tree/main/research).
