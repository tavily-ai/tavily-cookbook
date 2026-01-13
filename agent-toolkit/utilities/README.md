# Utilities

Helper functions that power the tools and agents. Most are internal, but a few are useful on their own.

---

## `handle_research_stream`

Parse and display streaming responses from Tavily's Research API. Handles SSE events, tool calls, and content chunks.

```python
from utilities.research_stream import handle_research_stream

response = client.research(query="...", stream=True)
report = handle_research_stream(response, verbose=True)
```

---

## `clean_raw_content`

Remove web noise from raw HTML/markdown content—navigation elements, boilerplate, social buttons, markdown artifacts.

```python
from utilities.utils import clean_raw_content

cleaned = clean_raw_content(raw_webpage_content)
```

---

## `format_web_results`

Format search results into a structured string optimized for LLM consumption.

```python
from utilities.utils import format_web_results

formatted = format_web_results(search_results)
```

---

## `ainvoke_with_fallback`

**Model cascades are critical for LLM-heavy workflows.** This function handles automatic fallback when models fail—rate limits, outages, or errors trigger the next model in the chain.

```python
from utilities.utils import ainvoke_with_fallback
from models import ModelConfig, ModelObject

config = ModelConfig(
    model=ModelObject(model="gpt-4o"),
    fallback_models=[
        ModelObject(model="claude-sonnet-4-20250514"),
        ModelObject(model="gemini-2.0-flash"),
    ],
)

# If gpt-4o fails, automatically tries Claude, then Gemini
result = await ainvoke_with_fallback(config, messages)
```

**Retry behavior:**
- **With fallbacks**: Each model gets 1 attempt before moving to next
- **Without fallbacks**: Primary model gets 1 retry (2 attempts total)
