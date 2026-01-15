# Agents

## Contents

- [Why Hybrid Research?](#why-hybrid-research)
- [`hybrid_research`](#hybrid_research) — API reference
- [Modes](#modes) — Fast vs Multi-Agent
- [Structured Output](#structured-output)
- [Custom Synthesis](#custom-synthesis)
- [Data Enrichment](#data-enrichment) — Store web findings internally
- [Implementing Your RAG Function](#implementing-your-rag-function)
- [Deep Internal Research](#deep-internal-research-with-langchain-deep-agents) — LangChain Deep Agents option

---

## Why Hybrid Research?

The most powerful production agents combine internal knowledge with real-time web data.

**Your internal data is your edge.** It's the proprietary context that makes your agent unique—customer records, product specs, historical decisions, domain expertise. This is what lets your agent solve *your* business problems, not generic ones.

**But internal data is never complete.** Markets shift, competitors launch products, regulations change, and your knowledge base can't keep up. That's where the web comes in—it bridges the gap between what you know internally and the 100% coverage your agent needs to answer confidently.

The hybrid approach gives you:
- **Grounded answers** rooted in your proprietary data
- **Complete coverage** with real-time web context
- **Enrichment opportunities** by storing relevant web findings back into your knowledge base

> **Want deeper internal research?** For multi-agent mode, you can use [LangChain Deep Agents](#deep-internal-research-with-langchain-deep-agents) to iteratively explore your internal data before Tavily handles the web portion.

---

## `hybrid_research`

Combines your internal RAG with web research to produce comprehensive reports.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | str | Required | Tavily API key |
| `query` | str | Required | The research question |
| `model_config` | ModelConfig | Required | Your LLM configuration |
| `internal_rag_function` | Callable[[str], str] | Required | Function that queries your internal data |
| `mode` | "fast" \| "multi_agent" | "fast" | Research mode (see below) |
| `output_schema` | OutputSchema | None | Pydantic model for structured output |
| `research_synthesis_prompt` | str | None | Custom instructions for synthesis |

### Returns

```python
{
    "report": str | BaseModel,  # Synthesized report or structured output
    "web_sources": [            # Sources used from web research
        {"title": "...", "url": "..."},
        ...
    ]
}
```

---

## Modes

### Fast Mode

Best for: Quick answers, lower latency, cost-sensitive applications.

```
                                    ┌─────────────────────┐
                                    │   Generate Web      │
                         ┌─────────▶│   Subqueries        │
                         │          │   (based on gaps)   │
                         │          └──────────┬──────────┘
                         │                     │
                         │                     ▼
┌─────────┐    ┌─────────┴─────────┐    ┌─────────────────────┐
│  Query  │───▶│   Internal RAG    │    │  Parallel Search    │
└─────────┘    │   (your data)     │    │  + Deduplication    │
               └─────────┬─────────┘    └──────────┬──────────┘
                         │                         │
                         │    ┌────────────────────┘
                         │    │
                         ▼    ▼
               ┌───────────────────────┐
               │      Synthesize       │
               │   Internal + Web      │───▶  Report or structure output + Sources
               └───────────────────────┘
```

```python
from tavily_agent_toolkit import hybrid_research, ModelConfig, ModelObject

# Your RAG function - returns relevant context from your knowledge base
def my_rag(query: str) -> str:
    # Query your vector store, database, or documents
    results = vector_store.similarity_search(query, k=5)
    return "\n".join([doc.page_content for doc in results])

result = await hybrid_research(
    api_key="tvly-xxx",
    query="What's our competitor's current pricing strategy?",
    model_config=ModelConfig(model=ModelObject(model="openai:gpt-5.2")),
    internal_rag_function=my_rag,
    mode="fast",
)

print(result["report"])
print(f"Sources: {len(result['web_sources'])} web pages")
```

### Multi-Agent Mode

Best for: Comprehensive research, complex topics, when accuracy matters more than speed.

This mode uses Tavily's deep research endpoint—a **multi-agent system** that orchestrates sub-agents to iteratively search, extract, and analyze. It's optimized for context engineering: pruning irrelevant data, preserving high-value findings, and returning grounded reports with complete coverage.

```
┌─────────┐    ┌───────────────────────┐
│  Query  │───▶│   AGENT 1: Internal   │
└─────────┘    │   RAG (your data)     │
               └───────────┬───────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │   Identify Gaps       │
               │   (LLM analysis)      │
               └───────────┬───────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │   AGENT 2: Tavily     │
               │   Deep Research       │
               │   ┌─────────────────┐ │
               │   │ Sub-agents:     │ │
               │   │ search, extract │ │
               │   │ analyze, iterate│ │
               │   └─────────────────┘ │
               │   Context engineered  │
               │   for max relevance   │
               └───────────┬───────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │      Synthesize       │
               │   Internal + Deep     │───▶  Report or structure output + Sources
               │   Research Findings   │
               └───────────────────────┘
```

```python
result = await hybrid_research(
    api_key="tvly-xxx",
    query="Full competitive analysis of the AI search market",
    model_config=ModelConfig(model=ModelObject(model="anthropic:claude-sonnet-4-20250514")),
    internal_rag_function=my_rag,
    mode="multi_agent",  # Uses Tavily's deep research endpoint
)
```

---

## Structured Output

Use `output_schema` to get consistent, parseable results:

```python
from pydantic import Field
from models import OutputSchema

class CompetitorAnalysis(OutputSchema):
    company_name: str = Field(description="Name of the competitor")
    products: list[str] = Field(description="Main products or services")
    pricing: str = Field(description="Pricing strategy or model")
    strengths: list[str] = Field(description="Key competitive strengths")
    weaknesses: list[str] = Field(description="Known weaknesses or gaps")

result = await hybrid_research(
    api_key="tvly-xxx",
    query="Analyze Perplexity as a competitor",
    model_config=ModelConfig(model=ModelObject(model="groq:openai/gpt-oss-120b")),
    internal_rag_function=my_rag,
    mode="fast",
    output_schema=CompetitorAnalysis,
)

analysis = CompetitorAnalysis.model_validate_json(result["report"])
print(f"Strengths: {analysis.strengths}")
```

---

## Custom Synthesis

Guide how the report is structured with `research_synthesis_prompt`:

```python
result = await hybrid_research(
    api_key="tvly-xxx",
    query="What are the latest developments in AI agents?",
    model_config=ModelConfig(model=ModelObject(model="groq:llama-3.3-70b-versatile")),
    internal_rag_function=my_rag,
    mode="fast",
    research_synthesis_prompt="""
    Structure the report as:
    1. Executive Summary (2-3 sentences)
    2. Key Developments (bullet points)
    3. Impact on Our Product (specific recommendations)
    4. Sources
    
    Keep it under 500 words. Focus on actionable insights.
    """,
)
```

---

## Data Enrichment

When your agent hits the web to fill gaps, that data is relevant to your users—otherwise the agent wouldn't have needed it. This creates a flywheel:

1. Agent queries internal data → finds gaps
2. Agent searches web to fill gaps
3. Web results get synthesized into the answer
4. **Store those web results internally** for future queries

Over time, your knowledge base grows with exactly the information your users need. The `web_sources` returned by `hybrid_research` make this easy:

```python
result = await hybrid_research(...)

# Store web findings for future queries
for source in result["web_sources"]:
    store_in_knowledge_base(
        url=source["url"],
        title=source["title"],
        content=fetch_content(source["url"]),  # Optional: extract full content
        query_context=original_query,
    )
```

---

## Implementing Your RAG Function

The `internal_rag_function` is simple: take a query, return relevant context as a string.

```python
def my_rag(query: str) -> str:
    # Query your internal data source (vector store, database, API, files, etc.)
    results = your_retrieval_method(query)
    
    # Format as a string - include source attribution if available
    return "\n\n".join([
        f"Source: {r.source}\n{r.content}" 
        for r in results
    ])
```

**Tips:**
- Return 3-10 relevant chunks—enough context without overwhelming
- Include source metadata (file names, URLs, doc IDs) for traceability
- The hybrid researcher handles the rest: gap detection, web search, synthesis

---

## Deep Internal Research with LangChain Deep Agents

For multi-agent mode, you can upgrade the internal research step from a simple RAG call to a full iterative agent using [LangChain Deep Agents](https://docs.langchain.com/oss/python/deepagents/quickstart).

Instead of a single retrieval, a Deep Agent can autonomously plan, query multiple internal sources, and iterate until it has comprehensive internal context—*before* handing off to Tavily for web research.

```python
from deepagents import create_deep_agent

# Your internal tools
def search_docs(query: str) -> str:
    """Search internal documentation"""
    return vector_store.search(query)

def query_metrics(sql: str) -> str:
    """Query internal metrics database"""
    return db.execute(sql)

# Create internal research agent
internal_agent = create_deep_agent(
    tools=[search_docs, query_metrics],
    system_prompt="Gather all relevant internal context for the query."
)

# Use as your internal_rag_function
def deep_internal_rag(query: str) -> str:
    result = internal_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

# Now use with hybrid_research - Tavily handles the web portion
result = await hybrid_research(
    api_key="tvly-xxx",
    query="Analyze our Q3 performance vs competitors",
    model_config=model_config,
    internal_rag_function=deep_internal_rag,  # Deep internal research
    mode="multi_agent",  # Tavily deep research for web
)
```

This gives you iterative depth on *both* sides: Deep Agents for internal, Tavily for web.
