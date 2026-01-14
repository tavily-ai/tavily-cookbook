# Tavily Agent Toolkit

Build production-grade research agents with patterns we've battle-tested across customer deployments and our deep research endpoint.

## What This Library Provides

**Tools** — Optimized configurations of Tavily API calls for common research patterns. We handle the context engineering (formatting, deduplication, token management, summarization) so your agent gets clean, relevant data.

**Bring Your Own Model** — Every tool that needs an LLM accepts a `ModelConfig`. We support 20+ providers via [LangChain](https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model) with automatic fallback chains. You control the model, we bring our prompts and context engineering strategies.

**Agents** — Pre-built research strategies that combine internal knowledge with web research. Two modes: fast or deep multi-agent research.

## Project Structure

```
agent-toolkit/
├── tools/           # Research primitives
├── agents/          # Pre-built research agents
├── utilities/       # Context engineering utilities
├── models.py        # Type definitions
└── tests/           # Test suite
├── evals/           # Evaluation code and examples for your research agents (coming soon...)
```

## Tools

Research primitives that combine Tavily endpoints with context engineering. Each tool handles a specific retrieval pattern so your agent can focus on reasoning while we handle the complexity of getting clean, relevant data.

| Tool | Use Case |
|------|----------|
| `search_and_answer` | Answer questions with web research + LLM synthesis |
| `search_dedup` | Run multiple queries in parallel, deduplicate results |
| `crawl_and_summarize` | Extract and summarize entire websites |
| `extract_and_summarize` | Get focused summaries from specific URLs |
| `social_media_search` | Search Reddit, X, LinkedIn, TikTok, etc. |

# Example: Answer a question with comprehensive web research
```bash
pip install git+https://github.com/tavily-ai/tavily-cookbook.git#subdirectory=agent-toolkit
```

```python

from tavily_agent_toolkit import social_media_search
from models import ModelConfig, ModelObject

result = await social_media_search(
    query="Who is Leo Messi?",
    api_key="tvly-xxx",
)
print(result["result"])
```

> 📖 See [`tools/README.md`](tools/README.md) for detailed documentation, parameters, and examples.

## Agents

### `hybrid_research`

**Use case:** Combine your internal knowledge base with real-time web research.

You provide a RAG function that queries your internal data. The agent identifies gaps and fills them with web research.

**Fast mode:**
1. Query your internal RAG
2. Generate subqueries based on what's missing
3. Parallel web search with deduplication
4. Synthesize everything into a report

**Multi-agent mode:**
1. Query your internal RAG
2. LLM identifies knowledge gaps
3. Tavily's deep research endpoint fills those gaps
4. Synthesize into a comprehensive report

```python
from agents.hybrid_researcher import hybrid_research
from models import ModelConfig, ModelObject

result = await hybrid_research(
    api_key="tvly-xxx",
    query="What is our competitor's pricing strategy?",
    model_config=ModelConfig(model=ModelObject(model="groq:openai/gpt-oss-120b")),
    internal_rag_function=my_rag_function,  # Your RAG
    mode="fast",  # or "multi_agent" for deep research
    output_schema=CompetitorAnalysis,  # Optional structured output
)
```

> 📖 See [`agents/README.MD`](agents/README.MD) for full documentation.

## Utilities

Helper functions that power the tools and agents. A few are reusable on their own: `handle_research_stream`, `clean_raw_content`, `format_web_results`, and `ainvoke_with_fallback` for model cascades.

> 📖 See [`utilities/README.md`](utilities/README.md) for details.

## Installation

**Option 1: Install directly** (recommended for quick start)

```bash
pip install git+https://github.com/tavily-ai/tavily-cookbook.git#subdirectory=agent-toolkit
```

Then import and use:

```python
from tavily_agent_toolkit import search_and_answer, crawl_and_summarize, hybrid_research
```

**What's included in the install:**

| Category | Exports |
|----------|---------|
| **Tools** | `search_and_answer`, `search_and_format`, `search_dedup`, `crawl_and_summarize`, `extract_and_summarize`, `social_media_search` |
| **Agents** | `hybrid_research` |
| **Utilities** | `handle_research_stream`, `format_web_results`, `clean_raw_content`, `ainvoke_with_fallback`, and more |
| **Models** | `ModelConfig`, `ModelObject`, `OutputSchema`, and various TypedDicts |

We'll do our best to keep these tools up to date as patterns evolve and the Tavily API improves. Check back for updates or watch the repo.

---

**Option 2: Copy the code** (recommended for customization)

Clone the repo and copy `agent-toolkit/` into your project. Modify the tools and prompts to fit your needs.

---

For LLM features, install your preferred provider:

```bash
pip install langchain-openai       # OpenAI
pip install langchain-anthropic    # Anthropic  
pip install langchain-google-genai # Google
pip install langchain-groq         # Groq
# See models.py for all 20+ supported providers
```

## Model Configuration

All tools accept a `ModelConfig` for LLM operations. Use the `"provider:model"` format:

```python
from models import ModelConfig, ModelObject

config = ModelConfig(
    model=ModelObject(model="openai:gpt-5.2"),
    fallback_models=[  # Optional fallback chain
        ModelObject(model="anthropic:claude-sonnet-4-20250514"),
        ModelObject(model="groq:llama-3.3-70b-versatile"),
    ],
    temperature=0.7,
)
```

**20+ providers supported** via LangChain's [`init_chat_model`](https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model): OpenAI, Anthropic, Google, Groq, Mistral, Cohere, Together, Fireworks, AWS Bedrock, Azure, and more.

## Tests

The `tests/` directory contains integration tests for all tools and agents. These tests are a great way to see what the tools look like when they run and understand expected inputs/outputs. With AI coding assistants like Claude Code and Cursor getting better, test-driven development is more accessible and fast than ever—use these as a starting point.

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_search_and_answer.py
```

## License

See [LICENSE](../LICENSE) for details.
