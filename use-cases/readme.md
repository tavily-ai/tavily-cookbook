# Tavily Use Cases

Production-ready agent implementations demonstrating practical applications of Tavily's web intelligence APIs. These examples combine multiple Tavily tools (search, crawl, extract) with LLM agents to solve real-world problems. 

Use them as starter code to get up and running quickly, or deploy them in your applications.

## Prerequisites

### API Keys

Set these environment variables (or create a `.env` file in the repository root):

```bash
export TAVILY_API_KEY="your-tavily-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

### Installation

```bash
cd use-cases
pip install -r requirements.txt
```

---

## Available Use Cases

### 1. Conversational Chatbot

**File:** `chatbot.py`

An intelligent chatbot that dynamically routes between quick web searches and deep research based on query complexity.

#### Features

- **Smart tool selection**: Uses lightweight search for simple factual questions, deep research for complex topics
- **Multi-turn conversation**: Maintains context across conversation turns
- **Citation support**: Provides numbered citations linking to sources

#### Tools Used

| Tool | When Used |
|------|-----------|
| `search_and_answer` | Simple, factual questions (e.g., "What is the capital of France?") |
| `research` | Complex queries requiring multiple sources and synthesis |

#### How It Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER QUERY                                     │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLM AGENT                                         │
│                    (Analyzes query complexity)                              │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
        Simple Question?                  Complex Question?
        (single fact, quick lookup)       (analysis, comparison, trends)
                 │                                 │
                 ▼                                 ▼
┌─────────────────────────────┐     ┌─────────────────────────────┐
│     search_and_answer       │     │         research            │
│  ┌───────────────────────┐  │     │  ┌───────────────────────┐  │
│  │   Tavily Search API   │  │     │  │  Tavily Research API  │  │
│  │   (5 web results)     │  │     │  │  (deep multi-source)  │  │
│  └───────────────────────┘  │     │  └───────────────────────┘  │
│             │               │     │             │               │
│             ▼               │     │             ▼               │
│  ┌───────────────────────┐  │     │  ┌───────────────────────┐  │
│  │   LLM Synthesizes     │  │     │  │   Streaming Report    │  │
│  │   Quick Answer        │  │     │  │   Generation          │  │
│  └───────────────────────┘  │     │  └───────────────────────┘  │
└──────────────┬──────────────┘     └──────────────┬──────────────┘
               │                                   │
               │ Can call multiple times           │ One-shot only
               │ until satisfied                   │
               │                                   │
               └─────────────┬─────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │      Enough Information?     │
              └──────────────┬───────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              No                           Yes
              │                             │
              ▼                             ▼
   ┌──────────────────┐         ┌───────────────────────────┐
   │  Call search_    │         │   Generate Final Response │
   │  and_answer      │         │   with Citations [1][2]   │
   │  again           │         └─────────────┬─────────────┘
   └────────┬─────────┘                       │
            │                                 ▼
            └──────────────────►   ┌─────────────────────────┐
                                   │      USER RESPONSE      │
                                   │  (Answer + Sources)     │
                                   └─────────────────────────┘
```

**Key behavior:**
- The agent can call `search_and_answer` **multiple times** to gather enough facts
- The agent can only call `research` **once** per query (it's comprehensive but expensive)
- All responses include numbered citations linking to sources

#### Usage

```bash
python chatbot.py
```

```
Chatbot ready! Type 'quit' to exit.

You: What are the latest developments in quantum computing?
Assistant: [Researches and provides comprehensive answer with citations]
```

---

### 2. Company Intelligence Agent

**File:** `company_intelligence_deep_agent.py`

A ReAct agent that conducts comprehensive research on companies by combining website crawling with web search.

#### Features

- **Website crawling**: Discovers and summarizes pages from a company's website
- **Targeted extraction**: Extracts specific information from known URLs
- **External research**: Searches for news, funding, reviews, and competitive intelligence
- **Streaming output**: Shows tool calls as they happen for transparency

#### Tools Used

| Tool | Purpose |
|------|---------|
| `crawl_company_website` | Crawl and summarize company website pages |
| `extract_from_urls` | Extract detailed content from specific URLs |
| `tavily_search` | Search the web for external information |

#### Usage

```bash
python company_intelligence_deep_agent.py
```

```
============================================================
Company Intelligence Research Agent
============================================================

Enter company name: Anthropic
Enter website URL: https://anthropic.com
Enter research focus (or press Enter for general research): leadership team and recent funding

Researching: Anthropic
Website: https://anthropic.com
Focus: leadership team and recent funding
------------------------------------------------------------

🔧 Calling: crawl_company_website
✅ crawl_company_website completed
🔧 Calling: tavily_search
✅ tavily_search completed

============================================================
RESEARCH REPORT
============================================================

[Comprehensive report with citations]
```

#### Example Research Topics

- Company overview and products
- Leadership team and organizational structure
- Recent funding rounds and investors
- Competitive landscape
- Customer reviews and reputation

---

### 3. Social Media Research Agent

**File:** `social_media_research.py`

A general-purpose agent for researching any topic across social media platforms.

#### Features

- **Multi-platform search**: Searches TikTok, Instagram, Reddit, X, Facebook, and LinkedIn
- **Flexible queries**: Research any topic—products, trends, opinions, events, people
- **Platform-specific targeting**: Agent can choose to search specific platforms (e.g., Reddit for honest reviews, TikTok for trends)
- **Streaming output**: Shows tool calls as they happen for transparency
- **Interactive CLI**: Continuous conversation loop for multiple queries

#### Tools Used

| Tool | Platforms |
|------|-----------|
| `search_social_media` | TikTok, Instagram, Reddit, X, Facebook, LinkedIn, or combined |

The agent can customize search parameters including:
- `platform` - Target a specific platform or search all
- `max_results` - Control how many results to fetch
- `time_range` - Filter by day, week, month, or year

#### Usage

```bash
python social_media_research.py
```

```
============================================================
Social Media Research Agent
============================================================

This agent searches across TikTok, Reddit, Instagram, X,
Facebook, and LinkedIn to research any topic.

Type 'quit' or 'exit' to end the session.

What would you like to research?
> What are people saying about the new iPhone?
------------------------------------------------------------
Researching...

🔧 Calling: search_social_media
✅ search_social_media completed
🔧 Calling: search_social_media
✅ search_social_media completed

============================================================
REPORT
============================================================

[Comprehensive report with insights and citations]
```

#### Example Research Topics

- Product reviews and sentiment ("What do people think of the Dyson Airwrap?")
- Trending discussions ("What's viral on TikTok this week?")
- Public opinion ("How do people feel about remote work in 2025?")
- Event reactions ("What are people saying about the Super Bowl?")
- Travel recommendations ("Best hiking spots according to Reddit?")

---

## Architecture

These use cases are built on top of the `solution-patterns/` library, which provides:

- **Tools**: Reusable Tavily tool wrappers (`search_and_answer`, `crawl_and_summarize`, `extract_and_summarize`, `social_media_search`)
- **Utilities**: Summarization, result formatting
- **Models**: Pydantic models for type-safe configuration

The `use-cases/utils.py` file contains shared utilities like `stream_agent_response` for streaming agent execution with progress indicators.

```
use-cases/
├── chatbot.py                      # Conversational agent
├── company_intelligence_deep_agent.py  # Company research agent
├── social_media_research.py        # Social media research agent
├── utils.py                        # Shared utilities
├── requirements.txt
└── README.md

solution-patterns/
├── tools/                          # Tavily tool wrappers
├── utilities/                      # Shared utilities
└── models.py                       # Data models
```

## Extending These Examples

1. **Add new tools**: Create wrappers in `solution-patterns/tools/` and import them into your agent
2. **Customize prompts**: Modify the system prompts to change agent behavior
3. **Change models**: Update `ModelConfig` to use different LLMs (Claude, Gemini, etc.)
4. **Add memory**: Integrate vector stores for long-term conversation memory
