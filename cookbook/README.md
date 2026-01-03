# Tavily Cookbook

The Tavily Cookbook provides code and guides designed to help developers build with Tavily's search, extract, crawl, and map APIs, offering copy-able code snippets that you can easily integrate into your own projects.


## Table of recipes

### Fundemantals
If you're new to working with the Tavily API, we recommend starting with our [Tavily API Fundamentals course](./getting-started/) to get a solid foundation. This tutorial series follows a step-by-step learning path with three stand-alone tutorials:
- [Intro](./getting-started/search-extract-crawl.ipynb): Learn the **basics of web access**.
- [Web Agent](./getting-started/web-agent-tutorial.ipynb): Build a **web agent** that can search, scrape, and crawl the web.
- [Hybrid Agent](./getting-started/hybrid-agent-tutorial.ipynb):Build a system that **combines real-time web information with private knowledge base data**.

### Search
- [Profile Search](./search/linkedin_profile_search.ipynb): Learn how to search and extract professional background information from LinkedIn profiles.
- [Product News Tracker](./search/product_news_tracker.ipynb): Monitor and collect product-related news using time-filtered and specialized news search capabilities

### Research
> **Note:** Research is our newest API offering! Features and capabilities are evolving rapidly. For detailed usage and the latest updates, see the [Research API Reference](https://docs.tavily.com/documentation/api-reference/endpoint/research).
- [Polling](./research/polling.ipynb): Asynchronous polling for background research requests.
- [Streaming](./research/streaming.ipynb): Stream real-time progress and answers during research.
- [Structured Output](./research/structured_output.ipynb): Get results in custom schema formats.
- [Clarification](./research/clarification.ipynb): Refine user prompts through multi-turn clarification before research.
- [Hybrid Research](./research/hybrid_research.ipynb): Combine Tavily research with your internal data.

### Crawl
- [Getting Started](./crawl/getting_started.ipynb): Learn the fundamentals of web crawling with Tavily's crawl and map endpoints, including URL discovery, recursive navigation, and content extraction.
- [Crawl to RAG](./crawl/crawl_to_rag.ipynb): Build a complete RAG system by crawling websites and converting content into a searchable knowledge base with vector embeddings and question-answering capabilities.
- [Agent Grounding](./crawl/agent_grounding.ipynb): Create an intelligent research agent that uses Tavily-LangChain integration to autonomously search, map, and extract web data with LangGraph.
- [Data Collection](./crawl/data_collection.ipynb): Collect and export website content as organized PDF files, perfect for creating offline archives, research documentation, and building local knowledge bases.