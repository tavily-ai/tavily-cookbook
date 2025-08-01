# Tavily Cookbook

The Tavily Cookbook provides code and guides designed to help developers build with Tavily's search, extract, crawl, and map APIs, offering copy-able code snippets that you can easily integrate into your own projects.

## Prerequisites

**Sign up** for Tavily at [app.tavily.com](https://app.tavily.com/home/?utm_source=github&utm_medium=referral&utm_campaign=nir_diamant) to get your API key.

While the code examples are primarily written in Python, the concepts can be adapted to any programming language that supports interaction with the Tavily API.

If you're new to working with the Tavily API, we recommend starting with our [Tavily API Fundamentals course](https://github.com/NirDiamant/agents-towards-production/tree/main/tutorials/agent-with-tavily-web-access) to get a solid foundation.

## Explore Further

Looking for more resources to enhance your experience with Tavily and AI Agents? Check out these helpful links:

- [Tavily developer documentation](https://docs.tavily.com/welcome)
- [Tavily support](https://www.tavily.com/contact)
- [Tavily community](https://community.tavily.com/)
- [Tavily github](https://github.com/tavily-ai)


## Table of recipes

### Web Search
- [Profile Search](./search/linkedin_profile_search.ipynb): Learn how to search and extract professional background information from LinkedIn profiles.
- [Product News Tracker](./search/product_news_tracker.ipynb): Monitor and collect product-related news using time-filtered and specialized news search capabilities

### Web Crawl
- [Getting Started](./crawl/getting_started.ipynb): Learn the fundamentals of web crawling with Tavily's crawl and map endpoints, including URL discovery, recursive navigation, and content extraction.
- [Crawl to RAG](./crawl/crawl_to_rag.ipynb): Build a complete RAG system by crawling websites and converting content into a searchable knowledge base with vector embeddings and question-answering capabilities.
- [Agent Grounding](./crawl/agent_grounding.ipynb): Create an intelligent research agent that uses Tavily-LangChain integration to autonomously search, map, and extract web data with LangGraph.
- [Data Collection](./crawl/data_collection.ipynb): Collect and export website content as organized PDF files, perfect for creating offline archives, research documentation, and building local knowledge bases.

### Use Cases
- [Company Research](./use-cases/company-research/company_research.ipynb): Build an automated research system for portfolio companies using Tavily and LangGraph agents to generate comprehensive weekly reports with the latest news, developments, and insights.
- [Data Enrichment](./use-cases/data-enrichment/data_enrichment_agent.ipynb): Enhance your datasets by automatically filling missing values using Tavily's search capabilities, supporting CSV, Excel, and Google Sheets with intelligent data completion.