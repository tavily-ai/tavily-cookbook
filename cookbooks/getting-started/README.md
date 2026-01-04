![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--agent-with-tavily-web-access--readme)

# Supercharge Your Agent with Web Access Using Tavily


## Overview

This tutorial series is designed for Python developers who want to empower their AI agents with real-time web access, enabling agents to utilize up-to-date information as context. Live web information is critical for AI agents tasked with performing research, answering questions accurately, monitoring trends, or providing up-to-date recommendations. You'll learn how to build AI agents that search the web, extract valuable content, navigate websites intelligently, and integrate real-time web information into private knowledge bases. By the end, you'll be equipped to build sophisticated web agents that can fetch timely data, improving your agent's accuracy and reliability.

## Agenda

This tutorial series follows a step-by-step learning path with three stand-alone tutorials:
1. In [tutorial #1](./search-extract-crawl.ipynb), we'll cover the **basics of web access**.

2. In [tutorial #2](./web-agent-tutorial.ipynb), we'll **build a web agent** that can search, scrape, and crawl the web.

3. Finally, In [tutorial #3](./hybrid-agent-tutorial.ipynb), we'll develop a system that **combines real-time web information with private knowledge base data**.


## Directory Structure

```
📁 agent-with-tavily-web-access/
├── 📓 search-extract-crawl.ipynb  # Tutorial notebook 1
├── 📓 web-agent-tutorial.ipynb    # Tutorial notebook 2
├── 📓 hybrid-agent-tutorial.ipynb # Tutorial notebook 3
├── 📁 assets/                     # Diagrams and screenshots
│   ├── 🖼️ web-agent.svg
|   ├── 🖼️ hybrid.svg
|   ├── 🖼️ api-key.png
│   └── 🖼️ sign-up.png
├── 📁 supplemental/                # Supplemental materials
│   ├── 📓 vectorize_tutorial.ipynb # Vectorize your own documents
│   ├── 📁 docs/                    # Sample CRM documents
│   └── 📁 db/                      # Pre-built Chroma vector database
└── 📄 README.md                    # This file
```