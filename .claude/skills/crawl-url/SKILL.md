---
name: crawl-url
description: "Crawl any website and save pages as local markdown files. Use when you need to download documentation, knowledge bases, or web content for offline access or analysis. No code required - just provide a URL."
---

# URL Crawler

Crawls websites using Tavily Crawl API and saves each page as a separate markdown file in a flat directory structure.

## When to Use

Use this skill when the user wants to:
- Crawl and extract content from a website
- Download API documentation, framework docs, or knowledge bases
- Save web content locally for offline access or analysis

## Usage

Execute the crawl script with a URL and optional instruction:

```bash
python scripts/crawl_url.py <URL> [--instruction "guidance text"]
```

### Required Parameters

- **URL**: The website to crawl (e.g., `https://docs.stripe.com/api`)

### Optional Parameters

- `--instruction, -i`: Natural language guidance for the crawler (e.g., "Focus on API endpoints only")
- `--output, -o`: Output directory (default: `<repo_root>/crawled_context/<domain>`)
- `--depth, -d`: Max crawl depth (default: 2, range: 1-5)
- `--breadth, -b`: Max links per level (default: 50)
- `--limit, -l`: Max total pages to crawl (default: 50)

### Output

The script creates a flat directory structure at `<repo_root>/crawled_context/<domain>/` with one markdown file per crawled page. Filenames are derived from URLs (e.g., `docs_stripe_com_api_authentication.md`).

Each markdown file includes:
- Frontmatter with source URL and crawl timestamp
- The extracted content in markdown format

## Examples

### Basic Crawl

```bash
python scripts/crawl_url.py https://docs.anthropic.com
```

Crawls the Anthropic docs with default settings, saves to `<repo_root>/crawled_context/docs_anthropic_com/`.

### With Instruction

```bash
python scripts/crawl_url.py https://react.dev --instruction "Focus on API reference pages and hooks documentation"
```

Uses natural language instruction to guide the crawler toward specific content.

### Custom Output Directory

```bash
python scripts/crawl_url.py https://docs.stripe.com/api -o ./stripe-api-docs
```

Saves results to a custom directory.

### Adjust Crawl Parameters

```bash
python scripts/crawl_url.py https://nextjs.org/docs --depth 3 --breadth 100 --limit 200
```

Increases crawl depth, breadth, and page limit for more comprehensive coverage.

## Important Notes

- **API Key Required**: Set `TAVILY_API_KEY` environment variable (loads from `.env` if available)
- **Crawl Time**: Deeper crawls take longer (depth 3+ may take many minutes)
- **Filename Safety**: URLs are converted to safe filenames automatically
- **Flat Structure**: All files saved in `<repo_root>/crawled_context/<domain>/` directory regardless of original URL hierarchy
- **Duplicate Prevention**: Files are overwritten if URLs generate identical filenames
