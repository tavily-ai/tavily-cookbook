---
name: research
description: General-purpose research skill. Conducts comprehensive research on any topic with optional structured JSON output. Supports streaming and polling modes. Use when you need AI-synthesized research grounded in web data with citations on any subject.
---

# Research Skill

Conduct comprehensive research on any topic with automatic source gathering, analysis, and response generation with citations.

## When to Use

- Researching any topic requiring web-sourced information
- Generating structured research reports with custom schemas
- Building pipelines that need AI-synthesized insights with citations

## Prerequisites

**Tavily API Key Required** - Get your key at https://tavily.com

Add to `~/.claude/settings.json`:
```json
{
  "env": {
    "TAVILY_API_KEY": "tvly-your-api-key-here"
  }
}
```

## Quick Start

### Basic Research (Polling Mode)

```bash
python scripts/research.py "Latest developments in quantum computing"
```

### With Custom Schema

```bash
python scripts/research.py "Electric vehicle market analysis" \
  --schema ./schemas/market_analysis.json \
  --model pro
```

### Streaming Mode

```bash
python scripts/research.py "AI agent frameworks comparison" --stream
```

### Save to File

```bash
python scripts/research.py "Rust async ecosystem" \
  --output ./reports/rust_async.json \
  --model pro
```

## CLI Reference

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `topic` | - | Required | Research topic or question |
| `--schema` | `-s` | None | Path to JSON schema file or inline JSON |
| `--stream` | - | False | Enable streaming mode |
| `--model` | `-m` | `auto` | Model: `mini`, `pro`, `auto` |
| `--citation` | `-c` | `numbered` | Citation format: `numbered`, `mla`, `apa`, `chicago` |
| `--output` | `-o` | stdout | Output file path |
| `--poll-interval` | `-p` | 5 | Seconds between polls (polling mode) |
| `--quiet` | `-q` | False | Suppress progress output |

## Output Format

```json
{
  "meta": {
    "topic": "Your research topic",
    "model": "pro",
    "completed_at": "2025-01-08T14:30:00Z",
    "response_time_seconds": 45.2
  },
  "content": "...",
  "sources": [
    {"url": "https://...", "title": "Source Title", "citation": "[1]"}
  ]
}
```

- **content**: Markdown string (default) or structured JSON (when schema provided)
- **sources**: Array of citations used in the research

## Schema Usage

Schemas make output structured and predictable. Provide via file path or inline JSON.

### File Path

```bash
python scripts/research.py "topic" --schema ./my_schema.json
```

### Inline JSON

```bash
python scripts/research.py "topic" --schema '{"properties": {"summary": {"type": "string", "description": "Executive summary"}}}'
```

### Schema Requirements

Every property **MUST** include both `type` and `description`:

```json
{
  "properties": {
    "summary": {
      "type": "string",
      "description": "2-3 sentence executive summary"
    },
    "key_points": {
      "type": "array",
      "description": "Main takeaways",
      "items": {"type": "string"}
    }
  },
  "required": ["summary", "key_points"]
}
```

See `references/schema.json` for complete schema rules and examples.

## Model Selection

**Rule of thumb**: "what does X do?" → mini. "X vs Y vs Z" or "best way to..." → pro.

| Model | Use Case | Speed |
|-------|----------|-------|
| `mini` | Single topic, targeted research | Fast |
| `pro` | Comprehensive multi-angle analysis, open ended | Slower |
| `auto` | API chooses based on topic complexity | Varies |


## Examples

### Market Research

```bash
python scripts/research.py "Fintech startup landscape 2025" \
  --schema ./schemas/market_research.json \
  --model pro \
  --output ./reports/fintech_2025.json
```

### Technical Comparison

```bash
python scripts/research.py "LangGraph vs CrewAI for multi-agent systems" \
  --model pro \
  --citation mla
```

### Quick Overview

```bash
python scripts/research.py "What is retrieval augmented generation?" \
  --model mini \
  --quiet
```

