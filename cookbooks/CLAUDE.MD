# Cookbooks Directory

This directory contains Jupyter notebook tutorials demonstrating Tavily API usage patterns. Each notebook is standalone and executable.

## Styling Guidelines

Notebooks should be **simple, clear, and concise**. Follow these principles:

- **Minimal code** - Show the essential pattern, avoid boilerplate
- **One concept per notebook** - Each notebook demonstrates a single API feature or use case
- **Progressive complexity** - Start with the simplest example, layer in options
- **Copy-paste ready** - Code snippets should work immediately when copied
- **Brief explanations** - Use markdown cells sparingly; let code speak for itself
- **No unnecessary abstractions** - Prefer inline code over helper functions

## Notebook Conventions

**Environment Variables**: All notebooks expect `TAVILY_API_KEY`. Many also require `OPENAI_API_KEY` or other LLM provider keys.

**Execution**: Notebooks are designed to run top-to-bottom. Code cells build on previous outputs.

## When Updating Notebooks

1. **Test all cells** - Run the full notebook before committing
2. **Clear outputs** - Remove cell outputs to keep diffs clean (optional based on team preference)
3. **Update [README.md](./README.md)** - If adding a new notebook, add an entry to the table of recipes
4. **Check API compatibility** - Ensure code matches current Tavily SDK patterns documented in `/.claude/skills/tavily-api/`

## Tavily API Skill

**Always leverage the `tavily-api` skill when building new cookbooks.** This skill lives at `/.claude/skills/tavily-api/` and contains up-to-date API reference documentation with correct patterns, parameters, and best practices.
