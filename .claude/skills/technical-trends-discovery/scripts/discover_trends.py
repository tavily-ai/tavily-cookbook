#!/usr/bin/env python3
"""
Technical Trends Discovery - Two-Step Workflow

Step 1: Search X for what thought leaders are discussing (xAI API)
Step 2: Deep research on the #1 trend (Tavily Research API with structured output)

This workflow leverages X as the source for real-time opinions from top voices,
then uses Tavily to do comprehensive research on the single most important trend.

Output: Structured JSON with trend metadata, packages, resources, and insights.
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Output directory at repo root
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parents[3]
TRENDS_REPORTS_DIR = REPO_ROOT / "trends-reports"

# =============================================================================
# Structured Output Schema for Tavily Research
# =============================================================================

TREND_RESEARCH_SCHEMA = {
    "properties": {
        "trend": {
            "type": "object",
            "description": "The single most important trend - focused on actionable info to help developers get started",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Short, concise name of the trend (e.g., 'Model Context Protocol', 'LangGraph')"
                },
                "summary": {
                    "type": "string",
                    "description": "2-3 sentence summary of what this is and why developers should care"
                },
                "why_important": {
                    "type": "string",
                    "description": "Why this is the #1 trend right now - what problem it solves, why it's gaining traction"
                },
                "docs_url": {
                    "type": "string",
                    "description": "Primary documentation URL (single URL). Empty string if none."
                },
                "github_repo": {
                    "type": "string",
                    "description": "Main GitHub repository URL. Empty string if not open source."
                },
                "quickstart": {
                    "type": "object",
                    "description": "Everything needed to get a working example in 5 minutes",
                    "properties": {
                        "prerequisites": {
                            "type": "array",
                            "description": "What you need before starting (e.g., 'Python 3.10+', 'OpenAI API key')",
                            "items": {"type": "string"}
                        },
                        "install_commands": {
                            "type": "string",
                            "description": "Copy-paste shell commands to install with specific versions"
                        },
                        "hello_world_code": {
                            "type": "string",
                            "description": "Minimal working Python code (5-20 lines) from official docs that demonstrates the core concept"
                        },
                        "expected_output": {
                            "type": "string",
                            "description": "What the code produces when run successfully"
                        }
                    },
                    "required": ["prerequisites", "install_commands", "hello_world_code"]
                },
                "use_cases": {
                    "type": "array",
                    "description": "3-5 concrete things developers can build",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Short name (e.g., 'RAG Chatbot')"},
                            "description": {"type": "string", "description": "One sentence on what it does"},
                            "complexity": {"type": "string", "description": "'beginner', 'intermediate', or 'advanced'"}
                        },
                        "required": ["name", "description", "complexity"]
                    }
                },
                "common_pitfalls": {
                    "type": "array",
                    "description": "Common mistakes from GitHub issues, Stack Overflow, or forums",
                    "items": {"type": "string", "description": "A pitfall and how to avoid it"}
                },
                "key_packages": {
                    "type": "array",
                    "description": "Packages to install with exact versions",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Package name (e.g., 'langgraph')"},
                            "latest_version": {"type": "string", "description": "Latest version from package registry"},
                            "package_manager": {"type": "string", "description": "'pip', 'npm', etc."}
                        },
                        "required": ["name", "latest_version", "package_manager"]
                    }
                },
                "key_concepts": {
                    "type": "array",
                    "description": "5-10 terms/concepts to understand",
                    "items": {"type": "string"}
                },
                "additional_resources": {
                    "type": "array",
                    "description": "URLs to tutorials or guides for deeper learning",
                    "items": {"type": "string"}
                }
            },
            "required": ["name", "summary", "why_important", "quickstart", "use_cases", "key_packages"]
        },
        "meta": {
            "type": "object",
            "description": "Metadata about the research",
            "properties": {
                "research_date": {"type": "string", "description": "YYYY-MM-DD"}
            },
            "required": ["research_date"]
        }
    },
    "required": ["trend", "meta"]
}


def get_output_dir() -> Path:
    """Generate timestamped output directory."""
    TRENDS_REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = TRENDS_REPORTS_DIR / f"trends_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# =============================================================================
# Step 1: X Search
# =============================================================================

DEFAULT_HANDLES = [
    "hwchase17",      # Harrison Chase (LangChain)
    "rlancemartin",   # Lance Martin (LangChain)
    "simonw",         # Simon Willison
    "karpathy",       # Andrej Karpathy
    "cherny",         # Boris Cherny
    "swyx",           # Swyx
    "alexalbert__",   # Alex Albert (Anthropic)
]

X_DISCOVERY_PROMPT = """Analyze the recent posts from these AI thought leaders and identify the **SINGLE MOST IMPORTANT** emerging trend.

Your task:
1. **Identify THE #1 Trend** - What is the single most significant topic that multiple thought leaders are discussing? Be specific about the trend name.
2. **Specific Tools/Projects Mentioned** - What specific tools, frameworks, or projects are thought leaders actually using or discussing in relation to this trend?
3. **Why It's #1** - Explain why this trend stands out above others (momentum, impact, novelty, adoption)
4. **Key Voices** - Which thought leaders mentioned it and what did they say about it?
5. **Supporting Context** - What other topics are related to this main trend?

Focus on DEPTH over breadth. We want to deeply understand ONE trend, not superficially cover many.
The trend should be something developers can act on - a tool, framework, methodology, or paradigm shift."""


def search_x_trends(handles: list[str], days_back: int) -> dict:
    """Step 1: Search X for trends from thought leaders."""
    try:
        from xai_sdk import Client
        from xai_sdk.chat import user
        from xai_sdk.search import SearchParameters, x_source
    except ImportError:
        raise ImportError("xai-sdk not installed. Run: pip install xai-sdk")

    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY environment variable not set")

    to_date = datetime.now()
    from_date = to_date - timedelta(days=days_back)

    print("=" * 60)
    print("STEP 1: X SEARCH")
    print("=" * 60)
    print(f"Searching posts from {len(handles)} thought leaders...")
    print(f"Date range: {from_date.date()} to {to_date.date()}")
    print(f"Handles: {', '.join(handles)}\n")

    client = Client(api_key=api_key)

    chat = client.chat.create(
        model="grok-3",
        search_parameters=SearchParameters(
            mode="on",
            from_date=from_date,
            to_date=to_date,
            return_citations=True,
            sources=[
                x_source(
                    included_x_handles=handles,
                    post_favorite_count=100,  # Filter for posts with decent engagement
                )
            ],
        ),
    )

    chat.append(user(X_DISCOVERY_PROMPT))
    response = chat.sample()

    print("X search complete.\n")
    return {
        "content": response.content,
        "citations": getattr(response, "citations", []),
    }


# =============================================================================
# Step 2: Tavily Deep Research
# =============================================================================

def build_research_prompt(x_trends: str) -> str:
    """Build a research prompt focused on actionable developer content."""
    return f"""Based on recent discussions from AI thought leaders on X, this trend has been identified:

{x_trends}

Your task: Research this trend with a focus on ACTIONABLE INFORMATION that helps a developer get started in 5 minutes.

## Required Research

### 1. Quickstart (MOST IMPORTANT)
- **Prerequisites**: What does a developer need? (Python version, API keys, etc.)
- **Install commands**: Exact pip/npm install commands WITH specific version numbers
- **Hello World code**: Find the simplest working example from official docs (5-20 lines of Python). Must be REAL, RUNNABLE code - not pseudocode.
- **Expected output**: What does the code produce when successful?

### 2. Use Cases
Find 3-5 concrete things developers are building with this. Look at:
- GitHub repos using this technology
- Blog posts showing implementations
- Official examples/cookbooks
For each: name, one-sentence description, complexity (beginner/intermediate/advanced)

### 3. Common Pitfalls
Search GitHub issues, Stack Overflow, Discord/Slack communities for common problems:
- Setup issues people hit
- Configuration mistakes
- Version compatibility problems
Include how to avoid or fix each one.

### 4. Packages
Find exact package names and their LATEST versions from PyPI/npm. Check the actual package registry pages.

### 5. Key Concepts
What 5-10 terms/concepts must a developer understand?

## Research Sources Priority
1. Official documentation quickstart/getting-started pages
2. Package registry pages (PyPI, npm) for exact versions
3. GitHub issues for common pitfalls
4. Developer blog posts for real use cases

Focus on PRACTICAL, COPY-PASTE-ABLE information. A developer should be able to read the output and have working code in 5 minutes."""


def research_trends(prompt: str, poll_interval: int = 5) -> dict:
    """
    Step 2: Deep research on trends using Tavily with structured output.

    Args:
        prompt: Research prompt based on X discoveries
        poll_interval: Seconds between status polls

    Returns:
        Dict with status, content (structured JSON), and sources
    """
    try:
        from tavily import TavilyClient
    except ImportError:
        raise ImportError("tavily-python not installed. Run: pip install tavily-python")

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set")

    print("=" * 60)
    print("STEP 2: TAVILY DEEP RESEARCH")
    print("=" * 60)
    print("Initiating structured research on identified trend...\n")

    client = TavilyClient(api_key=api_key)

    try:
        result = client.research(
            input=prompt,
            model="pro",
            output_schema=TREND_RESEARCH_SCHEMA
        )
        request_id = result["request_id"]
        print(f"Research initiated (request_id: {request_id})")
    except Exception as e:
        print(f"ERROR: Failed to initiate research: {e}")
        return {"status": "failed", "error": str(e)}

    elapsed = 0
    while True:
        try:
            response = client.get_research(request_id)
            status = response.get("status", "unknown")

            if status == "completed":
                print(f"Research completed in {elapsed}s\n")
                return {
                    "status": "completed",
                    "content": response.get("content"),
                    "sources": response.get("sources", []),
                }
            elif status == "failed":
                error_msg = response.get("error", "Unknown error")
                print(f"Research failed: {error_msg}")
                return {"status": "failed", "error": error_msg}
            else:
                print(f"Status: {status}... waiting {poll_interval}s (elapsed: {elapsed}s)")
                time.sleep(poll_interval)
                elapsed += poll_interval
        except Exception as e:
            print(f"ERROR polling research status: {e}")
            print("Retrying...")
            time.sleep(poll_interval)
            elapsed += poll_interval


# =============================================================================
# Main Orchestration
# =============================================================================

def discover_trends(
    handles: list[str] = None,
    days_back: int = 20,
) -> dict:
    """
    Two-step trend discovery pipeline:
    1. Search X for what thought leaders are discussing
    2. Deep research on THE most important trend via Tavily (structured JSON output)

    Args:
        handles: X handles to search
        days_back: Days back to search

    Returns:
        Dict with x_trends, research (structured JSON), and output_dir
    """
    handles = handles or DEFAULT_HANDLES
    output_dir = get_output_dir()

    # Step 1: X Search
    x_result = search_x_trends(handles, days_back)

    # Step 2: Tavily Deep Research (structured JSON output)
    research_prompt = build_research_prompt(x_result["content"])
    research_result = research_trends(research_prompt)

    return {
        "x_trends": x_result,
        "research": research_result,
        "output_dir": str(output_dir),
    }


def save_results(output_dir: Path, results: dict):
    """Save all results to a single consolidated JSON file."""
    report_path = output_dir / "report.json"

    # Build consolidated report
    report = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "pipeline": "x_discovery → tavily_research",
        },
        "x_discovery": {
            "content": results["x_trends"]["content"],
            "citations": results["x_trends"].get("citations", []),
        },
    }

    # Add research results if completed
    if results["research"] and results["research"].get("status") == "completed":
        content = results["research"]["content"]

        # Parse structured data
        if isinstance(content, str):
            try:
                research_data = json.loads(content)
            except json.JSONDecodeError:
                research_data = {"raw": content}
        else:
            research_data = content

        # Simplify sources
        sources = [
            {"url": s.get("url", ""), "title": s.get("title", "Untitled")}
            for s in results["research"].get("sources", [])
        ]

        report["research"] = research_data
        report["sources"] = sources
        report["meta"]["sources_count"] = len(sources)
    else:
        report["research"] = None
        report["sources"] = []
        report["meta"]["status"] = results["research"].get("status", "unknown")
        if results["research"].get("error"):
            report["meta"]["error"] = results["research"]["error"]

    # Save consolidated report
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Discover AI trends: X search → Tavily deep research → Structured JSON"
    )
    parser.add_argument(
        "--handles", "-H",
        nargs="+",
        help="X handles to search (default: AI thought leaders)"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=20,
        help="Days back to search (default: 20)"
    )

    args = parser.parse_args()

    # Run discovery pipeline
    results = discover_trends(
        handles=args.handles,
        days_back=args.days,
    )

    # Save results
    if results.get("output_dir"):
        save_results(Path(results["output_dir"]), results)

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Output directory: {results['output_dir']}")

    if results["research"] and results["research"].get("status") == "completed":
        print("\n--- STRUCTURED OUTPUT ---")
        content = results["research"]["content"]
        if isinstance(content, dict):
            print(json.dumps(content, indent=2))
        else:
            print(content)


if __name__ == "__main__":
    main()
