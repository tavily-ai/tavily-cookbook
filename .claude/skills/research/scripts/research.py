#!/usr/bin/env python3
"""
Research Script

A flexible CLI for conducting research using Tavily's Research API.
Becomes domain-specific at runtime through topics, prompts, and optional schemas.

Supports both streaming (--stream) and polling modes.

Usage:
    python research.py "Your research topic"
    python research.py "Topic" --schema ./schema.json --model pro
    python research.py "Topic" --stream --output ./report.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tavily import TavilyClient
except ImportError:
    print("Error: tavily-python not installed. Run: pip install tavily-python")
    sys.exit(1)

# Constants
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent

DEFAULT_POLL_INTERVAL = 5
DEFAULT_MODEL = "auto"
DEFAULT_CITATION_FORMAT = "numbered"
MAX_POLL_TIME = 600  # 10 minutes

VALID_MODELS = ["mini", "pro", "auto"]
VALID_CITATION_FORMATS = ["numbered", "mla", "apa", "chicago"]


def load_schema(schema_arg: str | None) -> dict | None:
    """
    Load JSON schema from file path or inline JSON string.

    Args:
        schema_arg: Path to JSON file or inline JSON string, or None

    Returns:
        Parsed schema dict or None if no schema provided

    Raises:
        ValueError: If schema_arg is neither valid file path nor valid JSON
    """
    if schema_arg is None:
        return None

    # Try as file path first
    schema_path = Path(schema_arg)
    if schema_path.exists():
        try:
            with open(schema_path) as f:
                schema = json.load(f)
            return schema
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in schema file {schema_path}: {e}")

    # Try as inline JSON
    try:
        return json.loads(schema_arg)
    except json.JSONDecodeError:
        raise ValueError(
            f"Schema argument is neither a valid file path nor valid JSON: {schema_arg}"
        )


def validate_schema(schema: dict) -> None:
    """
    Validate that schema meets Tavily Research API requirements.

    Requirements:
    - Must have 'properties' key
    - Every property must have 'type' and 'description'

    Args:
        schema: JSON schema dict to validate

    Raises:
        ValueError: If schema doesn't meet requirements
    """
    if "properties" not in schema:
        raise ValueError("Schema must have 'properties' key at root level")

    def check_properties(props: dict, path: str = ""):
        for name, prop in props.items():
            prop_path = f"{path}.{name}" if path else name

            if "type" not in prop:
                raise ValueError(
                    f"Property '{prop_path}' missing required 'type' field"
                )

            if "description" not in prop:
                raise ValueError(
                    f"Property '{prop_path}' missing required 'description' field. "
                    "Tavily requires descriptions to guide content extraction."
                )

            # Recursively check nested objects
            if prop.get("type") == "object" and "properties" in prop:
                check_properties(prop["properties"], prop_path)

            # Check array items if they're objects
            if prop.get("type") == "array" and "items" in prop:
                items = prop["items"]
                if items.get("type") == "object" and "properties" in items:
                    check_properties(items["properties"], f"{prop_path}[]")

    check_properties(schema["properties"])


def format_output(
    topic: str,
    model: str,
    content: Any,
    sources: list[dict],
    response_time: float | None = None
) -> dict:
    """
    Format research results into standardized output structure.

    Args:
        topic: Original research topic
        model: Model used for research
        content: Research content (markdown string or structured dict)
        sources: List of source citations
        response_time: Time in seconds to complete research

    Returns:
        Formatted output dict
    """
    return {
        "meta": {
            "topic": topic,
            "model": model,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "response_time_seconds": response_time
        },
        "content": content,
        "sources": sources
    }


def save_output(output: dict, path: Path) -> None:
    """Save output dict to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def research_streaming(
    client: TavilyClient,
    topic: str,
    schema: dict | None,
    model: str,
    citation_format: str,
    quiet: bool
) -> dict:
    """
    Execute research using streaming mode (SSE events).

    Args:
        client: TavilyClient instance
        topic: Research topic
        schema: Optional output schema
        model: Research model (mini/pro/auto)
        citation_format: Citation format
        quiet: Suppress progress output

    Returns:
        Dict with content, sources, and response_time
    """
    start_time = time.time()

    stream = client.research(
        input=topic,
        model=model,
        stream=True,
        output_schema=schema,
        citation_format=citation_format
    )

    content_chunks = []
    sources = []
    current_tool = None

    for chunk in stream:
        try:
            data = chunk.decode('utf-8')

            # Skip empty lines and SSE prefixes
            if not data.strip() or data.startswith(':'):
                continue

            # Parse SSE data lines
            for line in data.split('\n'):
                if line.startswith('data: '):
                    json_str = line[6:]  # Remove 'data: ' prefix

                    if json_str.strip() == '[DONE]':
                        continue

                    try:
                        event = json.loads(json_str)

                        # Handle different event types
                        choices = event.get('choices', [])
                        for choice in choices:
                            delta = choice.get('delta', {})

                            # Tool calls (progress indicators)
                            if 'tool_calls' in delta:
                                for tool_call in delta['tool_calls']:
                                    func = tool_call.get('function', {})
                                    tool_name = func.get('name')
                                    if tool_name and not quiet:
                                        current_tool = tool_name
                                        print(f"[{tool_name}]", end=' ', flush=True)

                            # Content chunks
                            if 'content' in delta and delta['content']:
                                content_chunks.append(delta['content'])

                            # Sources in finish metadata
                            if 'sources' in delta:
                                sources = delta['sources']

                        # Check for sources at top level
                        if 'sources' in event:
                            sources = event['sources']

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            if not quiet:
                print(f"\nWarning: Error processing stream chunk: {e}")
            continue

    if not quiet:
        print()  # Newline after progress indicators

    response_time = time.time() - start_time

    # Parse content if schema was provided (expects JSON)
    content = ''.join(content_chunks)
    if schema and content:
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            pass  # Keep as string if not valid JSON

    return {
        "content": content,
        "sources": sources,
        "response_time": response_time
    }


def research_polling(
    client: TavilyClient,
    topic: str,
    schema: dict | None,
    model: str,
    citation_format: str,
    poll_interval: int,
    quiet: bool
) -> dict:
    """
    Execute research using polling mode (two-step async pattern).

    Args:
        client: TavilyClient instance
        topic: Research topic
        schema: Optional output schema
        model: Research model (mini/pro/auto)
        citation_format: Citation format
        poll_interval: Seconds between status polls
        quiet: Suppress progress output

    Returns:
        Dict with content, sources, and response_time

    Raises:
        RuntimeError: If research fails
        TimeoutError: If research times out
    """
    # Step 1: Initiate research
    if not quiet:
        print(f"Initiating research on: {topic}")

    result = client.research(
        input=topic,
        model=model,
        output_schema=schema,
        citation_format=citation_format
    )

    request_id = result.get("request_id")
    if not request_id:
        raise RuntimeError("Failed to get request_id from research initiation")

    if not quiet:
        print(f"Request ID: {request_id}")

    # Step 2: Poll for completion
    elapsed = 0
    while elapsed < MAX_POLL_TIME:
        response = client.get_research(request_id)
        status = response.get("status")

        if status == "completed":
            return {
                "content": response.get("content"),
                "sources": response.get("sources", []),
                "response_time": response.get("response_time")
            }
        elif status == "failed":
            error = response.get("error", "Unknown error")
            raise RuntimeError(f"Research failed: {error}")

        if not quiet:
            print(f"Status: {status}... waiting {poll_interval}s")

        time.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(f"Research timed out after {MAX_POLL_TIME}s")


def research(
    topic: str,
    schema_arg: str | None = None,
    stream: bool = False,
    model: str = DEFAULT_MODEL,
    citation_format: str = DEFAULT_CITATION_FORMAT,
    output: str | None = None,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    quiet: bool = False
) -> dict:
    """
    Main research orchestration function.

    Args:
        topic: Research topic or question
        schema_arg: Path to JSON schema file or inline JSON
        stream: Enable streaming mode
        model: Research model (mini/pro/auto)
        citation_format: Citation format
        output: Output file path (None for stdout)
        poll_interval: Seconds between polls (polling mode)
        quiet: Suppress progress output

    Returns:
        Formatted research results dict
    """
    # Validate API key
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY environment variable not set. "
            "Get your key at https://tavily.com"
        )

    # Validate model
    if model not in VALID_MODELS:
        raise ValueError(f"Invalid model '{model}'. Must be one of: {VALID_MODELS}")

    # Validate citation format
    if citation_format not in VALID_CITATION_FORMATS:
        raise ValueError(
            f"Invalid citation format '{citation_format}'. "
            f"Must be one of: {VALID_CITATION_FORMATS}"
        )

    # Load and validate schema
    schema = None
    if schema_arg:
        schema = load_schema(schema_arg)
        validate_schema(schema)
        if not quiet:
            print(f"Using custom schema with {len(schema.get('properties', {}))} properties")

    # Initialize client
    client = TavilyClient(api_key=api_key)

    # Execute research
    if stream:
        if not quiet:
            print("Starting research (streaming mode)...")
        result = research_streaming(
            client, topic, schema, model, citation_format, quiet
        )
    else:
        if not quiet:
            print("Starting research (polling mode)...")
        result = research_polling(
            client, topic, schema, model, citation_format, poll_interval, quiet
        )

    # Format output
    formatted = format_output(
        topic=topic,
        model=model,
        content=result["content"],
        sources=result["sources"],
        response_time=result.get("response_time")
    )

    # Save or return
    if output:
        output_path = Path(output)
        save_output(formatted, output_path)
        if not quiet:
            print(f"Results saved to: {output_path}")

    return formatted


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Tavily Research API - General Purpose Wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Latest developments in quantum computing"
  %(prog)s "AI market analysis" --schema ./market_schema.json --model pro
  %(prog)s "Compare React vs Vue" --stream --output ./report.json
  %(prog)s "Quick overview of RAG" --model mini --quiet
        """
    )

    parser.add_argument(
        "topic",
        help="Research topic or question"
    )

    parser.add_argument(
        "--schema", "-s",
        metavar="PATH_OR_JSON",
        help="Path to JSON schema file or inline JSON string for structured output"
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming mode (real-time progress)"
    )

    parser.add_argument(
        "--model", "-m",
        choices=VALID_MODELS,
        default=DEFAULT_MODEL,
        help=f"Research model (default: {DEFAULT_MODEL})"
    )

    parser.add_argument(
        "--citation", "-c",
        choices=VALID_CITATION_FORMATS,
        default=DEFAULT_CITATION_FORMAT,
        help=f"Citation format (default: {DEFAULT_CITATION_FORMAT})"
    )

    parser.add_argument(
        "--output", "-o",
        metavar="PATH",
        help="Output file path (default: stdout)"
    )

    parser.add_argument(
        "--poll-interval", "-p",
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Seconds between status polls (default: {DEFAULT_POLL_INTERVAL})"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    try:
        result = research(
            topic=args.topic,
            schema_arg=args.schema,
            stream=args.stream,
            model=args.model,
            citation_format=args.citation,
            output=args.output,
            poll_interval=args.poll_interval,
            quiet=args.quiet
        )

        # Print to stdout if no output file specified
        if not args.output:
            print(json.dumps(result, indent=2, ensure_ascii=False))

        sys.exit(0)

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Research failed: {e}", file=sys.stderr)
        sys.exit(1)
    except TimeoutError as e:
        print(f"Timeout: {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nResearch cancelled.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
