import json
from typing import Any


def handle_research_stream(response: Any, verbose: bool = True, stream_content_generation: bool = True) -> str:
    """
    Handle streaming response from Tavily Research API.
    
    Parses Server-Sent Events (SSE) and extracts:
    - Tool calls (Planning, WebSearch, ResearchSubtopic, Generating)
    - Tool responses with sources
    - Content chunks (the final report)
    - Sources list
    
    Args:
        response: The streaming response from Tavily Research API.
        verbose: If True, prints tool calls, sources, and status messages.
        stream_content_generation: If True, prints content chunks as they arrive. If False,
                        content is accumulated silently and only returned at the end.
    
    Returns the full concatenated report content.
    """
    full_report = ""
    sources = []
    current_event_type = None
    data_buffer = ""
    
    # Track what we've already printed to avoid duplicates
    printed_tool_calls = set()
    
    try:
        for chunk in response:
            line = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
            
            # Process each line in the chunk (may contain multiple lines)
            for raw_line in line.split("\n"):
                stripped = raw_line.strip()
                
                if not stripped:
                    # Empty line signals end of event - process buffered data
                    if data_buffer and current_event_type:
                        full_report = _process_event(
                            current_event_type, 
                            data_buffer, 
                            verbose,
                            stream_content_generation,
                            printed_tool_calls,
                            full_report_ref=[full_report],
                            sources_ref=sources
                        )
                    data_buffer = ""
                    current_event_type = None
                    continue
                
                if stripped.startswith("event:"):
                    current_event_type = stripped.split("event:", 1)[1].strip()
                    # Handle "done" event immediately
                    if current_event_type == "done":
                        if verbose:
                            print("\n✅ Research complete")
                        current_event_type = None
                elif stripped.startswith("data:"):
                    data_buffer = stripped.split("data:", 1)[1].strip()
                    # Process immediately if we have event type
                    if current_event_type:
                        full_report = _process_event(
                            current_event_type,
                            data_buffer,
                            verbose,
                            stream_content_generation,
                            printed_tool_calls,
                            full_report_ref=[full_report],
                            sources_ref=sources
                        )
                        data_buffer = ""

    except Exception as e:
        if verbose:
            print(f"\n⚠️ Stream error: {e}")

    if stream_content_generation and not full_report.endswith("\n"):
        print()
    
    return full_report


def _process_event(
    event_type: str,
    data_str: str,
    verbose: bool,
    stream_content_generation: bool,
    printed_tool_calls: set,
    full_report_ref: list,
    sources_ref: list
) -> str:
    """Process a single SSE event and return updated full_report."""
    full_report = full_report_ref[0]
    
    if event_type != "chat.completion.chunk":
        return full_report
    
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        return full_report
    
    choices = data.get("choices", [])
    if not choices:
        return full_report
    
    delta = choices[0].get("delta", {})
    
    # Handle content streaming (the final report)
    if "content" in delta:
        content = delta["content"]
        if isinstance(content, str):
            full_report += content
            if stream_content_generation:
                print(content, end="", flush=True)
        elif isinstance(content, dict):
            # Structured output mode
            if stream_content_generation:
                print(f"\n📊 Structured output: {json.dumps(content, indent=2)}")
            full_report = json.dumps(content)
    
    # Handle sources event
    if "sources" in delta:
        sources_ref.extend(delta["sources"])
        if verbose:
            source_count = len(delta["sources"])
            print(f"\n📚 Received {source_count} sources")
    
    # Handle tool calls
    if "tool_calls" in delta:
        tool_data = delta["tool_calls"]
        if isinstance(tool_data, dict):
            call_type = tool_data.get("type", "")
            
            if call_type == "tool_call":
                # Tool is being invoked
                for item in tool_data.get("tool_call", []):
                    if not isinstance(item, dict):
                        continue
                    
                    tool_id = item.get("id", "")
                    tool_name = item.get("name", "")
                    arguments = item.get("arguments", "")
                    queries = item.get("queries", [])
                    
                    # Create unique key to avoid duplicate prints
                    call_key = f"{tool_id}:{tool_name}:call"
                    if call_key in printed_tool_calls:
                        continue
                    printed_tool_calls.add(call_key)
                    
                    if verbose:
                        icon = _get_tool_icon(tool_name)
                        if tool_name == "WebSearch" and queries:
                            print(f"\n{icon} {tool_name}: {arguments}")
                            for q in queries[:5]:  # Show up to 5 queries
                                print(f"   • {q}")
                        elif tool_name == "Planning":
                            print(f"\n{icon} Planning research...")
                        elif tool_name == "Generating":
                            print(f"\n{icon} Generating report...")
                        elif tool_name == "ResearchSubtopic":
                            print(f"\n{icon} Researching subtopic: {arguments}")
                        else:
                            print(f"\n{icon} {tool_name}: {arguments}")
            
            elif call_type == "tool_response":
                # Tool completed
                for item in tool_data.get("tool_response", []):
                    if not isinstance(item, dict):
                        continue
                    
                    tool_id = item.get("id", "")
                    tool_name = item.get("name", "")
                    arguments = item.get("arguments", "")
                    tool_sources = item.get("sources", [])
                    
                    # Create unique key to avoid duplicate prints
                    resp_key = f"{tool_id}:{tool_name}:response"
                    if resp_key in printed_tool_calls:
                        continue
                    printed_tool_calls.add(resp_key)
                    
                    if verbose:
                        if tool_sources:
                            print(f"   ✓ {tool_name} found {len(tool_sources)} sources")
                        elif tool_name == "Planning":
                            print(f"   ✓ Research plan ready")
                        elif tool_name == "Generating":
                            print(f"   ✓ Report generation started")
                        else:
                            print(f"   ✓ {tool_name}: {arguments}")
    
    return full_report


def _get_tool_icon(tool_name: str) -> str:
    """Get an icon for the tool type."""
    icons = {
        "Planning": "📋",
        "WebSearch": "🔍",
        "ResearchSubtopic": "🔬",
        "Generating": "📝",
    }
    return icons.get(tool_name, "🔧")
