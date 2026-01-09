"""
Shared utilities for use-cases agents.
"""


async def stream_agent_response(agent, inputs: dict) -> str:
    """Stream agent execution, printing tool calls/completions, and return final response."""
    final_response = None
    async for chunk in agent.astream(inputs, stream_mode="updates"):
        for node_output in chunk.values():
            for msg in node_output.get("messages", []):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        print(f"🔧 Calling: {tool_call['name']}")
                elif hasattr(msg, "name") and msg.name:
                    print(f"✅ {msg.name} completed")
                elif hasattr(msg, "content") and msg.content:
                    final_response = msg.content
    return final_response or "No response generated"
