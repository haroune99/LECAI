from typing import TypedDict, Literal, Optional
from dataclasses import dataclass


@dataclass
class ToolResult:
    call_id: str
    tool_name: str
    status: Literal["success", "error", "timeout"]
    content: dict
    latency_ms: int
    error_message: Optional[str] = None
    tokens_used: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "status": self.status,
            "content": self.content,
            "latency_ms": self.latency_ms,
            "error_message": self.error_message,
            "tokens_used": self.tokens_used,
        }


def format_tool_results(results: list[ToolResult]) -> str:
    lines = []
    for r in results:
        lines.append(f"Tool: {r.tool_name}")
        lines.append(f"Status: {r.status}")
        lines.append(f"Latency: {r.latency_ms}ms")
        if r.status == "error":
            lines.append(f"Error: {r.error_message}")
        else:
            import json
            lines.append(f"Result: {json.dumps(r.content, ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)
