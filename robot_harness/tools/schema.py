"""ToolSchema — JSON-Schema-based input/output spec compatible with MCP and
OpenAI function calling format.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, field_validator


class ToolSchema(BaseModel):
    """Canonical schema for a single tool.

    ``input_schema`` and ``output_schema`` follow JSON Schema draft-07,
    which is the format used by both MCP and OpenAI function calling.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Tool name must not be empty")
        return v

    @field_validator("input_schema")
    @classmethod
    def _input_has_type(cls, v: dict[str, Any]) -> dict[str, Any]:
        if "type" not in v and "$ref" not in v and "oneOf" not in v and "anyOf" not in v:
            raise ValueError("input_schema must have a 'type' key (or $ref / oneOf / anyOf)")
        return v

    def to_openai_function(self) -> dict[str, Any]:
        """Export as OpenAI function-calling spec."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def to_mcp_tool(self) -> dict[str, Any]:
        """Export as MCP tool definition."""
        d: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
        if self.output_schema:
            d["outputSchema"] = self.output_schema
        return d


ToolBackend = Literal["mcp", "http", "native", "inproc"]
