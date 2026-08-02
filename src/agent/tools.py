"""Local tool registration and fail-closed execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent.models import ToolCall, ToolResult


ToolFunction = Callable[..., Any]


class ToolRegistry:
    """Map stable tool names to local callables."""

    def __init__(self) -> None:
        self._functions: dict[str, ToolFunction] = {}

    def register(self, name: str, function: ToolFunction) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("tool name must be a non-empty string")
        if not callable(function):
            raise TypeError("tool function must be callable")
        if name in self._functions:
            raise ValueError(f"tool already registered: {name}")
        self._functions[name] = function

    def execute(self, call: ToolCall) -> ToolResult:
        function = self._functions.get(call.tool_name)
        if function is None:
            return ToolResult(
                tool_name=call.tool_name,
                ok=False,
                error=f"unknown tool: {call.tool_name}",
            )
        try:
            output = function(**call.arguments)
        except Exception as error:
            return ToolResult(
                tool_name=call.tool_name,
                ok=False,
                error=f"{type(error).__name__}: {error}",
            )
        return ToolResult(tool_name=call.tool_name, ok=True, output=output)
