"""LangGraph runtime for the deterministic flat-skill agent baseline."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.models import (
    AgentRunResult,
    SkillBundle,
    SkillCandidate,
    ToolCall,
    ToolResult,
    VerificationResult,
)
from agent.organizers import FlatOrganizer
from agent.selectors import Selection, SkillSelector
from agent.tools import ToolRegistry
from agent.verifiers import VerifierRegistry


TerminationReason = Literal[
    "verified",
    "verification_failed",
    "selection_failed",
    "execution_failed",
]


class RuntimeState(TypedDict, total=False):
    task_id: str
    task: str
    candidates: list[SkillCandidate]
    verifier_id: str
    expected: Any
    bundle: SkillBundle
    selection: Selection | None
    tool_result: ToolResult | None
    verification: VerificationResult
    termination_reason: TerminationReason
    trace: list[dict[str, Any]]


class FlatSkillAgent:
    """Execute one selected skill and verify its result through explicit states."""

    def __init__(
        self,
        organizer: FlatOrganizer,
        selector: SkillSelector,
        tools: ToolRegistry,
        verifiers: VerifierRegistry,
    ) -> None:
        self._organizer = organizer
        self._selector = selector
        self._tools = tools
        self._verifiers = verifiers
        self._graph = self._build_graph()

    def run(
        self,
        task_id: str,
        task: str,
        candidates: Iterable[SkillCandidate],
        verifier_id: str,
        expected: Any,
    ) -> AgentRunResult:
        final_state = self._graph.invoke(
            RuntimeState(
                task_id=task_id,
                task=task,
                candidates=list(candidates),
                verifier_id=verifier_id,
                expected=expected,
                trace=[],
            )
        )
        selection = final_state.get("selection")
        return AgentRunResult(
            task_id=task_id,
            selected_skill_id=selection.skill_id if selection else None,
            tool_result=final_state.get("tool_result"),
            verification=final_state["verification"],
            termination_reason=final_state["termination_reason"],
            trace=final_state["trace"],
        )

    def _build_graph(self):
        graph = StateGraph(RuntimeState)
        graph.add_node("organize", self._organize)
        graph.add_node("select", self._select)
        graph.add_node("selection_failed", self._selection_failed)
        graph.add_node("execute", self._execute)
        graph.add_node("execution_failed", self._execution_failed)
        graph.add_node("verify", self._verify)

        graph.add_edge(START, "organize")
        graph.add_edge("organize", "select")
        graph.add_conditional_edges(
            "select",
            self._route_after_selection,
            {
                "execute": "execute",
                "selection_failed": "selection_failed",
            },
        )
        graph.add_edge("selection_failed", END)
        graph.add_conditional_edges(
            "execute",
            self._route_after_execution,
            {
                "verify": "verify",
                "execution_failed": "execution_failed",
            },
        )
        graph.add_edge("execution_failed", END)
        graph.add_edge("verify", END)
        return graph.compile()

    def _organize(self, state: RuntimeState) -> RuntimeState:
        bundle = self._organizer.organize(state["candidates"])
        return {
            "bundle": bundle,
            "trace": _append_trace(
                state,
                "organized",
                strategy=bundle.strategy,
                skill_ids=[skill.skill_id for skill in bundle.skills],
            ),
        }

    def _select(self, state: RuntimeState) -> RuntimeState:
        selection = self._selector.select(state["task"], state["bundle"])
        if selection is None:
            return {
                "selection": None,
                "trace": _append_trace(state, "selection_failed"),
            }
        return {
            "selection": selection,
            "trace": _append_trace(
                state,
                "selected",
                skill_id=selection.skill_id,
                tool_name=selection.tool_name,
                reason=selection.reason,
            ),
        }

    def _selection_failed(self, state: RuntimeState) -> RuntimeState:
        return {
            "tool_result": None,
            "verification": VerificationResult(
                passed=False,
                verifier_id=state["verifier_id"],
                details="verification not run: no skill selected",
            ),
            "termination_reason": "selection_failed",
        }

    def _execute(self, state: RuntimeState) -> RuntimeState:
        selection = state["selection"]
        if selection is None:
            raise RuntimeError("execute reached without a selection")
        result = self._tools.execute(
            ToolCall(
                tool_name=selection.tool_name,
                arguments=selection.arguments,
            )
        )
        return {
            "tool_result": result,
            "trace": _append_trace(
                state,
                "tool_executed",
                tool_name=result.tool_name,
                ok=result.ok,
                error=result.error,
            ),
        }

    def _execution_failed(self, state: RuntimeState) -> RuntimeState:
        result = state["tool_result"]
        return {
            "verification": VerificationResult(
                passed=False,
                verifier_id=state["verifier_id"],
                details=f"verification not run: {result.error if result else 'tool failed'}",
            ),
            "termination_reason": "execution_failed",
        }

    def _verify(self, state: RuntimeState) -> RuntimeState:
        result = state["tool_result"]
        if result is None or not result.ok:
            raise RuntimeError("verify reached without a successful tool result")
        verification = self._verifiers.verify(
            state["verifier_id"],
            actual=result.output,
            expected=state["expected"],
        )
        termination_reason: TerminationReason = (
            "verified" if verification.passed else "verification_failed"
        )
        return {
            "verification": verification,
            "termination_reason": termination_reason,
            "trace": _append_trace(
                state,
                "verified",
                verifier_id=verification.verifier_id,
                passed=verification.passed,
                details=verification.details,
            ),
        }

    @staticmethod
    def _route_after_selection(
        state: RuntimeState,
    ) -> Literal["execute", "selection_failed"]:
        return "execute" if state.get("selection") is not None else "selection_failed"

    @staticmethod
    def _route_after_execution(
        state: RuntimeState,
    ) -> Literal["verify", "execution_failed"]:
        result = state.get("tool_result")
        return "verify" if result is not None and result.ok else "execution_failed"


def _append_trace(
    state: RuntimeState,
    event: str,
    **payload: Any,
) -> list[dict[str, Any]]:
    return [*state.get("trace", []), {"event": event, **payload}]
