"""LangGraph topology for the Jev decision-driven research agent.

Flow:

    understand_task  (LLM: understanding, plan, candidate queries)
        -> select_action  (Jev: action Choice)
            -> finish? generate_answer
            -> else execute_tool (harness runs the search)
                -> decide_continue (Jev: sufficiency Noul)
                    -> keep going? back to select_action
                    -> else generate_answer (LLM synthesizes)
                        -> END
"""

from __future__ import annotations

from typing import Callable, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.decision.policy import (
    DecisionLayer,
    StopPolicy,
    create_action_policy,
    create_stop_policy,
)
from app.decision.jev import create_decision_layer
from app.decision.schemas import ResearchAction
from app.graph.nodes import (
    make_decide_continue_node,
    make_execute_tool_node,
    make_generate_answer_node,
    make_select_action_node,
    make_understand_task_node,
)
from app.graph.state import ResearchState
from app.llm.model import get_chat_model
from app.tools.news import search_news
from app.tools.search import search_web


def route_after_action(state: ResearchState) -> str:
    """From select_action: execute the chosen search, or finish immediately."""
    action = state.get("action")
    if action is ResearchAction.FINISH:
        return "generate_answer"
    return "execute_tool"


def route_after_continue(state: ResearchState) -> str:
    """From decide_continue: keep researching or write the final answer."""
    if state.get("stop_research"):
        return "generate_answer"
    return "select_action"


def build_graph(
    *,
    chat_model=None,
    decision_layer: Optional[DecisionLayer] = None,
    web_search: Callable = search_web,
    news_search: Callable = search_news,
    max_steps: Optional[int] = None,
    action_policy=None,
    stop_policy: Optional[StopPolicy] = None,
) -> CompiledStateGraph:
    """Build the compiled research graph.

    Dependency injection keeps the graph testable: pass fakes for any of the
    components. Defaults wire real LangChain / Jev components from the
    environment.
    """
    if chat_model is None:
        chat_model = get_chat_model()
    if action_policy is None:
        action_policy = create_action_policy()
    if decision_layer is None:
        decision_layer = create_decision_layer(policy=action_policy)
    if stop_policy is None:
        stop_policy = create_stop_policy()

    builder = StateGraph(ResearchState)

    builder.add_node("understand_task", make_understand_task_node(chat_model))
    builder.add_node(
        "select_action",
        make_select_action_node(decision_layer, max_steps=max_steps),
    )
    builder.add_node(
        "execute_tool",
        make_execute_tool_node(web_search=web_search, news_search=news_search),
    )
    builder.add_node(
        "decide_continue",
        make_decide_continue_node(decision_layer, stop_policy=stop_policy, max_steps=max_steps),
    )
    builder.add_node("generate_answer", make_generate_answer_node(chat_model))

    builder.add_edge(START, "understand_task")
    builder.add_edge("understand_task", "select_action")

    builder.add_conditional_edges(
        "select_action",
        route_after_action,
        {"execute_tool": "execute_tool", "generate_answer": "generate_answer"},
    )
    builder.add_edge("execute_tool", "decide_continue")
    builder.add_conditional_edges(
        "decide_continue",
        route_after_continue,
        {"select_action": "select_action", "generate_answer": "generate_answer"},
    )
    builder.add_edge("generate_answer", END)

    return builder.compile()


__all__ = ["build_graph", "route_after_action", "route_after_continue"]