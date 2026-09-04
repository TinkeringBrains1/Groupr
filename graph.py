"""
LangGraph wrapper around the pipeline. Every node calls the exact same,
already-tested functions from agents/ -- no logic is rewritten or
duplicated. This module only wires them into a graph with shared state.

Graph shape (linear, matches the locked architecture):
    dedup_and_blocking -> cluster -> direct_match -> transitive_link -> drift -> END

Direct Match's confirm/reject/retry loop with Cluster Agent stays INTERNAL
to the direct_match node rather than being separate bouncing graph edges --
restructuring proven, validated per-cluster retry logic into LangGraph's
shared-state model was assessed as real correctness risk for what would be
a purely cosmetic diagram difference. The flow below is what actually runs.
"""

import os
import sys
from typing import TypedDict, Optional, Callable

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "agents"))

from langgraph.graph import StateGraph, END

import dedup
import blocking
import cluster
import direct_match
import transitive_link
import drift


class PipelineState(TypedDict):
    dataset: dict
    clean_gateway: list
    clean_ledger: list
    dedup_exception_report: list
    candidate_pools: dict
    clusters: dict
    decision_log: dict


def node_dedup_and_blocking(state: PipelineState) -> dict:
    dataset = state["dataset"]
    dedup_result = dedup.run_duplicate_check(dataset)
    candidate_pools = blocking.run_blocking(
        dedup_result["clean_gateway"], dedup_result["clean_ledger"], dataset["settlement"]
    )
    return {
        "clean_gateway": dedup_result["clean_gateway"],
        "clean_ledger": dedup_result["clean_ledger"],
        "dedup_exception_report": dedup_result["exception_report"],
        "candidate_pools": candidate_pools,
    }


def node_cluster(state: PipelineState) -> dict:
    clusters = cluster.run_cluster_agent(state["candidate_pools"])
    return {"clusters": clusters}


def make_node_direct_match(llm_batch_fn: Optional[Callable] = None):
    def node(state: PipelineState) -> dict:
        kwargs = {"llm_batch_fn": llm_batch_fn} if llm_batch_fn is not None else {}
        clusters = direct_match.run_direct_match(
            state["clusters"], state["dataset"], state["decision_log"], **kwargs
        )
        return {"clusters": clusters, "decision_log": state["decision_log"]}
    return node


def make_node_transitive_link(llm_call_fn: Optional[Callable] = None):
    def node(state: PipelineState) -> dict:
        clusters = transitive_link.run_transitive_link(
            state["clusters"], state["dataset"], state["decision_log"], llm_call_fn=llm_call_fn
        )
        return {"clusters": clusters, "decision_log": state["decision_log"]}
    return node


def make_node_drift(llm_call_fn: Optional[Callable] = None):
    def node(state: PipelineState) -> dict:
        clusters = drift.run_drift(
            state["clusters"], state["dataset"], state["decision_log"], llm_call_fn=llm_call_fn
        )
        return {"clusters": clusters, "decision_log": state["decision_log"]}
    return node


def build_graph(direct_match_llm_fn=None, transitive_llm_fn=None, drift_llm_fn=None):
    graph = StateGraph(PipelineState)
    graph.add_node("dedup_and_blocking", node_dedup_and_blocking)
    graph.add_node("cluster", node_cluster)
    graph.add_node("direct_match", make_node_direct_match(direct_match_llm_fn))
    graph.add_node("transitive_link", make_node_transitive_link(transitive_llm_fn))
    graph.add_node("drift", make_node_drift(drift_llm_fn))

    graph.set_entry_point("dedup_and_blocking")
    graph.add_edge("dedup_and_blocking", "cluster")
    graph.add_edge("cluster", "direct_match")
    graph.add_edge("direct_match", "transitive_link")
    graph.add_edge("transitive_link", "drift")
    graph.add_edge("drift", END)

    return graph.compile()


def run_pipeline(dataset, direct_match_llm_fn=None, transitive_llm_fn=None, drift_llm_fn=None):
    app = build_graph(direct_match_llm_fn, transitive_llm_fn, drift_llm_fn)
    initial_state = {"dataset": dataset, "decision_log": {}}
    return app.invoke(initial_state)
