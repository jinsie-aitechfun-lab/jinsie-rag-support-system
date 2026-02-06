from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Protocol

State = Dict[str, Any]


class Node(Protocol):
    name: str

    def __call__(self, state: State) -> State: ...


@dataclass
class RetrieverNode:
    name: str = "retriever"
    retrieve_fn: Callable[[State], State] = lambda s: s  # returns {"docs":..., "context":..., "augmented":..., "resp_mode":...}

    def __call__(self, state: State) -> State:
        out = self.retrieve_fn(state)
        state.update(out)
        return state


@dataclass
class AnswerNode:
    name: str = "answer"
    answer_fn: Callable[[State], State] = lambda s: s  # returns {"result":...}

    def __call__(self, state: State) -> State:
        out = self.answer_fn(state)
        state.update(out)
        return state


class GraphRunner:
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes

    def run(self, state: State) -> State:
        for node in self.nodes:
            state = node(state)
        return state
