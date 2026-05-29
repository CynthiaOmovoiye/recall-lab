"""Tests for the vector-retrieval control's plumbing.

The model call in `respond` needs an API key, so it is not exercised here.
What these tests pin down is the retrieval store: documents go in, the top-k
most similar come back, and each instance is isolated from the next. A
deterministic embedding function keeps the tests offline and key-free.
"""

from __future__ import annotations

import chromadb
import pytest

from recall_lab.controls.vector import VectorRetrievalAgent


class _BagOfCharsEF(chromadb.EmbeddingFunction):
    """A tiny deterministic embedding function over character counts.

    Not semantic, but stable and offline: documents that share characters land
    near each other, which is enough to test that retrieval returns the right
    neighbours. Subclasses Chroma's EmbeddingFunction so it inherits the
    query/document embedding plumbing the newer API expects.
    """

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def __call__(self, input):  # noqa: A002 - Chroma's interface names it `input`
        vectors = []
        for document in input:
            vector = [0.0] * self.dim
            for index, char in enumerate(document.lower()):
                vector[ord(char) % self.dim] += 1.0
            vectors.append(vector)
        return vectors

    @staticmethod
    def name() -> str:
        return "bag-of-chars-test-ef"


def _make_agent(top_k: int = 2, name: str = "recall_vec_unit") -> VectorRetrievalAgent:
    return VectorRetrievalAgent(
        top_k=top_k,
        collection_name=name,
        embedding_function=_BagOfCharsEF(),
        client_factory=chromadb.EphemeralClient,
    )


def test_empty_store_retrieves_nothing() -> None:
    agent = _make_agent()
    assert agent._retrieve("where do I ship?") == []


def test_stored_exchanges_come_back_by_similarity() -> None:
    agent = _make_agent(top_k=1)
    agent._store("Ship my orders to Lagos.", "Got it, shipping to Lagos.")
    agent._store("My favorite color is green.", "Noted, green it is.")

    top = agent._retrieve("what is my favorite color?")
    assert len(top) == 1
    assert "green" in top[0].lower()


def test_top_k_is_capped_by_what_is_stored() -> None:
    agent = _make_agent(top_k=5)
    agent._store("Ship to Lagos.", "Okay.")
    # Only one document exists; query must not fail asking for five.
    assert len(agent._retrieve("shipping")) == 1


def test_instances_do_not_share_state() -> None:
    first = _make_agent(name="recall_vec_iso")
    first._store("Ship to Lagos.", "Okay.")
    second = _make_agent(name="recall_vec_iso")
    # A fresh instance reuses the collection name but must start empty.
    assert second.collection.count() == 0


def test_zero_budget_agent_construction_is_independent_of_key() -> None:
    # Construction must not require an API key; only respond() does.
    agent = _make_agent()
    assert agent.last_input_tokens == 0
    assert agent._next_id == 0


def test_respond_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("recall_lab.controls.vector.OPENROUTER_API_KEY", "")
    agent = _make_agent()
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        agent.respond("hello")
