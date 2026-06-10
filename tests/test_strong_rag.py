"""Tests for the industry-standard RAG control.

The model call needs an API key, so `respond` is not exercised end to end. What
matters here is the retrieval plumbing: RRF fusion combines two rankings, the
recency boost lifts newer exchanges, reranking reorders by the relevance scorer,
and the full pipeline composes those stages. Every external piece is injected so
nothing here touches ChromaDB, BM25, or the network. The key guard is checked
too.
"""

from __future__ import annotations

import pytest

from recall_lab.controls.strong_rag import StrongRAGAgent, _Exchange


def _agent(**kwargs) -> StrongRAGAgent:
    """Build an agent with the dense path injected, so ChromaDB is never imported."""
    kwargs.setdefault("dense_retriever", lambda query, k: [])
    kwargs.setdefault("query_rewriter", lambda message: message)
    return StrongRAGAgent(**kwargs)


def _seed(agent: StrongRAGAgent, texts: list[str]) -> None:
    """Populate exchanges with explicit, increasing turn indices (oldest first)."""
    for i, text in enumerate(texts):
        agent.exchanges.append(_Exchange(doc_id=i, text=text, turn=i + 1))
    agent._turn = len(texts)


def test_fuse_rewards_documents_in_both_rankings() -> None:
    agent = _agent()
    fused = agent._fuse(dense_ids=[0, 1], lexical_ids=[1, 2])
    # Doc 1 is ranked by both retrievers, so it should outscore 0 and 2.
    assert fused[1] > fused[0]
    assert fused[1] > fused[2]


def test_recency_boost_prefers_newer_on_equal_relevance() -> None:
    agent = _agent(recency_weight=1.0)
    _seed(agent, ["oldest", "newest"])
    fused = {0: 0.10, 1: 0.10}  # identical fusion scores
    boosted = agent._apply_recency(fused)
    # Same relevance, so recency must break the tie toward the newer doc.
    assert boosted[1] > boosted[0]


def test_recency_weight_zero_is_a_noop() -> None:
    agent = _agent(recency_weight=0.0)
    _seed(agent, ["a", "b"])
    fused = {0: 0.2, 1: 0.1}
    assert agent._apply_recency(fused) == fused


def test_rerank_orders_by_injected_scores() -> None:
    agent = _agent(reranker=lambda query, docs: [0.1, 0.9, 0.5])
    _seed(agent, ["doc0", "doc1", "doc2"])
    order = agent._rerank("q", [0, 1, 2])
    # Highest score first: doc1, then doc2, then doc0.
    assert order == [1, 2, 0]


def test_pipeline_surfaces_the_current_fact_via_recency() -> None:
    agent = _agent(
        top_k=1,
        recency_weight=1.0,
        dense_retriever=lambda query, k: [0, 1, 2],
        query_rewriter=lambda message: message,
        reranker=lambda query, docs: [1.0] * len(docs),  # neutral reranker
    )
    _seed(
        agent,
        [
            "User: Ship to Lagos\nAssistant: ok, Lagos",
            "User: Ship to Berlin\nAssistant: ok, Berlin",
            "User: Ship to Nairobi\nAssistant: ok, Nairobi",
        ],
    )
    snippets = agent._retrieve("What city should you ship to today?")
    # With a neutral reranker, recency decides, so the newest city wins.
    assert len(snippets) == 1
    assert "Nairobi" in snippets[0]


def test_reranker_can_override_recency() -> None:
    agent = _agent(
        top_k=1,
        recency_weight=1.0,
        dense_retriever=lambda query, k: [0, 1, 2],
        # Content-aware reranker: forces the Lagos doc regardless of candidate order.
        reranker=lambda query, docs: [1.0 if "Lagos" in d else 0.0 for d in docs],
    )
    _seed(agent, ["first city Lagos", "then Berlin", "now Nairobi"])
    snippets = agent._retrieve("Where did the user first live?")
    assert "Lagos" in snippets[0]


def test_lexical_retrieval_ranks_the_matching_exchange() -> None:
    agent = _agent()
    _seed(agent, ["my favorite color is green", "ship to Nairobi", "I enjoy jazz"])
    ids = agent._lexical_ranked_ids("what is my favorite color")
    assert ids, "expected at least one lexical hit"
    assert ids[0] == 0  # the color exchange matches the query tokens best


def test_empty_store_retrieves_nothing() -> None:
    agent = _agent(dense_retriever=lambda query, k: [])
    assert agent._retrieve("anything") == []


def test_respond_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("recall_lab.controls.strong_rag.OPENROUTER_API_KEY", "")
    agent = _agent()
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        agent.respond("hello")


# ── date-metadata variant ────────────────────────────────────────────────────

_DAY = 86400.0


def test_set_clock_stamps_added_at() -> None:
    agent = _agent()
    agent.set_clock(1234.0)
    agent._store("ship to Lagos", "ok")
    assert agent.exchanges[-1].added_at == 1234.0


def test_timestamp_recency_prefers_newer_by_date() -> None:
    agent = _agent(recency_weight=1.0, recency_mode="timestamp")
    agent.exchanges = [
        _Exchange(doc_id=0, text="old", turn=1, added_at=1000.0),
        _Exchange(doc_id=1, text="new", turn=2, added_at=2000.0),
    ]
    boosted = agent._apply_recency({0: 0.10, 1: 0.10})
    assert boosted[1] > boosted[0]


def test_date_filter_drops_candidates_outside_the_window() -> None:
    agent = _agent(recency_mode="timestamp", recency_window_days=1.0)
    agent.exchanges = [
        _Exchange(doc_id=0, text="day0", turn=1, added_at=0.0),
        _Exchange(doc_id=1, text="day1", turn=2, added_at=1 * _DAY),
        _Exchange(doc_id=2, text="day2", turn=3, added_at=2 * _DAY),
    ]
    # cutoff = newest (2*DAY) - 1 day = 1*DAY, so day0 is filtered out.
    assert agent._date_filter([0, 1, 2]) == [1, 2]


def test_date_filter_is_noop_in_turn_mode() -> None:
    agent = _agent(recency_mode="turn", recency_window_days=1.0)
    agent.exchanges = [
        _Exchange(doc_id=0, text="a", turn=1, added_at=0.0),
        _Exchange(doc_id=1, text="b", turn=2, added_at=2 * _DAY),
    ]
    # Turn mode has no date semantics, so the window must not filter anything.
    assert agent._date_filter([0, 1]) == [0, 1]


def test_date_filter_never_returns_empty() -> None:
    agent = _agent(recency_mode="timestamp", recency_window_days=0.0)
    agent.exchanges = [
        _Exchange(doc_id=0, text="old", turn=1, added_at=0.0),
        _Exchange(doc_id=1, text="new", turn=2, added_at=5 * _DAY),
    ]
    kept = agent._date_filter([0, 1])
    assert kept  # an empty context is worse than a stale hit
    assert 1 in kept  # the newest always survives


# ── fair-shot: timestamps visible in context ─────────────────────────────────

def _epoch(y, m, d) -> float:
    import datetime as dt
    return dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp()


def test_show_timestamps_prefixes_snippets_with_date() -> None:
    agent = _agent(
        top_k=2,
        show_timestamps=True,
        dense_retriever=lambda query, k: [0, 1],
        reranker=lambda query, docs: [1.0] * len(docs),
    )
    agent.exchanges = [
        _Exchange(doc_id=0, text="ship to Lagos", turn=1, added_at=_epoch(2026, 5, 24)),
        _Exchange(doc_id=1, text="ship to Nairobi", turn=2, added_at=_epoch(2026, 5, 26)),
    ]
    agent._turn = 2
    snips = agent._retrieve("where do I ship now")
    # Each snippet is tagged with its real date so the model can order the chain.
    assert any(s.startswith("[2026-05-24]") for s in snips)
    assert any(s.startswith("[2026-05-26]") for s in snips)


def test_show_timestamps_off_has_no_prefix() -> None:
    agent = _agent(
        top_k=2,
        show_timestamps=False,
        dense_retriever=lambda query, k: [0, 1],
        reranker=lambda query, docs: [1.0] * len(docs),
    )
    agent.exchanges = [
        _Exchange(doc_id=0, text="ship to Lagos", turn=1, added_at=_epoch(2026, 5, 24)),
        _Exchange(doc_id=1, text="ship to Nairobi", turn=2, added_at=_epoch(2026, 5, 26)),
    ]
    agent._turn = 2
    snips = agent._retrieve("where do I ship now")
    assert all(not s.startswith("[") for s in snips)


def test_chronological_orders_snippets_oldest_first() -> None:
    agent = _agent(
        top_k=3,
        chronological=True,
        dense_retriever=lambda query, k: [2, 0, 1],  # deliberately scrambled order
        reranker=lambda query, docs: [1.0] * len(docs),  # neutral, no reordering
    )
    agent.exchanges = [
        _Exchange(doc_id=0, text="Lagos", turn=1, added_at=_epoch(2026, 5, 24)),
        _Exchange(doc_id=1, text="Berlin", turn=2, added_at=_epoch(2026, 5, 25)),
        _Exchange(doc_id=2, text="Nairobi", turn=3, added_at=_epoch(2026, 5, 26)),
    ]
    agent._turn = 3
    snips = agent._retrieve("where do I ship")
    joined = " | ".join(snips)
    # Regardless of retrieval or rerank order, the context is oldest-first.
    assert joined.index("Lagos") < joined.index("Berlin") < joined.index("Nairobi")
