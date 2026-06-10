"""Industry-standard RAG baseline.

The flat vector control (`controls/vector.py`) is the naive end of retrieval:
top-k cosine similarity over raw exchanges, nothing else. Beating it proves
little, because no serious team ships retrieval that bare. This control is the
strong end. It stacks the techniques a production RAG pipeline actually uses:

1. Query rewriting. The raw user turn is rewritten into a retrieval query
   before search, so "where do you ship today" becomes an explicit query for the
   current shipping city. This is the standard rewrite/expansion step.
2. Hybrid retrieval. Dense vector search (ChromaDB) and lexical search (BM25)
   run in parallel and are fused with Reciprocal Rank Fusion. Dense catches
   paraphrase, lexical catches exact tokens, fusion takes both.
3. Recency-aware ranking. A recency boost lifts newer exchanges so a fresh fact
   outranks a stale one of equal similarity. Two modes: "turn" uses insertion
   order; "timestamp" uses a real `added_at` date stored as chunk metadata,
   sourced from the scenario dates via `set_clock`.
4. Date-metadata filtering. In timestamp mode, `recency_window_days` drops
   candidates older than the window before reranking. This is the explicit
   metadata-by-date filter a production RAG applies. It helps current-fact
   queries and, by design, cannot surface a fact that predates the window.
5. Reranking. The fused candidates are reranked by a relevance scorer before the
   top few are kept, the cross-encoder / LLM-reranker step that usually buys the
   biggest quality jump.

The point of this control is the honest test behind Chapter 3. Flat vector fails
the relocation chain because similarity has no notion of what is current. The
open question is whether this whole stack closes that gap. Recency boosting is a
heuristic that approximates authority. The hypothesis under test is that it
clears single corrections and still breaks on a chain or an adversarial
re-assertion, because recency is not the same as a validity decision. This
control is what makes "RAG misses authority" a measured claim instead of a
claim against a strawman.

Dependencies follow the repo pattern: ChromaDB is imported lazily so the rest of
the lab does not pay its import cost, and `rank_bm25` is optional with a
token-overlap fallback so the control still runs without it (the fallback is
logged, never silent). Every external piece (dense retriever, reranker, query
rewriter) is injectable, so the plumbing is tested offline with deterministic
fakes and no API key.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone


def _fmt_stamp(ex: "_Exchange") -> str:
    """Human-readable recency tag for a snippet: a real date if known, else turn."""
    if ex.added_at is not None:
        return datetime.fromtimestamp(ex.added_at, tz=timezone.utc).strftime("%Y-%m-%d")
    return f"turn {ex.turn}"

from recall_lab.config import (
    AGENT_MODEL,
    MAX_OUTPUT_TOKENS,
    OPENROUTER_API_KEY,
)
from recall_lab.eval.metrics import estimate_tokens
from recall_lab.llm import chat_client, complete

# Reciprocal Rank Fusion constant. 60 is the value from the original RRF paper
# (Cormack et al., 2009) and the de facto default in hybrid-search stacks.
RRF_K = 60

STRONG_RAG_PREAMBLE = (
    "You are a helpful assistant with access to retrieved snippets of this "
    "conversation's history, ranked by a retrieval pipeline. Treat the snippets "
    "as your memory of earlier turns. The snippets are ordered best-first. If a "
    "later snippet corrects an earlier one, the later statement is what is "
    "currently true. If the snippets do not contain a personal fact the user "
    "asks for, say you do not know rather than guessing."
)

QUERY_REWRITE_PROMPT = (
    "Rewrite the user's message into a single, explicit search query for "
    "retrieving the relevant earlier turns of this conversation. Resolve "
    "references like 'today' or 'now' into what is actually being asked "
    "(for example, the current shipping city). Return only the rewritten "
    "query, no preamble.\n\nUser message: {message}"
)

RERANK_PROMPT = (
    "Score how relevant each snippet is for answering the query, from 0.0 "
    "(irrelevant) to 1.0 (directly answers it). Prefer snippets that state the "
    "currently true value when the query asks about the present.\n\n"
    "Query: {query}\n\n"
    "Snippets:\n{snippets}\n\n"
    "Return one line per snippet in the form `index: score`, indices starting "
    "at 1, nothing else."
)


@dataclass
class _Exchange:
    """One stored, retrievable past exchange."""

    doc_id: int
    text: str
    turn: int  # monotonic recency signal from insertion order; higher is newer
    added_at: float | None = None  # real timestamp (epoch seconds) when stored


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens for the BM25 path and its fallback."""
    return re.findall(r"[a-z0-9]+", text.lower())


class StrongRAGAgent:
    """Production-grade RAG: rewrite, hybrid retrieve, recency-boost, rerank.

    Implements the `.respond(str) -> str` agent protocol used by the harness.
    It has no validity state by design: the question is whether the retrieval
    stack alone can order superseded facts. Authority is exactly what it lacks,
    so any success on the relocation chain comes from recency heuristics, and
    any failure isolates the gap Recall Lab's validity layer is built to fill.
    """

    def __init__(
        self,
        top_k: int = 5,
        candidate_k: int = 12,
        recency_weight: float = 0.30,
        recency_mode: str = "turn",
        recency_window_days: float | None = None,
        show_timestamps: bool = False,
        chronological: bool = False,
        collection_name: str = "recall_strong_rag",
        embedding_function: Callable | None = None,
        client_factory: Callable | None = None,
        dense_retriever: Callable[[str, int], list[int]] | None = None,
        query_rewriter: Callable[[str], str] | None = None,
        reranker: Callable[[str, Sequence[str]], list[float]] | None = None,
    ) -> None:
        """Wire the pipeline.

        top_k: snippets kept in the final context after reranking.
        candidate_k: snippets pulled from fusion before reranking.
        recency_weight: how hard newer exchanges are boosted, 0 disables it.
        recency_mode: "turn" boosts by insertion order (the default, no date
            metadata). "timestamp" boosts by each chunk's real `added_at`, the
            date-metadata version teams ship. The runner sets the clock per day
            via `set_clock` so timestamps come from the scenario dates.
        recency_window_days: when set, this is metadata filtering by date added.
            Candidates older than `latest_added_at - window` are dropped before
            reranking. None disables the filter. Only takes effect in timestamp
            mode, since turn mode has no real dates to filter on.
        dense_retriever / query_rewriter / reranker: injection points. When left
            None they use ChromaDB and the agent model. Tests inject fakes so the
            fusion, recency, filter, and rerank logic run without network or key.
        """
        if recency_mode not in {"turn", "timestamp"}:
            raise ValueError(f"recency_mode must be 'turn' or 'timestamp', got {recency_mode!r}")
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.recency_weight = recency_weight
        self.recency_mode = recency_mode
        self.recency_window_days = recency_window_days
        self.show_timestamps = show_timestamps
        self.chronological = chronological
        self.collection_name = collection_name
        self.last_input_tokens = 0
        self._turn = 0
        self._clock: float | None = None
        self.exchanges: list[_Exchange] = []

        self._injected_dense = dense_retriever
        self._injected_rewriter = query_rewriter
        self._injected_reranker = reranker
        self._bm25_unavailable_logged = False

        # ChromaDB is only needed for the default dense path. Skip the import and
        # the collection entirely when a dense retriever is injected.
        self.collection = None
        if dense_retriever is None:
            import chromadb

            client = client_factory() if client_factory else chromadb.EphemeralClient()
            try:
                client.delete_collection(collection_name)
            except Exception:  # noqa: BLE001 - collection simply did not exist
                pass
            create_kwargs = {"name": collection_name}
            if embedding_function is not None:
                create_kwargs["embedding_function"] = embedding_function
            self.collection = client.create_collection(**create_kwargs)

    # ── retrieval stages ─────────────────────────────────────────────────────

    def _rewrite_query(self, message: str) -> str:
        """Resolve references into an explicit retrieval query."""
        if self._injected_rewriter is not None:
            return self._injected_rewriter(message)
        client = chat_client()
        completion = complete(
            client,
            model=AGENT_MODEL,
            messages=[{"role": "user", "content": QUERY_REWRITE_PROMPT.format(message=message)}],
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        rewritten = (completion.choices[0].message.content or "").strip()
        return rewritten or message

    def _dense_ranked_ids(self, query: str) -> list[int]:
        """Doc ids from dense vector search, best-first."""
        if self._injected_dense is not None:
            return self._injected_dense(query, self.candidate_k)
        if self.collection is None or self.collection.count() == 0:
            return []
        result = self.collection.query(
            query_texts=[query],
            n_results=min(self.candidate_k, self.collection.count()),
        )
        ids = (result.get("ids") or [[]])[0]
        return [int(i) for i in ids]

    def _lexical_ranked_ids(self, query: str) -> list[int]:
        """Doc ids from BM25, best-first, with a token-overlap fallback."""
        if not self.exchanges:
            return []
        corpus = [_tokenize(ex.text) for ex in self.exchanges]
        query_tokens = _tokenize(query)
        try:
            from rank_bm25 import BM25Okapi

            scores = BM25Okapi(corpus).get_scores(query_tokens)
        except Exception:  # noqa: BLE001 - optional dep or empty query
            if not self._bm25_unavailable_logged:
                print("[strong_rag] rank_bm25 unavailable, using token-overlap fallback")
                self._bm25_unavailable_logged = True
            query_set = set(query_tokens)
            scores = [sum(1 for t in doc if t in query_set) for doc in corpus]
        ranked = sorted(range(len(self.exchanges)), key=lambda i: scores[i], reverse=True)
        return [self.exchanges[i].doc_id for i in ranked if scores[i] > 0][: self.candidate_k]

    def _fuse(self, dense_ids: list[int], lexical_ids: list[int]) -> dict[int, float]:
        """Reciprocal Rank Fusion of two ranked id lists into a score map."""
        fused: dict[int, float] = {}
        for ranked in (dense_ids, lexical_ids):
            for rank, doc_id in enumerate(ranked):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        return fused

    def _recency_score(self, ex: _Exchange, oldest: float, newest: float) -> float:
        """Normalized 0 (oldest) to 1 (newest) recency for one exchange.

        In turn mode the signal is insertion order. In timestamp mode it is the
        real `added_at` date, falling back to turn order when a chunk has no
        timestamp so a partially dated store still ranks sanely.
        """
        if self.recency_mode == "timestamp" and ex.added_at is not None and newest > oldest:
            return (ex.added_at - oldest) / (newest - oldest)
        if self._turn == 0:
            return 0.0
        return ex.turn / self._turn

    def _apply_recency(self, fused: dict[int, float]) -> dict[int, float]:
        """Add a recency boost so newer exchanges of equal relevance rank higher.

        The boost is normalized to the same scale as the fused RRF scores so
        recency_weight reads as a fraction of the top fusion score, not an
        arbitrary unit. recency_weight of 0 leaves fusion untouched.
        """
        if self.recency_weight <= 0 or not fused:
            return fused
        by_id = {ex.doc_id: ex for ex in self.exchanges}
        stamps = [ex.added_at for ex in self.exchanges if ex.added_at is not None]
        oldest = min(stamps) if stamps else 0.0
        newest = max(stamps) if stamps else 0.0
        top_fused = max(fused.values())
        boosted = dict(fused)
        for doc_id in fused:
            ex = by_id.get(doc_id)
            if ex is None:
                continue
            recency = self._recency_score(ex, oldest, newest)
            boosted[doc_id] += self.recency_weight * top_fused * recency
        return boosted

    def _date_filter(self, candidate_ids: list[int]) -> list[int]:
        """Metadata filtering by date added: drop candidates outside the window.

        When `recency_window_days` is set in timestamp mode, candidates whose
        `added_at` is older than `latest_added_at - window` are removed before
        reranking. This is the explicit date filter a production RAG applies on a
        timestamp metadata field. It helps current-fact queries and, by design,
        cannot surface a fact that predates the window. No-op without timestamps.
        """
        if self.recency_window_days is None or self.recency_mode != "timestamp":
            return candidate_ids
        by_id = {ex.doc_id: ex for ex in self.exchanges}
        stamps = [by_id[i].added_at for i in candidate_ids if by_id.get(i) and by_id[i].added_at is not None]
        if not stamps:
            return candidate_ids
        cutoff = max(stamps) - self.recency_window_days * 86400
        kept = [i for i in candidate_ids if by_id.get(i) and (by_id[i].added_at or 0) >= cutoff]
        # Never filter down to nothing; an empty context is worse than a stale hit.
        return kept or candidate_ids

    def _rerank(self, query: str, candidate_ids: list[int]) -> list[int]:
        """Reorder candidates by a relevance scorer, best-first."""
        if not candidate_ids:
            return []
        by_id = {ex.doc_id: ex for ex in self.exchanges}
        docs = [by_id[i].text for i in candidate_ids if i in by_id]
        ids = [i for i in candidate_ids if i in by_id]
        if not ids:
            return []
        scores = (
            self._injected_reranker(query, docs)
            if self._injected_reranker is not None
            else self._llm_rerank_scores(query, docs)
        )
        order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
        return [ids[i] for i in order]

    def _llm_rerank_scores(self, query: str, docs: Sequence[str]) -> list[float]:
        """Ask the agent model to score each candidate's relevance to the query."""
        snippet_block = "\n".join(f"{i}. {d}" for i, d in enumerate(docs, start=1))
        client = chat_client()
        completion = complete(
            client,
            model=AGENT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": RERANK_PROMPT.format(query=query, snippets=snippet_block),
                }
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        text = completion.choices[0].message.content or ""
        scores = [0.0] * len(docs)
        for line in text.splitlines():
            match = re.match(r"\s*(\d+)\s*[:.)]\s*([0-9]*\.?[0-9]+)", line)
            if not match:
                continue
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(scores):
                scores[idx] = float(match.group(2))
        return scores

    def _retrieve(self, message: str) -> list[str]:
        """Full retrieval stack: rewrite, hybrid, recency, rerank, top-k."""
        query = self._rewrite_query(message)
        dense_ids = self._dense_ranked_ids(query)
        lexical_ids = self._lexical_ranked_ids(query)
        fused = self._fuse(dense_ids, lexical_ids)
        if not fused:
            return []
        boosted = self._apply_recency(fused)
        ranked_ids = sorted(boosted, key=lambda i: boosted[i], reverse=True)[: self.candidate_k]
        ranked_ids = self._date_filter(ranked_ids)
        reranked_ids = self._rerank(query, ranked_ids)[: self.top_k]
        by_id = {ex.doc_id: ex for ex in self.exchanges}
        if self.chronological:
            # Present the kept snippets oldest-first, so a temporal question is
            # not handed a relevance-scrambled order. This is the fair way to
            # give retrieval a shot at reconstructing the order of a chain.
            reranked_ids = sorted(
                reranked_ids,
                key=lambda i: (
                    by_id[i].added_at
                    if i in by_id and by_id[i].added_at is not None
                    else (by_id[i].turn if i in by_id else 0)
                ),
            )
        snippets = []
        for i in reranked_ids:
            if i not in by_id:
                continue
            ex = by_id[i]
            # When timestamps are shown, prefix each snippet with its date so the
            # model can reconstruct the order of a correction chain from context.
            snippets.append(f"[{_fmt_stamp(ex)}] {ex.text}" if self.show_timestamps else ex.text)
        return snippets

    def set_clock(self, timestamp: float | None) -> None:
        """Set the wall-clock time used to stamp the next stored exchanges.

        The runner calls this once per simulated day with the scenario date, so
        timestamp-mode recency and the date filter use real dates rather than
        insertion order. Left unset, `added_at` stays None and the agent falls
        back to turn-order recency.
        """
        self._clock = timestamp

    def _store(self, user_message: str, response: str) -> None:
        """Index one completed exchange for future retrieval."""
        doc_id = len(self.exchanges)
        document = f"User: {user_message}\nAssistant: {response}"
        self.exchanges.append(
            _Exchange(doc_id=doc_id, text=document, turn=self._turn, added_at=self._clock)
        )
        if self.collection is not None:
            metadatas = [{"added_at": self._clock}] if self._clock is not None else None
            self.collection.add(documents=[document], ids=[str(doc_id)], metadatas=metadatas)

    # ── agent protocol ───────────────────────────────────────────────────────

    def respond(self, user_message: str) -> str:
        """Run the retrieval stack, compose context, answer, then store."""
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is required to run StrongRAGAgent.")

        self._turn += 1
        snippets = self._retrieve(user_message)

        lines = [STRONG_RAG_PREAMBLE, "", "## Retrieved memory (best first)"]
        if snippets:
            lines.extend(f"{i}. {snippet}" for i, snippet in enumerate(snippets, start=1))
        else:
            lines.append("(nothing retrieved yet)")
        lines.extend(["", "## Current user message", user_message])
        prompt = "\n".join(lines).strip()
        self.last_input_tokens = estimate_tokens(prompt)

        client = chat_client()
        completion = complete(
            client,
            model=AGENT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        response = completion.choices[0].message.content or ""

        self._store(user_message, response)
        return response
