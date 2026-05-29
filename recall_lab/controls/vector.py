"""Vector-retrieval baseline.

Embeds every past exchange in ChromaDB. On each turn, retrieves the top-k most
similar past exchanges and prepends them. This is the Mem0 / standard-RAG
default for "agent memory."

The point of this control is to establish how far "store everything, retrieve
the relevant" goes before it plateaus. Unlike Recall Lab, it has no notion of
validity: a superseded fact (Lagos) and the correction that replaced it
(Berlin) are both just documents in the index. Whichever scores more similar to
the question wins, regardless of which is still true. That is the failure this
control is meant to expose.

Isolation: each instance gets its own in-memory ChromaDB client, so nothing
leaks across runs or across variance campaigns.

Embeddings: defaults to ChromaDB's bundled embedding function (a local ONNX
MiniLM model, downloaded on first use). Tests inject a deterministic function so
they need neither network nor an API key.
"""

from __future__ import annotations

from recall_lab.config import (
    AGENT_MODEL,
    MAX_OUTPUT_TOKENS,
    OPENROUTER_API_KEY,
)
from recall_lab.eval.metrics import estimate_tokens
from recall_lab.llm import chat_client, complete

RETRIEVAL_PREAMBLE = (
    "You are a helpful assistant with access to retrieved snippets of this "
    "conversation's history. Treat the snippets as your memory of earlier "
    "turns. If the snippets do not contain a personal fact the user asks for, "
    "say you do not know rather than guessing."
)


class VectorRetrievalAgent:
    """Flat vector retrieval over every past exchange.

    Implements the `.respond(str) -> str` agent protocol used by the harness.
    """

    def __init__(
        self,
        top_k: int = 5,
        collection_name: str = "recall_vector_control",
        embedding_function=None,
        client_factory=None,
    ) -> None:
        self.top_k = top_k
        self.collection_name = collection_name
        self.last_input_tokens = 0
        self._next_id = 0

        # Imported lazily so the rest of the lab does not pay ChromaDB's import
        # cost (and so a missing optional dep only breaks this control).
        import chromadb

        client = client_factory() if client_factory else chromadb.EphemeralClient()
        # A fresh client is empty, but guard anyway so re-runs in one process
        # never inherit a stale collection.
        try:
            client.delete_collection(collection_name)
        except Exception:  # noqa: BLE001 - collection simply did not exist
            pass

        create_kwargs = {"name": collection_name}
        if embedding_function is not None:
            create_kwargs["embedding_function"] = embedding_function
        self.collection = client.create_collection(**create_kwargs)

    def _retrieve(self, query: str) -> list[str]:
        """Return up to top_k stored exchange snippets most similar to query."""
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(
            query_texts=[query],
            n_results=min(self.top_k, count),
        )
        documents = result.get("documents") or [[]]
        return documents[0]

    def _store(self, user_message: str, response: str) -> None:
        """Index one completed exchange for future retrieval."""
        document = f"User: {user_message}\nAssistant: {response}"
        self.collection.add(documents=[document], ids=[str(self._next_id)])
        self._next_id += 1

    def respond(self, user_message: str) -> str:
        """Retrieve top-k past exchanges, compose context, answer, then store."""
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is required to run VectorRetrievalAgent.")

        snippets = self._retrieve(user_message)

        lines = [RETRIEVAL_PREAMBLE, "", "## Retrieved memory"]
        if snippets:
            for i, snippet in enumerate(snippets, start=1):
                lines.append(f"{i}. {snippet}")
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
