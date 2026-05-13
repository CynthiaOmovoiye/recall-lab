"""Vector-retrieval baseline.

Embeds every past exchange in ChromaDB. On each turn, retrieves the top-k most
similar past exchanges and prepends them. This is the Mem0 / standard-RAG
default for "agent memory."

The point of this control is to establish how far "store everything, retrieve
the relevant" goes before it plateaus.
"""

from __future__ import annotations


class VectorRetrievalAgent:
    """Flat vector retrieval over every past exchange."""

    def __init__(self, top_k: int = 5, collection_name: str = "recall_vector_control") -> None:
        self.top_k = top_k
        self.collection_name = collection_name
        # TODO: init ChromaDB client and collection

    def respond(self, user_message: str) -> str:
        """Embed query, retrieve top-k past exchanges, compose context, respond."""
        # TODO: embed user_message, query Chroma, build context, call model, store new exchange
        raise NotImplementedError(
            "Wire up ChromaDB + OpenRouter client and return the response."
        )
