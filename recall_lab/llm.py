"""Shared OpenRouter client factory and request helper.

Every agent and the judge talk to OpenRouter through the OpenAI client. They
used to each construct their own client inline with the default two retries and
no control over which provider served the call. Two failure modes followed:

- A transient connection blip during a variance batch wiped nine of ten runs,
  because a multi-minute outage blows past two retries.
- A run died when OpenRouter routed a benign prompt to Azure, whose content
  filter false-flagged it as a jailbreak.

Centralizing construction here fixes both at once: `chat_client()` sets shared
retry and timeout settings, and `complete()` injects provider-routing
preferences so the same constraints apply to every call. See config.py for the
routing knobs.
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from recall_lab.config import (
    OPENROUTER_ALLOW_FALLBACKS,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_IGNORE_PROVIDERS,
    OPENROUTER_MAX_RETRIES,
    OPENROUTER_PROVIDER_ORDER,
    OPENROUTER_TIMEOUT_SECONDS,
)


def chat_client() -> OpenAI:
    """Build an OpenRouter-backed OpenAI client with shared resilience settings."""
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        max_retries=OPENROUTER_MAX_RETRIES,
        timeout=OPENROUTER_TIMEOUT_SECONDS,
    )


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _order_applies_to(model: str | None, order: list[str]) -> bool:
    """Whether the pinned provider order should be applied to this model.

    The pinned providers only serve their own vendor's models (the "OpenAI"
    provider serves `openai/*`, not `anthropic/*`). Applying an order to a model
    the provider cannot serve, with fallbacks off, would error the call. So the
    order is applied only when the model's vendor prefix matches the first
    pinned provider. A model with no match (the Anthropic judge and
    contradiction classifier) keeps its own routing. When the model is unknown,
    apply the order rather than silently dropping the pin.
    """
    if not order:
        return False
    if model is None:
        return True
    vendor = model.split("/", 1)[0].strip().lower()
    return vendor == order[0].strip().lower()


def provider_routing(model: str | None = None) -> dict[str, Any] | None:
    """Build OpenRouter's `provider` preference object, or None if unconstrained.

    Constraints come from config: an ignore list (default: Azure), a pinned
    provider order (default: OpenAI), and a fallback flag (default: off). The
    order is applied per model family so the OpenAI pin does not break the
    Anthropic judge. Azure is excluded on every call regardless of model.
    Returns None only if nothing applies, so callers can skip the field.
    """
    routing: dict[str, Any] = {}
    order = _csv(OPENROUTER_PROVIDER_ORDER)
    ignore = _csv(OPENROUTER_IGNORE_PROVIDERS)
    if _order_applies_to(model, order):
        routing["order"] = order
        routing["allow_fallbacks"] = OPENROUTER_ALLOW_FALLBACKS
    if ignore:
        routing["ignore"] = ignore
    if not routing:
        return None
    return routing


def complete(client: OpenAI, **kwargs: Any):
    """Create a chat completion with provider routing applied.

    A thin wrapper over `client.chat.completions.create` that injects the
    configured provider preferences into `extra_body`. Routing is keyed off the
    call's own `model`, so the pin follows the model family. Use this instead of
    calling `.create` directly so routing constraints hold everywhere.
    """
    routing = provider_routing(kwargs.get("model"))
    if routing is not None:
        extra_body = dict(kwargs.get("extra_body") or {})
        extra_body["provider"] = routing
        kwargs["extra_body"] = extra_body
    return client.chat.completions.create(**kwargs)
