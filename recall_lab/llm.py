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
    OPENROUTER_JUDGE_PROVIDER_ORDER,
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
    """Whether a pinned provider order should be applied to this model.

    A pinned provider only serves its own vendor's models (the "OpenAI" provider
    serves `openai/*`, not `anthropic/*`). Applying an order to a model the
    provider cannot serve, with fallbacks off, would error the call. So an order
    applies only when the model's vendor prefix matches the order's first
    provider. When the model is unknown, the agent pin is assumed to apply
    rather than silently dropping it.
    """
    if not order:
        return False
    if model is None:
        return True
    vendor = model.split("/", 1)[0].strip().lower()
    return vendor == order[0].strip().lower()


def _pinned_orders() -> list[list[str]]:
    """The configured provider orders, agent pin first then judge pin.

    Each model family is pinned to its own provider: PROVIDER_ORDER for the
    agent (openai/*), JUDGE_PROVIDER_ORDER for the Anthropic judge and
    contradiction classifier. provider_routing applies whichever order matches
    the call's model vendor.
    """
    orders = []
    for raw in (OPENROUTER_PROVIDER_ORDER, OPENROUTER_JUDGE_PROVIDER_ORDER):
        order = _csv(raw)
        if order:
            orders.append(order)
    return orders


def provider_routing(model: str | None = None) -> dict[str, Any] | None:
    """Build OpenRouter's `provider` preference object, or None if unconstrained.

    Constraints come from config: an ignore list (default: Azure) and per-vendor
    pinned orders (agent: OpenAI, judge: Anthropic) with fallbacks off. The order
    whose provider serves the call's model vendor is applied, so each model
    family lands on a single provider without breaking the other. Azure is
    excluded on every call regardless of model. Returns None only if nothing
    applies, so callers can skip the field.
    """
    routing: dict[str, Any] = {}
    ignore = _csv(OPENROUTER_IGNORE_PROVIDERS)
    for order in _pinned_orders():
        if _order_applies_to(model, order):
            routing["order"] = order
            routing["allow_fallbacks"] = OPENROUTER_ALLOW_FALLBACKS
            break
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
