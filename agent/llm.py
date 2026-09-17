"""
Pluggable natural-language layer.

Supported providers:
- OpenAI
- Anthropic
- none / template fallback

IMPORTANT:
The LLM receives:
    - selected customer's booking only
    - rules
    - already-computed policy decision
    - grounded draft

The LLM cannot change the policy decision.
"""

from __future__ import annotations

import os

from time import perf_counter

from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

from .usage import LLMUsage


load_dotenv()


class LLM:

    def __init__(
        self,
        provider: Optional[str] = None
    ):

        self.provider = (
            provider
            or os.getenv(
                "LLM_PROVIDER",
                "none"
            )
        ).lower().strip()

        self.openai_model = os.getenv(
            "OPENAI_MODEL",
            "gpt-4o"
        )

        self.anthropic_model = os.getenv(
            "ANTHROPIC_MODEL",
            "claude-sonnet-5"
        )

    def enabled(self) -> bool:

        if self.provider == "openai":
            return bool(
                os.getenv(
                    "OPENAI_API_KEY"
                )
            )

        if self.provider == "anthropic":
            return bool(
                os.getenv(
                    "ANTHROPIC_API_KEY"
                )
            )

        return False

    def rewrite(
        self,
        *,
        grounded_draft: str,
        booking_context: Dict[str, Any],
        rules_context: Dict[str, Any],
        decision: Dict[str, Any],
    ) -> str:

        """
        Rewrite a grounded draft.

        The LLM is explicitly instructed that it cannot
        change the decision.

        Returns (text, usage). When no provider is configured the draft
        is returned unchanged with an empty usage record, so callers get
        the same shape either way.
        """

        if not self.enabled():
            return grounded_draft, LLMUsage(
                provider=self.provider,
            )

        system_prompt = """
You are a customer support phrasing assistant.

You are NOT a policy decision-maker.

Your only job is to improve the wording of an already-computed
customer support response.

NEVER:
- add a new policy
- invent a flight
- invent availability
- invent compensation
- invent a price
- invent a refund method
- invent a deadline
- invent an exception
- change an escalation decision
- change an allowed action
- add external knowledge

The supplied computed decision is authoritative.

Every policy statement must reference the applicable rule using
phrasing such as "under our cancellation rebooking rule",
"under our delay compensation rule", or
"under our fare difference rule".

Be empathetic, calm and professional.

Return only the customer-facing response.
"""

        user_prompt = {
            "grounded_draft": grounded_draft,

            "selected_customer_booking_only": booking_context,

            "rules": rules_context,

            "computed_decision": decision,
        }

        call = (
            self._openai
            if self.provider == "openai"
            else self._anthropic
            if self.provider == "anthropic"
            else None
        )

        if call is None:
            return grounded_draft, LLMUsage(provider=self.provider)

        started = perf_counter()

        try:
            text, usage = call(
                system_prompt,
                str(user_prompt),
            )

        except Exception as exc:  # noqa: BLE001 - surfaced in the UI

            # A phrasing failure must never cost the customer an answer:
            # fall back to the grounded draft, which is already complete
            # and correct, and report the error alongside it.
            return grounded_draft, LLMUsage(
                provider=self.provider,
                model=self._model_id(),
                latency_ms=(perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )

        usage.latency_ms = (perf_counter() - started) * 1000

        return (text or grounded_draft), usage

    def _model_id(self) -> Optional[str]:

        if self.provider == "openai":
            return self.openai_model

        if self.provider == "anthropic":
            return self.anthropic_model

        return None

    def _openai(
        self,
        system: str,
        user: str
    ) -> Tuple[str, LLMUsage]:

        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ[
                "OPENAI_API_KEY"
            ]
        )

        response = client.chat.completions.create(
            model=self.openai_model,

            temperature=0,

            messages=[
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": user,
                },
            ],
        )

        text = (
            response
            .choices[0]
            .message
            .content
        )

        reported = getattr(response, "usage", None)

        usage = LLMUsage(
            provider="openai",
            model=self.openai_model,
            input_tokens=getattr(reported, "prompt_tokens", 0) or 0,
            output_tokens=getattr(reported, "completion_tokens", 0) or 0,
        )

        return (
            text.strip() if text else "",
            usage,
        )

    def _anthropic(
        self,
        system: str,
        user: str
    ) -> Tuple[str, LLMUsage]:

        from anthropic import Anthropic

        client = Anthropic(
            api_key=os.environ[
                "ANTHROPIC_API_KEY"
            ]
        )

        # No temperature: sampling parameters are rejected with a 400 on
        # Sonnet 5 and the other current models. Thinking is off and effort
        # is low because this call only rephrases an already-final
        # decision — there is nothing here to reason about.
        response = client.messages.create(
            model=self.anthropic_model,

            max_tokens=1024,

            thinking={"type": "disabled"},

            output_config={"effort": "low"},

            system=system,

            messages=[
                {
                    "role": "user",
                    "content": user,
                }
            ],
        )

        parts = []

        for block in response.content:

            if getattr(
                block,
                "type",
                None
            ) == "text":

                parts.append(
                    block.text
                )

        reported = getattr(response, "usage", None)

        usage = LLMUsage(
            provider="anthropic",
            model=self.anthropic_model,
            input_tokens=getattr(reported, "input_tokens", 0) or 0,
            output_tokens=getattr(reported, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(
                reported, "cache_read_input_tokens", 0
            ) or 0,
            cache_write_tokens=getattr(
                reported, "cache_creation_input_tokens", 0
            ) or 0,
        )

        return (
            "\n".join(parts).strip(),
            usage,
        )