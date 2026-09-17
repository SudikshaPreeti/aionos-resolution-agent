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

from typing import Any, Dict, Optional

from dotenv import load_dotenv


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
            "claude-3-5-sonnet-latest"
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
        """

        if not self.enabled():
            return grounded_draft

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

        if self.provider == "openai":

            return self._openai(
                system_prompt,
                str(user_prompt)
            )

        if self.provider == "anthropic":

            return self._anthropic(
                system_prompt,
                str(user_prompt)
            )

        return grounded_draft

    def _openai(
        self,
        system: str,
        user: str
    ) -> str:

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

        return (
            text.strip()
            if text
            else user
        )

    def _anthropic(
        self,
        system: str,
        user: str
    ) -> str:

        from anthropic import Anthropic

        client = Anthropic(
            api_key=os.environ[
                "ANTHROPIC_API_KEY"
            ]
        )

        response = client.messages.create(
            model=self.anthropic_model,

            max_tokens=500,

            temperature=0,

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

        return "\n".join(
            parts
        ).strip()