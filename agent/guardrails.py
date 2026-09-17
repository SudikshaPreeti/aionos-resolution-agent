"""
Pre-LLM guardrails.

Hard prohibited requests are detected before an LLM can be called.

If a hard guardrail fires:
    Guardrails -> Orchestrator -> Escalation Response

The request does NOT reach the LLM.
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field

from typing import List, Optional


@dataclass
class GuardrailResult:
    triggered: bool

    category: Optional[str] = None

    reason: Optional[str] = None

    matched_terms: List[str] = field(
        default_factory=list
    )

    fare_difference: Optional[float] = None

    # Set when the customer asked for more than the policy allows, but a
    # stated entitlement still applies to the same disruption. The agent
    # grants the entitled portion and escalates only the excess.
    partial_grant: bool = False


class Guardrails:

    # ==========================================================
    # LEGAL / FORMAL COMPLAINT
    # ==========================================================

    LEGAL_PATTERNS = [
        r"\blegal action\b",
        r"\blawyer\b",
        r"\bcourt\b",
        r"\bformal complaint\b",
        r"\bfile a complaint\b",
        r"\bsue\b",
    ]

    # ==========================================================
    # FREE BUSINESS CLASS / FREE UPGRADE
    # ==========================================================

    FREE_UPGRADE_PATTERNS = [
        r"\bfree\b.{0,30}\bupgrade\b",
        r"\bupgrade\b.{0,30}\bfree\b",
        r"\bbusiness class\b",
        r"\bfree business\b",
    ]

    # ==========================================================
    # FULL NIGHT HOTEL
    # ==========================================================

    FULL_NIGHT_PATTERNS = [
        r"\bfull night\b",
        r"\bfull night's\b",
        r"\bwhole night\b",
        r"\bentire night\b",
        r"\bfull-night\b",
    ]

    # ==========================================================
    # OUTSIDE-POLICY COMPENSATION
    # ==========================================================

    NONPOLICY_COMP_PATTERNS = [
        r"\bextra compensation\b",
        r"\badditional compensation\b",
        r"\bcompensation\b.{0,30}\bupgrade\b",
        r"\bfor the trouble\b",
    ]

    # ==========================================================
    # HOTEL INSISTENCE
    # ==========================================================

    HOTEL_INSIST_PATTERNS = [
        r"\bno\b.{0,15}\bhotel\b",
        r"\bbut\b.{0,30}\bhotel\b",
        r"\bstill want\b.{0,30}\bhotel\b",
        r"\binsist\b.{0,30}\bhotel\b",
        r"\bwant\b.{0,20}\bhotel\b.{0,30}\banyway\b",
    ]

    # ==========================================================
    # FARE DIFFERENCE
    # ==========================================================

    FARE_PATTERN = re.compile(
        r"(?:₹|rs\.?|inr)\s*"
        r"([0-9][0-9,]*(?:\.[0-9]+)?)"
        r"|"
        r"\b([0-9][0-9,]*(?:\.[0-9]+)?)\s*"
        r"(?:rupees?|rs\.?|inr)\b",
        re.IGNORECASE,
    )

    def check(
        self,
        message: str,
        *,
        delay_hours: float | None = None,
        conversation_history: list[dict] | None = None,
    ) -> GuardrailResult:

        text = message.lower()

        history_text = " ".join(
            str(m.get("content", ""))
            for m in (conversation_history or [])
        ).lower()

        combined = f"{history_text} {text}"

        # ======================================================
        # LEGAL THREAT
        # ======================================================

        for pattern in self.LEGAL_PATTERNS:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )

            if match:

                return GuardrailResult(
                    triggered=True,
                    category="legal_threat",
                    reason=(
                        "Legal-action or formal-complaint language "
                        "requires immediate human escalation."
                    ),
                    matched_terms=[
                        match.group(0)
                    ],
                )

        # ======================================================
        # FARE DIFFERENCE HARD GATE
        # ======================================================

        fare = self._extract_fare_difference(text)

        if fare is not None and fare > 1500:

            return GuardrailResult(
                triggered=True,
                category="fare_difference",
                reason=(
                    f"The requested fare difference of ₹{fare:,.0f} "
                    "exceeds the ₹1,500 waiver limit and requires "
                    "supervisor approval."
                ),
                matched_terms=[
                    f"₹{fare:,.0f}"
                ],
                fare_difference=fare,
            )

        # ======================================================
        # FREE BUSINESS CLASS
        # ======================================================

        if any(
            re.search(
                pattern,
                text,
                re.IGNORECASE
            )
            for pattern in self.FREE_UPGRADE_PATTERNS
        ):

            return GuardrailResult(
                triggered=True,
                category="beyond_policy",
                reason=(
                    "A free business-class upgrade is additional "
                    "compensation not stated in the supplied policy."
                ),
                matched_terms=self._matched_patterns(
                    text,
                    self.FREE_UPGRADE_PATTERNS,
                ),
            )

        # ======================================================
        # FULL NIGHT HOTEL
        # ======================================================

        if any(
            re.search(
                pattern,
                text,
                re.IGNORECASE
            )
            for pattern in self.FULL_NIGHT_PATTERNS
        ):

            if (
                delay_hours is not None
                and delay_hours > 5
            ):

                return GuardrailResult(
                    triggered=True,
                    category="beyond_policy",
                    reason=(
                        "The delay rule covers hotel accommodation "
                        "for delayed hours only, not a full night's stay."
                    ),
                    matched_terms=self._matched_patterns(
                        text,
                        self.FULL_NIGHT_PATTERNS,
                    ),
                    # The delay itself qualifies under the more-than-5-hours
                    # rule, so the delayed-hours entitlement still stands.
                    partial_grant=True,
                )

        # ======================================================
        # HOTEL INSISTENCE
        # ======================================================

        if delay_hours is not None and delay_hours <= 5:

            if any(
                re.search(
                    pattern,
                    combined,
                    re.IGNORECASE
                )
                for pattern in self.HOTEL_INSIST_PATTERNS
            ):

                return GuardrailResult(
                    triggered=True,
                    category="beyond_policy",
                    reason=(
                        "Hotel accommodation is not provided by the "
                        "supplied delay rule for this delay duration; "
                        "repeated insistence requires human escalation."
                    ),
                    matched_terms=self._matched_patterns(
                        combined,
                        self.HOTEL_INSIST_PATTERNS,
                    ),
                )

        # ======================================================
        # ADDITIONAL COMPENSATION
        # ======================================================

        if any(
            re.search(
                pattern,
                text,
                re.IGNORECASE
            )
            for pattern in self.NONPOLICY_COMP_PATTERNS
        ):

            return GuardrailResult(
                triggered=True,
                category="beyond_policy",
                reason=(
                    "The customer is requesting compensation outside "
                    "the stated policy and it must be escalated."
                ),
                matched_terms=self._matched_patterns(
                    text,
                    self.NONPOLICY_COMP_PATTERNS,
                ),
            )

        return GuardrailResult(
            triggered=False
        )

    @classmethod
    def _extract_fare_difference(
        cls,
        text: str
    ) -> Optional[float]:

        match = cls.FARE_PATTERN.search(text)

        if not match:
            return None

        raw = match.group(1) or match.group(2)

        try:
            return float(
                raw.replace(",", "")
            )
        except ValueError:
            return None

    @staticmethod
    def _matched_patterns(
        text: str,
        patterns: list[str]
    ) -> list[str]:

        matches = []

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )

            if match:
                matches.append(
                    match.group(0)
                )

        return matches