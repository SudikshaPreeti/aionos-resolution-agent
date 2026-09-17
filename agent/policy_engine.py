"""
Deterministic policy engine.

IMPORTANT:
This module does not use an LLM.

All policy decisions are made deterministically from:
1. customer data
2. booking data
3. rules.json
4. classified intent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PolicyDecision:
    """
    Structured result produced by the deterministic policy engine.
    """

    intent: str

    outcome: str

    allowed_actions: List[str] = field(default_factory=list)

    rule_ids: List[str] = field(default_factory=list)

    explanation: str = ""

    escalation_category: Optional[str] = None

    escalation_reason: Optional[str] = None


class PolicyEngine:
    """
    Applies the airline service rules deterministically.
    """

    def __init__(self, rules: Dict[str, Any]):
        self.rules = rules["rules"]
        self.allowed_actions = rules["allowed_actions"]

    @staticmethod
    def _delay_bucket(delay_hours: float) -> str:
        """
        Convert delay duration into the exact policy buckets.
        """

        if delay_hours > 5:
            return "more_than_5_hours"

        if delay_hours > 3:
            return "more_than_3_hours"

        return "under_or_equal_3_hours"

    def evaluate(
        self,
        intent: str,
        customer: Dict[str, Any],
        booking: Dict[str, Any],
        request: Dict[str, Any] | None = None,
    ) -> PolicyDecision:

        request = request or {}

        tier = customer.get("loyalty_tier", "")

        flights = booking.get("flights", [])

        affected = next(
            (
                flight
                for flight in flights
                if flight.get("disruption_type") in {
                    "cancellation",
                    "delay"
                }
            ),
            flights[0] if flights else {},
        )

        # ========================================================
        # CANCELLATION
        # ========================================================

        if intent == "cancellation":

            if affected.get("disruption_type") != "cancellation":

                return PolicyDecision(
                    intent=intent,
                    outcome="escalate",
                    escalation_category="beyond_policy",
                    escalation_reason=(
                        "The selected booking does not contain an "
                        "airline-caused cancellation in the supplied data."
                    ),
                )

            actions = [
                "Offer free rebooking on the next available flight within 24 hours",
                "OR initiate a full refund",
            ]

            if tier in self.rules["loyalty_tier"]["priority_tiers"]:

                actions.append(
                    f"{tier} priority: first access to next-available seats"
                )

            return PolicyDecision(
                intent=intent,
                outcome="allowed",
                allowed_actions=actions,
                rule_ids=[
                    "cancellation_rebooking",
                    "refund_processing",
                    "loyalty_tier",
                ],
                explanation=(
                    "Under the cancellation rebooking rule, an airline-caused "
                    "cancellation gives the customer a choice between free "
                    "rebooking within 24 hours and a full refund. "
                    "Under the refund processing rule, the refund is processed "
                    "within 7 business days to the original payment method."
                ),
            )

        # ========================================================
        # REFUND
        # ========================================================

        if intent == "refund":

            if affected.get("disruption_type") == "cancellation":

                return PolicyDecision(
                    intent=intent,
                    outcome="allowed",
                    allowed_actions=[
                        "Initiate a full refund request"
                    ],
                    rule_ids=[
                        "cancellation_rebooking",
                        "refund_processing",
                    ],
                    explanation=(
                        "Under the cancellation rebooking rule, a full refund "
                        "is a customer choice for an airline-caused cancellation. "
                        "Under the refund processing rule, it is processed "
                        "within 7 business days to the original payment method."
                    ),
                )

            return PolicyDecision(
                intent=intent,
                outcome="escalate",
                escalation_category="beyond_policy",
                escalation_reason=(
                    "The supplied refund rule applies to airline-caused "
                    "cancellations. No refund entitlement is stated for "
                    "this selected delay."
                ),
            )

        # ========================================================
        # DELAY
        # ========================================================

        if intent == "delay":

            delay = float(
                affected.get("delay_hours", 0)
            )

            if affected.get("disruption_type") != "delay":

                return PolicyDecision(
                    intent=intent,
                    outcome="escalate",
                    escalation_category="beyond_policy",
                    escalation_reason=(
                        "The selected booking does not contain "
                        "the stated delay scenario."
                    ),
                )

            bucket = self._delay_bucket(delay)

            # ----------------------------------------------------
            # MORE THAN 5 HOURS
            # ----------------------------------------------------

            if bucket == "more_than_5_hours":

                return PolicyDecision(
                    intent=intent,
                    outcome="allowed",
                    allowed_actions=[
                        "Issue meal voucher",
                        "Provide lounge access",
                        "Arrange hotel accommodation for delayed hours only",
                    ],
                    rule_ids=[
                        "delay_compensation"
                    ],
                    explanation=(
                        f"Under the delay compensation rule, a {delay:g}-hour "
                        "delay is more than 5 hours, so meal voucher, lounge "
                        "access and hotel accommodation covering only the "
                        "delayed hours apply. The rule does not provide "
                        "a full night's hotel stay."
                    ),
                )

            # ----------------------------------------------------
            # MORE THAN 3 HOURS BUT NOT MORE THAN 5
            # ----------------------------------------------------

            if bucket == "more_than_3_hours":

                return PolicyDecision(
                    intent=intent,
                    outcome="allowed",
                    allowed_actions=[
                        "Issue meal voucher",
                        "Provide lounge access",
                    ],
                    rule_ids=[
                        "delay_compensation"
                    ],
                    explanation=(
                        f"Under the delay compensation rule, a {delay:g}-hour "
                        "delay is more than 3 hours, so meal voucher and lounge "
                        "access apply. Hotel accommodation is stated only "
                        "for delays more than 5 hours."
                    ),
                )

            # ----------------------------------------------------
            # UNDER 3 HOURS
            # ----------------------------------------------------

            return PolicyDecision(
                intent=intent,
                outcome="allowed",
                allowed_actions=[
                    "Issue ₹500 meal voucher"
                ],
                rule_ids=[
                    "delay_compensation"
                ],
                explanation=(
                    "Under the delay compensation rule, a delay under "
                    "3 hours qualifies for a ₹500 meal voucher."
                ),
            )

        # ========================================================
        # FARE DIFFERENCE
        # ========================================================

        if intent == "fare_difference":

            amount = request.get("fare_difference")

            threshold = self.rules[
                "fare_difference"
            ]["hard_escalation_threshold"]

            if amount is not None and amount > threshold:

                return PolicyDecision(
                    intent=intent,
                    outcome="escalate",
                    escalation_category="fare_difference",
                    escalation_reason=(
                        f"The requested fare-difference waiver is "
                        f"₹{amount:,.0f}, which is above the ₹"
                        f"{threshold:,.0f} limit. Supervisor approval "
                        "is required."
                    ),
                    rule_ids=[
                        "fare_difference"
                    ],
                    explanation=(
                        "Under the fare difference rule, voluntary "
                        "higher-fare rebooking requires the customer "
                        "to pay the difference. Agents cannot waive "
                        "more than ₹1,500 without supervisor approval."
                    ),
                )

            return PolicyDecision(
                intent=intent,
                outcome="allowed",
                allowed_actions=[
                    "Customer pays the fare difference for voluntary higher-fare rebooking"
                ],
                rule_ids=[
                    "fare_difference"
                ],
                explanation=(
                    "Under the fare difference rule, voluntary rebooking "
                    "onto a higher-fare flight requires the customer "
                    "to pay the fare difference."
                ),
            )

        # ========================================================
        # BEYOND POLICY / LEGAL THREAT
        # ========================================================

        if intent in {
            "beyond_policy",
            "legal_threat",
        }:

            return PolicyDecision(
                intent=intent,
                outcome="escalate",
                escalation_category=intent,
                escalation_reason=(
                    "This request requires human handling under "
                    "the prohibited-actions section of the policy."
                ),
                rule_ids=[],
                explanation=(
                    "The prohibited-actions section requires escalation."
                ),
            )

        # ========================================================
        # UNKNOWN
        # ========================================================

        return PolicyDecision(
            intent="unknown",
            outcome="clarify",
            explanation=(
                "I can provide the selected customer's own booking and "
                "flight status information and apply the stated cancellation, "
                "delay, refund and fare-difference rules."
            ),
        )