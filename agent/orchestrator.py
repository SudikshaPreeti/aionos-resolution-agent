"""
End-to-end orchestration.

Execution order:

1. Identify selected customer
2. Identify booking
3. Run guardrails BEFORE LLM
4. Classify intent deterministically
5. Apply policy deterministically
6. Generate grounded response
7. Optional LLM rewrite ONLY when not escalated
"""

from __future__ import annotations

import re

from typing import Any, Dict, List, Optional


from .guardrails import Guardrails

from .llm import LLM

from .policy_engine import PolicyEngine

from .response_generator import ResponseGenerator


INTENTS = [
    "cancellation",
    "delay",
    "refund",
    "fare_difference",
    "beyond_policy",
    "legal_threat",
    "unknown",
]


class Orchestrator:

    def __init__(
        self,
        customers: Dict[str, Any],
        bookings: Dict[str, Any],
        rules: Dict[str, Any],
    ):

        self.customers = customers[
            "customers"
        ]

        self.bookings = bookings[
            "bookings"
        ]

        self.rules = rules

        self.guardrails = Guardrails()

        self.policy = PolicyEngine(
            rules
        )

        self.responses = ResponseGenerator()

        self.llm = LLM()

    # ==========================================================
    # CUSTOMER LOOKUP
    # ==========================================================

    def _customer(
        self,
        name: str
    ) -> Dict[str, Any]:

        return next(
            customer
            for customer in self.customers
            if customer["name"] == name
        )

    # ==========================================================
    # BOOKING LOOKUP
    # ==========================================================

    def _booking(
        self,
        customer: Dict[str, Any]
    ) -> Dict[str, Any]:

        return next(
            booking
            for booking in self.bookings
            if booking["booking_reference"]
            == customer["booking_reference"]
        )

    # ==========================================================
    # AFFECTED FLIGHT
    # ==========================================================

    @staticmethod
    def _affected_flight(
        booking: Dict[str, Any]
    ) -> Dict[str, Any]:

        return next(
            (
                flight
                for flight in booking["flights"]
                if flight.get("disruption_type")
                in {
                    "cancellation",
                    "delay",
                }
            ),
            booking["flights"][0],
        )

    # ==========================================================
    # DETERMINISTIC INTENT CLASSIFIER
    # ==========================================================

    def classify_intent(
        self,
        message: str,
        booking: Dict[str, Any],
    ) -> str:

        text = message.lower()

        affected = self._affected_flight(
            booking
        )

        # ------------------------------------------------------
        # LEGAL
        # ------------------------------------------------------

        if re.search(
            r"\b("
            r"legal action|"
            r"lawyer|"
            r"court|"
            r"formal complaint|"
            r"file a complaint|"
            r"sue"
            r")\b",
            text,
        ):

            return "legal_threat"

        # ------------------------------------------------------
        # FARE DIFFERENCE
        # ------------------------------------------------------

        if re.search(
            r"\b("
            r"fare difference|"
            r"higher-fare|"
            r"higher fare|"
            r"fare is|"
            r"difference is"
            r")\b",
            text,
        ):

            if (
                re.search(
                    r"(£|gbp|\bpounds?\b)\s*[0-9]",
                    text,
                )
                or re.search(
                    r"\b[0-9][0-9,]*\s*"
                    r"(?:pounds?|gbp)\b",
                    text,
                )
            ):

                return "fare_difference"

        # ------------------------------------------------------
        # REFUND
        # ------------------------------------------------------

        if re.search(
            r"\b("
            r"refund|"
            r"cash back|"
            r"money back"
            r")\b",
            text,
        ):

            return "refund"

        # ------------------------------------------------------
        # CANCELLATION
        # ------------------------------------------------------

        if (
            affected.get(
                "disruption_type"
            ) == "cancellation"
            or re.search(
                r"\b("
                r"cancelled|"
                r"canceled|"
                r"cancellation"
                r")\b",
                text,
            )
        ):

            return "cancellation"

        # ------------------------------------------------------
        # DELAY
        # ------------------------------------------------------

        if (
            affected.get(
                "disruption_type"
            ) == "delay"
            or re.search(
                r"\b("
                r"delay|"
                r"delayed|"
                r"late"
                r")\b",
                text,
            )
        ):

            return "delay"

        # ------------------------------------------------------
        # BEYOND POLICY
        # ------------------------------------------------------

        if re.search(
            r"\b("
            r"upgrade|"
            r"business class|"
            r"extra compensation|"
            r"additional compensation|"
            r"free hotel|"
            r"full night"
            r")\b",
            text,
        ):

            return "beyond_policy"

        return "unknown"

    # ==========================================================
    # MAIN HANDLER
    # ==========================================================

    def handle(
        self,
        *,
        customer_name: str,
        message: str,
        history: Optional[
            List[Dict[str, str]]
        ] = None,
    ) -> Dict[str, Any]:

        customer = self._customer(
            customer_name
        )

        booking = self._booking(
            customer
        )

        affected = self._affected_flight(
            booking
        )

        delay_hours = affected.get(
            "delay_hours"
        )

        # ======================================================
        # STEP 1
        # HARD GUARDRAILS
        #
        # This occurs BEFORE ANY LLM call.
        # ======================================================

        guardrail = self.guardrails.check(
            message,

            delay_hours=delay_hours,

            conversation_history=history,
        )

        # ======================================================
        # STEP 2
        # IMMEDIATE ESCALATION
        #
        # No LLM is called.
        # ======================================================

        if guardrail.triggered:

            decision_intent = (
                guardrail.category
                or "beyond_policy"
            )

            decision = self.policy.evaluate(
                decision_intent,

                customer,

                booking,

                {
                    "fare_difference":
                        guardrail.fare_difference
                },
            )

            response = (
                self.responses.escalation_response(
                    customer_name,
                    guardrail,
                )
            )

            return {
                "response": response,

                "intent": decision_intent,

                "escalated": True,

                "escalation_category":
                    guardrail.category,

                "escalation_reason":
                    guardrail.reason,

                "rule_ids":
                    decision.rule_ids,

                "actions": [],

                "llm_used": False,

                "customer": customer,

                "booking": booking,
            }

        # ======================================================
        # STEP 3
        # DETERMINISTIC INTENT
        # ======================================================

        intent = self.classify_intent(
            message,
            booking,
        )

        # ======================================================
        # STEP 4
        # DETERMINISTIC POLICY
        # ======================================================

        decision = self.policy.evaluate(
            intent,

            customer,

            booking,

            {},
        )

        # ======================================================
        # STEP 5
        # POLICY-LEVEL ESCALATION
        # ======================================================

        if decision.outcome == "escalate":

            response = (
                f"I'm sorry for the disruption, "
                f"{customer_name}. "

                "I want to make sure this gets the "
                "right attention — I'm escalating "
                "this to our specialist support team "
                "right now, and they'll review the request. "

                f"Under the prohibited-actions section, "
                f"{decision.escalation_reason}"
            )

            return {
                "response": response,

                "intent": intent,

                "escalated": True,

                "escalation_category":
                    decision.escalation_category,

                "escalation_reason":
                    decision.escalation_reason,

                "rule_ids":
                    decision.rule_ids,

                "actions": [],

                "llm_used": False,

                "customer": customer,

                "booking": booking,
            }

        # ======================================================
        # STEP 6
        # GROUNDED RESPONSE
        # ======================================================

        draft = self.responses.policy_response(
            customer_name,

            customer,

            booking,

            decision,
        )

        # ======================================================
        # STEP 7
        # OPTIONAL LLM
        #
        # Only happens when there was no escalation.
        # ======================================================

        final_response = self.llm.rewrite(
            grounded_draft=draft,

            booking_context={
                "booking_reference":
                    booking[
                        "booking_reference"
                    ],

                "flights":
                    booking["flights"],
            },

            rules_context=self.rules,

            decision={
                "intent":
                    decision.intent,

                "outcome":
                    decision.outcome,

                "allowed_actions":
                    decision.allowed_actions,

                "rule_ids":
                    decision.rule_ids,
            },
        )

        return {
            "response": final_response,

            "intent": intent,

            "escalated": False,

            "escalation_category": None,

            "escalation_reason": None,

            "rule_ids":
                decision.rule_ids,

            "actions":
                decision.allowed_actions,

            "llm_used":
                self.llm.enabled(),

            "customer":
                customer,

            "booking":
                booking,
        }