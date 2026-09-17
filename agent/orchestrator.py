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

from time import perf_counter

from typing import Any, Dict, List, Optional


from .guardrails import Guardrails

from .llm import LLM

from .policy_engine import PolicyEngine

from .response_generator import ResponseGenerator

from .usage import LLMUsage, Pricing, TurnMetrics


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

        self.pricing = Pricing()

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
                    r"(₹|\brs\.?\b|\binr\b|\brupees?\b)\s*[0-9]",
                    text,
                )
                or re.search(
                    r"\b[0-9][0-9,]*\s*"
                    r"(?:rupees?|rs\.?|inr)\b",
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
    # METRICS
    # ==========================================================

    def _metrics(
        self,
        *,
        intent: str,
        outcome: str,
        rule_ids: List[str],
        escalated: bool,
        partial_grant: bool,
        guardrail_ms: float,
        policy_ms: float,
        started: float,
        response: str,
        usage: Optional[LLMUsage] = None,
        llm_skip_reason: Optional[str] = None,
    ) -> TurnMetrics:
        """
        Assemble the accounting for one turn.

        Turns that never reach a model still get a record — zero tokens
        and zero cost is the useful number, not a missing one.
        """

        usage = usage or LLMUsage()

        cost = self.pricing.cost(
            usage.model,
            usage.input_tokens,
            usage.output_tokens,
        )

        spent_tokens = usage.total_tokens > 0

        return TurnMetrics(
            intent=intent,
            outcome=outcome,
            rule_ids=rule_ids,
            escalated=escalated,
            partial_grant=partial_grant,

            guardrail_ms=guardrail_ms,
            policy_ms=policy_ms,
            llm_ms=usage.latency_ms,
            total_ms=(perf_counter() - started) * 1000,

            llm_used=spent_tokens,
            llm_skip_reason=llm_skip_reason,

            provider=usage.provider,
            model=usage.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,

            cost=cost if cost is not None else (0.0 if not spent_tokens else None),
            currency=self.pricing.currency,
            unpriced=spent_tokens and cost is None,
            llm_error=usage.error,

            response_chars=len(response),
        )

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

        started = perf_counter()

        # ======================================================
        # STEP 1
        # HARD GUARDRAILS
        #
        # This occurs BEFORE ANY LLM call.
        # ======================================================

        guardrail_started = perf_counter()

        guardrail = self.guardrails.check(
            message,

            delay_hours=delay_hours,

            conversation_history=history,
        )

        guardrail_ms = (perf_counter() - guardrail_started) * 1000

        # ======================================================
        # STEP 2
        # IMMEDIATE ESCALATION
        #
        # No LLM is called.
        # ======================================================

        if guardrail.triggered:

            # --------------------------------------------------
            # PARTIAL GRANT
            #
            # The request exceeds policy, but a stated entitlement
            # still applies to the same disruption. Grant the
            # entitled portion, escalate only the excess.
            #
            # Still no LLM call: the excess is an escalation.
            # --------------------------------------------------

            if guardrail.partial_grant:

                policy_started = perf_counter()

                entitlement = self.policy.evaluate(
                    "delay",

                    customer,

                    booking,

                    {},
                )

                policy_ms = (perf_counter() - policy_started) * 1000

                if entitlement.outcome == "allowed":

                    response = (
                        self.responses.partial_grant_response(
                            customer_name,
                            customer,
                            booking,
                            entitlement,
                            guardrail,
                        )
                    )

                    return {
                        "response": response,

                        "intent": "delay",

                        "escalated": True,

                        "escalation_category":
                            guardrail.category,

                        "escalation_reason":
                            guardrail.reason,

                        "rule_ids":
                            entitlement.rule_ids,

                        "actions":
                            entitlement.allowed_actions,

                        "partial_grant": True,

                        "llm_used": False,

                        "metrics": self._metrics(
                            intent="delay",
                            outcome="partial_grant",
                            rule_ids=entitlement.rule_ids,
                            escalated=True,
                            partial_grant=True,
                            guardrail_ms=guardrail_ms,
                            policy_ms=policy_ms,
                            started=started,
                            response=response,
                            llm_skip_reason=(
                                "Escalation path — wording is deterministic "
                                "by design, so no model is called."
                            ),
                        ).to_dict(),

                        "customer": customer,

                        "booking": booking,
                    }

            decision_intent = (
                guardrail.category
                or "beyond_policy"
            )

            policy_started = perf_counter()

            decision = self.policy.evaluate(
                decision_intent,

                customer,

                booking,

                {
                    "fare_difference":
                        guardrail.fare_difference
                },
            )

            policy_ms = (perf_counter() - policy_started) * 1000

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

                "partial_grant": False,

                "llm_used": False,

                "metrics": self._metrics(
                    intent=decision_intent,
                    outcome="escalate",
                    rule_ids=decision.rule_ids,
                    escalated=True,
                    partial_grant=False,
                    guardrail_ms=guardrail_ms,
                    policy_ms=policy_ms,
                    started=started,
                    response=response,
                    llm_skip_reason=(
                        "Guardrail fired before any model call."
                    ),
                ).to_dict(),

                "customer": customer,

                "booking": booking,
            }

        # ======================================================
        # STEP 3
        # DETERMINISTIC INTENT
        # ======================================================

        policy_started = perf_counter()

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

        policy_ms = (perf_counter() - policy_started) * 1000

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

                "partial_grant": False,

                "llm_used": False,

                "metrics": self._metrics(
                    intent=intent,
                    outcome="escalate",
                    rule_ids=decision.rule_ids,
                    escalated=True,
                    partial_grant=False,
                    guardrail_ms=guardrail_ms,
                    policy_ms=policy_ms,
                    started=started,
                    response=response,
                    llm_skip_reason=(
                        "Policy engine escalated — wording is "
                        "deterministic by design."
                    ),
                ).to_dict(),

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

        final_response, llm_usage = self.llm.rewrite(
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

            "partial_grant": False,

            "llm_used":
                self.llm.enabled(),

            "metrics": self._metrics(
                intent=intent,
                outcome=decision.outcome,
                rule_ids=decision.rule_ids,
                escalated=False,
                partial_grant=False,
                guardrail_ms=guardrail_ms,
                policy_ms=policy_ms,
                started=started,
                response=final_response,
                usage=llm_usage,
                llm_skip_reason=(
                    None
                    if self.llm.enabled()
                    else "No provider configured (LLM_PROVIDER=none) — "
                         "using the grounded template."
                ),
            ).to_dict(),

            "customer":
                customer,

            "booking":
                booking,
        }