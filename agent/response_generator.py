"""
Grounded natural-language response construction.

This module does not make policy decisions.
It turns already-computed decisions into customer-facing text.
"""

from __future__ import annotations

from typing import Any, Dict


class ResponseGenerator:

    def escalation_response(
        self,
        customer_name: str,
        guardrail
    ) -> str:

        return (
            f"I'm sorry this has been such a frustrating experience, "
            f"{customer_name}. "

            "I want to make sure this gets the right attention — "
            "I'm escalating this to our specialist support team "
            "right now, and they'll review the request. "

            f"Under the prohibited-actions section of our policy, "
            f"{guardrail.reason}"
        )

    def policy_response(
        self,
        customer_name: str,
        customer: Dict[str, Any],
        booking: Dict[str, Any],
        decision,
    ) -> str:

        flights = booking.get(
            "flights",
            []
        )

        affected = next(
            (
                flight
                for flight in flights
                if flight.get("disruption_type")
                in {"cancellation", "delay"}
            ),
            flights[0] if flights else {},
        )

        status = affected.get(
            "status",
            "status available in the booking data"
        )

        opening = (
            f"I'm sorry for the disruption, "
            f"{customer_name}. "

            f"Your booking "
            f"{customer['booking_reference']} "
            f"shows {status}. "
        )

        # ======================================================
        # CANCELLATION
        # ======================================================

        if decision.intent == "cancellation":

            return (
                opening

                + "Under our cancellation rebooking rule, "
                "because the airline cancelled the flight, "
                "you can choose either free rebooking on the "
                "next available flight within 24 hours or a "
                "full refund. "

                "Under our refund processing rule, a full refund "
                "is processed within 7 business days to the "
                "original payment method. "

                + self._loyalty_note(customer)
            )

        # ======================================================
        # REFUND
        # ======================================================

        if decision.intent == "refund":

            return (
                opening

                + "Under our cancellation rebooking rule, "
                "a full refund is available for this "
                "airline-caused cancellation. "

                "Under our refund processing rule, it is "
                "processed within 7 business days to the "
                "original payment method."
            )

        # ======================================================
        # DELAY
        # ======================================================

        if decision.intent == "delay":

            return (
                opening

                + decision.explanation

                + " These are the actions I can take under "
                "the stated policy: "

                + "; ".join(
                    decision.allowed_actions
                )

                + "."
            )

        # ======================================================
        # FARE DIFFERENCE
        # ======================================================

        if decision.intent == "fare_difference":

            return (
                opening

                + decision.explanation

                + " I can proceed only within those "
                "stated limits."
            )

        # ======================================================
        # UNKNOWN
        # ======================================================

        return (
            f"I can help with the booking information and "
            f"the stated airline disruption rules. "

            f"Under the applicable policy: "
            f"{decision.explanation}"
        )

    @staticmethod
    def _loyalty_note(
        customer: Dict[str, Any]
    ) -> str:

        tier = customer.get(
            "loyalty_tier"
        )

        if tier in {
            "Gold",
            "Platinum"
        }:

            return (
                f"As a {tier} customer, you receive priority "
                "rebooking with first access to next-available "
                "seats, but the loyalty rule does not provide "
                "additional compensation."
            )

        return ""