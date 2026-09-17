"""
Console test suite.

Mirrors the way app.py drives the agent:

- load the three JSON data files
- build one Orchestrator
- call handle(customer_name=..., message=..., history=...)
- keep per-customer history so repeat-insistence guardrails fire
"""

import json
from pathlib import Path

from agent.orchestrator import Orchestrator


BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# LOAD JSON
# ============================================================
def load_json(relative_path: str):
    with open(BASE_DIR / relative_path, "r", encoding="utf-8") as f:
        return json.load(f)


customers_data = load_json("data/customers.json")
bookings_data = load_json("data/bookings.json")
rules_data = load_json("data/rules.json")


# ============================================================
# PNR -> CUSTOMER NAME
# ============================================================
PNR_TO_NAME = {
    c["booking_reference"]: c["name"]
    for c in customers_data["customers"]
}


# ============================================================
# PER-CUSTOMER CONVERSATION HISTORY
# ============================================================
histories: dict[str, list[dict]] = {}


def run_test(agent, pnr, message):
    customer_name = PNR_TO_NAME[pnr]
    history = histories.setdefault(pnr, [])

    print("\n" + "=" * 70)
    print(f"CUSTOMER: {customer_name} ({pnr})")
    print(f"USER: {message}")
    print("-" * 70)

    result = agent.handle(
        customer_name=customer_name,
        message=message,
        history=list(history),
    )

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": result.get("response", "")})

    print(f"INTENT: {result.get('intent')}")
    print(f"ESCALATED: {result.get('escalated')}")

    if result.get("escalation_category"):
        print(f"ESCALATION CATEGORY: {result.get('escalation_category')}")

    if result.get("rule_ids"):
        print(f"RULE IDS: {', '.join(result['rule_ids'])}")

    if result.get("actions"):
        print(f"ACTIONS: {result['actions']}")

    print(f"\nAGENT RESPONSE:\n{result.get('response')}")


def main():
    agent = Orchestrator(customers_data, bookings_data, rules_data)

    print("\n" + "#" * 70)
    print("AGENTIC AI FACTORY - AGENT TEST SUITE")
    print("#" * 70)

    # ============================================================
    # SCENARIO 1 - PRIYA NAIR
    # ============================================================

    run_test(
        agent,
        "SK4821X",
        "My flight SK-204 is cancelled. What can you do for me?"
    )

    run_test(
        agent,
        "SK4821X",
        "I want a full cash refund plus a free upgrade to business class."
    )

    # ============================================================
    # SCENARIO 2 - ARVIND KULKARNI
    # ============================================================

    run_test(
        agent,
        "TR1190B",
        "My flight is delayed 4 hours. I want hotel accommodation."
    )

    run_test(
        agent,
        "TR1190B",
        "I insist on getting a hotel because this delay has caused me problems."
    )

    # ============================================================
    # SCENARIO 3 - MEHER KAUR
    # ============================================================

    run_test(
        agent,
        "WL7742",
        "My flight is delayed 6 hours. I want a full night's hotel stay."
    )

    run_test(
        agent,
        "WL7742",
        "I insist that you give me a full night's hotel."
    )

    run_test(
        agent,
        "WL7742",
        "Move me to a different higher-fare flight. The fare difference is ₹2,000."
    )

    # ============================================================
    # LEGAL THREAT TEST
    # ============================================================

    run_test(
        agent,
        "SK4821X",
        "If you don't give me the upgrade, I will take legal action."
    )

    print("\n" + "#" * 70)
    print("TEST SUITE COMPLETED")
    print("#" * 70)


if __name__ == "__main__":
    main()
