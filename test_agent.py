from agent.orchestrator import AgentOrchestrator


def run_test(agent, customer, message):
    print("\n" + "=" * 70)
    print(f"CUSTOMER: {customer}")
    print(f"USER: {message}")
    print("-" * 70)

    result = agent.handle_message(
        customer_id=customer,
        message=message
    )

    print(f"INTENT: {result.get('intent')}")
    print(f"ESCALATED: {result.get('escalated')}")
    
    if result.get("escalation_category"):
        print(f"ESCALATION CATEGORY: {result.get('escalation_category')}")

    print(f"\nAGENT RESPONSE:\n{result.get('response')}")


def main():
    agent = AgentOrchestrator()

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
        "Move me to a different higher-fare flight. The fare difference is £2,000."
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