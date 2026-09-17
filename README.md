# AIONOS Assignment 3 — Customer-Facing Resolution Agent

## Airline Disruption Resolution Agent

A Python + Streamlit prototype for the AIONOS Customer-Facing Resolution Agent assignment.

It handles the three supplied airline-disruption scenarios while keeping every policy decision deterministic, auditable, and grounded in the supplied data pack.

---

# 1. Project Overview

The agent is designed around one core principle:

> The LLM generates language. The deterministic policy engine makes decisions.

An LLM is never asked whether a customer gets a refund, a hotel, or a fare waiver. Those answers come from `data/rules.json` via a rule engine that returns a structured decision. The LLM — when enabled at all — is handed that already-final decision and may only rephrase it.

This matters for three reasons:

- **Auditability.** Every answer carries the rule IDs that produced it, visible in the UI's decision trace.
- **Safety.** A prohibited request cannot be talked into approval, because the guardrails run *before* any model call.
- **Reproducibility.** The same input always produces the same decision, with or without an API key.

The project contains:

- customer, booking, and service-rule data (verbatim from the data pack)
- pre-LLM guardrails
- deterministic intent classification
- deterministic policy engine
- grounded response generator
- optional OpenAI / Anthropic phrasing layer
- Streamlit frontend
- template fallback when no API key is available

**The application runs and demonstrates all three scenarios with no API key.**

---

# 2. Architecture

```text
                              CUSTOMER
                                  |
                                  v
                        +-------------------+
                        | Streamlit Chat UI |
                        +-------------------+
                                  |
                                  v
                        +-------------------+
                        |   ORCHESTRATOR    |
                        +-------------------+
                                  |
                                  v
                        +-------------------+
                        |    GUARDRAILS     |
                        |   BEFORE  LLM     |
                        +-------------------+
                          |       |        |
                     HARD |  PARTIAL |     | SAFE
                 ESCALATE |    GRANT |     |
                          v          v     v
                  +-----------+  +---------------------+
                  |  HUMAN    |  |  INTENT CLASSIFIER  |
                  | ESCALATION|  |    DETERMINISTIC    |
                  +-----------+  +---------------------+
                       ^                    |
                       |                    v
                       |          +---------------------+
                       |          |    POLICY ENGINE    |
                       |          |    DETERMINISTIC    |
                       |          +---------------------+
                       |             |               |
                       |        ALLOWED          ESCALATE
                       |             |               |
                       |             v               |
                       |   +------------------+      |
                       +---| GROUNDED         |      |
                  entitled | RESPONSE         |      |
                  part +   +------------------+      |
                  escalated         |                |
                  excess            v                v
                            +----------------+  +----------+
                            | OPTIONAL LLM   |  | HUMAN    |
                            | PHRASE ONLY    |  | SUPPORT  |
                            +----------------+  +----------+
                                     |                |
                                     v                v
                                        CUSTOMER
```

### Execution order

1. Resolve the selected customer and their booking.
2. **Guardrails run first**, before any model call.
3. Classify intent deterministically (regex over the message + the booking's disruption type).
4. Apply the policy engine to produce a structured `PolicyDecision`.
5. Build a grounded response from that decision.
6. Optionally hand the finished draft to an LLM for phrasing — **only when nothing was escalated**.

Steps 5 and 6 never run for an escalation, so escalation wording is fully deterministic.

### The three outcomes

| Outcome | When | LLM called? |
|---|---|---|
| **Allowed** | The request is covered by a stated rule | Yes, phrasing only |
| **Escalated** | Prohibited action, legal threat, or over the fare-waiver limit | No |
| **Partial grant** | The ask exceeds policy, but a stated entitlement still applies to the same disruption | No |

**Partial grant** is the interesting one. When Meher (6h delay) asks for a *full night's* hotel, a naive agent either wrongly grants it or escalates the whole turn and never tells her what she is actually owed. Here the agent applies her more-than-5-hours entitlement — meal voucher, lounge access, hotel covering the delayed hours — states which rule limits the rest, and escalates only that excess. This mirrors the tone of Sample B in the data pack.

---

# 3. Running it

Requires Python 3.10+.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/streamlit run app.py
```

Then open http://localhost:8501 and click one of the three scenario buttons.

No API key or `.env` file is needed — the agent falls back to deterministic templates, and every policy decision is identical either way.

### Console walkthrough

To see all three scenarios plus the legal-threat case without the UI:

```bash
.venv/bin/python test_agent.py
```

This prints each turn with its classified intent, escalation status, rule IDs, and allowed actions.

### Optional LLM phrasing

Copy `.env.example` to `.env` and set a provider:

```bash
cp .env.example .env
```

| Variable | Values |
|---|---|
| `LLM_PROVIDER` | `none` (default), `openai`, `anthropic` |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | key, and model id |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | key, and model id |

The LLM receives only the selected customer's booking, the rules, the computed decision, and the grounded draft. It cannot change the decision, and it is never called on an escalation path.

---

# 4. The scenarios

### Scenario 1 — Priya Nair (Gold, SK4821X)

Her SK-204 is cancelled by the airline.

- **Turn 1** — *"What can you do for me?"* → `cancellation`, allowed. Offers free rebooking within 24h **or** a full refund (7 business days, original payment method), plus the Gold priority-rebooking note and an explicit statement that loyalty tier adds no extra compensation.
- **Turn 2** — *"I'm furious… full cash refund plus a free upgrade to business class for the trouble."* → `beyond_policy`, escalated. A free upgrade is compensation beyond the stated policy. Caught by guardrails before any model call.

### Scenario 2 — Arvind Kulkarni (Silver, TR1190B)

SK-118 delayed 4 hours; he asks for a hotel.

→ `delay`, allowed. 4 hours clears the 3-hour threshold but not the 5-hour one, so he gets meal voucher + lounge access, and the agent states plainly that hotel accommodation begins above 5 hours. It declines without escalating, because nothing prohibited was requested.

If he then *insists* on a hotel, repeat-insistence on an entitlement he does not have escalates to a human.

### Scenario 3 — Meher Kaur (Platinum, WL7742)

SK-305 delayed 6 hours. The data pack describes two separate asks, so the demo plays them as two turns.

- **Turn 1** — *"a full night's hotel stay, not just the delayed hours."* → **partial grant.** Applies the >5h entitlement, then escalates only the full-night excess.
- **Turn 2** — *"move me to a higher-fare flight, the difference is ₹2,000."* → `fare_difference`, escalated. ₹2,000 exceeds the ₹1,500 agent waiver limit, so it requires supervisor approval.

### Cross-cutting — legal threats

Any message containing legal-action or formal-complaint language escalates immediately, ahead of every other check, and never reaches an LLM.

---

# 5. Project layout

```text
app.py                        Streamlit UI, scenario playback, decision trace
agent/orchestrator.py         Execution order, intent classification
agent/guardrails.py           Pre-LLM prohibited-request detection
agent/policy_engine.py        Deterministic rule application
agent/response_generator.py   Grounded response text
agent/llm.py                  Optional OpenAI / Anthropic phrasing
data/customers.json           Customer profiles
data/bookings.json            Bookings and disruption status
data/rules.json               Service rules, allowed and prohibited actions
test_agent.py                 Console walkthrough of all scenarios
```

---

# 6. Design notes

**Currency is ₹ (INR) throughout**, matching the data pack. The fare parser accepts `₹`, `Rs`, `Rs.`, `INR`, and `rupees`.

**Guardrails run before the LLM, not after.** Filtering a model's output means the model has already reasoned about a prohibited request. Checking first means it never sees it.

**Intent classification is regex, not a model.** For a fixed rule set this is more predictable and fully explainable in the decision trace. A production version would keep the deterministic policy engine and swap this layer for a classifier with a confidence threshold, falling back to escalation when uncertain.

**Nothing is invented.** Every customer-facing claim traces to a rule ID in `data/rules.json`. Where the data pack is silent, the agent escalates rather than filling the gap.

**Known limits.** Rebooking and refunds are surfaced as decisions, not executed against a reservation system; there is no auth layer, so the customer is chosen from a dropdown; and conversation state is per-session only.
