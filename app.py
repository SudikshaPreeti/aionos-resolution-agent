import json
from pathlib import Path

import streamlit as st

from agent.orchestrator import Orchestrator
from agent.usage import summarize


# ============================================================
# PAGE CONFIG — MUST BE THE FIRST STREAMLIT COMMAND
# ============================================================
st.set_page_config(
    page_title="AIONOS Customer Resolution Agent",
    page_icon="✈️",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# METRIC FORMATTING
# ============================================================
def fmt_cost(value, currency="USD", unpriced=False):
    """
    Money is shown at the precision it actually has. A single rephrasing
    call costs a fraction of a cent, so two decimal places would print
    every turn as $0.00 and hide the thing we are measuring.
    """

    if unpriced:
        return "unpriced"

    if value is None:
        return "—"

    symbol = "$" if currency == "USD" else f"{currency} "

    if value == 0:
        return f"{symbol}0.00"

    if value < 0.01:
        return f"{symbol}{value:.6f}"

    return f"{symbol}{value:.4f}"


def fmt_ms(value):
    if value is None:
        return "—"

    if value < 1:
        return f"{value:.2f} ms"

    if value < 1000:
        return f"{value:.0f} ms"

    return f"{value / 1000:.2f} s"


def render_turn_metrics(metrics):
    """
    The per-response accounting strip.
    """

    if not metrics:
        return

    llm_used = metrics.get("llm_used")

    # Three metrics, not four: a fourth column is narrow enough that
    # Streamlit truncates the value text.
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Tokens",
        f"{metrics.get('total_tokens', 0):,}",
        help=(
            f"{metrics.get('input_tokens', 0):,} in · "
            f"{metrics.get('output_tokens', 0):,} out"
        ),
    )

    c2.metric(
        "Cost",
        fmt_cost(
            metrics.get("cost"),
            metrics.get("currency", "USD"),
            metrics.get("unpriced", False),
        ),
        help="Rates come from data/model_pricing.json.",
    )

    c3.metric(
        "Latency",
        fmt_ms(metrics.get("total_ms")),
    )

    st.caption(
        "**Path:** "
        + (
            f"{metrics.get('provider')} · `{metrics.get('model')}`"
            if llm_used
            else "deterministic — no model call"
        )
        + f" &nbsp;•&nbsp; **Intent:** `{metrics.get('intent')}`"
        + f" &nbsp;•&nbsp; **Outcome:** `{metrics.get('outcome')}`"
    )

    st.caption(
        "**Where the time went:** "
        f"guardrails {fmt_ms(metrics.get('guardrail_ms'))}"
        f" &nbsp;•&nbsp; policy {fmt_ms(metrics.get('policy_ms'))}"
        f" &nbsp;•&nbsp; phrasing {fmt_ms(metrics.get('llm_ms'))}"
    )

    if metrics.get("llm_error"):
        st.warning(
            "Phrasing call failed, so the grounded draft was sent "
            f"unchanged: {metrics['llm_error']}"
        )

    if not llm_used and metrics.get("llm_skip_reason"):
        st.caption(f"⚡ {metrics['llm_skip_reason']}")

    if metrics.get("cache_read_tokens"):
        st.caption(
            f"Cache: {metrics['cache_read_tokens']:,} read · "
            f"{metrics.get('cache_write_tokens', 0):,} written"
        )


# ============================================================
# LOAD JSON
# ============================================================
@st.cache_data
def load_json(relative_path: str):
    with open(BASE_DIR / relative_path, "r", encoding="utf-8") as f:
        return json.load(f)


customers_data = load_json("data/customers.json")
bookings_data  = load_json("data/bookings.json")
rules_data     = load_json("data/rules.json")


# ============================================================
# AGENT
# ============================================================
agent = Orchestrator(customers_data, bookings_data, rules_data)


# ============================================================
# TITLE
# ============================================================
st.title("✈️ Customer-Facing Resolution Agent")
st.caption("AIONOS Assignment 3 • Airline Disruption • Policy-Grounded Prototype")


# ============================================================
# SIDEBAR — CUSTOMER + BOOKING
# ============================================================
customer_names = [c["name"] for c in customers_data["customers"]]

# A scenario button queues a customer switch and reruns. Streamlit forbids
# writing a widget's state after that widget exists, so apply the switch
# here — before the selectbox below is instantiated.
if "pending_customer" in st.session_state:
    st.session_state.customer_select = st.session_state.pop("pending_customer")

with st.sidebar:
    st.header("Customer")

    selected = st.selectbox(
        "Select customer",
        customer_names,
        key="customer_select",
    )

    customer = next(
        c for c in customers_data["customers"] if c["name"] == selected
    )
    booking = next(
        b for b in bookings_data["bookings"]
        if b["booking_reference"] == customer["booking_reference"]
    )

    st.divider()
    st.subheader("Customer profile")
    st.write(f"**Name:** {customer['name']}")
    st.write(f"**Tier:** {customer['loyalty_tier']}")
    st.write(f"**PNR:** {customer['booking_reference']}")

    st.divider()
    st.subheader("Live booking status")
    for f in booking["flights"]:
        st.markdown(
            f"**{f['flight']}**  \n"
            f"{f['route']}  \n"
            f"{f['date']} • {f['scheduled_departure']}  \n"
            f"**Status:** {f['status']}"
        )
        st.write("")

    st.divider()
    st.subheader("Session cost")

    session = summarize([
        m.get("metrics")
        for m in st.session_state.get("messages", [])
        if m.get("role") == "assistant"
    ])

    if session["turns"] == 0:
        st.caption("No responses yet.")
    else:
        s1, s2 = st.columns(2)
        s1.metric("Responses", session["turns"])
        s2.metric("Tokens", f"{session['total_tokens']:,}")

        s3, s4 = st.columns(2)
        s3.metric(
            "Total cost",
            fmt_cost(
                session["cost"],
                session["currency"],
                session["unpriced"],
            ),
        )
        s4.metric("Avg latency", fmt_ms(session["avg_ms"]))

        # The headline number for this design: how much of the work never
        # reached a model at all.
        st.metric(
            "Resolved without an LLM",
            f"{session['deterministic_share'] * 100:.0f}%",
            help=(
                f"{session['deterministic_turns']} of {session['turns']} "
                "responses cost zero tokens — guardrails and the policy "
                "engine answered them outright."
            ),
        )

        st.caption(
            f"{session['escalated']} of {session['turns']} escalated to a human."
        )

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# SESSION STATE
# ============================================================
if "selected_customer" not in st.session_state:
    st.session_state.selected_customer = selected

if st.session_state.selected_customer != selected:
    st.session_state.selected_customer = selected
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# DEMO SCENARIO BUTTONS
# ============================================================
st.subheader("Try a scenario")

# Each scenario is a list of turns, played in order. Meher's two asks are
# separate turns because the data pack describes them that way, and because
# one combined message would only ever surface whichever rule fires first.
scenario_prompts = {
    "Priya Nair": [
        "My flight SK-204 is cancelled. What can you do for me?",
        "I'm furious. I want a full cash refund plus a free upgrade "
        "to business class on my return flight for the trouble.",
    ],
    "Arvind Kulkarni": [
        "SK-118 is delayed 4 hours and I'm missing a connecting meeting. "
        "I want hotel accommodation since it's been such a long delay.",
    ],
    "Meher Kaur": [
        "SK-305 is delayed 6 hours. I want a full night's hotel stay, "
        "not just coverage for the delayed hours.",
        "Then move me onto a different, higher-fare flight instead of "
        "waiting. The fare difference is ₹2,000.",
    ],
}

c1, c2, c3 = st.columns(3)

def start_scenario(customer_name: str) -> None:
    """
    Switch to that scenario's customer and queue their turns.

    Both are needed: sending Meher's message while Priya is selected would
    answer it against Priya's booking.
    """

    st.session_state.pending_customer = customer_name
    st.session_state.pending_prompts = list(scenario_prompts[customer_name])
    st.rerun()


with c1:
    if st.button("Scenario 1 — Priya", use_container_width=True):
        start_scenario("Priya Nair")
with c2:
    if st.button("Scenario 2 — Arvind", use_container_width=True):
        start_scenario("Arvind Kulkarni")
with c3:
    if st.button("Scenario 3 — Meher", use_container_width=True):
        start_scenario("Meher Kaur")


# ============================================================
# PROCESS PENDING DEMO TURNS
#
# One turn per rerun, so each answer is generated against the
# history the turns before it produced.
# ============================================================
queued = st.session_state.get("pending_prompts") or []

pending = queued.pop(0) if queued else None

if not queued:
    st.session_state.pop("pending_prompts", None)

if pending:
    st.session_state.messages.append({"role": "user", "content": pending})
    result = agent.handle(
        customer_name=selected,
        message=pending,
        history=st.session_state.messages[:-1],
    )
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["response"],
        "intent": result["intent"],
        "escalated": result["escalated"],
        "escalation_category": result["escalation_category"],
        "rule_ids": result["rule_ids"],
        "actions": result["actions"],
        "partial_grant": result.get("partial_grant", False),
        "llm_used": result["llm_used"],
        "metrics": result.get("metrics", {}),
    })
    st.rerun()


# ============================================================
# CONVERSATION
# ============================================================
st.divider()
st.subheader("Conversation")

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            if message.get("partial_grant"):
                st.warning(
                    "⚖️ Entitlement applied • excess escalated: "
                    + str(message.get("escalation_category", "human review"))
                )
            elif message.get("escalated"):
                st.error(
                    "⚠️ Escalated: "
                    + str(message.get("escalation_category", "human review"))
                )
            st.caption("Intent: " + str(message.get("intent", "unknown")))
            st.write(message["content"])
            if message.get("rule_ids"):
                st.caption("Rules used: " + ", ".join(message["rule_ids"]))
            with st.expander("📊 Cost & performance for this response"):
                render_turn_metrics(message.get("metrics"))


# ============================================================
# CHAT INPUT
# ============================================================
prompt = st.chat_input("Describe the customer's issue...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    result = agent.handle(
        customer_name=selected,
        message=prompt,
        history=st.session_state.messages[:-1],
    )
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["response"],
        "intent": result["intent"],
        "escalated": result["escalated"],
        "escalation_category": result["escalation_category"],
        "rule_ids": result["rule_ids"],
        "actions": result["actions"],
        "partial_grant": result.get("partial_grant", False),
        "llm_used": result["llm_used"],
        "metrics": result.get("metrics", {}),
    })
    st.rerun()


# ============================================================
# TECHNICAL TRACE
# ============================================================
with st.expander("🔍 Technical decision trace"):
    if not st.session_state.messages:
        st.write("Run a scenario or send a message to see the decision trace.")
    else:
        for message in reversed(st.session_state.messages):
            if message["role"] == "assistant":
                st.write({
                    "intent": message.get("intent"),
                    "escalated": message.get("escalated"),
                    "escalation_category": message.get("escalation_category"),
                    "rule_ids": message.get("rule_ids"),
                    "allowed_actions": message.get("actions"),
                    "partial_grant": message.get("partial_grant"),
                    "llm_used": message.get("llm_used"),
                    "metrics": message.get("metrics"),
                })