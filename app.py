import json
from pathlib import Path

import streamlit as st

from agent.orchestrator import Orchestrator


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

with st.sidebar:
    st.header("Customer")

    selected = st.selectbox("Select customer", customer_names)

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

scenario_prompts = {
    "Priya Nair": (
        "My flight SK-204 is cancelled. I want a full cash refund "
        "plus a free upgrade to business class on my return flight."
    ),
    "Arvind Kulkarni": (
        "SK-118 is delayed 4 hours. I want hotel accommodation."
    ),
    "Meher Kaur": (
        "SK-305 is delayed 6 hours. I want a full night's hotel stay "
        "and to move to a higher-fare flight. The fare difference is £2,000."
    ),
}

c1, c2, c3 = st.columns(3)

with c1:
    if st.button("Scenario 1 — Priya", use_container_width=True):
        st.session_state.pending_prompt = scenario_prompts["Priya Nair"]
with c2:
    if st.button("Scenario 2 — Arvind", use_container_width=True):
        st.session_state.pending_prompt = scenario_prompts["Arvind Kulkarni"]
with c3:
    if st.button("Scenario 3 — Meher", use_container_width=True):
        st.session_state.pending_prompt = scenario_prompts["Meher Kaur"]


# ============================================================
# PROCESS PENDING DEMO PROMPT
# ============================================================
pending = st.session_state.pop("pending_prompt", None)

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
        "llm_used": result["llm_used"],
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
            if message.get("escalated"):
                st.error(
                    "⚠️ Escalated: "
                    + str(message.get("escalation_category", "human review"))
                )
            st.caption("Intent: " + str(message.get("intent", "unknown")))
            st.write(message["content"])
            if message.get("rule_ids"):
                st.caption("Rules used: " + ", ".join(message["rule_ids"]))


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
        "llm_used": result["llm_used"],
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
                    "llm_used": message.get("llm_used"),
                })