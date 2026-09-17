import html
import json
from pathlib import Path

import streamlit as st

from agent.orchestrator import Orchestrator
from agent.usage import summarize


# ============================================================
# PAGE CONFIG — MUST BE THE FIRST STREAMLIT COMMAND
# ============================================================
st.set_page_config(
    page_title="Resolution Agent",
    page_icon="◆",
    layout="centered",
)


# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# STYLE
#
# Colours are rgba tints over whatever the active Streamlit theme
# provides, so the same values stay legible in light and dark.
# ============================================================
st.markdown(
    """
    <style>
      .block-container { padding-top: 3rem; max-width: 46rem; }

      /* The default h1 is far too loud for a single-purpose tool. */
      .app-title {
        font-size: 1.45rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        margin: 0 0 0.15rem 0;
      }
      .app-sub {
        font-size: 0.82rem;
        opacity: 0.55;
        margin: 0 0 0.35rem 0;
      }

      .eyebrow {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.45;
        margin: 0 0 0.5rem 0;
      }

      .pill {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        padding: 0.12rem 0.5rem;
        border-radius: 999px;
        margin: 0 0.3rem 0.5rem 0;
        border: 1px solid transparent;
      }
      .pill-escalated {
        background: rgba(220, 70, 70, 0.12);
        border-color: rgba(220, 70, 70, 0.35);
        color: rgb(215, 90, 90);
      }
      .pill-partial {
        background: rgba(220, 150, 40, 0.12);
        border-color: rgba(220, 150, 40, 0.38);
        color: rgb(205, 145, 50);
      }
      .pill-resolved {
        background: rgba(60, 170, 110, 0.12);
        border-color: rgba(60, 170, 110, 0.35);
        color: rgb(70, 165, 115);
      }
      .pill-quiet {
        background: rgba(128, 128, 128, 0.1);
        border-color: rgba(128, 128, 128, 0.25);
        opacity: 0.75;
      }

      /* The customer's own words: quieter than the agent's answer. */
      .turn-user {
        font-size: 0.9rem;
        opacity: 0.72;
        border-left: 2px solid rgba(128, 128, 128, 0.3);
        padding: 0.1rem 0 0.1rem 0.7rem;
        margin: 1.4rem 0 0.7rem 0;
      }

      .turn-body { font-size: 0.93rem; line-height: 1.6; }

      /* Metrics live inline and always visible — an expander per turn
         buried the one thing this UI exists to show. */
      .turn-meta {
        font-size: 0.72rem;
        opacity: 0.5;
        margin-top: 0.6rem;
        font-variant-numeric: tabular-nums;
      }
      .turn-meta code {
        font-size: 0.72rem;
        background: rgba(128, 128, 128, 0.12);
        padding: 0.05rem 0.3rem;
        border-radius: 3px;
      }

      .sb-name { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.1rem; }
      .sb-meta { font-size: 0.75rem; opacity: 0.55; margin-bottom: 0.9rem; }
      .sb-flight {
        font-size: 0.78rem;
        line-height: 1.5;
        padding: 0.5rem 0.65rem;
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 6px;
        margin-bottom: 0.5rem;
      }
      .sb-stat {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        padding: 0.18rem 0;
        font-variant-numeric: tabular-nums;
      }
      .sb-stat span:last-child { opacity: 0.6; }
      .sb-head {
        font-size: 1.1rem;
        font-weight: 650;
        font-variant-numeric: tabular-nums;
      }

      section[data-testid="stSidebar"] hr { margin: 0.9rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FORMATTING
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


def pill(label, kind="quiet"):
    return f'<span class="pill pill-{kind}">{html.escape(str(label))}</span>'


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
# SIDEBAR — CUSTOMER, BOOKING, SESSION COST
# ============================================================
customer_names = [c["name"] for c in customers_data["customers"]]

# A scenario button queues a customer switch and reruns. Streamlit forbids
# writing a widget's state after that widget exists, so apply the switch
# here — before the selectbox below is instantiated.
if "pending_customer" in st.session_state:
    st.session_state.customer_select = st.session_state.pop("pending_customer")

with st.sidebar:
    st.markdown('<p class="eyebrow">Customer</p>', unsafe_allow_html=True)

    selected = st.selectbox(
        "Select customer",
        customer_names,
        key="customer_select",
        label_visibility="collapsed",
    )

    customer = next(
        c for c in customers_data["customers"] if c["name"] == selected
    )
    booking = next(
        b for b in bookings_data["bookings"]
        if b["booking_reference"] == customer["booking_reference"]
    )

    # The name is already in the selectbox above — repeating it here just
    # takes a line. Tier and PNR are what the reviewer needs alongside it.
    st.markdown(
        f'<div class="sb-meta">{html.escape(customer["loyalty_tier"])}'
        f' · {html.escape(customer["booking_reference"])}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="eyebrow">Booking</p>', unsafe_allow_html=True)

    for f in booking["flights"]:
        st.markdown(
            '<div class="sb-flight">'
            f'<b>{html.escape(f["flight"])}</b> &nbsp; '
            f'{html.escape(f["route"])}<br>'
            f'<span style="opacity:.55">{html.escape(f["date"])} · '
            f'{html.escape(f["scheduled_departure"])}</span><br>'
            f'{html.escape(f["status"])}'
            "</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown('<p class="eyebrow">Session</p>', unsafe_allow_html=True)

    session = summarize([
        m.get("metrics")
        for m in st.session_state.get("messages", [])
        if m.get("role") == "assistant"
    ])

    if session["turns"] == 0:
        st.caption("No responses yet.")
    else:
        # The headline number for this design: how much of the work never
        # reached a model at all.
        st.markdown(
            f'<div class="sb-head">'
            f'{session["deterministic_share"] * 100:.0f}% resolved'
            f"</div>"
            f'<div class="sb-meta">without an LLM — '
            f'{session["deterministic_turns"]} of {session["turns"]} '
            f"responses cost zero tokens</div>",
            unsafe_allow_html=True,
        )

        rows = [
            ("Responses", f"{session['turns']}"),
            ("Escalated", f"{session['escalated']}"),
            ("Tokens", f"{session['total_tokens']:,}"),
            (
                "Cost",
                fmt_cost(
                    session["cost"],
                    session["currency"],
                    session["unpriced"],
                ),
            ),
            ("Avg latency", fmt_ms(session["avg_ms"])),
        ]

        st.markdown(
            "".join(
                f'<div class="sb-stat"><span>{k}</span>'
                f"<span>{html.escape(v)}</span></div>"
                for k, v in rows
            ),
            unsafe_allow_html=True,
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
# HEADER
# ============================================================
st.markdown(
    '<p class="app-title">Customer Resolution Agent</p>'
    '<p class="app-sub">Airline disruption · decisions are deterministic, '
    "the model only phrases them</p>",
    unsafe_allow_html=True,
)


# ============================================================
# DEMO SCENARIOS
# ============================================================

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

SCENARIO_LABELS = {
    "Priya Nair": "Priya · cancellation",
    "Arvind Kulkarni": "Arvind · 4h delay",
    "Meher Kaur": "Meher · 6h delay",
}


def start_scenario(customer_name: str) -> None:
    """
    Switch to that scenario's customer and queue their turns.

    Both are needed: sending Meher's message while Priya is selected would
    answer it against Priya's booking.
    """

    st.session_state.pending_customer = customer_name
    st.session_state.pending_prompts = list(scenario_prompts[customer_name])
    st.rerun()


st.markdown('<p class="eyebrow">Scenarios</p>', unsafe_allow_html=True)

cols = st.columns(3)

for col, name in zip(cols, scenario_prompts):
    with col:
        if st.button(SCENARIO_LABELS[name], use_container_width=True):
            start_scenario(name)


# ============================================================
# TURN HANDLING
# ============================================================
def record(message: str) -> None:
    """
    Send one customer message and store the answer with its accounting.
    """

    st.session_state.messages.append({"role": "user", "content": message})

    result = agent.handle(
        customer_name=selected,
        message=message,
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


# One queued turn per rerun, so each answer is generated against the
# history the turns before it produced.
queued = st.session_state.get("pending_prompts") or []

pending = queued.pop(0) if queued else None

if not queued:
    st.session_state.pop("pending_prompts", None)

if pending:
    record(pending)
    st.rerun()


# ============================================================
# CONVERSATION
# ============================================================
def render_status(message) -> str:
    if message.get("partial_grant"):
        return pill(
            "entitlement applied · excess escalated",
            "partial",
        )

    if message.get("escalated"):
        return pill(
            f"escalated · {message.get('escalation_category', 'human review')}",
            "escalated",
        )

    return pill("resolved in policy", "resolved")


def render_meta(message) -> str:
    """
    The single always-visible accounting line under each answer.
    """

    metrics = message.get("metrics") or {}

    parts = []

    if message.get("rule_ids"):
        parts.append(
            "rules "
            + " · ".join(
                f"<code>{html.escape(r)}</code>"
                for r in message["rule_ids"]
            )
        )

    parts.append(f"{metrics.get('total_tokens', 0):,} tokens")

    parts.append(
        fmt_cost(
            metrics.get("cost"),
            metrics.get("currency", "USD"),
            metrics.get("unpriced", False),
        )
    )

    parts.append(fmt_ms(metrics.get("total_ms")))

    parts.append(
        f"{metrics.get('provider')}/{metrics.get('model')}"
        if metrics.get("llm_used")
        else "no model call"
    )

    return '<div class="turn-meta">' + " &nbsp;·&nbsp; ".join(parts) + "</div>"


if st.session_state.messages:
    st.markdown(
        '<p class="eyebrow" style="margin-top:1.8rem">Conversation</p>',
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:

    if message["role"] == "user":
        st.markdown(
            f'<div class="turn-user">{html.escape(message["content"])}</div>',
            unsafe_allow_html=True,
        )
        continue

    with st.container(border=True):
        st.markdown(
            render_status(message)
            + pill(message.get("intent", "unknown")),
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="turn-body">{html.escape(message["content"])}</div>'
            + render_meta(message),
            unsafe_allow_html=True,
        )

        metrics = message.get("metrics") or {}

        if metrics.get("llm_error"):
            st.warning(
                "Phrasing call failed, so the grounded draft was sent "
                f"unchanged: {metrics['llm_error']}"
            )


# ============================================================
# CHAT INPUT
# ============================================================
prompt = st.chat_input("Describe the customer's issue…")

if prompt:
    record(prompt)
    st.rerun()


# ============================================================
# TECHNICAL TRACE
# ============================================================
with st.expander("Decision trace"):
    if not st.session_state.messages:
        st.caption("Run a scenario or send a message.")
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
