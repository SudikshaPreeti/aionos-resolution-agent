# AIONOS Assignment 3 — Customer-Facing Resolution Agent

## Airline Disruption Resolution Agent

A complete Python + Streamlit prototype for the AIONOS Customer-Facing Resolution Agent assignment.

The application handles the three supplied airline-disruption scenarios while keeping policy decisions deterministic and auditable.

---

# 1. Project Overview

The agent is designed around one core principle:

> The LLM generates language. The deterministic policy engine makes decisions.

The project contains:

- customer data
- booking data
- service rules
- deterministic intent classification
- pre-LLM guardrails
- deterministic policy engine
- grounded response generator
- optional OpenAI integration
- optional Anthropic integration
- Streamlit frontend
- template fallback when no API key is available

The application can therefore be demonstrated without any API key.

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
                 |   BEFORE LLM      |
                 +-------------------+
                    |             |
             HARD   |             | SAFE
          ESCALATE  |             |
                    v             v
             +-----------+   +------------------+
             |  HUMAN    |   | INTENT CLASSIFIER|
             | ESCALATION|   +------------------+
             +-----------+            |
                                      v
                             +------------------+
                             |  POLICY ENGINE   |
                             |  DETERMINISTIC   |
                             +------------------+
                                |           |
                           ALLOWED       ESCALATE
                                |           |
                                v           v
                     +----------------+  +----------+
                     | GROUNDED       |  | HUMAN    |
                     | RESPONSE       |  | SUPPORT  |
                     +----------------+  +----------+
                                |
                                v
                       +----------------+
                       | OPTIONAL LLM   |
                       | PHRASE ONLY    |
                       +----------------+
                                |
                                v
                           CUSTOMER