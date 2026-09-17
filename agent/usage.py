"""
Per-turn token, cost and latency accounting.

Every turn produces a TurnMetrics, whether or not an LLM was involved.
A turn resolved entirely by the deterministic path reports zero tokens
and zero cost — which is the point: it makes the cost of the guardrail
and policy layers visible as *nothing*.

Prices come from data/model_pricing.json, never from code. A model with
no rate configured still reports its token counts and is marked
unpriced rather than being costed with a guessed number.
"""

from __future__ import annotations

import json

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


PRICING_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "model_pricing.json"
)


# ==============================================================
# PRICING
# ==============================================================

class Pricing:
    """
    Token rates, loaded from data/model_pricing.json.
    """

    def __init__(self, path: Path | None = None):

        self.path = path or PRICING_PATH

        try:
            raw = json.loads(
                self.path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            raw = {}

        self.currency = raw.get("currency", "USD")

        self.models: Dict[str, Any] = raw.get("models", {})

    def rates(
        self,
        model: Optional[str]
    ) -> Optional[Dict[str, float]]:
        """
        Return {input, output} per 1M tokens, or None when the model
        is unknown or deliberately unpriced.
        """

        if not model:
            return None

        entry = self.models.get(model)

        if not entry:
            return None

        if entry.get("input") is None or entry.get("output") is None:
            return None

        return {
            "input": float(entry["input"]),
            "output": float(entry["output"]),
        }

    def cost(
        self,
        model: Optional[str],
        input_tokens: int,
        output_tokens: int,
    ) -> Optional[float]:
        """
        Cost in `currency`, or None when the model has no configured rate.
        """

        rates = self.rates(model)

        if rates is None:
            return None

        return (
            input_tokens * rates["input"]
            + output_tokens * rates["output"]
        ) / 1_000_000


# ==============================================================
# LLM CALL USAGE
# ==============================================================

@dataclass
class LLMUsage:
    """
    What one provider call consumed. Absent when no call was made.
    """

    provider: str = "none"

    model: Optional[str] = None

    input_tokens: int = 0

    output_tokens: int = 0

    # Anthropic reports these separately; they are a subset of input work
    # and are surfaced because a nonzero cache read is the clearest signal
    # that prompt caching is doing something.
    cache_read_tokens: int = 0

    cache_write_tokens: int = 0

    latency_ms: float = 0.0

    # Set when the call was attempted and failed, so the UI can show the
    # fallback to the grounded draft rather than silently reporting zeros.
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ==============================================================
# TURN METRICS
# ==============================================================

@dataclass
class TurnMetrics:
    """
    Everything worth knowing about how one response was produced.
    """

    # --- decision ---
    intent: str = "unknown"

    outcome: str = "allowed"

    rule_ids: List[str] = field(default_factory=list)

    escalated: bool = False

    partial_grant: bool = False

    # --- where the time went ---
    guardrail_ms: float = 0.0

    policy_ms: float = 0.0

    llm_ms: float = 0.0

    total_ms: float = 0.0

    # --- what it cost ---
    llm_used: bool = False

    # Why no model ran, when none did. Escalations skip the LLM by design,
    # so this distinguishes "policy forbade it" from "no key configured".
    llm_skip_reason: Optional[str] = None

    provider: str = "none"

    model: Optional[str] = None

    input_tokens: int = 0

    output_tokens: int = 0

    cache_read_tokens: int = 0

    cache_write_tokens: int = 0

    cost: Optional[float] = None

    currency: str = "USD"

    # True when tokens were spent on a model with no configured rate.
    unpriced: bool = False

    llm_error: Optional[str] = None

    # --- output shape ---
    response_chars: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def deterministic_ms(self) -> float:
        """Time spent in the parts that never call a model."""
        return self.guardrail_ms + self.policy_ms

    def to_dict(self) -> Dict[str, Any]:
        # The derived properties are folded in explicitly: asdict() only
        # emits declared fields, and the UI reads these from the dict
        # after a Streamlit rerun has discarded the dataclass.
        data = asdict(self)

        data["total_tokens"] = self.total_tokens
        data["deterministic_ms"] = self.deterministic_ms

        return data


# ==============================================================
# SESSION ROLLUP
# ==============================================================

def summarize(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Roll a conversation's per-turn metrics into session totals.

    Takes plain dicts because Streamlit stores messages as dicts across
    reruns; a TurnMetrics survives the round trip only as its to_dict().
    """

    turns = [m for m in metrics if m]

    if not turns:
        return {
            "turns": 0,
            "llm_turns": 0,
            "deterministic_turns": 0,
            "deterministic_share": 0.0,
            "escalated": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "unpriced": False,
            "currency": "USD",
            "avg_ms": 0.0,
            "total_ms": 0.0,
        }

    llm_turns = sum(1 for m in turns if m.get("llm_used"))

    costed = [
        m.get("cost")
        for m in turns
        if m.get("cost") is not None
    ]

    return {
        "turns": len(turns),

        "llm_turns": llm_turns,

        "deterministic_turns": len(turns) - llm_turns,

        "deterministic_share": (len(turns) - llm_turns) / len(turns),

        "escalated": sum(
            1 for m in turns if m.get("escalated")
        ),

        "input_tokens": sum(
            int(m.get("input_tokens", 0)) for m in turns
        ),

        "output_tokens": sum(
            int(m.get("output_tokens", 0)) for m in turns
        ),

        "total_tokens": sum(
            int(m.get("input_tokens", 0)) + int(m.get("output_tokens", 0))
            for m in turns
        ),

        "cost": sum(costed),

        # Flagged so a total is never shown as complete when some turn
        # burned tokens on a model we have no rate for.
        "unpriced": any(m.get("unpriced") for m in turns),

        "currency": next(
            (m.get("currency") for m in turns if m.get("currency")),
            "USD",
        ),

        "avg_ms": sum(
            float(m.get("total_ms", 0.0)) for m in turns
        ) / len(turns),

        "total_ms": sum(
            float(m.get("total_ms", 0.0)) for m in turns
        ),
    }
