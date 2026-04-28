"""Build Hermes-format tool schemas for Qwen3-30B-A3B-Thinking.

The model must only propose tool calls whose parameters we can actually
execute. Therefore every schema here is derived from a real executor — no
hallucinatable parameters. The four categories mirror proposal §3.2:

    A. change_instrument, layer_instrument          (re-rendering)
    B. apply_fx                                     (MultiAFX, 85 effects)
    C. humanize_timing, change_articulation                    (performance)
    D. add_track, remove_track, double_track, mute_and_replace  (arrangement)

apply_fx is the only "polymorphic" tool: it takes an ``effect`` enum and a
``params`` object whose keys depend on which effect was selected. We enumerate
all 85 MultiAFX effects and emit one JSON-schema oneOf branch per effect so
the model gets full parameter visibility.
"""

from __future__ import annotations

import json
from typing import Any

from multiafx import registry
from multiafx.types import ParamRange


# ---------------------------------------------------------------------------
# Helpers — translate MultiAFX registry entries into JSON Schema
# ---------------------------------------------------------------------------
INTEGER_PARAMS = frozenset({"order", "numtaps", "n_poles"})


def _param_to_schema(name: str, rng: ParamRange) -> dict[str, Any]:
    s: dict[str, Any] = {
        "type": "integer" if name in INTEGER_PARAMS else "number",
        "minimum": rng.min_val,
        "maximum": rng.max_val,
    }
    if rng.log_scale:
        s["description"] = f"log-scale range [{rng.min_val}, {rng.max_val}]"
    return s


def _effect_schema_branch(effect_name: str) -> dict[str, Any]:
    """oneOf branch for a single MultiAFX effect inside apply_fx."""
    eff = registry.get(effect_name)
    properties = {
        "effect": {"type": "string", "const": effect_name},
        "params": {
            "type": "object",
            "properties": {p: _param_to_schema(p, r) for p, r in eff.param_ranges.items()},
            "additionalProperties": False,
        },
    }
    return {
        "type": "object",
        "properties": properties,
        "required": ["effect", "params"],
        "additionalProperties": False,
        "description": f"{eff.library} / {eff.macro_category.value} / {eff.category}",
    }


# ---------------------------------------------------------------------------
# Category A — instrument change / layering
# ---------------------------------------------------------------------------
def change_instrument_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "change_instrument",
            "description": (
                "Re-render one track of the MIDI with a different General MIDI program. "
                "Only valid for track names present in the mixture metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "track": {"type": "string"},
                    "from_program": {"type": "integer", "minimum": 0, "maximum": 127,
                                     "description": "Original GM program number."},
                    "to_program": {"type": "integer", "minimum": 0, "maximum": 127,
                                   "description": "Target GM program number."},
                },
                "required": ["track", "to_program"],
                "additionalProperties": False,
            },
        },
    }


def layer_instrument_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "layer_instrument",
            "description": "Duplicate a track's MIDI with an additional GM program, blended at mix_ratio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "track": {"type": "string"},
                    "additional_program": {"type": "integer", "minimum": 0, "maximum": 127},
                    "mix_ratio": {"type": "number", "minimum": 0.1, "maximum": 1.0,
                                  "description": "Linear gain applied to the added layer (0.5 = -6dB)."},
                },
                "required": ["track", "additional_program", "mix_ratio"],
                "additionalProperties": False,
            },
        },
    }


# ---------------------------------------------------------------------------
# Category B — audio effects (MultiAFX)
# ---------------------------------------------------------------------------
def apply_fx_schema() -> dict[str, Any]:
    """apply_fx with a oneOf branch per effect so params are fully typed."""
    effect_names = sorted(registry.list_effects())
    oneof = [_effect_schema_branch(n) for n in effect_names]
    return {
        "type": "function",
        "function": {
            "name": "apply_fx",
            "description": (
                "Apply one MultiAFX audio effect to a specific track OR to the full "
                "mix bus. There are only two kinds of targets: (1) a track name from "
                "the Mixture metadata, applied to that stem only; (2) the literal "
                "string 'mix', applied to the entire summed mixture. We do not "
                "support per-group sub-buses (e.g. a 'guitars bus'); for the same "
                "treatment on several stems, emit one apply_fx call per stem. "
                "Exactly one effect per call; parameters must match the chosen "
                f"effect's schema. {len(effect_names)} effects across 7 libraries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "track": {"type": "string", "default": "mix",
                              "description": "Track name from the Mixture metadata, "
                                             "or 'mix' for the full mix bus."},
                    "call": {
                        "oneOf": oneof,
                        "description": "The effect and its parameters.",
                    },
                },
                "required": ["track", "call"],
                "additionalProperties": False,
            },
        },
    }


# ---------------------------------------------------------------------------
# Category C — performance modifications (MIDI-level)
# ---------------------------------------------------------------------------
def humanize_timing_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "humanize_timing",
            "description": "Add uniform random onset jitter to the named track within ±max_offset_ms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "track": {"type": "string"},
                    "max_offset_ms": {"type": "number", "minimum": 1.0, "maximum": 40.0},
                },
                "required": ["track", "max_offset_ms"],
                "additionalProperties": False,
            },
        },
    }


def change_articulation_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "change_articulation",
            "description": "Reshape note durations on the named track.",
            "parameters": {
                "type": "object",
                "properties": {
                    "track": {"type": "string"},
                    "style": {"type": "string", "enum": ["legato", "staccato", "tenuto"]},
                },
                "required": ["track", "style"],
                "additionalProperties": False,
            },
        },
    }


# ---------------------------------------------------------------------------
# Category D — arrangement (content-modifying operations)
# ---------------------------------------------------------------------------
def add_track_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "add_track",
            "description": (
                "Re-introduce a track that exists in the original MIDI but was withheld "
                "from the input mixture. See proposal §3.2 Category D."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "track_name": {"type": "string"},
                    "program": {"type": "integer", "minimum": 0, "maximum": 127},
                    "gain_db": {"type": "number", "minimum": -12.0, "maximum": 6.0, "default": 0.0},
                },
                "required": ["track_name", "program"],
                "additionalProperties": False,
            },
        },
    }


def remove_track_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "remove_track",
            "description": "Drop the named track's stem from the output mixture.",
            "parameters": {
                "type": "object",
                "properties": {"track_name": {"type": "string"}},
                "required": ["track_name"],
                "additionalProperties": False,
            },
        },
    }


def double_track_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "double_track",
            "description": "Duplicate a track with slight timing/pitch offset for thickening.",
            "parameters": {
                "type": "object",
                "properties": {
                    "track_name": {"type": "string"},
                    "offset_ms": {"type": "number", "minimum": 0.0, "maximum": 30.0},
                    "detune_cents": {"type": "number", "minimum": -25.0, "maximum": 25.0},
                },
                "required": ["track_name", "offset_ms", "detune_cents"],
                "additionalProperties": False,
            },
        },
    }


def mute_and_replace_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "mute_and_replace",
            "description": "Remove a track and add a new one playing the same MIDI with a different GM program.",
            "parameters": {
                "type": "object",
                "properties": {
                    "track_name": {"type": "string"},
                    "new_program": {"type": "integer", "minimum": 0, "maximum": 127},
                },
                "required": ["track_name", "new_program"],
                "additionalProperties": False,
            },
        },
    }


# ---------------------------------------------------------------------------
# Top-level: assemble Hermes tool list for the system prompt
# ---------------------------------------------------------------------------
def all_tool_schemas() -> list[dict[str, Any]]:
    return [
        change_instrument_schema(),
        layer_instrument_schema(),
        apply_fx_schema(),
        humanize_timing_schema(),
        change_articulation_schema(),
        add_track_schema(),
        remove_track_schema(),
        double_track_schema(),
        mute_and_replace_schema(),
    ]


HERMES_SYSTEM_PREAMBLE = (
    "You are a professional music producer making one production decision per turn. "
    "The mixtures you work with are INSTRUMENTAL multitracks rendered from MIDI; "
    "they never contain vocals, rap, or any voice. "
    "You have access to the tools below. First reason inside <think>...</think>: "
    "diagnose what is genuinely weak in the current mixture (referring ONLY to the "
    "tracks present in the 'Mixture metadata' below), and pick the tools that would "
    "address it. Then output one 'Production motivation: ...' sentence and 1-4 "
    "<tool_call>...</tool_call> blocks. Each tool call is a JSON object with 'name' "
    "and 'arguments'. Only use tools and parameter ranges listed below; never invent "
    "effect names or reference instruments not in the metadata."
)


def build_hermes_system_prompt(track_metadata: str) -> str:
    """Build the full Hermes-format system prompt.

    ``track_metadata`` is a short string describing the input mixture, e.g.
    ``"tracks: piano (program 0), bass (program 33), drums (channel 10); tempo 92 bpm; key C minor"``.
    This is the same metadata needed at inference for the model to produce valid tool calls
    (see proposal §4.6, "Inference requirement").
    """
    tools_json = json.dumps(all_tool_schemas(), separators=(",", ":"))
    return (
        f"{HERMES_SYSTEM_PREAMBLE}\n\n"
        f"<tools>{tools_json}</tools>\n\n"
        f"Mixture metadata: {track_metadata}"
    )


MOTIVATION_ONLY_PREAMBLE = (
    "You are a professional music producer narrating ONE production move "
    "per turn. The mixtures are INSTRUMENTAL multitracks rendered from MIDI; "
    "they never contain vocals, rap, or any voice. This turn does not "
    "require any tool calls — the operation is performed deterministically "
    "by the pipeline. You only need to: first reason inside "
    "<think>...</think> about WHY the producer is making this move (write "
    "specifically about the tracks listed under 'Mixture metadata'); then "
    "outside the think block, write exactly ONE line beginning with "
    "'Production motivation: ' followed by a natural-language sentence in "
    "the role's voice. Do NOT emit any <tool_call> blocks."
)


def build_motivation_only_system_prompt(track_metadata: str) -> str:
    """System prompt for motivation-only intents (e.g. ``extract_track``).

    Drops the ``<tools>`` block and the tool-call instructions; the model
    only needs to reason about the production move and write a motivation
    sentence. The pipeline performs the audio operation deterministically.
    """
    return (
        f"{MOTIVATION_ONLY_PREAMBLE}\n\n"
        f"Mixture metadata: {track_metadata}"
    )
