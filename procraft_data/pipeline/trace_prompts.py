"""Motivation-diverse prompt construction (role × abstraction × hook × intent).

The dataset's value to downstream training depends on the **motivation text
covering the full language space** of anyone who might describe a production
change — not only producers. A model trained on this data must respond to a
film supervisor asking for "music that feels like dusk", a session drummer
saying "my snare is getting buried", a non-musician client saying "it sounds
like they're in a cardboard box", and a mixing engineer's self-note "-4 dB
@ 250 Hz narrow Q". Each describes the *same* kind of underlying move in a
different voice.

A prompt is built from FOUR sampled dimensions:

1. **role** (13 options) — who is speaking.
2. **abstraction_level** (6 options) — how they phrase it.
3. **hook** — a concrete anchor drawn from the level's vocabulary pool.
4. **primary_intent** (10 options) — WHAT the main production move is. Sampled
   uniformly across the 4 proposal categories (instrument / effects /
   performance / arrangement) with sub-actions. For arrangement intents the
   target track is pre-committed so the mixture state is arranged accordingly
   and the LLM's motivation is guaranteed to match the executed action.

All four dimensions are recorded on every DatasetEntry so downstream code can
stratify training batches or audit the distribution.

Hard rules still hold: instrumental-only, mixture-grounded, real tool names.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from procraft_data.pipeline.vocab_pools import (
    EMOTIONAL_HOOKS,
    INSTRUMENT_SPECIFIC_HOOKS,
    NEGATIVE_SPACE_HOOKS,
    REFERENCE_HOOKS,
    SCENE_METAPHORS,
    TECHNICAL_HOOKS,
)
from procraft_data.tools.schemas import build_hermes_system_prompt


# ---------------------------------------------------------------------------
# Hard rules — apply to every entry regardless of role
# ---------------------------------------------------------------------------
_HARD_RULES = """
Hard rules (NEVER violate):
1. INSTRUMENTAL ONLY — no vocals, no singing, no rap, no spoken word, no voice
   processing. If a role that normally talks about vocals (e.g. artist_vocalist)
   is speaking, translate their intent onto the instrumental tracks that are
   actually present — never reference "my voice", "vocal clarity", de-essing,
   etc.
2. MIXTURE-GROUNDED — every track name in a tool_call must appear in the
   Mixture metadata. Never invent instruments that are not listed.
3. REAL TOOL / EFFECT NAMES — use exactly the names declared in <tools>.
   Never invent pseudo-names like "tape_saturation"; pick sox_overdrive or
   am_tanh_distortion from the cookbook below.
4. NATURAL LANGUAGE IN MOTIVATION — the Mixture metadata lists each track as
   `<identifier> "<program name>" (GM N)`. Internal identifiers (e.g.
   `guitar_1`, `strings__continued`, `organ_3`) are MIDI bookkeeping, NOT
   how a real producer/player/client would talk. In the motivation sentence:
     * Use the quoted program name or a natural paraphrase of it:
       `guitar_1 "Distortion Guitar"` → "the distortion guitar", "the
       crunchy guitar", "that driven 6-string"; `strings__continued
       "Choir Aahs"` → "the choir pad", "the vocal strings"; `organ_1
       "Drawbar Organ"` → "the tonewheel organ", "the Hammond".
     * If two stems share the same program (e.g., two `Distortion Guitar`
       tracks), say "the first distortion guitar" / "the second distortion
       guitar" or "the rhythm guitar" / "the lead guitar".
     * NEVER write `guitar_1`, `organ_2`, `strings__continued` literally in
       the motivation sentence.
     * NEVER write GM program numbers literally in the motivation: no
       "GM 85", "GM 12", "program 0", "to_program: 33". Real producers
       say "string warmth" / "Hammond" / "fretless bass" — paraphrase
       into what the new instrument SOUNDS like, not its number.
   In tool_call ``arguments``, always use the exact identifier from the
   metadata (e.g. `"track": "guitar_1"`) and the literal program
   number. Motivation = natural / paraphrased, tool_calls = exact.
"""


# ---------------------------------------------------------------------------
# Effect cookbook — unchanged
# ---------------------------------------------------------------------------
_EFFECT_COOKBOOK = """
Effect cookbook (all 85 MultiAFX effects are available in <tools> — these are
the production-tested defaults you'll reach for most often):

WARMTH / VINTAGE: sox_overdrive (tube sat), am_tanh_distortion (soft clip),
  am_bit_crush (lo-fi grit), am_aliasing, am_air_absorption (distance feel)
EQ / TONE: sox_equalizer (surgical), am_peaking_filter, ta_bass_biquad,
  ta_treble_biquad, am_highpass_filter (kill rumble), am_lowpass_filter
  (tame highs), am_low_shelf_filter, am_high_shelf_filter, sox_bass, sox_treble
DYNAMICS & LEVEL: sox_compand (comp), sox_contrast (presence),
  sox_gain / am_gain / ta_gain (per-track balance — ALWAYS use this for
  loud/quiet issues before reaching for EQ), sox_vol (linear mult),
  am_loudness_normalization / pln_loudness_normalize (LUFS target)
BUSSING: apply_fx supports two kinds of targets — a specific track name
from the Mixture metadata, or the literal "mix" for the full mix bus
(e.g. glue compression, master EQ, overall reverb send). There are NO
per-group sub-buses (no "guitars bus" or "drum bus"); if you want the
same treatment on several stems, emit one apply_fx call per stem.
SPACE: sox_reverb (hall/room), sox_echo / sox_echos (delay)
MOTION: sox_chorus (thicken), sox_flanger (sweep), sox_phaser (rotating),
  sox_tremolo (amplitude wobble)
SPATIAL: npy_stereo_widener, npy_lr_pan, sox_oops (mid-side trick)
DESTRUCTIVE: am_add_gaussian_noise / am_add_color_noise (tape hiss / vinyl),
  am_bit_crush, am_polarity_inversion

EFFECT STRENGTH — pick values by INTENT, not by "always push it further".
Choose light / medium / strong based on what the motivation calls for:
  - EQ cut/boost
      light:  ±2-4 dB    (subtle tilt)
      medium: ±5-9 dB    (clear shaping)
      strong: ±10-15 dB  (aggressive carve / clearing mud)
  - Compression (sox_compand)
      light:  1-3 dB GR with soft_knee 3-6 dB (glue)
      medium: 4-6 dB GR, attack 5-15 ms, decay 50-150 ms (tighten drums)
      strong: 6-10 dB GR, fast attack (punchy, pumping)
  - Reverb (sox_reverb)
      light:  reverberance 20-40, wet_gain -8 to -5 dB (hint of room)
      medium: reverberance 40-65, wet_gain -5 to -2 dB (audible space)
      strong: reverberance 65-85, wet_gain -2 to 0 dB (drenched / dreamy)
  - sox_overdrive gain_db
      light:  5-10 dB    (tube warmth, mix glue)
      medium: 10-18 dB   (clear drive, guitar-amp character)
      strong: 18-30 dB   (crunchy grit — use only when the brief says so)
      Avoid 30+ dB unless the motivation explicitly asks for fuzz/metal.
  - am_tanh_distortion
      light:  0.05-0.12  (barely saturated warmth)
      medium: 0.15-0.25  (soft-clip character)
      strong: 0.25-0.40  (aggressive saturation)
      Avoid 0.5+ — output becomes a square wave.
  - am_bit_crush bit_depth
      light:  12-14 bits (gentle digital haze)
      medium: 9-11 bits  (clear lo-fi character)
      strong: 6-8 bits   (crushed 8-bit aesthetic)
  - Stereo widener width
      light:  1.1-1.25  (gentle spread)
      medium: 1.3-1.5
      strong: 1.6-1.8   (dramatic spread)
      Avoid 1.0 (no-op) and 2.0 (mono-collapse risk on sides).
  - Gain / volume: -8 to +8 dB, pick the magnitude that matches the
    imbalance you described (don't boost +8 for a minor nudge).
  - Humanize max_offset_ms
      light:  5-10 ms   (tight human feel)
      medium: 10-20 ms  (noticeable groove)
      strong: 20-30 ms  (loose, drunken swing)
  - Velocity scale
      light:  0.85-1.15
      medium: 0.7-1.3
      strong: 0.55-1.45
Match the magnitude to the motivation's vocabulary — "subtle" / "gentle" →
light; "punchy" / "driven" / "aggressive" → medium or strong.

BOLD MOVES — a real production decision usually stacks multiple changes:
  - Cross-category instrument swaps: piano → Rhodes (GM 4), piano →
    vibraphone (11), piano → synth pad (88), strings → synth lead (81),
    guitar → organ (17), guitar → synth brass (62). Don't just stay
    within the same family (GM 30 → GM 31).
  - Multi-track arrangement: swap TWO instruments at once; remove TWO of
    three redundant tracks; double a melody AND re-voice the harmony.
  - FX chains: EQ + compression + saturation + reverb on the SAME track,
    each with a distinct job.
Prefer decisive moves. If a role would feel nervous about a strong call,
either own it or pick a different move.
"""


_PRODUCTION_PRINCIPLES = """
Production principles (apply when the role's brief doesn't give explicit
technical guidance):
- BALANCE BEFORE TONE: if a track dominates or hides, start with a gain
  effect before reaching for EQ/compression.
- CARVE SPACE: when two tracks fight in the same frequency range, cut one
  rather than boosting the other.
- COMBINE FREELY: a real production decision mixes instrument swaps, effects,
  performance tweaks, and arrangement moves in one motivation.
"""


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Role:
    name: str
    voice: str
    example_sentences: tuple[str, ...]
    requires_track: str | None = None   # skip this role if the named track not in mix
    # Abstraction levels this role would NEVER naturally use. Lets us kill
    # semantically nonsensical combinations (e.g. non_musician_client producing
    # a technical_parametric motivation). Empty = compatible with all.
    forbidden_abstractions: frozenset[str] = frozenset()

ROLES: list[Role] = [
    Role(
        name="producer",
        voice=(
            "Vision-driven and commercially aware. Anchors decisions with references "
            "(songs, artists, albums, eras), thinks about how the track will sit in a "
            "playlist or radio lineup. Confident, directive, uses producer shorthand."
        ),
        example_sentences=(
            "Give this that Dilla Donuts grit — crooked kick, dusty snares, roll off the air above 10k.",
            "Think Air's Moon Safari production: wide stereo, warm Rhodes, gentle filter sweeps.",
        ),
    ),
    Role(
        name="artist_vocalist",
        voice=(
            "Emotional and subjective. Talks about how the track makes them FEEL, "
            "often in physical or spatial terms (breath, chest, room size). They "
            "are describing an INSTRUMENTAL bed they'll later sing over; they "
            "never reference their own vocals in the tool calls."
        ),
        example_sentences=(
            "This feels too crowded — I need room to breathe, like the band is a few steps back from me.",
            "Right now it's sitting too bright, almost clinical. I want something warmer to sink into.",
        ),
    ),
    Role(
        name="artist_instrumentalist",
        voice=(
            "Cares specifically about how THEIR part reads in the mix. Names "
            "their instrument directly and describes what character they want "
            "it to have (tone, feel, placement)."
        ),
        example_sentences=(
            "My piano is getting swallowed by the guitars — can we carve some space for it around 1k?",
            "The bass should feel round and supportive, not pushing to the front.",
        ),
    ),
    Role(
        name="session_drummer",
        voice=(
            "Focused on how the drums sit and feel. Thinks in terms of kick/snare/"
            "hat balance, attack vs sustain, room sound, front/back placement."
        ),
        example_sentences=(
            "Snare's getting lost — can you bring it up and tighten the attack?",
            "The kit feels too clean; give it some room, a little grit around the edges.",
        ),
        requires_track="drums",
    ),
    Role(
        name="session_bassist",
        voice=(
            "Cares about how the bass sits underneath. Thinks in terms of low-end "
            "weight, definition vs fullness, and how the bass interacts with the kick."
        ),
        example_sentences=(
            "The low end's getting flubby — can we tighten it? Maybe HPF around 40 and push presence at 800.",
            "The bass needs more body in the 80-120 range without stepping on the kick.",
        ),
        requires_track="bass",
    ),
    Role(
        name="session_guitarist",
        voice=(
            "Talks about guitar tone character, pick vs strum, space around the part, "
            "where it sits in the stereo field. References pedals and amps naturally."
        ),
        example_sentences=(
            "My guitars are stepping on the piano — maybe narrow the stereo and roll back some 3k.",
            "Too much clean and clear; I want a bit of breakup, like a tweed running hot.",
        ),
    ),
    Role(
        name="session_keyboardist",
        voice=(
            "Talks about key instruments (piano, Rhodes, organ) and their touch/tone. "
            "Notes whether the parts sit in a sympathetic or clashing register."
        ),
        example_sentences=(
            "The organ's fighting with the piano in the midrange — pull the organ back, maybe a high shelf cut at 4k.",
            "Rhodes wants to be drifting just behind the drums, not lining up with them.",
        ),
    ),
    Role(
        name="non_musician_client",
        voice=(
            "No music vocabulary. Uses everyday metaphors, scene descriptions, "
            "comparisons to real-world environments, vague feeling-words. "
            "Never says Hz, dB, ms, or tool names."
        ),
        example_sentences=(
            "It sounds like they're playing in a cardboard box — can it be bigger?",
            "Make it feel like the music is in the room with you, not in another apartment.",
        ),
        forbidden_abstractions=frozenset({"technical_parametric",
                                          "instrument_specific"}),
    ),
    Role(
        name="mixing_engineer_self_note",
        voice=(
            "Terse, parametric, diagnostic — but written in flowing English, "
            "not as a dump of tool_call arguments. Uses Hz, dB, ms, and ratios "
            "as natural units while still diagnosing what is wrong and why. "
            "Should read like a seasoned engineer thinking aloud in a session "
            "note, not like a CSV of effect parameters."
        ),
        example_sentences=(
            "Bass is loose below 40 — needs a highpass, and the 250 Hz mud is "
            "getting in the way of the kick.",
            "Kick and bass masking each other; shelf the mix a touch down at "
            "3.5k and tighten the bass body.",
        ),
    ),
    Role(
        name="mastering_engineer",
        voice=(
            "Delivery-focused. LUFS targets, true-peak ceilings, platform translation. "
            "Worries about how the mix holds up after loudness normalization."
        ),
        example_sentences=(
            "Need -14 LUFS integrated, -1 dBTP ceiling, PLR above 10 — low end's eating headroom.",
            "The mix translates dark to Spotify; need 1-2 dB of 3-5k to survive the platform EQ curve.",
        ),
    ),
    Role(
        name="label_AnR",
        voice=(
            "Market and platform oriented. References charts, playlist placement, "
            "streaming competition. Less precise but pressure-forward."
        ),
        example_sentences=(
            "This needs to cut through on a phone speaker — loudness and punch, nothing shy.",
            "Lane is chill-lofi playlist territory. Tame it down, push the warm tape sound harder.",
        ),
    ),
    Role(
        name="band_consensus",
        voice=(
            "Multiple conflicting voices in a single request — quote members "
            "disagreeing, then land on a compromise. Keep it one coherent paragraph."
        ),
        example_sentences=(
            "Guitarist wants it wetter, drummer wants it drier. Compromise — short plate on the snare only, keep the rest dry.",
            "Bassist says too much low end, keys says not enough. Cut 60-80 on the bass, boost shelf at 120 for fullness.",
        ),
    ),
    Role(
        name="film_tv_music_supervisor",
        voice=(
            "Narrative-driven. Describes the onscreen moment the music underscores. "
            "Thinks about how the mix supports dialogue, scene pacing, emotional beat."
        ),
        example_sentences=(
            "This cue plays under a car driving away at dusk — it should recede, feel smaller by the end.",
            "Scene's a restaurant kitchen at rush — music needs to punch through but not crowd the diegetic sound.",
        ),
    ),
]


# ---------------------------------------------------------------------------
# Abstraction levels + hook pools
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AbstractionLevel:
    name: str
    description: str
    hook_pool: tuple[str, ...]

ABSTRACTION_LEVELS: dict[str, AbstractionLevel] = {
    "reference_based": AbstractionLevel(
        name="reference_based",
        description=(
            "Name a specific song, artist, album, era, or recording style as the "
            "target. The motivation should reference the hook by name or clearly "
            "evoke it."
        ),
        hook_pool=REFERENCE_HOOKS,
    ),
    "emotional_perceptual": AbstractionLevel(
        name="emotional_perceptual",
        description=(
            "Use subjective, feeling-forward, perceptual words — warm, intimate, "
            "aggressive, dreamy, nostalgic, claustrophobic, etc. Physical or "
            "emotional adjectives, not engineering language."
        ),
        hook_pool=EMOTIONAL_HOOKS,
    ),
    "scene_metaphor": AbstractionLevel(
        name="scene_metaphor",
        description=(
            "Use everyday scenes, objects, or real-world environments as similes. "
            "'Like <scene>' / 'as if <situation>'. No tool or frequency vocabulary."
        ),
        hook_pool=SCENE_METAPHORS,
    ),
    "instrument_specific": AbstractionLevel(
        name="instrument_specific",
        description=(
            "Call out specific named tracks from the Mixture metadata and describe "
            "what should happen to each individually — 'the drums need X, the bass "
            "needs Y, the piano is fine'. Use the exact track names from metadata."
        ),
        hook_pool=INSTRUMENT_SPECIFIC_HOOKS,
    ),
    "technical_parametric": AbstractionLevel(
        name="technical_parametric",
        description=(
            "Precise engineering numbers — Hz, dB, ms, ratios, LUFS. Terse, as if "
            "writing a session note. Parameters should translate directly into "
            "tool_call arguments."
        ),
        hook_pool=TECHNICAL_HOOKS,
    ),
    "negative_space": AbstractionLevel(
        name="negative_space",
        description=(
            "Frame the motivation partly in terms of what you DON'T want — 'not "
            "too clean', 'avoid boom', 'no vintage simulation'. The motivation "
            "must still imply a concrete positive move."
        ),
        hook_pool=NEGATIVE_SPACE_HOOKS,
    ),
}


# ---------------------------------------------------------------------------
# Primary intents — 10 sub-actions spanning the 4 proposal categories.
# Sampled uniformly so every dataset entry is tagged with a primary_intent.
# ---------------------------------------------------------------------------
PRIMARY_INTENTS = [
    "instrument_swap",            # category A — multi-target
    "instrument_layer",           # category A
    "effects",                    # category B (free choice of N apply_fx)
    "arrangement_add",            # category D — multi-target
    "arrangement_remove",         # category D — multi-target
    "arrangement_mute_replace",   # category D
    "extract_track",              # category E — solo a single stem (deterministic, motivation-only)
    "remix",                      # category E — kept-set drop/swap + add-back from removed-set
]
# Intents intentionally excluded from PRIMARY_INTENTS (audio changes too
# subtle to anchor a primary motivation), but their tools remain
# registered so the LLM can still call them from secondary slots of any
# other intent: ``arrangement_double`` (double_track), ``performance_timing``
# (humanize_timing), ``performance_articulation`` (change_articulation).

# Legacy — kept only for test back-compat.
MOTIVATION_TYPES = [
    "instrument_change", "effects_change", "performance_change",
    "arrangement_add", "arrangement_remove",
]


@dataclass(frozen=True)
class IntentCommitment:
    """Pre-committed primary-move for a prompt.

    The ``target_track`` is pre-selected by ``sample_primary_intent`` for
    intents where the concrete track must match the mixture state (e.g.
    ``arrangement_add`` needs a withheld track; ``arrangement_remove`` prefers
    a redundant instrument). For other intents the target is free for the LLM
    to pick from the mixture.

    ``forced_calls`` carries pre-committed tool_calls (used by ``remix``) that
    the LLM is required to emit verbatim alongside any free secondary calls.
    Empty for intents that pre-commit only one move via ``primary_tool`` /
    ``target_track``.

    ``plan`` carries non-prompt metadata for downstream stages — e.g. for
    ``remix``, the kept-set / removed-set partition + the per-track
    drop-or-swap decisions, used by the validator. Empty otherwise.
    """
    intent: str
    primary_tool: str               # canonical tool, or sentinel for motivation-only intents
    target_track: str | None        # pre-committed when applicable
    target_program: int | None      # for instrument_swap / mute_replace
    description: str                # human-readable one-liner for the prompt
    forced_calls: tuple = ()        # tuple[dict, ...] — frozen-friendly default
    plan: tuple = ()                # tuple of (key, value) pairs — frozen-friendly dict


# Two intents are not backed by a registered executor tool: ``extract_track``
# is a deterministic stem solo (no LLM tool_calls) and ``remix`` is a
# multi-tool composite where the primary "move" is the union of forced_calls.
# Sentinel names below carry intent identity through PromptSpec without
# pretending a real tool exists.
_INTENT_TOOL: dict[str, str] = {
    "instrument_swap":            "change_instrument",
    "instrument_layer":           "layer_instrument",
    "effects":                    "apply_fx",
    "arrangement_add":            "add_track",
    "arrangement_remove":         "remove_track",
    "arrangement_mute_replace":   "mute_and_replace",
    "extract_track":              "extract_track_op",
    "remix":                      "remix_op",
}


# Intents that don't go through the LLM tool-call path. For these,
# generate_one builds the audio + MIDI deterministically; the LLM only
# generates the motivation sentence and gets a tool-free system prompt.
MOTIVATION_ONLY_INTENTS: frozenset[str] = frozenset({"extract_track"})


# ---------------------------------------------------------------------------
# PromptSpec
# ---------------------------------------------------------------------------
@dataclass
class PromptSpec:
    system: str
    user: str
    role: str
    abstraction_level: str
    hook: str
    primary_intent: str              # one of PRIMARY_INTENTS
    primary_tool: str                # canonical tool name (or sentinel for motivation-only)
    target_track: str | None         # pre-committed target (may be None)
    target_program: int | None       # pre-committed new GM program (may be None)
    category_focus: str              # "free" (back-compat for legacy tests)
    temperature: float
    top_p: float = 0.95
    max_tokens: int = 6144
    tool_count_hint: tuple[int, int] = (10, 15)
    withhold_for_add: list[str] = field(default_factory=list)   # filled for arrangement_add
    forced_calls: list[dict] = field(default_factory=list)      # filled for remix
    plan: dict = field(default_factory=dict)                    # remix bookkeeping
    motivation_only: bool = False    # True for extract_track — no <tools> block
    chosen_count: int = 0            # the EXACTLY-N value embedded in the user prompt

    def as_messages(self) -> list[dict]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


# ---------------------------------------------------------------------------
# Primary-intent sampling
# ---------------------------------------------------------------------------
def _parse_present_tracks(metadata: str) -> list[tuple[str, str]]:
    """Return ordered list of ``(track_name, program_tag)`` parsed from metadata."""
    import re
    m = re.search(r"tracks:\s*(.*?)(?:;\s*(?:available_to_add|source):|$)",
                  metadata, re.DOTALL)
    if not m:
        return []
    out: list[tuple[str, str]] = []
    for chunk in m.group(1).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        # New format: `<name> "<program name>" (GM N)` or `<name> "Drum Kit"`.
        # Old format: `<name> (GM program N)`. Handle both.
        name_part, _, rest = chunk.partition(" ")
        out.append((name_part.strip(), rest.strip()))
    return out


def _natural_names_from_metadata(metadata: str) -> dict[str, str]:
    """Parse the metadata string directly to extract track→natural-name mapping.

    The metadata format emitted by ``describe_mixture`` is::

        tracks: guitar_1 "Distortion Guitar" (GM 30), drums "Drum Kit", ...;
            available_to_add: strings__continued "Choir Aahs" (GM 52, available-to-add);
            source: slakh/Track00001

    We extract the identifier and the quoted program name, disambiguating
    duplicates with "first" / "second" ordinals.
    """
    import re
    out: dict[str, str] = {}
    prog_for_name: dict[str, str] = {}
    seen_count: dict[str, int] = {}

    regions = re.findall(r"(?:tracks:|available_to_add:)(.*?)(?=;\s*\S+:|$)",
                         metadata, re.DOTALL)
    for region in regions:
        for chunk in region.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            m = re.match(r'(\S+)\s+"([^"]+)"', chunk)
            if not m:
                continue
            ident, prog = m.group(1), m.group(2)
            prog_for_name[ident] = prog
            seen_count[prog] = seen_count.get(prog, 0) + 1

    ordinals = ["first", "second", "third", "fourth", "fifth"]
    ord_idx: dict[str, int] = {}

    def _clean_prog(prog: str) -> str:
        """Strip GM parenthetical suffixes: 'Electric Bass (finger)' → 'electric bass'."""
        base = re.sub(r"\s*\([^)]*\)\s*$", "", prog).strip().lower()
        return base or prog.lower()

    for ident, prog in prog_for_name.items():
        if prog == "Drum Kit":
            out[ident] = "the drums"
            continue
        prog_l = _clean_prog(prog)
        if seen_count[prog] > 1:
            i = ord_idx.get(prog, 0)
            ord_idx[prog] = i + 1
            ord_w = ordinals[i] if i < len(ordinals) else f"{i + 1}th"
            out[ident] = f"the {ord_w} {prog_l}"
        else:
            out[ident] = f"the {prog_l}"
    return out


def _parse_inst_class_from_name(name: str) -> str:
    """Strip a trailing ``_N`` disambiguator to recover the inst_class slug."""
    import re
    return re.sub(r"_\d+$", "", name)


def _strip_program_quote(name: str) -> str:
    """Strip the trailing `"program"` quoted program name from parsed track names.

    The new metadata format includes the program name in quotes after the
    identifier — when ``sample_primary_intent`` records a ``target_track``,
    we want only the identifier portion. Covers both the new format and a
    fallback where the whole thing is passed as a single string.
    """
    import re
    return re.sub(r'\s+".*$', "", name).strip()


def _pick_redundant_target(metadata: str, rng: random.Random) -> str | None:
    """Pick a track whose inst_class is shared with another track (safest to drop)."""
    tracks = _parse_present_tracks(metadata)
    if not tracks:
        return None
    cls_counts: Counter = Counter(_parse_inst_class_from_name(n) for n, _ in tracks)
    redundant = [n for n, _ in tracks
                 if cls_counts[_parse_inst_class_from_name(n)] > 1]
    if redundant:
        return rng.choice(redundant)
    # Fallback: any non-drum track.
    non_drum = [n for n, tag in tracks if "drum" not in tag.lower()]
    if non_drum:
        return rng.choice(non_drum)
    return rng.choice([n for n, _ in tracks])


def _pick_any_track(metadata: str, rng: random.Random,
                    exclude_drums: bool = False) -> str | None:
    tracks = _parse_present_tracks(metadata)
    candidates = [n for n, tag in tracks
                  if not (exclude_drums and "drum" in tag.lower())]
    if not candidates:
        return None
    return rng.choice(candidates)


def _random_gm_program(rng: random.Random) -> int:
    """Pick a plausible melodic GM program (avoid sound-effect range)."""
    # Skip the high SFX range (96-127) for realism.
    return rng.randint(0, 95)


def _n_arrangement_targets(
    records: list[dict], rng: random.Random,
    exclude_drums: bool = False,
) -> int:
    """How many tracks to pre-commit for a multi-target arrangement intent.

    Returns ``0`` when the eligible pool is too small (the caller then
    falls back to ``effects``). Otherwise returns
    ``min(rng.randint(2, 4), eligible // 2)``.

    Eligibility: drums excluded for intents that don't make sense on a
    drum kit (currently ``instrument_swap`` because ``change_instrument``
    keeps ``is_drum=True``). Total eligible must be ≥ 4 for a multi-
    target intent — at smaller pools the user prefers to fall back to
    a different intent rather than reduce the requested target count.
    """
    pool = [r for r in records if not (exclude_drums and r["is_drum"])]
    if len(pool) < 4:
        return 0
    return min(rng.randint(2, 4), len(pool) // 2)


def _parse_track_records(metadata: str) -> list[dict]:
    """Parse ``tracks: ...`` into ``[{name, program, is_drum}]`` records.

    For drum tracks the metadata writes ``"Drum Kit"`` with no GM number;
    we report ``program=128`` (the Slakh sentinel) and ``is_drum=True``.
    """
    import re
    out: list[dict] = []
    for name, tag in _parse_present_tracks(metadata):
        if "drum" in tag.lower():
            out.append({"name": name, "program": 128, "is_drum": True})
            continue
        m = re.search(r"\bGM\s+(\d+)", tag)
        prog = int(m.group(1)) if m else 0
        out.append({"name": name, "program": prog, "is_drum": False})
    return out


def _plan_remix(track_metadata: str, rng: random.Random) -> dict | None:
    """Plan a remix entry — partition + per-track decisions + add-backs.

    Returns ``None`` when the mixture is too small (N < 4) so the caller
    can fall back to a different intent. Otherwise the dict has::

        {
          "removed_set":  list[str],       # track names withheld → pending_tracks
          "kept_set":     list[str],       # the "input mix" the LLM sees
          "kept_programs": set[int],       # original programs of kept_set (forbidden zone)
          "decisions":    list[(name, "drop"|"swap", new_prog_or_None)],
          "add_backs":    list[(name, prog)],   # name from removed_set, prog ∉ kept_programs
        }

    Per the locked spec:
      * K = ⌊N/2⌋, kept = N - K (≥ 2 because N ≥ 4).
      * For each kept-set track: 50/50 drop or swap, with a hard guarantee
        that the entry has at least one drop AND at least one swap.
      * Swap target program ∉ kept_programs (collision with removed_set is OK).
      * ⌈K/2⌉ removed-set tracks are added back; their program is the
        track's original unless that program collides with kept_programs,
        in which case we pick a fresh non-colliding program (per C-γ).
    """
    records = _parse_track_records(track_metadata)
    n = len(records)
    if n < 4:
        return None

    k = n // 2
    shuffled = records[:]
    rng.shuffle(shuffled)
    removed = shuffled[:k]
    kept = shuffled[k:]
    kept_programs: set[int] = {r["program"] for r in kept}

    # Drop-or-swap decisions per kept-set track. Drum tracks always drop —
    # ``change_instrument`` on a drum stem keeps ``is_drum=True`` so a swap
    # to a melodic GM program would still render through channel 10 and
    # produce nonsense.
    decisions: list[tuple[str, str, int | None]] = []
    for r in kept:
        if r["is_drum"]:
            action = "drop"
        else:
            action = "drop" if rng.random() < 0.5 else "swap"
        decisions.append((r["name"], action, None))

    # Enforce ≥1 drop AND ≥1 swap. If unanimous, flip the first non-drum
    # record (drums must stay drop). If every kept-set track is a drum kit,
    # ≥1 swap can't be enforced — leave the all-drop plan as-is.
    actions = {a for _, a, _ in decisions}
    kept_records = {r["name"]: r for r in kept}
    if "drop" not in actions and decisions:
        n0, _, _ = decisions[0]
        decisions[0] = (n0, "drop", None)
    elif "swap" not in actions and decisions:
        for i, (n, a, p) in enumerate(decisions):
            if not kept_records[n]["is_drum"]:
                decisions[i] = (n, "swap", None)
                break

    def _pick_swap_prog(orig: int) -> int:
        # Avoid kept_programs (the no-survival constraint) AND avoid
        # ``orig`` itself (a no-op swap).
        forbidden = kept_programs | {orig}
        for _ in range(64):
            cand = rng.randint(0, 95)
            if cand not in forbidden:
                return cand
        # Pathological: every program in [0, 95] forbidden — fall back to
        # the first non-forbidden index.
        for cand in range(96):
            if cand not in forbidden:
                return cand
        return (orig + 1) % 128

    # Resolve swap programs.
    name_to_prog = {r["name"]: r["program"] for r in records}
    decisions = [
        (name, action,
         _pick_swap_prog(name_to_prog[name]) if action == "swap" else None)
        for name, action, _ in decisions
    ]

    # Add-back picks: ⌈K/2⌉ from removed_set.
    #
    # Drum add-backs need extra care: the Slakh sentinel program for drums
    # is 128, which is out of the ``add_track`` schema's valid range
    # (0-127). The FluidSynth renderer keys off ``track.is_drum`` rather
    # than ``track.program`` for drum stems (it forces bank=128, program=0
    # internally), so we can safely report ``program=0`` to the LLM and
    # have the audio still render as drums.
    n_add = (k + 1) // 2
    add_pool = removed[:]
    rng.shuffle(add_pool)
    add_backs: list[tuple[str, int]] = []
    for r in add_pool[:n_add]:
        if r["is_drum"]:
            # Drum routing is controlled by ``is_drum``; report the
            # schema-valid placeholder.
            add_backs.append((r["name"], 0))
            continue
        prog = r["program"]
        if prog in kept_programs:
            # Pick fresh program ∉ kept_programs.
            for _ in range(64):
                cand = rng.randint(0, 95)
                if cand not in kept_programs:
                    prog = cand
                    break
        add_backs.append((r["name"], prog))

    return {
        "removed_set": [r["name"] for r in removed],
        "kept_set": [r["name"] for r in kept],
        "kept_programs": kept_programs,
        "decisions": decisions,
        "add_backs": add_backs,
    }


def _remix_forced_calls(plan: dict) -> list[dict]:
    """Render a remix plan into the concrete tool_calls the LLM must echo."""
    calls: list[dict] = []
    for name, action, new_prog in plan["decisions"]:
        if action == "drop":
            calls.append({"name": "remove_track",
                          "arguments": {"track": name}})
        else:
            calls.append({"name": "change_instrument",
                          "arguments": {"track": name, "to_program": int(new_prog)}})
    for name, prog in plan["add_backs"]:
        calls.append({"name": "add_track",
                      "arguments": {"track_name": name, "program": int(prog)}})
    return calls


def _remix_description(forced: list[dict]) -> str:
    """Build the human-readable PRIMARY MOVE description from forced_calls."""
    drop_lines, swap_lines, add_lines = [], [], []
    for c in forced:
        a = c["arguments"]
        if c["name"] == "remove_track":
            drop_lines.append(f"  - remove_track {{track: '{a['track']}'}}")
        elif c["name"] == "change_instrument":
            swap_lines.append(
                f"  - change_instrument {{track: '{a['track']}', "
                f"to_program: {a['to_program']}}}"
            )
        elif c["name"] == "add_track":
            add_lines.append(
                f"  - add_track {{track_name: '{a['track_name']}', "
                f"program: {a['program']}}}"
            )
    forced_block = "\n".join(drop_lines + swap_lines + add_lines)
    return (
        "The main move is to REMIX the mixture: every track currently "
        "in the input mix is either dropped or re-voiced (so none of "
        "them survive as-is), and a few withheld tracks are introduced "
        "from the 'available_to_add' pool. You MUST emit each of the "
        "tool_calls below verbatim (in any order, alongside any "
        "additional apply_fx calls you choose to make the variant "
        "cohere). Required calls:\n"
        f"{forced_block}\n"
        "Your motivation should describe the remix concept in the "
        "role's voice — what the variant emphasises, the new texture, "
        "the new energy."
    )


def _frozen_plan(plan: dict) -> tuple:
    """Convert a remix plan dict to the immutable representation stored on
    ``IntentCommitment.plan``."""
    return (
        ("removed_set", tuple(plan["removed_set"])),
        ("kept_set", tuple(plan["kept_set"])),
        ("kept_programs", frozenset(plan["kept_programs"])),
        ("decisions", tuple(plan["decisions"])),
        ("add_backs", tuple(plan["add_backs"])),
    )


def thaw_plan(plan) -> dict:
    """Restore a remix plan dict from its frozen tuple-of-pairs form, or
    pass through if already a dict. Used by ``generate_one`` when
    rebuilding the spec from a previously-stashed plan."""
    if isinstance(plan, dict):
        return dict(plan)
    return {k: v for k, v in plan}


def make_remix_intent(plan) -> IntentCommitment:
    """Materialize a fully-populated ``IntentCommitment`` from a remix plan.

    Accepts either a fresh plan dict (from ``_plan_remix``) or the frozen
    tuple-of-pairs that ``IntentCommitment.plan`` carries between stages.
    Used by ``sample_primary_intent`` (first call) and by ``generate_one``
    (when rebuilding the prompt against post-withhold metadata, where
    re-sampling would discard the original plan).
    """
    plan_dict = thaw_plan(plan)
    forced = _remix_forced_calls(plan_dict)
    return IntentCommitment(
        intent="remix",
        primary_tool=_INTENT_TOOL["remix"],
        target_track=None, target_program=None,
        description=_remix_description(forced),
        forced_calls=tuple(forced),
        plan=_frozen_plan(plan_dict),
    )


def sample_primary_intent(
    track_metadata: str,
    *,
    forced_intent: str | None = None,
    seed: int | None = None,
) -> IntentCommitment:
    """Pick a primary intent uniformly, then pre-commit a target where needed.

    For ``arrangement_add``, the caller is responsible for actually
    withholding ``target_track`` from the ``MixtureState`` before generation
    (the track must be marked in ``available_to_add``). For the other
    arrangement/performance intents, the target is simply named in the prompt
    so the LLM's motivation references it concretely.

    If ``forced_intent`` is supplied and mixture makes it unusable (e.g.,
    requesting arrangement_remove on a single-track mix), falls back to
    ``effects`` as the universal default.
    """
    rng = random.Random(seed)
    tracks = _parse_present_tracks(track_metadata)

    if forced_intent is not None:
        intent = forced_intent
    else:
        intent = rng.choice(PRIMARY_INTENTS)

    tool = _INTENT_TOOL[intent]
    target_track: str | None = None
    target_program: int | None = None
    description = ""

    if intent in ("effects",):
        description = (
            "The main move this turn is one or more apply_fx calls. Pick the "
            "track(s) yourself based on what is weak in the mixture. Reach "
            "for audible parameter values (see 'EFFECT STRENGTH' in the "
            "cookbook)."
        )

    elif intent == "instrument_swap":
        records = _parse_track_records(track_metadata)
        n = _n_arrangement_targets(records, rng, exclude_drums=True)
        if n == 0:
            return IntentCommitment(
                intent="effects", primary_tool="apply_fx",
                target_track=None, target_program=None,
                description=_INTENT_TOOL["effects"],
            )
        candidates = [r for r in records if not r["is_drum"]]
        rng.shuffle(candidates)
        targets = candidates[:n]
        forced: list[dict] = []
        target_names: list[str] = []
        for r in targets:
            new_prog = _random_gm_program(rng)
            for _ in range(32):
                if new_prog != r["program"]:
                    break
                new_prog = _random_gm_program(rng)
            forced.append({"name": "change_instrument",
                           "arguments": {"track": r["name"],
                                         "to_program": int(new_prog)}})
            target_names.append(f"'{r['name']}' → GM {new_prog}")
        forced_block = "\n".join(
            f"  - change_instrument {{track: '{c['arguments']['track']}', "
            f"to_program: {c['arguments']['to_program']}}}"
            for c in forced
        )
        description = (
            f"The main move this turn is to swap {n} instruments — each "
            f"track listed below is re-voiced to a different GM program "
            f"with change_instrument. Emit each call below verbatim:\n"
            f"{forced_block}\n"
            f"Your motivation should describe the re-voicing across these "
            f"tracks: {', '.join(target_names)}."
        )
        return IntentCommitment(
            intent=intent, primary_tool=tool,
            target_track=", ".join(r["name"] for r in targets),
            target_program=None,
            description=description,
            forced_calls=tuple(forced),
        )

    elif intent == "instrument_layer":
        target_track = _pick_any_track(track_metadata, rng, exclude_drums=True)
        target_program = _random_gm_program(rng)
        description = (
            f"The main move this turn is to layer a second instrument on top "
            f"of '{target_track}' using layer_instrument with additional_program "
            f"{target_program} at mix_ratio 0.4-0.7. Your motivation should "
            f"describe the layered texture."
        )

    elif intent == "arrangement_add":
        # Multi-target: pick how many tracks to withhold here (using
        # metadata only). The concrete track names + forced_calls are
        # built by ``generate_one`` after the note-rich selector picks
        # the actual stems (it has MIDI access; we only have metadata).
        records = _parse_track_records(track_metadata)
        n = _n_arrangement_targets(records, rng)
        if n == 0:
            return IntentCommitment(
                intent="effects", primary_tool="apply_fx",
                target_track=None, target_program=None,
                description=_INTENT_TOOL["effects"],
            )
        description = (
            f"The main move is to INTRODUCE {n} NEW elements to the "
            "mixture — each track listed under 'available_to_add' is "
            "something the producer is adding FOR THE FIRST TIME this "
            "session. Use the add_track tool, once per track listed. "
            "Your motivation MUST use additive verbs like 'introduce', "
            "'bring in', 'pad in', 'layer in', 'add a', 'drop in' — and "
            "must NEVER use restoration words: no 'missing', 'lacking', "
            "'absent', 'empty spot', 'bring back', 'restore', 'return', "
            "'reintroduce', 'put back', 'needs the X back'. Pretend the "
            "tracks never existed before — these are new moves. Describe "
            "the COMBINED CONTRIBUTION (pad, lift, grit, warmth, "
            "counter-melody, low-end weight)."
        )
        # Stash N so generate_one knows how many tracks to withhold.
        return IntentCommitment(
            intent=intent, primary_tool=tool,
            target_track=None, target_program=None,
            description=description,
            plan=(("n_targets", n),),
        )

    elif intent == "arrangement_remove":
        records = _parse_track_records(track_metadata)
        n = _n_arrangement_targets(records, rng)
        if n == 0:
            return IntentCommitment(
                intent="effects", primary_tool="apply_fx",
                target_track=None, target_program=None,
                description=_INTENT_TOOL["effects"],
            )
        # Bias toward redundant tracks (same inst_class as another) the way
        # the single-target picker did, but extend to N tracks.
        cls_counts: Counter = Counter(_parse_inst_class_from_name(r["name"]) for r in records)
        redundant = [r for r in records
                     if cls_counts[_parse_inst_class_from_name(r["name"])] > 1]
        non_drum = [r for r in records if not r["is_drum"]]
        pool = redundant if len(redundant) >= n else non_drum if len(non_drum) >= n else records
        rng.shuffle(pool)
        targets = pool[:n]
        forced = [{"name": "remove_track",
                   "arguments": {"track": r["name"]}} for r in targets]
        forced_block = "\n".join(
            f"  - remove_track {{track: '{c['arguments']['track']}'}}"
            for c in forced
        )
        names = [r["name"] for r in targets]
        description = (
            f"The main move is to remove {n} tracks from the mixture so the "
            f"remaining instrumentation breathes. Emit each call below "
            f"verbatim:\n"
            f"{forced_block}\n"
            f"Your motivation should explain why dropping these tracks "
            f"({', '.join(names)}) clears up the mix."
        )
        return IntentCommitment(
            intent=intent, primary_tool=tool,
            target_track=", ".join(names),
            target_program=None,
            description=description,
            forced_calls=tuple(forced),
        )

    elif intent == "arrangement_mute_replace":
        target_track = _pick_any_track(track_metadata, rng, exclude_drums=True)
        target_program = _random_gm_program(rng)
        description = (
            f"The main move is to replace '{target_track}' with a different "
            f"instrument (GM program {target_program}) using mute_and_replace. "
            f"Your motivation should describe the re-voicing of '{target_track}'."
        )

    elif intent == "extract_track":
        # Pick any track (drums OK). The pipeline performs the extraction
        # deterministically; the LLM gets a tool-free prompt and emits
        # motivation only.
        target_track = _pick_any_track(track_metadata, rng)
        description = (
            f"The producer is extracting '{target_track}' from the full mix "
            f"and presenting it in isolation. Your motivation should describe "
            f"WHY soloing '{target_track}' is useful — what the listener gains "
            f"by hearing this stem alone (its line, groove, timbre, role in "
            f"the arrangement). Do NOT propose any production changes; this "
            f"turn is purely about isolating the existing track."
        )

    elif intent == "remix":
        plan = _plan_remix(track_metadata, rng)
        if plan is None:
            # Mixture too small for a meaningful remix — fall back so the
            # entry is still useful.
            return IntentCommitment(
                intent="effects", primary_tool="apply_fx",
                target_track=None, target_program=None,
                description=_INTENT_TOOL["effects"],
            )
        return make_remix_intent(plan)

    if target_track is None and tracks and intent != "effects" and intent != "arrangement_add":
        # Fallback — couldn't find a target; degrade to effects.
        return IntentCommitment(
            intent="effects", primary_tool="apply_fx",
            target_track=None, target_program=None,
            description=_INTENT_TOOL["effects"],  # harmless placeholder
        )

    return IntentCommitment(
        intent=intent, primary_tool=tool,
        target_track=target_track, target_program=target_program,
        description=description,
    )


# ---------------------------------------------------------------------------
# Role / mixture compatibility
# ---------------------------------------------------------------------------
def _present_track_names(metadata: str) -> set[str]:
    """Best-effort parse of the Mixture metadata string for active track names."""
    import re
    names: set[str] = set()
    m = re.search(r"tracks:\s*(.*?)(?:;\s*(?:available_to_add|source):|$)",
                  metadata, re.DOTALL)
    if not m:
        return names
    for chunk in m.group(1).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        # Each chunk is "name (GM program N)" or "name (drums)"
        name = chunk.split("(", 1)[0].strip()
        if name:
            names.add(name)
    return names


# ---------------------------------------------------------------------------
# Method 6: content-aware hook filtering — role-specific vocabulary.
# Keywords that strongly signal a hook is technical, parametric, or
# engineer-coded. Roles that "don't speak Hz/dB" get these filtered out of
# their hook pool even within the broad abstraction levels.
# ---------------------------------------------------------------------------
_TECHNICAL_HOOK_KEYWORDS = re.compile(
    r"\b(?:\d+(?:\.\d+)?\s*(?:Hz|kHz|dB|dBFS|dBTP|LUFS|ms|s)|"
    r"HPF|LPF|Q\d|Q\s*\d|compression|sidechain|ratio|"
    r"reverberance|pre-delay|decay|attack|release|transient|RMS|"
    r"bit.crush|notch|shelf|EQ|tilt\s+EQ|M/S|mid/side|dBTP|peak|limiter|"
    r"true-peak|crest\s+factor)\b",
    re.IGNORECASE,
)


def _hook_is_technical(hook: str) -> bool:
    return bool(_TECHNICAL_HOOK_KEYWORDS.search(hook))


def _role_refuses_technical_hook(role_name: str) -> bool:
    """Non-musician / emotionally-framed roles should never get a tech hook."""
    return role_name in {"non_musician_client", "artist_vocalist"}


def _role_prefers_technical_hook(role_name: str) -> bool:
    """Engineer-style roles should lean INTO technical hooks when possible."""
    return role_name in {"mixing_engineer_self_note", "mastering_engineer"}


def _filter_hook_pool_for_role(pool: tuple[str, ...], role_name: str) -> tuple[str, ...]:
    """Method 6 — slice the hook pool by role compatibility.

    - Roles that wouldn't use engineering vocab (non_musician_client,
      artist_vocalist): drop technical hooks entirely.
    - Engineer-style roles (mixing/mastering): leave the pool as-is.
    - Other roles: no filtering (the base abstraction level already picks a
      semantically-appropriate pool).
    Returns the original pool if filtering would leave nothing.
    """
    if _role_refuses_technical_hook(role_name):
        filtered = tuple(h for h in pool if not _hook_is_technical(h))
        return filtered or pool
    return pool


# ---------------------------------------------------------------------------
# Method 3: detect + resample conflicting hook after initial sample.
# ---------------------------------------------------------------------------
def _hook_conflicts_with_role(hook: str, role_name: str) -> bool:
    """Return True if the sampled hook is semantically wrong for this role."""
    if _role_refuses_technical_hook(role_name) and _hook_is_technical(hook):
        return True
    # More rules can go here as we spot them.
    return False


def _compatible_roles(metadata: str) -> list[Role]:
    present = _present_track_names(metadata)

    def ok(role: Role) -> bool:
        if role.requires_track is None:
            return True
        # Substring match: "bass" matches "bass" but also "guitar" matches
        # "guitar_1"/"guitar_2"/"guitar_3".
        return any(role.requires_track in n for n in present)

    return [r for r in ROLES if ok(r)]


def _roles_for_target(metadata: str, target_track: str | None) -> list[Role]:
    """Subset of compatible roles that are semantically OK with this target.

    If a role has ``requires_track`` set (e.g. ``session_drummer`` wants
    drums) and the pre-committed target is a *different* instrument, that
    role is excluded — a session drummer should not be the voice narrating a
    bass humanize or a strings swap.
    """
    base = _compatible_roles(metadata)
    if target_track is None:
        return base
    keep: list[Role] = []
    for r in base:
        if r.requires_track is None or r.requires_track in target_track:
            keep.append(r)
    return keep or base   # never return empty — fall back to full compat set


# ---------------------------------------------------------------------------
# Build a single spec
# ---------------------------------------------------------------------------
def build_spec(
    track_metadata: str,
    *,
    role: str | None = None,
    abstraction_level: str | None = None,
    hook: str | None = None,
    intent: IntentCommitment | None = None,
    tool_count_range: tuple[int, int] = (10, 15),
    temperature: float = 0.9,
    seed: int | None = None,
    # Back-compat shims kept so older callers don't break.
    category_focus: str | None = None,
    style: str | None = None,
    tool_count: int | None = None,
    mood: str | None = None,
    era_genre: str | None = None,
    artist_vibe: str | None = None,
    technical_goal: str | None = None,
) -> PromptSpec:
    """Sample role × abstraction × hook × intent and build the Qwen3 prompt."""
    rng = random.Random(seed)

    # 4a) Primary intent + target first — role/abstraction filter against it.
    if intent is None:
        intent = sample_primary_intent(track_metadata,
                                       seed=rng.randint(0, 2**31 - 1))

    # 1) Pick a role compatible with both the mixture AND the committed target
    # (so session_drummer doesn't end up narrating a bass humanize).
    role_objs = _roles_for_target(track_metadata, intent.target_track)
    role_obj = next((r for r in role_objs if r.name == role), None)
    if role_obj is None:
        role_obj = rng.choice(role_objs)

    # 2) Pick an abstraction level — skip ones forbidden for this role.
    if abstraction_level in ABSTRACTION_LEVELS:
        level = ABSTRACTION_LEVELS[abstraction_level]
        if level.name in role_obj.forbidden_abstractions:
            # Caller asked for an incompatible combo; pick a compatible one.
            compat = [l for l in ABSTRACTION_LEVELS.values()
                      if l.name not in role_obj.forbidden_abstractions]
            level = rng.choice(compat or list(ABSTRACTION_LEVELS.values()))
    else:
        compat = [l for l in ABSTRACTION_LEVELS.values()
                  if l.name not in role_obj.forbidden_abstractions]
        level = rng.choice(compat or list(ABSTRACTION_LEVELS.values()))

    # 3) Pick a concrete hook from the level's pool. Method 6:
    # content-aware filtering — some roles can't naturally speak engineering
    # vocabulary even if the abstraction level allows technical hooks.
    if hook:
        hook_str = hook
    else:
        filtered_pool = _filter_hook_pool_for_role(level.hook_pool, role_obj.name)
        hook_str = rng.choice(filtered_pool)
        # Method 3: detect & resample on conflict. After one filter pass, if
        # the sampled hook still reads as incompatible (e.g., fell through a
        # less-strict filter), retry up to 5 times before giving up.
        for _ in range(5):
            if not _hook_conflicts_with_role(hook_str, role_obj.name):
                break
            hook_str = rng.choice(filtered_pool)

    # 5) Tool count.
    if tool_count is not None:
        tool_count_range = (tool_count, tool_count)
    lo, hi = tool_count_range
    chosen_count = rng.randint(lo, hi)
    # Forced calls (remix, multi-target arrangement_*) take priority over
    # the sampled count: the LLM must echo every one before any free
    # apply_fx, and the sample range can otherwise be smaller than the
    # forced-call list.
    n_forced = len(intent.forced_calls)
    if n_forced > chosen_count:
        chosen_count = n_forced + 1   # +1 → at least one free apply_fx

    motivation_only = intent.intent in MOTIVATION_ONLY_INTENTS
    if motivation_only:
        # Tool-free system prompt — the pipeline performs the operation
        # deterministically. No <tools> block, no cookbook (the cookbook
        # only matters for tool parameter choices).
        from procraft_data.tools.schemas import build_motivation_only_system_prompt
        base_system = build_motivation_only_system_prompt(track_metadata)
        system = (
            f"{base_system}\n"
            f"{_HARD_RULES}\n"
        )
    else:
        base_system = build_hermes_system_prompt(track_metadata)
        system = (
            f"{base_system}\n"
            f"{_HARD_RULES}\n"
            f"{_EFFECT_COOKBOOK}\n"
            f"{_PRODUCTION_PRINCIPLES}\n"
        )

    # Natural-name translation table — inject explicitly so the model has a
    # pre-approved phrasing for every identifier in the metadata. This is
    # what turns "guitar_1" / "strings__continued" into "the distortion
    # guitar" / "the choir pad" in the motivation.
    from procraft_data.sources.slakh import natural_names
    try:
        # describe_mixture already built the metadata string with
        # track_metadata. But the MixtureState / TrackMeta pair we need for
        # natural_names() isn't available here — we parse from the metadata
        # string instead.
        natural_map = _natural_names_from_metadata(track_metadata)
    except Exception:
        natural_map = {}
    natural_block = ""
    if natural_map:
        rows = "\n".join(f"  - `{k}` → {v}" for k, v in natural_map.items())
        natural_block = (
            f"Natural-language phrasings to use in the motivation sentence "
            f"(use these EXACT phrasings, or a close paraphrase, in place of "
            f"the internal identifiers):\n{rows}\n"
            f"In tool_call `arguments`, use the internal identifier on the "
            f"left; in the motivation sentence, use the natural phrase on "
            f"the right.\n\n"
        )

    example_block = "\n".join(f'  - "{s}"' for s in role_obj.example_sentences)
    user_head = (
        f"{natural_block}"
        f"You are writing AS: {role_obj.name}\n"
        f"Their voice: {role_obj.voice}\n"
        f"Two example sentences this role would write (in general, not for this "
        f"mixture):\n{example_block}\n"
        f"\n"
        f"Phrasing style for this turn: {level.name} — {level.description}\n"
        f"Concrete hook to anchor the motivation: \"{hook_str}\"\n"
        f"\n"
        f"PRIMARY MOVE:\n  {intent.description}\n"
        f"\n"
    )

    if motivation_only:
        # ``extract_track`` uses a tiny, instruction-style prompt: a
        # short imperative ("Extract the piano.") with verb variation
        # and a paraphrased natural name. No "why" clause, no role /
        # abstraction / hook scaffolding — those are still recorded on
        # the spec for diversity bookkeeping but don't shape this text.
        target_id = intent.target_track or ""
        natural = natural_map.get(target_id, target_id)
        user = (
            f"EXTRACT MOVE: take the track '{target_id}' "
            f"({natural}) out of the mixture.\n\n"
            f"Output ONE line: 'Production motivation: ' followed by a "
            f"5-to-10-word imperative sentence telling someone to extract "
            f"this track. Vary the verb naturally — pick from {{extract, "
            f"pull out, solo, isolate, separate, lift, take out, single "
            f"out}}. Paraphrase the track name when it reads naturally "
            f"('piano' → 'keyboard track'; 'electric guitar' → 'rhythm "
            f"guitar part'). DO NOT add an explanation, a 'so we can…' "
            f"clause, or any other text. Just the imperative.\n\n"
            f"Examples (do NOT copy verbatim):\n"
            f"  - \"Extract the piano.\"\n"
            f"  - \"Solo the bass track.\"\n"
            f"  - \"Pull the drums out of the mix.\"\n\n"
            f"Do NOT emit any <tool_call> blocks. Do NOT write a "
            f"<think>...</think> block. After the 'Production motivation:' "
            f"line, stop."
        )
    else:
        user = (
            user_head
            + f"Think inside <think>...</think> about what is genuinely weak "
              f"or underused in THIS mixture for that role's concern, "
              f"referencing ONLY tracks that appear in the Mixture metadata. "
              f"Then, outside the think block, write exactly ONE line "
              f"beginning with 'Production motivation: ' followed by a "
              f"natural sentence that:\n"
              f"  (a) sounds like it was written by {role_obj.name} — adopt their voice;\n"
              f"  (b) uses the {level.name} phrasing style above;\n"
              f"  (c) references or clearly evokes the hook \"{hook_str}\";\n"
              f"  (d) describes the PRIMARY MOVE above in the role's natural language"
            + (f" — the sentence MUST name '{intent.target_track}' explicitly"
               if intent.target_track else "")
            + (f" and describe the outcome of reaching GM program {intent.target_program} "
               "(even if the role's voice would not say 'program N' literally — "
               "paraphrase into what that instrument SOUNDS like)."
               if intent.target_program is not None else ".")
            + "\n"
              f"  (e) describes the WHOLE production direction across "
              f"the {chosen_count} tool_calls you're about to emit — "
              f"not just the primary move. Your motivation is the prose "
              f"summary of every move on this turn. If you're going to "
              f"add reverb on guitar AND sidechain the bass AND widen "
              f"the strings, the motivation must signal all of those "
              f"(either by name or via a single phrase that plausibly "
              f"encompasses them — e.g. 'lush stadium space, tight low "
              f"end, wide top'). When the hook gives you a narrow recipe "
              f"(e.g. a parametric reverb setting), promote it into a "
              f"broader direction your secondary calls will fill out — "
              f"don't just restate the recipe.\n\n"
              f"Finally emit EXACTLY {chosen_count} tool_call block(s):\n"
        )
        if intent.forced_calls:
            n_forced = len(intent.forced_calls)
            n_extra = max(0, chosen_count - n_forced)
            user += (
                f"  - The first {n_forced} call(s) MUST be the forced "
                f"tool_calls listed in the PRIMARY MOVE above (each one "
                f"emitted verbatim, in any order).\n"
                f"  - The remaining {n_extra} call(s) are free apply_fx "
                f"calls. EACH ONE must (a) elaborate or directly support "
                f"the production direction stated in your motivation — "
                f"do NOT introduce unrelated moves on tracks the "
                f"motivation doesn't reference; (b) be audibly substantial "
                f"— pick medium or strong intensity from the EFFECT "
                f"STRENGTH table; never 'light' / 'subtle'. If the "
                f"motivation supports it, spread the secondary fx across "
                f"different tracks AND effect families (EQ, dynamics, "
                f"modulation, distortion, reverb, delay).\n"
            )
        else:
            user += (
                f"  - AT LEAST ONE call MUST be the primary-move tool "
                f"({intent.primary_tool}) targeting the committed track "
                f"where applicable.\n"
                f"  - The remaining {max(0, chosen_count - 1)} call(s) are "
                f"secondary. EACH ONE must (a) elaborate or directly "
                f"support the production direction stated in your "
                f"motivation — do NOT introduce unrelated moves on tracks "
                f"the motivation doesn't reference; (b) be audibly "
                f"substantial — pick medium or strong intensity from the "
                f"EFFECT STRENGTH table; never 'light' / 'subtle'. Bias "
                f"toward apply_fx. If the motivation supports it (a broad "
                f"aesthetic direction), spread fx across different tracks "
                f"AND effect families. If the motivation is narrowly "
                f"scoped (one parametric recipe), keep secondary calls "
                f"tight around that move (carve EQ space for the focal "
                f"effect, sidechain off it, mirror-treat a sister track).\n"
            )
        user += (
            f"  - Every call: <tool_call>{{\"name\": ..., \"arguments\": ...}}</tool_call>\n"
            f"    wrapped in literal tags, JSON with exactly two keys, UNIQUE.\n"
            f"  - In every call's ``track`` argument, use the EXACT identifier "
            f"from the Mixture metadata (e.g. `guitar_1`) — never the quoted "
            f"program name.\n"
            f"  - COUNT IS ENFORCED: emit exactly {chosen_count} tool_call "
            f"blocks. Outputs with fewer than {max(1, chosen_count - 2)} "
            f"calls will be DISCARDED and the request retried — do not stop "
            f"early just because you've covered the primary move. Fill the "
            f"remaining slots with apply_fx calls that elaborate the "
            f"motivation as instructed above.\n"
            f"After the {chosen_count}th </tool_call>, stop generating."
        )

    return PromptSpec(
        system=system, user=user,
        role=role_obj.name,
        abstraction_level=level.name,
        hook=hook_str,
        primary_intent=intent.intent,
        primary_tool=intent.primary_tool,
        target_track=intent.target_track,
        target_program=intent.target_program,
        category_focus="free",
        temperature=temperature,
        tool_count_hint=tool_count_range,
        forced_calls=[dict(c) for c in intent.forced_calls],
        plan={k: v for k, v in intent.plan},
        motivation_only=motivation_only,
        chosen_count=chosen_count if not motivation_only else 0,
    )


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------
def smoke_plan(track_metadata: str, seed: int = 0) -> list[PromptSpec]:
    """One spec per PRIMARY_INTENT for smoke inspection.

    Iterates the 10 primary intents in order so every run covers each
    intent at least once — uniform category representation. Role and
    abstraction level are sampled randomly per slot for voice diversity.
    """
    rng = random.Random(seed)
    role_objs = _compatible_roles(track_metadata) or ROLES
    levels = list(ABSTRACTION_LEVELS)
    out: list[PromptSpec] = []
    for intent_name in PRIMARY_INTENTS:
        role_obj = rng.choice(role_objs)
        level_name = rng.choice(levels)
        intent = sample_primary_intent(
            track_metadata,
            forced_intent=intent_name,
            seed=rng.randint(0, 2**31 - 1),
        )
        out.append(build_spec(
            track_metadata,
            role=role_obj.name,
            abstraction_level=level_name,
            intent=intent,
            seed=rng.randint(0, 2**31 - 1),
        ))
    return out


def coverage_plan(
    track_metadata: str,
    *,
    temperatures: Iterable[float] = (0.7, 0.9, 1.1),
    intents_per_temp: int | None = None,
    roles_per_intent: int = 1,
    seed: int = 0,
) -> list[PromptSpec]:
    """Grid across intents × roles × abstractions × temperatures.

    Primary intent is uniformly distributed — every temperature block covers
    all 10 intents unless ``intents_per_temp`` is set to limit it.
    """
    rng = random.Random(seed)
    role_objs = _compatible_roles(track_metadata) or ROLES
    levels = list(ABSTRACTION_LEVELS)
    intent_list = PRIMARY_INTENTS
    if intents_per_temp is not None:
        intent_list = rng.sample(PRIMARY_INTENTS, k=intents_per_temp)
    specs: list[PromptSpec] = []
    for temp in temperatures:
        for intent_name in intent_list:
            for _ in range(roles_per_intent):
                role_obj = rng.choice(role_objs)
                level_name = rng.choice(levels)
                intent = sample_primary_intent(
                    track_metadata,
                    forced_intent=intent_name,
                    seed=rng.randint(0, 2**31 - 1),
                )
                specs.append(build_spec(
                    track_metadata,
                    role=role_obj.name,
                    abstraction_level=level_name,
                    intent=intent,
                    temperature=temp,
                    seed=rng.randint(0, 2**31 - 1),
                ))
    return specs
