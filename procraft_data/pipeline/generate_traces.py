"""Drive Qwen3 through a prompt plan, execute tool calls, write dataset entries.

Each dataset entry:

    entry_{track_id}_{motivation_type}_{idx}.json         (text trace)
    entry_{track_id}_{motivation_type}_{idx}_original.wav (48 kHz stereo, 10s)
    entry_{track_id}_{motivation_type}_{idx}_modified.wav

The JSON follows proposal §3.2 Step 2:

    {
        "track_id": "Track00001", "source": "slakh/babyslakh",
        "mixture_metadata": "tracks: piano (GM 1), bass (GM 33), ...",
        "style": "for a lo-fi hip-hop playlist",
        "category_focus": "effects_change",
        "temperature": 0.9,
        "motivation": "one-sentence production motivation",
        "think": "… the full <think> block content …",
        "tool_calls": [...],           # parsed list[{"name", "arguments"}]
        "executed_ok": true,
        "executed_errors": [],
        "usage": {"prompt_tokens": ..., "completion_tokens": ...},
        "latency_sec": 4.2
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from configs import paths
from procraft_data.pipeline.trace_client import TraceResult, VLLMClient
from procraft_data.pipeline.trace_prompts import PromptSpec, build_spec
from procraft_data.rendering.fluidsynth_render import (
    FluidSynthRenderer, find_default_soundfont,
)
from procraft_data.sources.slakh import (
    TrackMeta, build_mixture_state, describe_mixture,
)
import re

from multiafx import registry as _mafx_registry

from procraft_data.tools.executors import EXECUTORS, MixtureState, _mixdown


@dataclass
class DatasetEntry:
    track_id: str
    source: str
    mixture_metadata: str
    role: str
    abstraction_level: str
    hook: str
    primary_intent: str
    primary_tool: str
    target_track: str | None
    target_program: int | None
    primary_move_executed: bool
    attempt_count: int                   # 1 = first try succeeded; >1 = retried
    retry_reasons: list[str]             # reason per retry, length == attempt_count-1
    temperature: float
    motivation: str
    think: str | None
    tool_calls: list[dict]
    executed_ok: bool
    executed_errors: list[str]
    # Per-tool-call audio verification: one entry per parsed tool_call, in order.
    # Fields: name, status ('ok'|'error'|'skipped'),
    #         mix_rms_before, mix_rms_after, mix_rms_delta.
    # mix_rms_delta near zero for a status='ok' call usually means the call
    # was a silent no-op (parameter range, applied to a silent track, etc.).
    tool_effects: list[dict]
    mix_rms_original: float
    mix_rms_modified: float
    mix_rms_delta: float
    usage: dict
    latency_sec: float
    raw_response: str
    original_wav: str = ""
    modified_wav: str = ""

    def to_json(self) -> dict:
        d = self.__dict__.copy()
        return d




def _safe_rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def _coerce_top_level_fx(tc: dict) -> dict:
    """If ``tc.name`` is a multiafx effect name, re-wrap as apply_fx."""
    name = tc.get("name")
    if name in EXECUTORS:
        return tc
    if name in _mafx_registry:
        args = dict(tc.get("arguments", {}) or {})
        track = args.pop("track", None) or "mix"
        # Everything else in arguments becomes the effect's params.
        return {
            "name": "apply_fx",
            "arguments": {
                "track": track,
                "call": {"effect": name, "params": args},
            },
        }
    return tc


_RESTORATION_REWRITES = [
    # (pattern, replacement) — applied only to arrangement_add motivations
    (re.compile(r"\bit'?s\s+missing\b", re.I),                      "we can introduce"),
    (re.compile(r"\b(?:is|are|was|were)\s+missing\b", re.I),         "would benefit from"),
    (re.compile(r"\bneeds?\s+to\s+have\b", re.I),                    "would carry"),
    (re.compile(r"\bbring(?:ing)?\s+back\b", re.I),                  "introducing"),
    (re.compile(r"\brestore\b", re.I),                               "add"),
    (re.compile(r"\brestoring\b", re.I),                             "introducing"),
    (re.compile(r"\breintroduc(?:e|ing)\b", re.I),                   "introducing"),
    (re.compile(r"\b(?:re)?add(?:ing)?\s+back\b", re.I),             "introducing"),
    (re.compile(r"\b(?:is|are|was|were)\s+lacking\b", re.I),         "needs"),
    (re.compile(r"\blacking\b", re.I),                               "missing"),  # leave as signal
    (re.compile(r"\b(?:is|are|was|were)\s+absent\b", re.I),          "can be added"),
    (re.compile(r"\bput(?:ting)?\s+back\b", re.I),                   "introducing"),
]


def _rewrite_restoration_words(motivation: str) -> str:
    """Replace restorative framing with additive framing for arrangement_add."""
    if not motivation:
        return motivation
    for pat, rep in _RESTORATION_REWRITES:
        motivation = pat.sub(rep, motivation)
    return motivation


def _clean_motivation(motivation: str, natural_map: dict[str, str]) -> str:
    """Replace any leaked internal track IDs with the approved natural phrase.

    Qwen3 is told via system prompt + per-entry translation table to use
    natural names in the motivation sentence. Even so, ~30% of outputs
    leak ``guitar_2`` / ``strings__continued`` / ``organ_3`` into the
    motivation. Post-process deterministically:

    - If the ID is preceded by "the " (case-insensitive), consume that
      "the " so the natural phrase's own leading "the " doesn't double up.
      ``"the organ_2"`` → ``"the first harmonica"``.
    - Otherwise substitute the natural phrase directly. ``"swap organ_2 to"``
      → ``"swap the first harmonica to"``.
    - Identifiers with more characters are replaced first so
      ``strings__continued`` doesn't get partially matched as ``strings``.
    """
    if not motivation or not natural_map:
        return motivation
    # Single-word identifiers that are also real English words — the model
    # uses them naturally in prose ("this bright acoustic piano is…") and
    # substituting produces doubled noun phrases like
    # "this bright acoustic the bright acoustic piano". Skip those entirely.
    ENGLISH_WORD_IDS = {"piano", "bass", "drums", "organ", "guitar", "strings"}
    for ident in sorted(natural_map, key=len, reverse=True):
        if ident.lower() in ENGLISH_WORD_IDS:
            continue
        phrase = natural_map[ident]
        pattern = re.compile(
            rf'(\bthe\s+)?\b{re.escape(ident)}\b', re.IGNORECASE)

        def _sub(m, p=phrase):
            the_part = m.group(1) or ""
            if the_part and the_part[0].isupper():
                return p[0].upper() + p[1:]
            return p

        motivation = pattern.sub(_sub, motivation)
    motivation = re.sub(r'\bthe\s+the\b', 'the', motivation, flags=re.I)
    return motivation


def _canonicalize_tool_call(tc: dict) -> dict:
    """Normalize a tool_call for dedup: round floats, sort dict keys."""
    def _norm(x):
        if isinstance(x, float):
            return round(x, 3)
        if isinstance(x, int):
            return float(x)
        if isinstance(x, dict):
            return {k: _norm(v) for k, v in sorted(x.items())}
        if isinstance(x, list):
            return [_norm(v) for v in x]
        return x
    return {"name": tc.get("name"), "arguments": _norm(tc.get("arguments", {}))}


def _shared_peak_normalize(
    original: np.ndarray, modified: np.ndarray, target: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Peak-normalize both clips with the SAME scale factor.

    Using a shared scale preserves the relative loudness delta between
    original and modified (so the modification is audible), while still
    lifting quiet clips (e.g. sparse-piano excerpts) to a usable level and
    taming blow-up clips (e.g. mix-bus overdrive) to avoid clipping. The
    scale is picked so the louder of the two peaks lands at ``target``.
    """
    peak = max(float(np.max(np.abs(original))), float(np.max(np.abs(modified))))
    if peak <= 0.0:
        return original, modified, 1.0
    scale = target / peak
    return original * scale, modified * scale, scale


def _tool_target_track(name: str, args: dict) -> str | None:
    """Pull the track-name arg most tools carry so we can measure per-track delta."""
    if not isinstance(args, dict):
        return None
    for key in ("track", "track_name"):
        v = args.get(key)
        if isinstance(v, str):
            return v
    return None


def _track_audio(state: MixtureState, name: str | None) -> np.ndarray:
    if not name or name == "mix":
        return _mixdown(state) if state.tracks else np.zeros((2, 1))
    ts = state.tracks.get(name) or state.pending_tracks.get(name)
    if ts is None or ts.audio is None:
        return np.zeros((2, 1))
    return ts.audio


def _result_is_valid(motivation: str, tool_calls: list[dict], primary_tool: str) -> tuple[bool, str]:
    """Return ``(valid, reason_if_not)`` for a freshly-generated response.

    A valid response: has a non-empty motivation text AND emitted the
    primary-move tool. Anything else is retryable.
    """
    if not motivation.strip():
        return False, "empty motivation line"
    if not any(tc.get("name") == primary_tool for tc in tool_calls):
        return False, f"primary tool {primary_tool!r} missing from tool_calls"
    return True, ""


def generate_one(
    track: TrackMeta,
    spec: PromptSpec,
    client: VLLMClient,
    renderer: FluidSynthRenderer,
    out_dir: Path,
    *,
    duration_sec: float = paths.CLIP_SECONDS,
    sample_rate: int = paths.SAMPLE_RATE,
    entry_idx: int = 0,
    withhold_for_add: list[str] | None = None,
    max_retries: int = 3,
) -> DatasetEntry:
    """One (prompt → response → execute → save) round-trip, with retry.

    If the first LLM call returns an empty motivation or doesn't emit the
    pre-committed primary tool, retry up to ``max_retries`` times. Each
    retry issues a fresh sampling call (same prompt, LLM non-determinism
    gives us different output); reasons are recorded on the saved entry.
    After the final attempt, whatever we have is stored with
    ``executed_ok = False`` and ``retry_reasons`` populated so a downstream
    quality filter can drop it.
    """
    import pretty_midi
    longest = max(pretty_midi.PrettyMIDI(str(track.midi_path(sid))).get_end_time()
                  for sid in track.stems)
    start = max(0.0, (longest - duration_sec) / 2.0)

    # Honor the pre-committed arrangement_add intent by withholding a track if
    # the caller didn't already supply one. Prefer note-rich tracks so the
    # added stem audibly changes the mix. Count notes INSIDE the 30s window
    # (not the whole-track note count) because a track with 500 total notes
    # might have only 3 in our window.
    if spec.primary_intent == "arrangement_add" and not withhold_for_add:
        import random as _rnd
        import pretty_midi as _pm
        from procraft_data.sources.slakh import _dedup_names
        rng = _rnd.Random(hash((track.track_id, entry_idx)))
        name_by_sid = _dedup_names(list(track.stems.values()))

        # Count notes per stem in the window.
        note_counts: list[tuple[str, int]] = []
        for sid, name in name_by_sid.items():
            try:
                midi = _pm.PrettyMIDI(str(track.midi_path(sid)))
                instr = midi.instruments[0] if midi.instruments else None
                if instr is None:
                    continue
                n = sum(1 for note in instr.notes
                        if note.end > start and note.start < start + duration_sec)
                note_counts.append((name, n))
            except Exception:
                continue

        if len(note_counts) >= 3:
            # Keep candidates with above-median note count; this biases toward
            # tracks that actually contribute material in the window.
            sorted_by_notes = sorted(note_counts, key=lambda x: x[1], reverse=True)
            top_half = sorted_by_notes[: max(3, len(sorted_by_notes) // 2)]
            # Guard against tiny-note-count trivial picks.
            top_half = [(n, c) for (n, c) in top_half if c >= 6] or top_half
            pick_name, _ = rng.choice(top_half)
            withhold_for_add = [pick_name]

    state = build_mixture_state(
        track, sample_rate, start, duration_sec,
        withhold=withhold_for_add,
    )
    for ts in state.tracks.values():
        ts.audio = renderer.render_track(ts, duration_sec)
    # Capture pre-execution metadata — tool calls mutate state in place.
    input_metadata = describe_mixture(state, track)

    # Rebuild the prompt with the real post-withhold metadata so the system
    # prompt agrees with the state the executors will see (especially for
    # arrangement_add, which depends on 'available_to_add' track names).
    # All role/abstraction/hook/intent fields are preserved verbatim.
    from procraft_data.pipeline.trace_prompts import IntentCommitment
    pinned_intent = IntentCommitment(
        intent=spec.primary_intent,
        primary_tool=spec.primary_tool,
        target_track=spec.target_track,
        target_program=spec.target_program,
        description="",   # rebuilt inside build_spec via sample_primary_intent fallback
    )
    # For arrangement_add we want the description to name the actual withheld
    # track; for other intents the previously-sampled description is fine.
    from procraft_data.pipeline.trace_prompts import sample_primary_intent
    if spec.primary_intent == "arrangement_add" and withhold_for_add:
        pinned_intent = IntentCommitment(
            intent="arrangement_add",
            primary_tool="add_track",
            target_track=withhold_for_add[0],
            target_program=None,
            description=(
                f"The main move is to bring back the withheld track "
                f"'{withhold_for_add[0]}' using add_track. Your motivation "
                f"should describe what it adds to the mixture."
            ),
        )
    else:
        pinned_intent = sample_primary_intent(
            input_metadata, forced_intent=spec.primary_intent,
            seed=hash((track.track_id, entry_idx, spec.primary_intent)),
        )
    spec = build_spec(
        track_metadata=input_metadata,
        role=spec.role,
        abstraction_level=spec.abstraction_level,
        hook=spec.hook,
        intent=pinned_intent,
        tool_count_range=spec.tool_count_hint,
        temperature=spec.temperature,
    )

    tr: TraceResult
    retry_reasons: list[str] = []
    attempt = 0
    while True:
        attempt += 1
        tr = client.complete(spec)

        # Coerce any top-level tool name that's actually a multiafx effect into
        # the proper apply_fx form. Qwen3 sometimes emits e.g.
        # ``{"name": "npy_stereo_widener", "arguments": {"track": "mix", "width": 1.5}}``
        # when the correct form is an apply_fx envelope.
        tr.tool_calls = [_coerce_top_level_fx(tc) for tc in tr.tool_calls]

        # Deduplicate tool_calls using a canonical key (rounded floats, sorted keys)
        # so Qwen3 repeat-emissions collapse correctly.
        seen: set[str] = set()
        deduped: list[dict] = []
        for tc in tr.tool_calls:
            key = json.dumps(_canonicalize_tool_call(tc), sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(tc)
        tr.tool_calls = deduped

        valid, why = _result_is_valid(
            tr.motivation_text, tr.tool_calls, spec.primary_tool
        )
        if valid or attempt > max_retries:
            break
        retry_reasons.append(f"attempt {attempt}: {why}")

    errors: list[str] = []
    executed = 0
    tool_effects: list[dict] = []
    for tc in tr.tool_calls:
        name = tc.get("name")
        args = tc.get("arguments", {}) or {}
        target = _tool_target_track(name, args)

        mix_before = _safe_rms(_mixdown(state)) if state.tracks else 0.0
        track_before = _safe_rms(_track_audio(state, target)) if target else None

        if name not in EXECUTORS:
            errors.append(f"unknown tool: {name!r}")
            tool_effects.append({
                "name": name, "target": target, "status": "error",
                "error": "unknown tool",
                "mix_rms_before": mix_before, "mix_rms_after": mix_before,
                "mix_rms_delta": 0.0,
                "track_rms_before": track_before, "track_rms_after": track_before,
                "track_rms_delta": 0.0,
            })
            continue
        try:
            EXECUTORS[name](args, state, renderer, duration_sec)
            state.executed.append(tc)
            executed += 1
            mix_after = _safe_rms(_mixdown(state)) if state.tracks else 0.0
            track_after = _safe_rms(_track_audio(state, target)) if target else None
            tool_effects.append({
                "name": name, "target": target, "status": "ok",
                "mix_rms_before": mix_before, "mix_rms_after": mix_after,
                "mix_rms_delta": abs(mix_after - mix_before),
                "track_rms_before": track_before, "track_rms_after": track_after,
                "track_rms_delta": (abs(track_after - track_before)
                                    if track_before is not None and track_after is not None
                                    else None),
            })
        except Exception as e:
            errors.append(f"{name}: {e}")
            tool_effects.append({
                "name": name, "target": target, "status": "error",
                "error": str(e),
                "mix_rms_before": mix_before, "mix_rms_after": mix_before,
                "mix_rms_delta": 0.0,
                "track_rms_before": track_before, "track_rms_after": track_before,
                "track_rms_delta": 0.0,
            })
    executed_ok = executed > 0 and executed == len(tr.tool_calls)

    original_mix = _compute_original_mix(
        track, sample_rate, start, duration_sec, renderer
    )
    modified_mix = _mixdown(state) if state.tracks else np.zeros_like(original_mix)
    mix_rms_original = _safe_rms(original_mix)
    mix_rms_modified = _safe_rms(modified_mix)

    stem = f"entry_{track.track_id}_{spec.category_focus}_{entry_idx:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    orig_path = out_dir / f"{stem}_original.wav"
    mod_path = out_dir / f"{stem}_modified.wav"
    original_out, modified_out, _ = _shared_peak_normalize(original_mix, modified_mix)
    sf.write(orig_path, original_out.T, sample_rate)
    sf.write(mod_path, modified_out.T, sample_rate)

    primary_move_executed = any(
        tc.get("name") == spec.primary_tool for tc in tr.tool_calls
    )

    # Final validity check — if retry loop gave up, mark executed_ok=False so
    # the downstream quality filter drops this entry.
    final_valid, final_why = _result_is_valid(
        tr.motivation_text, tr.tool_calls, spec.primary_tool
    )
    if not final_valid:
        errors.append(f"final-attempt invalid: {final_why}")
        executed_ok = False

    # Post-process the motivation: swap any leaked internal track IDs for
    # their natural-name phrase. (The system prompt forbids leaks, but Qwen3
    # still leaks ~30% of the time; this is the deterministic safety net.)
    from procraft_data.pipeline.trace_prompts import _natural_names_from_metadata
    natural_map = _natural_names_from_metadata(input_metadata)
    cleaned_motivation = _clean_motivation(tr.motivation_text, natural_map)
    # For arrangement_add, also rewrite any leftover restoration words into
    # additive framing. The model sometimes still says "it's missing X" even
    # with explicit "never use 'missing'" in the prompt.
    if spec.primary_intent == "arrangement_add":
        cleaned_motivation = _rewrite_restoration_words(cleaned_motivation)

    entry = DatasetEntry(
        track_id=track.track_id,
        source=f"slakh/{track.root.parent.name}",
        mixture_metadata=input_metadata,
        role=spec.role,
        abstraction_level=spec.abstraction_level,
        hook=spec.hook,
        primary_intent=spec.primary_intent,
        primary_tool=spec.primary_tool,
        target_track=spec.target_track,
        target_program=spec.target_program,
        primary_move_executed=primary_move_executed,
        attempt_count=attempt,
        retry_reasons=retry_reasons,
        temperature=spec.temperature,
        motivation=cleaned_motivation,
        think=tr.reasoning_text,
        tool_calls=tr.tool_calls,
        executed_ok=executed_ok,
        executed_errors=errors,
        tool_effects=tool_effects,
        mix_rms_original=mix_rms_original,
        mix_rms_modified=mix_rms_modified,
        mix_rms_delta=abs(mix_rms_modified - mix_rms_original),
        usage=tr.usage,
        latency_sec=tr.latency_sec,
        raw_response=tr.assistant_text,
        original_wav=orig_path.name,
        modified_wav=mod_path.name,
    )
    (out_dir / f"{stem}.json").write_text(json.dumps(entry.to_json(), indent=2))
    return entry


def _compute_original_mix(
    track: TrackMeta, sample_rate: int, start: float, dur: float,
    renderer: FluidSynthRenderer,
) -> np.ndarray:
    """Render the unmodified mixture (no withhold) for the 'original_audio' side.

    We render twice (once for input-to-model, once for original_audio) rather
    than caching the first because tool_calls that mutate the state would
    poison a shared reference.
    """
    state = build_mixture_state(track, sample_rate, start, dur)
    for ts in state.tracks.values():
        ts.audio = renderer.render_track(ts, dur)
    return _mixdown(state)


def _choose_withhold(track: TrackMeta, role: str | None = None,
                     rng_seed: int = 0,
                     probability: float = 0.5) -> list[str] | None:
    """Optionally withhold one track so add_track can materialize it.

    Any role can trigger add_track now (no category_focus gating), so we
    withhold a random track half the time. Skipped if the mix has fewer than
    3 tracks (keeping at least 2 for a meaningful modified mix).
    """
    import random
    from procraft_data.sources.slakh import _dedup_names
    rng = random.Random(rng_seed)
    if rng.random() > probability:
        return None
    names = list(_dedup_names(list(track.stems.values())).values())
    if len(names) < 3:
        return None
    return [rng.choice(names)]
