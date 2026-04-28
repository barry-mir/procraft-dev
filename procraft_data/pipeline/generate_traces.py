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
    # Structured instrument lists captured from MixtureState.tracks at the
    # exact same moments as the mix snapshots in §3.10 of IMPLEMENTATION.md:
    #   pre_instruments  = state.tracks BEFORE any tool_call executes.
    #                      For arrangement_add the withheld target lives in
    #                      state.pending_tracks at this point, so it is NOT
    #                      in this list — that matches what original_audio
    #                      actually contains.
    #   post_instruments = state.tracks AFTER all tool_calls execute.
    # Each entry: {name, program, is_drum, midi_program_name, inst_class}.
    # Use these for [TMB] supervision; downstream code can dedupe by program
    # if it wants set-level mixture labels.
    pre_instruments: list[dict]
    post_instruments: list[dict]
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
    latency_sec: float            # LLM-only call latency (sum across retries)
    raw_response: str
    original_wav: str = ""
    modified_wav: str = ""
    # Per-stage wall-clock breakdown for throughput analysis. ``llm_sec``
    # is the cumulative LLM time (== latency_sec, kept for convenience).
    # ``render_sec`` is FluidSynth render of all stems before the LLM
    # call. ``executor_sec`` is the time spent applying tool_calls
    # (mostly multiafx + any add_track re-render). ``total_sec`` is the
    # entry's wall-clock from generate_one start to write end.
    llm_sec: float = 0.0
    render_sec: float = 0.0
    executor_sec: float = 0.0
    total_sec: float = 0.0
    # MIDI sidecars written next to the WAVs. ``original.mid`` describes
    # ``state.tracks`` BEFORE any tool_call executes (matches original_audio).
    # ``modified.mid`` describes ``state.tracks`` AFTER all tool_calls execute
    # (matches modified_audio). Used as the content-branch supervision target.
    original_midi: str = ""
    modified_midi: str = ""

    def to_json(self) -> dict:
        d = self.__dict__.copy()
        return d




def _safe_rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def _state_to_midi(state: MixtureState) -> "pretty_midi.PrettyMIDI":
    """Bundle ``state.tracks`` into one multitrack ``PrettyMIDI``.

    Used as the snapshot source for ``original.mid`` (pre-execution) and
    ``modified.mid`` (post-execution). Each ``TrackState`` becomes a separate
    ``Instrument`` track, with the ``TrackState.name`` carried into
    ``Instrument.name`` so downstream tools can map a MIDI track back to the
    audio stem / metadata record. Withheld tracks (``state.pending_tracks``)
    are intentionally excluded from the *pre* snapshot so the file describes
    what ``original_audio`` actually contains; ``add_track`` will move the
    pending entry into ``state.tracks`` before the *post* snapshot.
    """
    import copy as _copy
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    for ts in state.tracks.values():
        inst = _copy.deepcopy(ts.midi)
        inst.is_drum = bool(ts.is_drum)
        # ``change_instrument`` keeps ts.midi.program in sync, but layered /
        # doubled / arrangement_add tracks may not — be defensive.
        try:
            inst.program = 0 if ts.is_drum else max(0, min(127, int(ts.program)))
        except Exception:
            inst.program = 0
        inst.name = ts.name
        pm.instruments.append(inst)
    return pm


def _capture_instruments(
    state: MixtureState,
    track,    # TrackMeta — kept untyped to avoid an import cycle with sources.slakh
) -> list[dict]:
    """Snapshot the active instrumentation in ``state.tracks``.

    Returns one dict per active TrackState. Slakh metadata
    (``midi_program_name``, ``inst_class``) is filled in for stems that map
    back to a Slakh ``StemMeta`` via the dedup-name lookup. Synthetic
    tracks introduced by ``layer_instrument`` (``<base>__layer<prog>``) or
    ``double_track`` (``<base>__dbl``) don't have Slakh metadata; for those
    we fall back to ``pretty_midi``'s GM program-name table.
    """
    import pretty_midi
    from procraft_data.sources.slakh import _dedup_names

    sid_by_name: dict[str, str] = {}
    if track is not None and getattr(track, "stems", None):
        names_by_sid = _dedup_names(list(track.stems.values()))
        sid_by_name = {v: k for k, v in names_by_sid.items()}

    out: list[dict] = []
    for name, ts in state.tracks.items():
        sid = sid_by_name.get(name)
        stem = track.stems.get(sid) if (sid and track is not None) else None
        if stem is not None:
            midi_name = stem.midi_program_name
            inst_class = stem.inst_class
        else:
            # Synthetic track (layer / double / new add) — derive name from GM.
            if ts.is_drum:
                midi_name = "Drum Kit"
            else:
                try:
                    midi_name = pretty_midi.program_to_instrument_name(
                        max(0, min(127, int(ts.program)))
                    )
                except Exception:
                    midi_name = ""
            inst_class = None
        out.append({
            "name": name,
            "program": int(ts.program),
            "is_drum": bool(ts.is_drum),
            "midi_program_name": midi_name,
            "inst_class": inst_class,
        })
    return out


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
    if not motivation:
        return motivation
    # GM program-number leakage scrub. The system prompt forbids
    # writing "GM 85" / "program 33" / "to_program: 0" literally in the
    # motivation, but the model still echoes ~25% of the time. Replace
    # the parenthetical numeric reference with the GM program name from
    # ``pretty_midi.program_to_instrument_name`` so the sentence reads
    # in plain English. Patterns covered:
    #   - "to GM 85" / "GM 85"   →  "to String Ensemble 1" / "String Ensemble 1"
    #   - "(GM 85)"              →  "" (drop — the surrounding word
    #                                 already names the instrument)
    #   - "program 33"           →  "Electric Bass (finger)"
    #   - "to_program: 33"       →  "Electric Bass (finger)"
    import pretty_midi as _pm
    def _gm_name(prog: int) -> str:
        try:
            return _pm.program_to_instrument_name(max(0, min(127, int(prog))))
        except Exception:
            return ""
    def _replace_gm(match: "re.Match") -> str:
        prog = int(match.group("prog"))
        name = _gm_name(prog)
        return name or match.group(0)
    motivation = re.sub(r"\(\s*GM\s+(?P<prog>\d+)\s*\)", "", motivation, flags=re.I)
    motivation = re.sub(r"\bGM\s+(?P<prog>\d+)\b", _replace_gm, motivation, flags=re.I)
    motivation = re.sub(r"\bto_program\s*[:=]\s*(?P<prog>\d+)\b", _replace_gm, motivation, flags=re.I)
    motivation = re.sub(r"\bprogram\s+(?P<prog>\d+)\b", _replace_gm, motivation, flags=re.I)
    # Collapse any double-spaces left behind by the parenthetical drop.
    motivation = re.sub(r" {2,}", " ", motivation)

    if not natural_map:
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


def _result_is_valid(motivation: str, tool_calls: list[dict],
                     spec: PromptSpec) -> tuple[bool, str]:
    """Return ``(valid, reason_if_not)`` for a freshly-generated response.

    Branching by spec:
      * ``motivation_only`` (extract_track) — only require non-empty
        motivation; tool_calls are ignored (and the pipeline overwrites
        them with ``[]`` later).
      * ``forced_calls`` non-empty (remix) — require every forced call to
        appear in ``tool_calls`` matching name + key arguments.
      * Otherwise — require non-empty motivation AND at least one
        ``tool_calls`` entry matching ``spec.primary_tool``.
    """
    if not motivation.strip():
        return False, "empty motivation line"
    if spec.motivation_only:
        return True, ""
    if spec.forced_calls:
        for fc in spec.forced_calls:
            if not _matches_forced_call(fc, tool_calls):
                return False, f"forced call missing: {fc['name']} {fc['arguments']}"
        # ``remix``-specific content-binding (Option C). The motivation
        # must reference at least one dropped track AND at least one
        # added track — by identifier or natural-name substring,
        # case-insensitive. This catches generic "playlist-ready" prose
        # that doesn't actually describe the variant.
        if spec.primary_intent == "remix":
            ok, why = _remix_motivation_grounded(motivation, spec)
            if not ok:
                return False, why
        return True, ""
    if not any(tc.get("name") == spec.primary_tool for tc in tool_calls):
        return False, f"primary tool {spec.primary_tool!r} missing from tool_calls"
    # Under-emission check: the prompt asks for ``chosen_count`` calls
    # exactly. Allow a 2-call slack to accommodate small dedup losses
    # from ``_canonicalize_tool_call``, but reject anything well below
    # the requested count — a 12-call request that comes back with 2
    # calls means the model gave up on the secondary slots and the
    # output won't match the "densely-treated mix" we promise downstream.
    min_required = max(1, int(spec.chosen_count) - 2)
    if spec.chosen_count and len(tool_calls) < min_required:
        return False, (
            f"too few tool_calls: got {len(tool_calls)}, need ≥ {min_required} "
            f"(chosen_count = {spec.chosen_count})"
        )
    return True, ""


def _finalize_extract_track(
    *, track, spec: PromptSpec, tr: TraceResult, state: MixtureState,
    snapshot_before: np.ndarray, pre_instruments: list[dict],
    pre_midi: "any", input_metadata: str, attempt: int,
    retry_reasons: list[str], entry_idx: int, out_dir: Path,
    sample_rate: int,
    render_sec: float = 0.0, total_start: float = 0.0,
) -> "DatasetEntry":
    """Build the DatasetEntry for an ``extract_track`` run.

    The "modification" is purely deterministic: ``modified_audio`` is the
    target stem rendered alone (the per-track audio captured during the
    pre-LLM render pass). LLM tool_calls are ignored — the spec's
    motivation-only prompt forbids them, but we strip any that leaked.
    """
    target_name = spec.target_track or ""
    # Build the post-execution single-track view.
    single_state = MixtureState(sample_rate=state.sample_rate)
    if target_name in state.tracks:
        single_state.tracks[target_name] = state.tracks[target_name]
    post_instruments = _capture_instruments(single_state, track)
    post_midi = _state_to_midi(single_state)

    if target_name in state.tracks and state.tracks[target_name].audio is not None:
        modified_mix = state.tracks[target_name].audio
    else:
        modified_mix = np.zeros_like(snapshot_before)

    original_mix = snapshot_before
    mix_rms_original = _safe_rms(original_mix)
    mix_rms_modified = _safe_rms(modified_mix)

    stem = f"entry_{track.track_id}_{spec.category_focus}_{entry_idx:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    orig_path = out_dir / f"{stem}_original.wav"
    mod_path = out_dir / f"{stem}_modified.wav"
    orig_midi_path = out_dir / f"{stem}_original.mid"
    mod_midi_path = out_dir / f"{stem}_modified.mid"
    original_out, modified_out, _ = _shared_peak_normalize(original_mix, modified_mix)
    sf.write(orig_path, original_out.T, sample_rate)
    sf.write(mod_path, modified_out.T, sample_rate)
    pre_midi.write(str(orig_midi_path))
    post_midi.write(str(mod_midi_path))

    cleaned_motivation = tr.motivation_text  # extract_track prompt has no
    # internal-id leakage paths to worry about in practice; if the model
    # writes an ID, it's the target_track which is also a valid English-y
    # identifier in most Slakh tracks.

    entry = DatasetEntry(
        track_id=track.track_id,
        source=f"slakh/{track.root.parent.name}",
        mixture_metadata=input_metadata,
        role=spec.role,
        abstraction_level=spec.abstraction_level,
        hook=spec.hook,
        primary_intent=spec.primary_intent,
        primary_tool=spec.primary_tool,
        target_track=target_name,
        target_program=None,
        primary_move_executed=True,   # extract is performed by the pipeline
        attempt_count=attempt,
        retry_reasons=retry_reasons,
        pre_instruments=pre_instruments,
        post_instruments=post_instruments,
        temperature=spec.temperature,
        motivation=cleaned_motivation,
        think=tr.reasoning_text,
        tool_calls=[],   # the operation is not LLM-driven
        executed_ok=bool(cleaned_motivation.strip()),
        executed_errors=[],
        tool_effects=[],
        mix_rms_original=mix_rms_original,
        mix_rms_modified=mix_rms_modified,
        mix_rms_delta=abs(mix_rms_modified - mix_rms_original),
        usage=tr.usage,
        latency_sec=tr.latency_sec,
        raw_response=tr.assistant_text,
        original_wav=orig_path.name,
        modified_wav=mod_path.name,
        original_midi=orig_midi_path.name,
        modified_midi=mod_midi_path.name,
        llm_sec=tr.latency_sec,
        render_sec=render_sec,
        executor_sec=0.0,    # extract_track has no executor stage
        total_sec=(__import__('time').perf_counter() - total_start) if total_start else 0.0,
    )
    (out_dir / f"{stem}.json").write_text(json.dumps(entry.to_json(), indent=2))
    return entry


def _remix_motivation_grounded(motivation: str, spec: PromptSpec) -> tuple[bool, str]:
    """Verify the remix motivation references at least one dropped AND at
    least one added track — by identifier or natural-name substring,
    case-insensitive. The prompt asks for this explicitly; the validator
    here is the deterministic safety net.
    """
    text = motivation.lower()

    def _name_variants(track_id: str) -> list[str]:
        out = [track_id.lower()]
        natural = (spec.natural_map or {}).get(track_id, "").lower()
        if natural:
            # Strip leading "the " and trailing parentheticals so the
            # surface form is the noun phrase we expect to see in prose.
            stripped = natural
            if stripped.startswith("the "):
                stripped = stripped[4:]
            out.append(stripped)
        return out

    dropped_ids = [c["arguments"]["track"] for c in spec.forced_calls
                   if c["name"] == "remove_track"]
    added_ids = [c["arguments"]["track_name"] for c in spec.forced_calls
                 if c["name"] == "add_track"]

    def _any_in_text(ids: list[str]) -> bool:
        for tid in ids:
            for variant in _name_variants(tid):
                if variant and variant in text:
                    return True
        return False

    if dropped_ids and not _any_in_text(dropped_ids):
        return False, f"motivation doesn't reference any dropped track ({dropped_ids})"
    if added_ids and not _any_in_text(added_ids):
        return False, f"motivation doesn't reference any added track ({added_ids})"
    return True, ""


def _matches_forced_call(forced: dict, tool_calls: list[dict]) -> bool:
    """Check whether ``forced`` appears in ``tool_calls``.

    Match policy by tool:
      * ``remove_track``        → same ``track``.
      * ``change_instrument``   → same ``track`` AND same ``to_program``.
      * ``add_track``           → same ``track_name`` AND same ``program``.
    Any extra key on the LLM call (e.g. ``gain_db`` defaulted) is OK.
    """
    fname = forced["name"]
    fargs = forced["arguments"]
    for tc in tool_calls:
        if tc.get("name") != fname:
            continue
        a = tc.get("arguments", {}) or {}
        if fname == "remove_track":
            if a.get("track") == fargs["track"] or a.get("track_name") == fargs["track"]:
                return True
        elif fname == "change_instrument":
            t_match = (a.get("track") == fargs["track"]
                       or a.get("track_name") == fargs["track"])
            if t_match and int(a.get("to_program", -1)) == int(fargs["to_program"]):
                return True
        elif fname == "add_track":
            t_match = (a.get("track_name") == fargs["track_name"]
                       or a.get("track") == fargs["track_name"])
            if t_match and int(a.get("program", -1)) == int(fargs["program"]):
                return True
    return False


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
    import time as _time
    _t_total_start = _time.perf_counter()
    _render_sec = 0.0
    _executor_sec = 0.0
    import pretty_midi
    longest = max(pretty_midi.PrettyMIDI(str(track.midi_path(sid))).get_end_time()
                  for sid in track.stems)
    start = max(0.0, (longest - duration_sec) / 2.0)

    # Honor the pre-committed arrangement_add intent by withholding a track if
    # the caller didn't already supply one. Prefer note-rich tracks so the
    # added stem audibly changes the mix. Count notes INSIDE the 30s window
    # (not the whole-track note count) because a track with 500 total notes
    # might have only 3 in our window.
    #
    # Remix uses a sibling mechanism: the plan was committed at sample-time
    # and named ⌊N/2⌋ tracks for the removed_set. Reuse the same withhold
    # path so those tracks land in ``state.pending_tracks`` and surface as
    # ``available_to_add`` in the metadata seen by the LLM.
    if spec.primary_intent == "remix" and not withhold_for_add and spec.plan:
        from procraft_data.pipeline.trace_prompts import thaw_plan
        plan_dict = thaw_plan(spec.plan)
        removed = list(plan_dict.get("removed_set", []))
        if removed:
            withhold_for_add = removed

    if spec.primary_intent == "arrangement_add" and not withhold_for_add:
        import random as _rnd
        import pretty_midi as _pm
        from procraft_data.sources.slakh import _dedup_names
        rng = _rnd.Random(hash((track.track_id, entry_idx)))
        name_by_sid = _dedup_names(list(track.stems.values()))

        # How many tracks to withhold — set by sample_primary_intent at
        # spec-build time, defaults to 1 if absent (back-compat).
        plan_d = dict(spec.plan) if spec.plan else {}
        n_targets = int(plan_d.get("n_targets", 1))

        # Count notes per stem in the window.
        note_counts: list[tuple[str, int]] = []
        for sid, name in name_by_sid.items():
            try:
                midi = _pm.PrettyMIDI(str(track.midi_path(sid)))
                instr = midi.instruments[0] if midi.instruments else None
                if instr is None:
                    continue
                nc = sum(1 for note in instr.notes
                         if note.end > start and note.start < start + duration_sec)
                note_counts.append((name, nc))
            except Exception:
                continue

        if len(note_counts) >= 3:
            # Top-N by note-count, biased toward tracks that actually
            # contribute material in the window.
            sorted_by_notes = sorted(note_counts, key=lambda x: x[1], reverse=True)
            # Filter out near-empty stems unless that would leave us with
            # too few candidates.
            non_trivial = [(n, c) for (n, c) in sorted_by_notes if c >= 6]
            pool = non_trivial if len(non_trivial) >= n_targets else sorted_by_notes
            # Sample top-half (note-rich half) then take the requested count.
            top_half = pool[: max(n_targets, len(pool) // 2)]
            rng.shuffle(top_half)
            withhold_for_add = [n for n, _ in top_half[:n_targets]]

    state = build_mixture_state(
        track, sample_rate, start, duration_sec,
        withhold=withhold_for_add,
    )
    _t = _time.perf_counter()
    for ts in state.tracks.values():
        ts.audio = renderer.render_track(ts, duration_sec)
    _render_sec += _time.perf_counter() - _t
    # Snapshot the pre-execution mixdown — this IS the original audio we
    # store. For arrangement_add the state has the target track withheld
    # (pending_tracks, not state.tracks), so this snapshot is the
    # *incomplete* mix; after add_track executes, modified_mix will contain
    # the complete mix with the new stem. For every other intent the state
    # already has all tracks, so original vs modified differ only by the
    # tool's effect. Copy because _mixdown returns a fresh array but we
    # want to be explicit.
    snapshot_before = _mixdown(state).copy() if state.tracks else np.zeros((2, 1), dtype=np.float32)
    # Same snapshot moment, structured form: list of instruments actually
    # present in original_audio. See DatasetEntry.pre_instruments docstring.
    pre_instruments = _capture_instruments(state, track)
    # Same snapshot moment, MIDI form: bundled multitrack PrettyMIDI describing
    # the *content* (pitch + rhythm) underneath ``original_audio``. Used as the
    # supervision target for the content branch (proposal §4.5 / §4.9).
    pre_midi = _state_to_midi(state)
    # Capture pre-execution metadata — tool calls mutate state in place.
    input_metadata = describe_mixture(state, track)

    # Rebuild the prompt with the real post-withhold metadata so the system
    # prompt agrees with the state the executors will see (especially for
    # arrangement_add, which depends on 'available_to_add' track names).
    # All role/abstraction/hook/intent fields are preserved verbatim.
    from procraft_data.pipeline.trace_prompts import (
        IntentCommitment, make_remix_intent, sample_primary_intent,
    )

    if spec.primary_intent == "arrangement_add" and withhold_for_add:
        # Build N forced add_track calls — one per withheld track. Use each
        # track's original GM program (drum stems get the schema-valid
        # placeholder ``program=0``; drum routing is keyed off ``is_drum``,
        # not ``program``). state.pending_tracks already contains the
        # withheld TrackStates.
        forced_add: list[dict] = []
        add_lines: list[str] = []
        for name in withhold_for_add:
            ts = state.pending_tracks.get(name)
            if ts is None:
                continue
            prog_arg = 0 if ts.is_drum else max(0, min(127, int(ts.program)))
            forced_add.append({
                "name": "add_track",
                "arguments": {"track_name": name, "program": prog_arg},
            })
            add_lines.append(
                f"  - add_track {{track_name: '{name}', program: {prog_arg}}}"
            )
        n_forced_add = len(forced_add)
        forced_block = "\n".join(add_lines)
        if n_forced_add == 1:
            description = (
                f"The main move is to introduce the withheld track "
                f"'{withhold_for_add[0]}' using add_track. Emit this call "
                f"verbatim:\n{forced_block}\n"
                "Your motivation should describe what it adds to the "
                "mixture using additive verbs ('introduce', 'bring in', "
                "'pad in', 'layer in', 'add a', 'drop in') — NEVER "
                "restoration words ('missing', 'bring back', 'restore', "
                "'return', 'reintroduce'). Pretend the track never existed "
                "before — describe the CONTRIBUTION (pad, lift, grit, "
                "warmth, counter-melody, low-end weight)."
            )
        else:
            description = (
                f"The main move is to introduce {n_forced_add} new "
                "elements to the mixture. Emit each call below verbatim:\n"
                f"{forced_block}\n"
                "Your motivation MUST use additive verbs like 'introduce', "
                "'bring in', 'pad in', 'layer in', 'add a', 'drop in' — "
                "and must NEVER use restoration words: no 'missing', "
                "'lacking', 'absent', 'empty spot', 'bring back', "
                "'restore', 'return', 'reintroduce', 'put back'. Pretend "
                "the tracks never existed before — describe the COMBINED "
                "CONTRIBUTION across the new additions."
            )
        pinned_intent = IntentCommitment(
            intent="arrangement_add",
            primary_tool="add_track",
            target_track=", ".join(withhold_for_add),
            target_program=None,
            description=description,
            forced_calls=tuple(forced_add),
        )
    elif spec.primary_intent == "remix" and spec.plan:
        # Reuse the original plan — re-sampling here would re-randomize
        # the partition + decisions and discard the per-entry plan that's
        # already baked into ``spec.forced_calls``.
        pinned_intent = make_remix_intent(spec.plan)
    elif spec.primary_intent == "extract_track":
        # Preserve the sample-time target_track; the prompt's only
        # metadata-dependent piece is the `tracks: ...` list at the top
        # of the system block, which gets refreshed by build_spec below.
        pinned_intent = IntentCommitment(
            intent="extract_track",
            primary_tool="extract_track_op",
            target_track=spec.target_track,
            target_program=None,
            description=(
                f"The producer is extracting '{spec.target_track}' from the "
                f"full mix and presenting it in isolation. Your motivation "
                f"should describe WHY soloing '{spec.target_track}' is "
                f"useful — what the listener gains by hearing this stem "
                f"alone (its line, groove, timbre, role in the "
                f"arrangement). Do NOT propose any production changes; "
                f"this turn is purely about isolating the existing track."
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
            tr.motivation_text, tr.tool_calls, spec
        )
        if valid or attempt > max_retries:
            break
        retry_reasons.append(f"attempt {attempt}: {why}")

    # ------------------------------------------------------------------
    # extract_track short-circuit: deterministic stem solo, no executors
    # ------------------------------------------------------------------
    if spec.primary_intent == "extract_track":
        return _finalize_extract_track(
            track=track, spec=spec, tr=tr, state=state,
            snapshot_before=snapshot_before, pre_instruments=pre_instruments,
            pre_midi=pre_midi, input_metadata=input_metadata,
            attempt=attempt, retry_reasons=retry_reasons,
            entry_idx=entry_idx, out_dir=out_dir, sample_rate=sample_rate,
            render_sec=_render_sec, total_start=_t_total_start,
        )

    errors: list[str] = []
    executed = 0
    tool_effects: list[dict] = []
    _exec_start = _time.perf_counter()
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
    _executor_sec += _time.perf_counter() - _exec_start
    executed_ok = executed > 0 and executed == len(tr.tool_calls)

    # Original = pre-execution snapshot (see comment above). Modified =
    # post-execution mixdown. For arrangement_add, this yields the proposal
    # §3.2 Category D pair: input=incomplete, output=complete.
    original_mix = snapshot_before
    modified_mix = _mixdown(state) if state.tracks else np.zeros_like(snapshot_before)
    post_instruments = _capture_instruments(state, track)
    post_midi = _state_to_midi(state)
    mix_rms_original = _safe_rms(original_mix)
    mix_rms_modified = _safe_rms(modified_mix)

    stem = f"entry_{track.track_id}_{spec.category_focus}_{entry_idx:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    orig_path = out_dir / f"{stem}_original.wav"
    mod_path = out_dir / f"{stem}_modified.wav"
    orig_midi_path = out_dir / f"{stem}_original.mid"
    mod_midi_path = out_dir / f"{stem}_modified.mid"
    original_out, modified_out, _ = _shared_peak_normalize(original_mix, modified_mix)
    sf.write(orig_path, original_out.T, sample_rate)
    sf.write(mod_path, modified_out.T, sample_rate)
    pre_midi.write(str(orig_midi_path))
    post_midi.write(str(mod_midi_path))

    if spec.forced_calls:
        # Remix: the "primary move" is the union of forced calls.
        primary_move_executed = all(
            _matches_forced_call(fc, tr.tool_calls)
            for fc in spec.forced_calls
        )
    else:
        primary_move_executed = any(
            tc.get("name") == spec.primary_tool for tc in tr.tool_calls
        )

    # Final validity check — if retry loop gave up, mark executed_ok=False so
    # the downstream quality filter drops this entry.
    final_valid, final_why = _result_is_valid(
        tr.motivation_text, tr.tool_calls, spec
    )
    if not final_valid:
        errors.append(f"final-attempt invalid: {final_why}")
        executed_ok = False

    # Remix-specific post-execution validator: enforce the "no kept-set
    # original program survives in the modified mix" invariant. The forced
    # calls are designed to satisfy this by construction (every kept-set
    # track is dropped or swapped to a non-colliding program), but a
    # secondary apply_fx call shouldn't be able to subvert it. If a
    # kept-set program slipped through (e.g. an LLM-emitted layer_instrument
    # with a kept-set program — which the prompt forbids), flag the entry
    # rather than silently shipping it.
    if spec.primary_intent == "remix" and spec.plan:
        from procraft_data.pipeline.trace_prompts import thaw_plan
        plan_dict = thaw_plan(spec.plan)
        kept_programs = set(plan_dict.get("kept_programs", set()))
        live_programs = {int(rec["program"]) for rec in post_instruments}
        survived = kept_programs & live_programs
        if survived:
            errors.append(
                f"remix-invariant violated: kept-set programs still in mix: "
                f"{sorted(survived)}"
            )
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
        pre_instruments=pre_instruments,
        post_instruments=post_instruments,
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
        original_midi=orig_midi_path.name,
        modified_midi=mod_midi_path.name,
        llm_sec=tr.latency_sec,
        render_sec=_render_sec,
        executor_sec=_executor_sec,
        total_sec=_time.perf_counter() - _t_total_start,
    )
    (out_dir / f"{stem}.json").write_text(json.dumps(entry.to_json(), indent=2))
    return entry


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
