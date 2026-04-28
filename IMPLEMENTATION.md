# ProCraft-Data Implementation

**Purpose:** the single source of truth for the *currently-implemented*
ProCraft-Data pipeline — methods in use, file map, open items. Reference
material for paper writing. Not a journal: do not add iteration logs,
abandoned approaches, or "fixed in v17" notes.

---

## 1. Project overview

ProCraft-Data pairs `(original_audio, think, motivation, tool_calls, modified_audio)`
per entry, covering the 4 proposal categories (instrument change, audio
effects, arrangement change, plus the new "stem extraction" / "remix"
operations) via 8 primary sub-intents. The motivation text is generated
by Qwen3-30B-A3B-Thinking and is meant to cover the full language space
of anyone who describes a production change — from a non-musician
client's scene metaphor to a mixing engineer's parametric session note.

Source: `prop_timbre_factorized_003.md` (proposal). This document is
authoritative for *what is actually implemented*.

---

## 2. Current pipeline (end-to-end)

```
Slakh MIDI (BabySlakh)
  ↓  procraft_data/sources/slakh.py : load_track + build_mixture_state
MixtureState (per-stem pretty_midi.Instrument + per-stem stereo audio)
  ↓  procraft_data/rendering/fluidsynth_render.py : FluidSynthRenderer
Rendered 30 s stems @ 48 kHz (no pre-edit normalization; MIDI velocity carries balance)
  ↓  sample_primary_intent + sample role × abstraction × hook
prompt built in procraft_data/pipeline/trace_prompts.py::build_spec
  ↓  procraft_data/pipeline/trace_client.py : VLLMClient.complete()
Qwen3-30B-A3B-Thinking (cpatonn/…-AWQ-4bit) on 3090 via vLLM
  ↓  parse <think>…</think>, motivation, <tool_call>…</tool_call> blocks
Post-process: dedup, top-level-FX coercion, ID→natural-name, restorative rewrite
  ↓  executors.EXECUTORS[tool_name] mutate MixtureState
  ↓  re-mixdown, shared peak-normalize, write WAV + MIDI + JSON
/nas/pro-craft/traces/smoke/Track00001/entry_*.{json,
  _original.wav,_modified.wav,_original.mid,_modified.mid}
```

**Concurrency:** `ThreadPoolExecutor(max_workers=4)` in the smoke script;
each worker owns its own `FluidSynthRenderer` (libfluidsynth is not
thread-safe after `sfload`). vLLM runs with `--max-num-seqs 6` to enable
continuous batching. Wall time for 10 entries ≈ 130–160 s.

---

## 3. Active methods

### 3.1 Prompt construction (4 sampled dimensions)
- **role** (13): producer, artist_vocalist, artist_instrumentalist,
  session_{drummer,bassist,guitarist,keyboardist}, non_musician_client,
  mixing_engineer_self_note, mastering_engineer, label_AnR, band_consensus,
  film_tv_music_supervisor.
- **abstraction_level** (6): reference_based, emotional_perceptual,
  scene_metaphor, instrument_specific, technical_parametric, negative_space.
- **hook**: one string from the abstraction level's pool in
  `procraft_data/pipeline/vocab_pools.py`
  (REFERENCE_HOOKS = 167 · EMOTIONAL_HOOKS = 149 · SCENE_METAPHORS = 198 ·
  INSTRUMENT_SPECIFIC_HOOKS = 79 · TECHNICAL_HOOKS = 154 ·
  NEGATIVE_SPACE_HOOKS = 149 — total ≈ 900).
- **primary_intent** (8): instrument_swap, instrument_layer, effects,
  arrangement_{add,remove,mute_replace}, extract_track, remix. Uniformly
  sampled so every run covers all categories evenly. Three intents are
  intentionally excluded from the pool because their audio delta is too
  subtle to anchor a primary motivation; their tools stay registered so
  the LLM can still call them from secondary slots of any other intent:
  ``arrangement_double`` (`double_track`), ``performance_timing``
  (`humanize_timing`), ``performance_articulation``
  (`change_articulation`).
- **tool_count_range** default `(10, 15)` for every abstraction level.
  Multi-target intents whose ``forced_calls`` list exceeds the sampled
  ceiling override the count via
  ``chosen_count = max(sampled_count, n_forced + 1)``. Earlier
  iterations clamped ``technical_parametric`` to `(4, 6)` to keep
  motivation/fx aligned, but the alignment language (below) makes that
  unnecessary — when the hook is a narrow parametric recipe, the model
  is told to promote it into a broader direction the 9-14 secondary
  fx fill out, not to restate the recipe.
- **motivation expansion.** The motivation prompt's clause (e) names
  the chosen count explicitly: "describes the WHOLE production
  direction across the N tool_calls you're about to emit". Narrow
  parametric hooks must be promoted to a broader direction the
  secondary calls elaborate; the motivation can name moves
  individually OR roll them up under a single phrase ("lush stadium
  space, tight low end, wide top").
- **secondary-call alignment.** Each secondary call must elaborate or
  directly support the production direction stated in the motivation —
  the prompt forbids unrelated moves on tracks the motivation doesn't
  reference. For broad aesthetic motivations the prompt asks for
  spread across tracks AND effect families (EQ, dynamics, modulation,
  distortion, reverb, delay) at medium / strong intensity (never light
  / subtle). For narrow parametric motivations the prompt asks
  secondary calls to tighten around the focal move (carve EQ space,
  sidechain, mirror-treat a sister track).

### 3.2 Pre-committed intent target
For most non-`effects` intents, `sample_primary_intent` pre-commits the
target — either a single track + program (e.g. `arrangement_mute_replace`,
`instrument_layer`) or a multi-target plan baked into
`IntentCommitment.forced_calls`. Multi-target intents:

- ``instrument_swap``, ``arrangement_remove``, ``arrangement_add`` —
  pick `N = min(rng.randint(2, 4), TRACK_NUM // 2)` targets at sample
  time. If `TRACK_NUM < 4` the intent falls back to ``effects`` (single-
  target versions of these intents are no longer used). Each target gets
  one forced tool_call; the validator (see §3.3) requires every forced
  call to appear in the LLM's response.
- ``instrument_swap`` swap programs are chosen ≠ that track's own
  original program (so each swap is non-trivial); collisions across
  targets or with kept tracks are not enforced.
- ``arrangement_remove`` biases toward redundant tracks (same
  `inst_class` as another) when there are enough; falls back to non-drum
  pool when the redundant pool is too small.
- ``arrangement_add`` defers track selection to ``generate_one`` (it
  needs MIDI access for note-richness); ``sample_primary_intent`` only
  stashes the target count `N` in `IntentCommitment.plan["n_targets"]`.
  ``generate_one`` picks the top-N note-rich stems for withholding,
  builds N `add_track` forced calls, and rebuilds the spec.

`extract_track` pre-commits a `target_track` (any class — drums OK) but
emits no tool_calls; the pipeline performs the stem solo deterministically
(see §3.11). `remix` pre-commits a partition + per-track decisions in
`IntentCommitment.plan` (see §3.12).

### 3.3 Retry validator
`generate_one` wraps the LLM call in a bounded retry loop (default 3
retries). `_result_is_valid(motivation, tool_calls, spec)` rejects the
attempt — and triggers a retry — when any of these are true:

1. **Empty motivation line.**
2. **Forced calls missing** (multi-target intents, remix). For each
   `spec.forced_calls` entry, ``_matches_forced_call`` looks for a
   matching `name` + key arguments (`track` for remove, `track`+
   `to_program` for swap, `track_name`+`program` for add).
3. **Primary tool missing** (single-target intents). At least one
   `tool_calls` entry must have `name == spec.primary_tool`.
4. **Under-emission.** The user prompt asks for exactly
   `spec.chosen_count` tool_calls; responses with fewer than
   `chosen_count - 2` are rejected (a 2-call slack accommodates
   ``_canonicalize_tool_call`` dedup losses). Empirically the model
   sometimes stops after one or two calls when the primary move feels
   resolved; the densely-treated mix we promise downstream needs the
   full N. ``motivation_only`` intents (``extract_track``) are exempt
   — their `chosen_count` is 0.

`attempt_count` and `retry_reasons` are recorded on every DatasetEntry.
If all retries are exhausted, the last response is saved with
`executed_ok = False` and the reason logged; the downstream quality
filter drops such entries.

### 3.4 Contradiction handling
- **Method 1 (hard compatibility matrix)** — `Role.forbidden_abstractions`
  blocks known-bad pairs (non_musician × technical/instrument_specific);
  `_roles_for_target` excludes session roles when the pre-committed target
  isn't their instrument.
- **Method 6 (content-aware hook filtering)** — per-role hook pool is
  filtered before sampling. Non-musician / artist_vocalist roles have all
  technical hooks (matching `_TECHNICAL_HOOK_KEYWORDS`) removed from their
  abstraction level's pool, so a `non_musician_client × technical_parametric`
  prompt can't grab "target -14 LUFS integrated" as a hook.
- **Method 3 (detect + resample)** — after filtered sampling, if the hook
  still conflicts with the role (`_hook_conflicts_with_role`), retry up to
  5× with resampling of only the hook dimension.
- Deferred: Method 2 (soft scoring), 4 (trust LLM), 5 (joint pre-sampling),
  7 (post-hoc quality filter), 8 (hierarchical sampling).

### 3.5 Natural-name hygiene
- `describe_mixture` advertises each track as
  `<identifier> "<midi_program_name>" (GM N)`.
- `_natural_names_from_metadata` builds a per-entry mapping
  `guitar_1 → "the first distortion guitar"`, handling duplicates with
  ordinals (first / second / third).
- System prompt hard rule #4 forbids raw identifiers in the motivation
  sentence and tells the model to use the quoted program name. The same
  rule also forbids GM program numbers in motivation prose: no
  ``GM 85``, ``program 33``, ``to_program: 33`` — the model is told
  to paraphrase into what the new instrument *sounds* like, not its
  number.
- `_clean_motivation` (post-process) is the deterministic safety net
  for both rules:
    - **Identifier substitution.** Leaked identifier (e.g.
      ``guitar_2``) is replaced with its natural phrase; leading "the"
      case preserved. Single-word English identifiers (`piano`, `bass`,
      `drums`, `organ`, `guitar`, `strings`) are skipped to avoid
      doubled noun phrases.
    - **GM-number substitution.** ``"GM 85"`` / ``"program 33"`` /
      ``"to_program: 33"`` are replaced with the GM program name from
      ``pretty_midi.program_to_instrument_name``. ``"(GM 85)"``
      parentheticals are dropped entirely (the surrounding word
      typically already names the instrument).

### 3.6 arrangement_add framing enforcement
System prompt forbids restorative words ("missing", "bring back",
"restore", "reintroduce", "absent", "lacking"). If the model slips,
`_rewrite_restoration_words` post-processes the motivation text to swap in
additive phrasing ("is missing" → "would benefit from", "bringing back" →
"introducing").

### 3.7 Tool-call post-processing
1. `_coerce_top_level_fx` — if the model emits a multiafx effect name as a
   top-level tool (e.g., `{"name": "npy_stereo_widener", "arguments": {…}}`),
   rewrite it into a proper `apply_fx` envelope.
2. `_canonicalize_tool_call` + dedup — round floats to 3 dp, sort keys,
   reject exact duplicates (Qwen3 emits duplicates ~20% of the time).
3. `_coerce_fx_params` — case-insensitive param name coercion for multiafx
   effects (handles `Q` vs `q` cross-contamination).
4. `_get_track_arg` — accept either `track` or `track_name` in any tool arg
   (schemas use different keys; model confuses them).

### 3.8 Level-preserving effects (RMS-match)
`_LEVEL_PRESERVING_EFFECTS = {sox_overdrive, ta_overdrive,
am_tanh_distortion, am_clipping_distortion, am_bit_crush, sox_contrast,
ta_contrast, sox_compand}`. For these effects, after applying in
`apply_fx`, the output is RMS-matched to the input so perceived loudness
is preserved while character changes come through.

The distortion family is in here because the DSP implementations push
output level up substantially (~+6-12 dB on overdrive); a producer
asking for "some grit" doesn't expect the track to also get +10 dB.

`sox_compand` is a special case: SoX's compand falls back to its
default transfer function ``[(-70,-70), (-60,-20), (0,0)]`` whenever
no `tf_points` argument is given (which is how multiafx wraps it),
applying ~+6 dB of free makeup at typical signal levels. Producers
calling compand expect dynamics shaping ("gain reduction"), not free
gain. RMS-matching the output kills the unintended boost while keeping
the dynamics character.

Gain / loudness-normalization / vol effects are deliberately excluded
— those exist specifically to change level.

### 3.9 Humanize timing (secondary tool)
Reachable from any intent's secondary tool_call slots — not a primary
intent (the audio delta is too subtle to anchor a motivation). Gaussian
jitter `σ = max_offset_ms/2`, clipped to ±max_offset_ms. Per-note seed
from `time.time_ns() ^ id(args) ^ id(state)` so repeated calls produce
independent takes. Coupled velocity jitter `1 + N(0, 0.08)` clipped to
[0.6, 1.25] — timing + velocity co-vary the way real human performance
does.

### 3.10 Per-entry artifacts (audio + MIDI + structured instrument lists)
Every entry persists five files plus the JSON trace. The audio and the
MIDI use the same two-snapshot semantics so original/modified pairs are
mutually consistent across modalities.

**Audio.** `original.wav` is captured with `snapshot_before =
_mixdown(state).copy()` immediately *before* any tool_call executes.
`modified.wav` is captured with `_mixdown(state)` *after* every tool_call
executes. Both are 48 kHz stereo 30 s clips. For `arrangement_add` the
withheld target lives in `state.pending_tracks` at the *before* moment, so
`original.wav` is the incomplete mix the proposal §3.2 Category D requires
as input; the executor moves the pending entry into `state.tracks` before
the *after* moment, so `modified.wav` is the complete mix.

**MIDI.** `original.mid` and `modified.mid` are written via
`_state_to_midi(state)` at the same two moments. Each `TrackState` becomes
a separate `Instrument` (with `name` set to the track identifier so MIDI
tracks can be matched back to their audio stem and metadata record).
These are the supervision target for the content branch (proposal §4.5 /
§4.9 — pianoroll predictor under the 8-dim content bottleneck). Content
operations (`arrangement_add`, `arrangement_remove`, `mute_and_replace`,
`extract_track`, `remix`) change the modified MIDI in addition to
changing the audio; non-content operations (`effects`,
`instrument_swap`, `instrument_layer`, and any secondary
`humanize_timing` / `change_articulation` / `double_track`) leave the
modified MIDI structurally identical to the original except for any
executor-driven mutations (timing / articulation / program number /
duplicated stems).

**Structured instrument lists.** `_capture_instruments(state, track)` is
called at the same two moments and produces `pre_instruments` /
`post_instruments` on the JSON trace. Each record is `{name, program,
is_drum, midi_program_name, inst_class}`. Slakh metadata is filled in for
stems that map back to a `StemMeta`; synthetic tracks introduced by
`layer_instrument` (`<base>__layer<prog>`) or `double_track` (`<base>__dbl`)
fall back to `pretty_midi.program_to_instrument_name`. These lists are
the source for [TMB] supervision labels — downstream code can dedupe by
program if it wants set-level mixture labels.

### 3.11 extract_track — deterministic stem solo
This intent does not call any tool. `sample_primary_intent` picks one
track (any class — drums OK). The LLM is given a stripped-down system
prompt (`EXTRACT_TRACK_PREAMBLE` — no `<tools>` block, no `<think>`
requirement) and a tiny instruction-style user prompt: produce a 5-to-
10-word imperative ("Extract the piano." / "Solo the bass track." /
"Pull the drums out of the mix.") with verb variation (extract / pull
out / solo / isolate / separate / lift / take out / single out) and a
paraphrased natural name (e.g. `piano` → `keyboard track`). The prompt
explicitly forbids "why" clauses, ``<think>`` blocks, or any text
beyond the imperative. role / abstraction_level / hook fields are
still sampled and recorded on the spec for diversity bookkeeping but
don't shape this text. `generate_one` short-circuits at
`_finalize_extract_track`: `original_audio` is the full mix;
`modified_audio` is the target track's per-stem audio rendered alone;
`pre_instruments` is the full mixture; `post_instruments` is
`[the one extracted track]`; `tool_calls = []`.

### 3.12 remix — kept-set drop/swap + add-back from removed-set
`_plan_remix` runs at sample time on the FULL mix metadata:

1. Skip when `N < 4` (fall back to `effects`).
2. Partition: `removed_set` (size `K = ⌊N/2⌋`) and `kept_set` (rest).
3. Per kept-set track, randomly pick "drop" or "swap" (50/50), with two
   guarantees: (a) drum kits always drop — `change_instrument` keeps
   `is_drum=True` and a melodic GM program on channel 10 produces
   nonsense; (b) the entry has at least one drop AND at least one swap
   (forced by flipping a non-drum track if either action is missing).
4. Swap target programs are sampled in [0, 95] excluding the kept-set's
   original programs (collision with removed-set programs is OK; that's
   the C-γ contract).
5. Add-back picks: ⌈K/2⌉ from `removed_set`. Add-back program defaults to
   the track's original program except (a) drum tracks report the
   schema-valid placeholder `program=0` (drum routing is controlled by
   `is_drum`, not `program`); (b) if the original program collides with
   `kept_programs`, a fresh non-colliding program is sampled.

The plan is stashed on `IntentCommitment.plan` (frozen tuple form) and
the resolved tool_calls are stashed on `IntentCommitment.forced_calls`.
`generate_one` reads `spec.plan["removed_set"]` to pick the withhold list
*before* rendering — `original_audio` therefore renders the kept set
only, and the removed set surfaces in the metadata's `available_to_add`
section.

The system prompt's PRIMARY MOVE block lists every forced call
verbatim; the LLM is told to emit each verbatim plus enough free
`apply_fx` calls to reach `chosen_count = max(sampled_count, n_forced + 1)`.
`_result_is_valid(spec)` — used in the retry loop — accepts the response
only when every forced call is matched by a tool_call (matching policy in
`_matches_forced_call`: name + key arguments). Post-execution, an
additional invariant check ensures no kept-set original program survives
in `post_instruments`; violations append to `executed_errors` and flip
`executed_ok=False` (no retry — the prompt's structure guarantees
satisfaction in the happy path).

### 3.13 Shared peak-normalize at output
`_shared_peak_normalize(original, modified, target=0.95)` computes one
scale factor from `max(|original|, |modified|)`; applies to both.
Preserves the relative loudness delta between the pair while preventing
clipping and lifting quiet clips to a usable level. No pre-edit
normalization (MIDI velocity carries balance; Slakh's per-stem LUFS
targets are NOT honored — simplification).

---

## 4. Datasets on disk

- **BabySlakh (16 kHz)** — 20 multitrack songs at
  `/nas/pro-craft/raw/slakh2100/babyslakh_16k/`. Per-stem MIDI + audio.
  Used by ``procraft_data/sources/slakh.py`` and the demo.
- **Lakh MIDI (LMD-full)** — 178,561 raw multi-instrument MIDIs at
  `/nas/pro-craft/raw/lakh_midi/lmd_full/<first-char>/<md5>.mid`,
  deduplicated to **20,797** canonical tracks via
  ``CAugBERT_0.99_with_CLaMP_0.99.json`` (Jeong et al.,
  github.com/jech2/LMD_Deduplication). The list of canonical paths
  lives at ``/nas/pro-craft/raw/lakh_midi/kept_paths.txt``. No loader
  yet — Lakh files are single multi-instrument MIDIs (not per-stem
  like Slakh) and need a thin splitter before they can feed the
  pipeline.

## 5. Open items / next

- Implement DAC tokenization + the factorized codec model (ProCraft-Model,
  proposal §4). Pipeline for this doesn't exist yet; data side is where
  effort has been so far.
- Loudness normalization — ITU-R BS.1770-4 LUFS via pyloudnorm (proposal
  §3.1.1). Currently peak-only at output. Not a blocker.
- FAD-based quality filtering (proposal §3.2 Step 4). Not a blocker.
- Scale from BabySlakh (20 tracks) to full Slakh2100 (2,100 tracks) once
  motivation quality review passes.
- Lakh loader — split each LMD-full MIDI by ``pretty_midi.Instrument``
  into a `MixtureState`-shaped record so the existing pipeline can
  source from Lakh in addition to Slakh.

---

## 6. File map

| File | Role |
|---|---|
| `configs/paths.py` | Path constants, 48 kHz, 30 s clips |
| `procraft_data/sources/slakh.py` | Slakh loader, `describe_mixture`, `natural_names` |
| `procraft_data/rendering/fluidsynth_render.py` | Per-worker stereo FluidSynth renderer |
| `procraft_data/tools/schemas.py` | 9 Hermes tool schemas; apply_fx oneOf over 85 effects |
| `procraft_data/tools/executors.py` | Per-tool mutators on MixtureState; level-preserving RMS-match |
| `procraft_data/tools/parse.py` | `<think>` / `<tool_call>` regex parsing |
| `procraft_data/pipeline/trace_prompts.py` | Role/abstraction/hook/intent sampling + prompt build |
| `procraft_data/pipeline/vocab_pools.py` | Large per-abstraction hook pools (~900 total) |
| `procraft_data/pipeline/trace_client.py` | vLLM HTTP client + tolerant JSON parser |
| `procraft_data/pipeline/generate_traces.py` | Per-entry orchestration; post-process cleaners |
| `scripts/serve_qwen3.sh` | vLLM launch (3090, 32K ctx, `max-num-seqs 6`) |
| `scripts/smoke_qwen_all_motivations.py` | 4-worker smoke run covering all PRIMARY_INTENTS |
| `scripts/build_demo.py` | One-case-per-intent demo page (random tracks per run); MP3 + MIDI sidecars |
| `tests/test_*.py` | Regression tests (14 currently passing) |
