# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

Repository holds the research proposal `prop_timbre_factorized_003.md` plus the initial data-pipeline scaffold. The proposal is the authoritative spec; when implementing, cite the section you are following.

## Environments and storage

- **Conda env: `pro-craft`** (Python 3.10) — installed with `pip install -e .`, pulls in multiafx (editable from `../multiafx`), pretty_midi, pyFluidSynth (with conda-forge libfluidsynth), torch 2.5.1+cu121, torchaudio, soundfile. Torch is pinned to cu121 because the box's driver is 550.54.14 / CUDA 12.4 — do **not** reinstall a cu13x wheel, it breaks CUDA.
- **GPU indexing** (critical — nvidia-smi and CUDA disagree here):
  - Default `CUDA_DEVICE_ORDER` (FASTEST_FIRST): **`cuda:0` = RTX 3090 (24 GB)**, `cuda:1` = RTX 3080 (10 GB)
  - `CUDA_DEVICE_ORDER=PCI_BUS_ID` swaps them (3080 at bus 03 becomes cuda:0). `nvidia-smi` prints PCI order.
  - Always target `cuda:0` for the 24 GB card unless you explicitly set PCI ordering. The LLM serving env relies on this default.
- **System FluidSynth**: `/usr/share/sounds/sf2/FluidR3_GM.sf2` is already installed and symlinked into `/nas/pro-craft/soundfonts/`. `find_default_soundfont()` in `procraft_data/rendering/fluidsynth_render.py` picks it up automatically.
- **All large data lives on `/nas/pro-craft/`** — never write mixtures/stems/traces to `$HOME`. Path constants are in `configs/paths.py`; override the root via `PROCRAFT_NAS_ROOT`. `HF_HOME` is auto-pointed at `/nas/pro-craft/cache/hf` by importing `configs.paths`, so HuggingFace weights never pollute the home directory.
- **NAS layout** (created, empty except for soundfonts):
  `raw/{slakh2100,gigamidi,lakh_midi,midicaps}`, `soundfonts/`, `rendered/`, `mixtures/`, `traces/`, `modified/`, `dataset/`, `cache/{dac_tokens,pianoroll,hf}`, `models/`, `logs/`.
- **DawDreamer (Tier 1) is not installed.** It needs Python 3.11+ and the box has no commercial VSTs licensed to it, so Tier 2 (FluidSynth + soundfonts) is the only active rendering path right now. If a Tier 1 run becomes necessary, use a separate env — don't upgrade `pro-craft`'s Python.

## Code layout

- `configs/paths.py` — all paths + `SAMPLE_RATE=48000`, `CLIP_SECONDS=30`.

## Qwen3 serving (pro-craft-llm env)

- **Conda env `pro-craft-llm`** (py 3.11, ~11 GB) — `vllm==0.19.0` (brings its own torch). Kept separate from `pro-craft` because vLLM pins its own torch build and drags in ~5 GB of deps that would conflict.
- **Model weights**: `cpatonn/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit` (17 GB, int4 AWQ group-size 32) sitting at `/nas/pro-craft/cache/hf/hub/models--cpatonn--Qwen3-30B-A3B-Thinking-2507-AWQ-4bit/`. HF_HOME is `/nas/pro-craft/cache/hf`.
- **Serve**: `bash scripts/serve_qwen3.sh` — sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=1` to pin the 3090 (bus 09), OpenAI-compatible at `http://127.0.0.1:8000/v1`, 16K max context, `--tool-call-parser hermes --reasoning-parser qwen3 --enable-auto-tool-choice`. The logs land at `/nas/pro-craft/logs/vllm/server.log`.
- **GPU selection gotcha**: UUID-form `CUDA_VISIBLE_DEVICES=GPU-…` works for PyTorch but **breaks vLLM** (it tries to `int()` the identifier). Use PCI_BUS_ID ordering + numeric index. Under PCI_BUS_ID the 3090 is `cuda:1`; under the default `FASTEST_FIRST` it would be `cuda:0` — the serve script explicitly sets PCI_BUS_ID so those two conventions can't collide.
- **Parser choice gotcha**: vLLM 0.19 has both `deepseek_r1` and the native `qwen3` reasoning parsers — use `qwen3`, it matches Qwen3-Thinking's template exactly. Older guides say `deepseek_r1`; that's only correct for DeepSeek-R1-style models.
- **Client**: `procraft_data.pipeline.trace_client.VLLMClient` re-inlines vLLM's OpenAI-format `tool_calls`/`reasoning_content` back into raw `<tool_call>`/`<think>` text so downstream parsing has one path. If you swap to a non-vLLM backend (llama.cpp, raw HF), no parser changes needed.

## Sample rate policy (48 kHz)

The whole data pipeline — FluidSynth rendering, stored mix.wav, every tool executor, future Qwen3 target audio — runs at **48 kHz stereo**. The proposal was updated to match (§3.1.1, §3.3, §4.2, §4.10). This overrides the original 44.1 kHz spec.

**DAC is still 44.1 kHz-native.** Because the public DAC checkpoint was trained at 44.1 kHz, audio must be resampled 48 → 44.1 kHz immediately before DAC encoding. Do this **GPU-side** with `torchaudio.functional.resample(..., 48000, 44100)` rather than pre-resampling to disk, so both sample rates coexist:
- disk / mix.wav / stems / modified audio → 48 kHz
- DAC token frames `z_t` → derived from the 44.1 kHz resampled view
- DAC decoder output → 44.1 kHz, resample back to 48 kHz before saving/evaluating if needed

If you later switch to a 48 kHz codec (e.g., a retrained DAC or Encodec-48k), the resample calls can just be removed — no other pipeline changes needed.
- `procraft_data/tools/schemas.py` — builds the Hermes system prompt Qwen3-30B-A3B-Thinking sees. The 10 tool names here are the **only** things the model is allowed to call; `apply_fx` uses a `oneOf` over all 85 MultiAFX effects so every parameter is typed. Changing an executor signature requires changing the matching schema, and vice versa (`tests/test_schemas_and_executors.py::test_schema_executor_parity` enforces this).
- `procraft_data/tools/executors.py` — Python implementations for those 10 tools. All operate on a mutable `MixtureState` (per-track `TrackState` with `pretty_midi.Instrument` + stereo float32 stem). Category D's `add_track` consumes from `state.pending_tracks` — Slakh/GigaMIDI loaders must pre-populate those when they withhold a track from the input mix (proposal §3.2 Category D implementation).
- `procraft_data/tools/parse.py` — extracts `<think>`, motivation, and `<tool_call>` JSON blocks from Qwen3 responses. Malformed JSON in a single block is dropped, not fatal.
- `procraft_data/rendering/fluidsynth_render.py` — `FluidSynthRenderer` owns a persistent `fluidsynth.Synth` per worker (libfluidsynth is **not fork-safe after sfload**, so instantiate inside each multiprocessing worker, never in the parent).
- `procraft_data/sources/slakh.py` — BabySlakh / Slakh2100 loader. Yields `TrackMeta` with per-stem `StemMeta` (inst_class, GM program, is_drum). **BabySlakh metadata.yaml ships with `midi_saved: false` for every stem even though the .mid files exist** — the loader trusts filesystem presence, not the flag. For drums Slakh writes `program_num: 128` as a sentinel; the renderer maps that to `bank=128 preset=0` in FluidR3_GM.sf2. Track names are deduplicated by `inst_class` slug (e.g., three `"Guitar"` stems become `guitar_1/guitar_2/guitar_3`); this is the same name the tool-call `track` field uses.
- `scripts/dump_tool_schemas.py` — `--json` prints the full ~38KB system prompt Qwen3 will receive.
- `scripts/smoke_test_pipeline.py` — synthetic 2-track MIDI smoke test (no Slakh dependency).
- `scripts/ingest_slakh_track.py` — render one Slakh track's 10s center-cropped window to `/nas/pro-craft/rendered/slakh/<TrackID>/` (stems + mix + meta.json).
- `scripts/apply_tool_calls_to_track.py` — full data-pair path: Slakh MIDI → render → canned Hermes response → executors → `original.wav` + `modified.wav` + `trace.json` under `/nas/pro-craft/modified/slakh/`. Accept real Qwen3 output via `--response file.txt`.

## Validation status

- BabySlakh (20 tracks, 883 MB) downloaded + extracted to `/nas/pro-craft/raw/slakh2100/babyslakh_16k/`.
- All 20 tracks render end-to-end in **~27s total** (~1.3s per 10s/multi-stem clip), zero failures. Tracks have 7–16 stems each. Peak levels vary 0.14–5.73 — peak-normalize at write time as a safety net; the proposal's ITU-R BS.1770-4 loudness normalization is still pending (use pyloudnorm before shipping the dataset).
- Tool-call execution verified on Track00001 with a 3-call chain (`sox_compand` on drums, `sox_equalizer` on mix bus, `ta_treble_biquad` on piano). RMS(diff) ≈ 0.1 vs original, audibly different, no clipping.

## Tool-call contract

- Schemas are emitted in **Hermes format** (`<tools>[…]</tools>` in system, `<tool_call>{…}</tool_call>` from assistant). This is what Qwen3's native chat template expects and what vLLM's `--tool-call-parser hermes` parses.
- `apply_fx` uses MultiAFX's real effect names (`sox_overdrive`, `am_tanh_distortion`, …) rather than pseudo-names like `tape_saturation`. The thinking model reasons directly in terms of what we can execute.
- `INTEGER_PARAMS = {order, numtaps, n_poles}` — these must be integers in schema and round-tripped before hitting multiafx (see `schemas.py` + `multiafx.registry.INTEGER_PARAMS`).

## The Two Deliverables

ProCraft is **two tightly coupled contributions**, not one. Keep them conceptually separate in code layout:

1. **ProCraft-Data** — an offline data-generation pipeline producing `(original_audio, motivation, thinking_trace, tool_calls, modified_audio)` tuples. Runs once (or iteratively) to produce the training corpus.
2. **ProCraft-Model** — a text-conditioned factorized codec trained on ProCraft-Data. Operates on frozen DAC tokens; learns to produce modified DAC tokens from `(original_audio, text_instruction)`.

## ProCraft-Data Pipeline (proposal §3)

Stages, in order:
1. **MIDI sourcing + filtering** — Slakh2100, GigaMIDI (filter: ≥3 tracks, NOMML expressivity threshold), Lakh+MidiCaps.
2. **Per-stem audio rendering** — two tiers that must coexist:
   - **Tier 1 (high-fidelity):** DawDreamer + commercial VSTi (EastWest/Kontakt/Arturia). Python multiprocessing; each worker owns a persistent RenderEngine reading from a shared Queue.
   - **Tier 2 (fast):** FluidSynth + GM soundfonts. Each worker is an independent FluidSynth instance. Used for development and soundfont-diversity augmentation.
   - Mix with ITU-R BS.1770-4 loudness normalization. Keep stems isolated — downstream tools need per-track access.
3. **Motivation + trace generation** — Qwen3-30B-A3B-Thinking via vLLM batching. Tools declared in Hermes-style `<tools>` XML; model emits `<think>...</think>` then `<tool_call>` blocks. Diversity via temperature {0.7, 0.9, 1.1} × style suffix × tool-count {1,2,3,4}.
4. **Tool execution** — four categories, all programmatic:
   - **A — Instrument change** (re-render track MIDI with new preset)
   - **B — Audio effects** (MultiAFX; fall back to pedalboard + librosa/scipy until MultiAFX releases)
   - **C — Performance mods** (pretty_midi: velocity scale, humanize, articulation)
   - **D — Arrangement** (add/remove/double/mute-and-replace tracks; these **change musical content**, unlike A–C)
5. **Quality filter** — drop entries where FAD(original, modified) > 2× batch median (too destructive), < 0.05 (inaudible), or tool execution fails / clips.

Split train/val/test **by track, not by entry** (80/10/10) to prevent leakage.

## ProCraft-Model Architecture (proposal §4)

Everything operates on **frozen DAC** tokens (Kumar et al. NeurIPS 2023; 12 RVQ levels, d_model=1024, T≈860 frames for 10s @ 44.1kHz). DAC encoder/decoder weights never update.

Three semantically named factors:
- **[TMB]** — timbre / instrument identity (supervised by mixture-level GM labels + GRL against pianoroll)
- **[EXP]** — expression (dynamics, effects character, articulation); defined residually via **dual GRL** against both instrument classifier and pianoroll — no positive target
- **Content (c)** — pitch/rhythm; passes through an **8-dim information bottleneck on the main path** (not a probe) that feeds a pianoroll predictor. The bottleneck is the primary structural disentanglement mechanism, not the GRL.

**Periodic token injection** (ALMTokenizer-style): insert `[TMB]_k` and `[EXP]_k` every w=8 DAC frames. Unified N=12-layer bidirectional transformer processes `[text_prefix | z̃_1..z̃_w [TMB]_1 [EXP]_1 | ...]` with ~20% DAC-token masking (MAE-style). Factor outputs are extracted by position.

**Text conditioning — no reference audio at inference.** The encoder always ingests *original* audio + text. Text changes *which labels supervise* the factor tokens and *which audio is the reconstruction target* — there is no separate "factor generator" or delta-prediction module, and no encoding of modified audio. See §4.6 examples and the §4.9 loss table for the exact label/target mapping per instruction type.

**Key invariants to preserve in any implementation:**
- Timbre labels are **mixture-level**, never single-instrument (e.g., swap piano→violin with bass+drums present → target label is `{violin, bass, drums}`).
- For **content-modifying** operations (add/remove track), the content pianoroll supervision target also changes to the modified pianoroll.
- ALL supervision losses are active in ALL modes — text only changes label/target, never gates a loss.
- GRL λ warmup is linear 0 → 0.1 over first 30K steps.

**Decoder:** per-window, M=8 transformer layers with cross-attention from content → factor tokens, output head predicts 12 RVQ indices (softmax over codebook size 1024). Indices go to frozen DAC decoder.

## Training Curriculum (proposal §4.8)

300K steps total, single A100-80GB, BF16, batch 32. Text-prompt probability schedule — do not shortcut this:
- Phase 1 (0–90K): 0% text, reconstruction-first, GRL warmup
- Phase 2 (90K–150K): linear 0% → 50% text
- Phase 3 (150K–240K): 50% text, full joint
- Phase 4 (240K–300K): 80% text, GRL λ decayed 0.5×

Reconstruction-first is load-bearing — early text conditioning lets the model use text as a shortcut around disentanglement.

## Go/No-Go Gate (proposal §7)

Before committing to the 300K-step run, train 50K steps on 5K pairs and require:
- Content Preservation Rate (piano-roll F1, ±50ms) ≥ 0.80
- Instrument Classification Accuracy ≥ 0.75
- FAD ≤ 2× the no-disentanglement baseline

If CPR < 0.80, check whether the pianoroll probe loss is starving reconstruction — halve the probe weight before other changes.

## Evaluation (proposal §6)

Primary metrics: FAD, KAD, Content Preservation Rate, Instrument Classification Accuracy, Effects Fidelity. MOS is supplementary only. Two test sets: 1,050 in-distribution Slakh pairs + 100 out-of-distribution MUSDB18-HQ pairs (the OOD set checks that the model isn't Slakh-rendered-audio-specific).

## Watch-Outs

- **MultiAFX is not yet released.** Effects Category B must ship with a pedalboard + librosa/scipy fallback implementation; the dataset's tool-call schemas should still reference all 85 effects so the corpus is forward-compatible.
- **Pedalboard** as a DawDreamer alternative: VST3-only and has segfault reports at 8K+ files — prefer DawDreamer for Tier 1 scale runs.
- **Neural synthesis** (MIDI-DDSP etc.) is GPU-bound and far too slow for full-corpus rendering; only use as a complement for specific solo instruments if quality demands it.
- **NSynth is not a valid data source** for this project (single notes, not multi-track mixtures). LP-Fx is cited but not a baseline.
