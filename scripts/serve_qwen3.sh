#!/usr/bin/env bash
# Start vLLM serving Qwen3-30B-A3B-Thinking-2507 AWQ-4bit on cuda:0 (the 3090, 24 GB).
#
# Default model is cpatonn's community AWQ-4bit (16.9 GB footprint). Native 262K
# context is overkill for our ~5KB system prompts — we cap at 16K to leave room
# for KV cache and concurrent batched generation.
#
# Tool parsing: the --tool-call-parser hermes flag makes vLLM emit parsed
# tool_calls in the OpenAI response structure, but we ignore that and parse the
# raw `<tool_call>…</tool_call>` text ourselves (keeps one parser path across
# offline experiments and any non-vLLM backend).
#
# Reasoning parsing: --reasoning-parser qwen3 surfaces the <think>…</think>
# span in `reasoning_content`. (vLLM 0.19 ships a dedicated qwen3 parser; earlier
# guides suggested deepseek_r1, but the native qwen3 one matches the exact
# template Qwen3-Thinking emits.) Our client re-inlines it if needed.

set -euo pipefail

# Pin the 3090 deterministically. UUID-form CUDA_VISIBLE_DEVICES breaks vLLM
# (it tries to int() the identifier), so we force PCI_BUS_ID ordering and pick
# index 1 — under PCI_BUS_ID the 3080 (bus 03) is 0 and the 3090 (bus 09) is 1.
# This matches vLLM's own startup warning recommendation.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HOME="${HF_HOME:-/nas/pro-craft/cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/nas/pro-craft/cache/hf/hub}"

MODEL="${PROCRAFT_QWEN_MODEL:-cpatonn/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit}"
PORT="${PROCRAFT_VLLM_PORT:-8765}"
MAX_LEN="${PROCRAFT_VLLM_MAX_LEN:-32768}"
GPU_UTIL="${PROCRAFT_VLLM_GPU_UTIL:-0.92}"

echo "Serving $MODEL on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES port=$PORT"
echo "HF_HOME=$HF_HOME"

# We bake the tool schema into the system prompt (Hermes-style) rather than
# passing `tools=[...]` in each request. vLLM's auto-tool-choice path only
# surfaces tool_calls when the request carries `tools`; since we don't, we
# disable it and let the model's raw `<tool_call>...</tool_call>` tags pass
# through in `content` for our own parser to handle. Reasoning parser still
# splits `<think>` into `reasoning_content`.
#
# --enforce-eager disables CUDA graph capture. Graph capture at the full
# max_model_len on a 24 GB card was taking 15+ min of apparent hang; eager
# execution starts in ~2 min and is plenty fast for batch-size 1 dataset gen.
# NO reasoning-parser: vLLM 0.19's qwen3 parser silently drops the <think>
# span for this AWQ build (completion_tokens reports 1-2K tokens but
# reasoning_content comes back empty). Running without it keeps the full
# <think>…</think> in content; our parser extracts it downstream.
#
# --max-num-seqs 6  lets vLLM continuous-batch up to 6 concurrent requests,
#                   which is what unlocks the 3090's decode throughput.
# No --enforce-eager  → vLLM captures CUDA graphs on first boot (adds ~10-15
#                   min one-time startup cost; subsequent boots reuse the
#                   compile cache at ~/.cache/vllm).
MAX_NUM_SEQS="${PROCRAFT_VLLM_MAX_NUM_SEQS:-16}"

exec vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --dtype auto
