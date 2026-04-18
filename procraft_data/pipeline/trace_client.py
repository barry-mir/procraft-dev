"""Client for the vLLM OpenAI-compatible server.

vLLM is started separately (``scripts/serve_qwen3.sh``). This module just
issues chat completions over HTTP and returns the raw assistant text. Parsing
and tool execution happen downstream in ``procraft_data.tools``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import requests

from procraft_data.pipeline.trace_prompts import PromptSpec


_WELL_FORMED_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_OPEN_TAG_ONLY_RE = re.compile(r"<tool_call>[\s\S]*$")   # dangling partial tag
_OPEN_TAG_RE = re.compile(r"<tool_call>", re.IGNORECASE)

_THINK_PAIR_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_LEADING_CLOSE_THINK_RE = re.compile(r"^(.*?)</think>", re.DOTALL)  # Qwen3 quirk


def _extract_think_span(text: str) -> str | None:
    """Return the model's reasoning span if present (handles Qwen3's missing-open-tag case)."""
    m = _THINK_PAIR_RE.search(text)
    if m:
        return m.group(1).strip() or None
    m = _LEADING_CLOSE_THINK_RE.search(text)
    if m and "</think>" in text:
        return m.group(1).strip() or None
    return None


def _strip_think_and_tool_fragments(text: str) -> str:
    """Remove <think>…</think> (and Qwen's orphan </think>) plus tool_call fragments."""
    text = _THINK_PAIR_RE.sub("", text, count=1)
    # If the open tag was missing but close tag present, drop everything up to and including </think>
    if "</think>" in text:
        text = _LEADING_CLOSE_THINK_RE.sub("", text, count=1)
    text = _WELL_FORMED_CALL_RE.sub("", text)
    text = _OPEN_TAG_ONLY_RE.sub("", text)
    return text


def _scan_json_object(s: str, start: int,
                      end_anchor: str | None = None) -> tuple[int, int] | None:
    """Return (start, end) of a JSON object beginning at ``s[start]``.

    ``end_anchor`` (optional): hard stop — the scan never crosses this literal
    substring. Used to scope the search inside a single ``<tool_call>…</tool_call>``
    block so a missing closing brace doesn't eat the next block.

    If the scan reaches the hard stop (or end-of-string) with depth>0,
    the partial span is returned anyway so the caller can try auto-closing.
    """
    limit = len(s)
    if end_anchor:
        idx = s.find(end_anchor, start)
        if idx != -1:
            limit = idx

    depth = 0
    in_str = False
    esc = False
    for i in range(start, limit):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, i + 1
    if depth > 0:
        return start, limit
    return None


def _parse_tool_call_blocks(text: str) -> list[dict]:
    """Extract Hermes tool calls from raw content with two fallbacks:

    1. `<tool_call>` opener anchors (handles missing `</tool_call>` — vLLM
       sometimes strips the closer).
    2. Bare JSON objects with ``"name"`` + ``"arguments"`` keys (handles the
       model forgetting the wrapper tags entirely).

    The anchor path runs first and claims its spans; leftover JSON objects
    only register if they don't overlap an already-claimed span, so a
    well-formed `<tool_call>{…}</tool_call>` block never double-counts.
    """
    out: list[dict] = []
    claimed: list[tuple[int, int]] = []

    for m in _OPEN_TAG_RE.finditer(text):
        after = m.end()
        while after < len(text) and text[after] not in "{":
            if text[after] not in " \t\r\n":
                after = -1
                break
            after += 1
        if after < 0 or after >= len(text):
            continue
        # Scope scan to the matching </tool_call> (or the next <tool_call> opener
        # if the close tag is missing) so one malformed block doesn't swallow
        # subsequent blocks.
        close_idx = text.find("</tool_call>", after)
        next_open = text.find("<tool_call>", after + 1)
        if close_idx == -1 or (next_open != -1 and next_open < close_idx):
            end_anchor = "<tool_call>"
        else:
            end_anchor = "</tool_call>"
        span = _scan_json_object(text, after, end_anchor=end_anchor)
        if not span:
            continue
        tc = _json_to_tool_call(text[span[0]:span[1]])
        if tc:
            out.append(tc)
            claimed.append(span)

    # Fallback: scan every balanced top-level {...} object and keep the ones
    # that look like tool calls.
    for start in range(len(text)):
        if text[start] != "{":
            continue
        if any(s <= start < e for s, e in claimed):
            continue
        span = _scan_json_object(text, start)
        if not span:
            continue
        if any(s <= span[0] and span[1] <= e for s, e in claimed):
            continue
        tc = _json_to_tool_call(text[span[0]:span[1]])
        if tc:
            out.append(tc)
            claimed.append(span)
    return out


def _json_to_tool_call(s: str) -> dict | None:
    obj = _loads_tolerant(s)
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    args = obj.get("arguments")
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    # Reject nested-payload-only objects like {"call": {...}} emitted as
    # stand-alone JSON — those belong inside an apply_fx envelope.
    return {"name": name, "arguments": args}


def _loads_tolerant(s: str) -> dict | None:
    """``json.loads`` with automatic close-brace / close-bracket repair.

    Qwen3-Thinking occasionally drops one or two closing braces on nested
    apply_fx objects (e.g. emits ``}}}`` when ``}}}}`` is needed). We retry
    parsing while appending up to 4 additional ``}`` before giving up.
    """
    s = s.strip()
    if not s:
        return None
    for extra in range(5):
        try:
            return json.loads(s + "}" * extra)
        except json.JSONDecodeError:
            continue
    # Last resort: try dropping a trailing stray character then closing.
    for trunc in range(1, 4):
        try:
            return json.loads(s[:-trunc] + "}" * 4)
        except (json.JSONDecodeError, IndexError):
            continue
    return None


@dataclass
class TraceResult:
    spec: PromptSpec
    assistant_text: str              # synthesized view (<think>…</think> + motivation + <tool_call>…)
    reasoning_text: str | None       # vLLM's parsed reasoning_content (qwen3 parser output)
    tool_calls: list[dict]           # structured: [{"name": str, "arguments": dict}, …]
    motivation_text: str             # content with tool_calls stripped
    raw_content: str                 # exactly what vLLM put in message.content
    usage: dict
    latency_sec: float


class VLLMClient:
    """Thin wrapper around the vLLM OpenAI endpoint.

    We prefer the raw ``/v1/chat/completions`` path (no tool-call JSON
    flattening) so downstream parsing can trust our own Hermes regex. vLLM
    with ``--reasoning-parser deepseek_r1`` will additionally expose the
    reasoning in a separate ``reasoning_content`` field — we capture both.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8765/v1",
                 model: str = "cpatonn/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit",
                 timeout_sec: float = 360.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec

    def complete(self, spec: PromptSpec) -> TraceResult:
        payload = {
            "model": self.model,
            "messages": spec.as_messages(),
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "max_tokens": spec.max_tokens,
        }
        t0 = time.time()
        r = requests.post(f"{self.base_url}/chat/completions",
                          json=payload, timeout=self.timeout_sec)
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        raw_content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content")   # populated only if server uses reasoning-parser

        # When the server runs without a reasoning-parser the <think>…</think>
        # span lives inside content — extract it ourselves.
        if not reasoning:
            reasoning = _extract_think_span(raw_content)

        # Tool calls arrive as raw Hermes <tool_call>{json}</tool_call> blocks
        # in content (server is launched without --enable-auto-tool-choice).
        tool_calls = _parse_tool_call_blocks(raw_content)
        if not tool_calls:
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args_field = fn.get("arguments")
                if isinstance(args_field, str):
                    try:
                        args = json.loads(args_field) if args_field else {}
                    except json.JSONDecodeError:
                        args = {}
                elif isinstance(args_field, dict):
                    args = args_field
                else:
                    args = {}
                name = fn.get("name") or tc.get("name")
                if name:
                    tool_calls.append({"name": name, "arguments": args})

        # Strip think/tool-call fragments so motivation text is clean.
        motivation = _strip_think_and_tool_fragments(raw_content).strip()

        # Synthesize a canonical assistant_text (<think>…</think> + motivation +
        # <tool_call>…</tool_call> blocks). This is what gets stored in the
        # dataset entry so downstream consumers don't depend on vLLM specifics.
        parts: list[str] = []
        if reasoning:
            parts.append(f"<think>\n{reasoning.strip()}\n</think>")
        if motivation:
            parts.append(motivation)
        for tc in tool_calls:
            parts.append(
                "<tool_call>\n"
                + json.dumps({"name": tc["name"], "arguments": tc["arguments"]})
                + "\n</tool_call>"
            )
        assistant_text = "\n\n".join(parts)

        return TraceResult(
            spec=spec,
            assistant_text=assistant_text,
            reasoning_text=reasoning,
            tool_calls=tool_calls,
            motivation_text=motivation,
            raw_content=raw_content,
            usage=data.get("usage", {}),
            latency_sec=time.time() - t0,
        )

    def wait_ready(self, max_wait_sec: float = 180.0, poll: float = 2.0) -> None:
        """Block until the server answers /v1/models or raise."""
        deadline = time.time() + max_wait_sec
        last_err = None
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.base_url}/models", timeout=5.0)
                if r.ok:
                    return
            except requests.RequestException as e:
                last_err = e
            time.sleep(poll)
        raise RuntimeError(f"vLLM server at {self.base_url} not ready: {last_err}")
