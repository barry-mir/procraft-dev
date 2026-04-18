"""Parse Qwen3-Thinking output: separate <think>, motivation, and <tool_call>s."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# Qwen3-Thinking sometimes emits only the closing </think> without the open tag
# (see model card: "Output may contain only </think> without an explicit opening
# <think> tag (normal behavior)"). Everything before the first </think> is still
# the reasoning trace.
_OPEN_THINK_MISSING_RE = re.compile(r"^(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass
class ParsedResponse:
    think: str | None
    motivation: str
    tool_calls: list[dict]     # each dict: {"name": str, "arguments": dict}
    raw: str

    def is_valid(self) -> bool:
        return bool(self.tool_calls) and all(
            isinstance(tc.get("name"), str) and isinstance(tc.get("arguments"), dict)
            for tc in self.tool_calls
        )


def parse_response(text: str) -> ParsedResponse:
    think_match = _THINK_RE.search(text)
    if think_match:
        think = think_match.group(1).strip()
        body = _THINK_RE.sub("", text, count=1)
    else:
        # Opening <think> missing but closing present — treat prefix as reasoning.
        alt = _OPEN_THINK_MISSING_RE.search(text)
        if alt:
            think = alt.group(1).strip()
            body = text[alt.end():]
        else:
            think = None
            body = text

    tool_calls: list[dict] = []
    for m in _TOOL_CALL_RE.finditer(body):
        try:
            tool_calls.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            continue

    motivation = _TOOL_CALL_RE.sub("", body).strip()
    return ParsedResponse(think=think, motivation=motivation, tool_calls=tool_calls, raw=text)
