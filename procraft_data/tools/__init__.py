"""Tool definitions and executors.

Proposal §3.2 defines four tool categories that Qwen3-30B-A3B-Thinking can call
during generation; this package exposes both:

1. ``schemas``        — Hermes-format JSON schemas emitted in the system prompt
2. ``executors``      — Python implementations that turn a parsed tool_call dict
                        back into modified audio / MIDI.

The same registry feeds both so the model can never ask for a tool we cannot
execute.
"""
