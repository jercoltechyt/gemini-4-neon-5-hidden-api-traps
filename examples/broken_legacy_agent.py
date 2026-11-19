# -*- coding: utf-8 -*-
"""
Example of a legacy Gemini 3 / Gemini 4 agent script migrated naively to `gemini-4-neon`.
Every single trap below triggers HTTP 400 errors, silent ignores, or 2x billing spikes.
(100% offline fixture for `gemini4_migration_audit.py` — never makes network calls.)
"""

from typing import Any


def build_broken_neon_request(client: Any, types: Any, parts: list[Any]) -> Any:
    # Trap #3: Explicit Context Caching is unsupported on gemini-4-neon
    cache = client.caches.create(model="gemini-4-neon", contents=["Large system handbook..."])

    # Trap #1: thinking_budget + MINIMAL return HTTP 400
    # Trap #4: temperature silently ignored (or 400 out-of-range), frequency_penalty & candidate_count=2 -> HTTP 400
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=1024,
            thinking_level="MINIMAL",
        ),
        cached_content=cache.name,
        temperature=0.2,
        frequency_penalty=0.5,
        candidate_count=2,
        max_output_tokens=128000,
    )

    # Trap #2: Broadcasting first thought_signature across sibling parallel functionCall parts -> HTTP 400
    first_sig = parts[0].thought_signature
    broken_parallel_parts = [
        types.Part(function_call=p.function_call, thought_signature=first_sig)
        for p in parts
    ]

    # Trap #3b: Ending multi-turn history with a prefilled model turn is rejected
    return client.models.generate_content(
        model="gemini-4-neon",
        contents=[
            {"role": "user", "parts": ["Run quarterly revenue audit."]},
            {"role": "model", "parts": broken_parallel_parts},
            {"role": "model", "parts": ['{"revenue": ']},
        ],
        config=config,
    )
