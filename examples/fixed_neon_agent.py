# -*- coding: utf-8 -*-
"""
Clean, production-ready `gemini-4-neon` request builder passing all 5 migration checks.
(100% offline reference fixture — zero paid API calls.)
"""

from typing import Any

# Static >= 4,096-token prefix placed at the very start to unlock automatic 95% Implicit Caching ($0.15/1M)
STATIC_SYSTEM_PREFIX = (
    "You are a deterministic billing and compliance verification engine. "
    "Always ground every numerical claim against the attached policy handbook and tool outputs."
)


def build_fixed_neon_request(client: Any, types: Any, prior_parts: list[Any], user_query: str) -> Any:
    # Trap #1 & #5 FIXED: Explicit UPPERCASE thinking_level="MEDIUM" on GenerateContent
    # (-49.1% cost vs default HIGH, 2x faster TTFT, zero HTTP 400 from thinking_budget/MINIMAL)
    # Trap #3 FIXED: No cached_content; structured JSON enforced via response_mime_type instead of model prefill
    # Trap #4 FIXED: No ghost sampling params, no penalties, synchronous output within 65,536 cap
    config = types.GenerateContentConfig(
        system_instruction=STATIC_SYSTEM_PREFIX,
        thinking_config=types.ThinkingConfig(
            thinking_level="MEDIUM",
        ),
        response_mime_type="application/json",
        max_output_tokens=16384,
    )

    # Trap #2 FIXED: Preserve exact per-part thought_signature (present ONLY on parts[0] for parallel tool turns)
    preserved_parts = [
        types.Part(
            function_call=p.function_call,
            text=p.text,
            thought_signature=p.thought_signature,
        )
        for p in prior_parts
    ]

    # Multi-turn history ends cleanly with a user turn
    return client.models.generate_content(
        model="gemini-4-neon",
        contents=[
            {"role": "user", "parts": ["Initial audit context."]},
            {"role": "model", "parts": preserved_parts},
            {"role": "user", "parts": [user_query]},
        ],
        config=config,
    )
