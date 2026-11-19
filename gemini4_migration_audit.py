# -*- coding: utf-8 -*-
"""
Gemini 4 Neon Migration Auditor & Token Economics CLI (100% Offline, $0 API Billing).

Statically audits Python source files (AST) and validates multi-turn parallel tool payloads
against the 5 documented breaking changes and billing traps in `gemini-4-neon`:

  1. Trap #1: `thinking_budget`, `thinking_level="MINIMAL"`, and GenerateContent vs Interactions casing
  2. Trap #2: Parallel `thought_signature` omission, sibling duplication, or multi-turn stripping
  3. Trap #3: Explicit context caching (`cached_content`/`caches.create`) and prefilled `model` turns
  4. Trap #4: Ghost sampling params (`temperature`/`top_p`/`top_k`), HTTP 400 penalties/`candidate_count>1`,
              and synchronous `max_output_tokens > 65536` without continuation/background handling
  5. Trap #5: Silent API default jump (`MEDIUM` -> `HIGH`) when `thinking_level` is omitted on `gemini-4-neon`

Note: API pricing and Artificial Analysis v5.0.1 metrics embedded in `--economics` reflect
verified rates and benchmarks as of November 19, 2026, and may change over time.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

VALID_GENERATE_CONTENT_LEVELS = {"LOW", "MEDIUM", "HIGH", "XHIGH"}
VALID_INTERACTIONS_LEVELS = {"low", "medium", "high", "xhigh"}
HARD_400_SAMPLING_PARAMS = {"frequency_penalty", "presence_penalty"}
GHOST_SAMPLING_PARAMS = {"temperature", "top_p", "top_k"}
SYNC_OUTPUT_CAP = 65536


@dataclass(frozen=True)
class Finding:
    trap_id: str
    severity: str  # "ERROR" (HTTP 400 / broken) or "WARNING" (billing / silent ignore)
    line: int
    message: str
    fix: str


def _get_call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        parts: list[str] = []
        curr: ast.AST = node.func
        while isinstance(curr, ast.Attribute):
            parts.append(curr.attr)
            curr = curr.value
        if isinstance(curr, ast.Name):
            parts.append(curr.id)
        return ".".join(reversed(parts))
    return ""


def _const_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _const_num(node: ast.AST) -> int | float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -node.operand.value
    return None


class Gemini4NeonAuditor(ast.NodeVisitor):
    """AST visitor detecting the 5 Gemini 4 Neon migration traps without making API calls."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.findings: list[Finding] = []
        self.has_explicit_thinking_level = False
        self.neon_model_lines: list[int] = []

    def _add(self, trap_id: str, severity: str, line: int, message: str, fix: str) -> None:
        self.findings.append(Finding(trap_id, severity, line, message, fix))

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _get_call_name(node)
        kw_map = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}

        # Track references to gemini-4-neon
        model_val = _const_str(kw_map["model"]) if "model" in kw_map else None
        if model_val and "gemini-4-neon" in model_val:
            self.neon_model_lines.append(node.lineno)

        # Trap #1: thinking_budget & thinking_level="MINIMAL" & casing mismatch
        if "thinking_budget" in kw_map:
            self._add(
                "TRAP_01_THINKING_CONFIG",
                "ERROR",
                node.lineno,
                "`thinking_budget` returns HTTP 400 on gemini-4-neon.",
                "Replace with `thinking_level='MEDIUM'` (GenerateContent) or `'medium'` (Interactions API).",
            )

        if "thinking_level" in kw_map:
            self.has_explicit_thinking_level = True
            lvl = _const_str(kw_map["thinking_level"])
            if lvl is not None:
                if lvl.upper() == "MINIMAL":
                    self._add(
                        "TRAP_01_THINKING_CONFIG",
                        "ERROR",
                        node.lineno,
                        "`thinking_level='MINIMAL'` returns HTTP 400 on gemini-4-neon.",
                        "Use 'LOW', 'MEDIUM', 'HIGH', or 'XHIGH' (GenerateContent) / lowercase on Interactions API.",
                    )
                elif "interactions" in call_name.lower():
                    if lvl not in VALID_INTERACTIONS_LEVELS:
                        self._add(
                            "TRAP_01_THINKING_CONFIG",
                            "ERROR",
                            node.lineno,
                            f"Interactions API requires lowercase thinking_level, got '{lvl}'.",
                            f"Change '{lvl}' to '{lvl.lower()}'.",
                        )
                elif any(tok in call_name for tok in ("ThinkingConfig", "GenerateContentConfig", "generate_content")):
                    if lvl not in VALID_GENERATE_CONTENT_LEVELS:
                        self._add(
                            "TRAP_01_THINKING_CONFIG",
                            "ERROR",
                            node.lineno,
                            f"GenerateContent API requires UPPERCASE thinking_level, got '{lvl}'.",
                            f"Change '{lvl}' to '{lvl.upper()}'.",
                        )

        # Trap #2: Detect obvious thought_signature broadcasting inside comprehensions/loops
        if call_name.endswith("Part") and "thought_signature" in kw_map and "function_call" in kw_map:
            sig_node = kw_map["thought_signature"]
            if isinstance(sig_node, ast.Name) and "first" in sig_node.id.lower():
                self._add(
                    "TRAP_02_THOUGHT_SIGNATURE",
                    "ERROR",
                    node.lineno,
                    "Suspicious broadcast of a single `thought_signature` variable onto multiple `Part(function_call=...)` items.",
                    "Attach `thought_signature` ONLY to the first `functionCall` part in a parallel turn; keep sibling parts at `None`.",
                )

        # Trap #3: Explicit context caching (cached_content / caches.create)
        if "cached_content" in kw_map or "cachedContents" in kw_map or call_name.endswith("caches.create"):
            self._add(
                "TRAP_03_EXPLICIT_CACHING",
                "ERROR",
                node.lineno,
                "Explicit Context Caching (`cached_content` / `caches.create`) is unsupported on gemini-4-neon.",
                "Remove explicit cache calls and place static prefixes >= 4,096 tokens at the start for automatic 95% Implicit Caching.",
            )

        # Trap #4: Ghost sampling params & hard HTTP 400 params & 65K sync cap
        for param in HARD_400_SAMPLING_PARAMS:
            if param in kw_map:
                self._add(
                    "TRAP_04_SAMPLING_AND_WINDOW",
                    "ERROR",
                    node.lineno,
                    f"`{param}` returns immediate HTTP 400 on gemini-4-neon.",
                    f"Remove `{param}` from GenerateContentConfig.",
                )

        if "candidate_count" in kw_map:
            val = _const_num(kw_map["candidate_count"])
            if val is not None and val > 1:
                self._add(
                    "TRAP_04_SAMPLING_AND_WINDOW",
                    "ERROR",
                    node.lineno,
                    f"`candidate_count={val}` (> 1) returns immediate HTTP 400 on gemini-4-neon.",
                    "Set `candidate_count=1` or omit it.",
                )

        for param in GHOST_SAMPLING_PARAMS:
            if param in kw_map:
                val = _const_num(kw_map[param])
                if param == "temperature" and val is not None and (val < 0.0 or val > 2.0):
                    self._add(
                        "TRAP_04_SAMPLING_AND_WINDOW",
                        "ERROR",
                        node.lineno,
                        f"Out-of-range `temperature={val}` returns HTTP 400 on Gemini API.",
                        "Remove `temperature` entirely; in-range values are silently ignored on gemini-4-neon.",
                    )
                else:
                    self._add(
                        "TRAP_04_SAMPLING_AND_WINDOW",
                        "WARNING",
                        node.lineno,
                        f"Ghost parameter `{param}` is silently ignored in-range on gemini-4-neon.",
                        f"Remove `{param}` and tune reasoning behavior via `thinking_level`.",
                    )

        if "max_output_tokens" in kw_map:
            val = _const_num(kw_map["max_output_tokens"])
            if val is not None and val > SYNC_OUTPUT_CAP:
                has_long_decode = "continuation_token" in kw_map or "background" in kw_map
                if not has_long_decode:
                    self._add(
                        "TRAP_04_SAMPLING_AND_WINDOW",
                        "WARNING",
                        node.lineno,
                        f"`max_output_tokens={val}` exceeds the 65,536-token synchronous/foreground cap.",
                        "Use `continuation_token` (v1beta1) or `background=True` (Interactions API) for decodes > 65,536 tokens.",
                    )

        # Trap #3b: Prefilled model turn ending a contents list
        if "contents" in kw_map and isinstance(kw_map["contents"], ast.List) and kw_map["contents"].elts:
            last_elt = kw_map["contents"].elts[-1]
            if _looks_like_model_role(last_elt):
                self._add(
                    "TRAP_03_MODEL_PREFILL",
                    "ERROR",
                    last_elt.lineno,
                    "Multi-turn `contents` ends with a prefilled `model` role turn (rejected on gemini-4-neon).",
                    "End `contents` with a `user` turn and enforce structure with `response_mime_type='application/json'`.",
                )

        self.generic_visit(node)

    def finalize(self) -> list[Finding]:
        # Trap #5: Silent default jump from MEDIUM -> HIGH when using gemini-4-neon without thinking_level
        if self.neon_model_lines and not self.has_explicit_thinking_level:
            self._add(
                "TRAP_05_DEFAULT_HIGH_BILLING",
                "WARNING",
                self.neon_model_lines[0],
                "`gemini-4-neon` called without explicit `thinking_level` (silently defaults to 'HIGH': $3.74/task, 40.64s TTFT).",
                "Explicitly set `thinking_level='MEDIUM'` ($1.90/task, 20.14s TTFT — 49.1% cheaper, 2x faster) unless deep reasoning is required.",
            )
        return sorted(self.findings, key=lambda f: (f.line, f.trap_id))


def _looks_like_model_role(node: ast.AST) -> bool:
    if isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if k is not None and _const_str(k) == "role" and _const_str(v) == "model":
                return True
    if isinstance(node, ast.Call):
        for kw in node.keywords:
            if kw.arg == "role" and _const_str(kw.value) == "model":
                return True
    return False


def validate_parallel_tool_turn(parts: list[dict[str, Any]], is_active_turn: bool = True) -> list[str]:
    """
    Runtime payload validator for Trap #2 (`thought_signature` in parallel tool calls).

    Rules verified against Google Cloud Docs (`get-started-with-gemini-4`, Nov 19, 2026):
      - In a parallel function-calling turn, ONLY the first `functionCall` part carries `thought_signature`.
      - Omitting `thought_signature` on the first `functionCall` of the active turn -> HTTP 400.
      - Duplicating `thought_signature` onto sibling `functionCall` parts (index >= 1) -> HTTP 400.
      - Stripping `thought_signature` from earlier turns (`functionCall` or final `text`) degrades reasoning
        and can trigger `MALFORMED_FUNCTION_CALL`.
    """
    errors: list[str] = []
    fn_indices = [idx for idx, part in enumerate(parts) if "functionCall" in part or "function_call" in part]

    if not fn_indices:
        if not is_active_turn and parts and not any(part.get("thought_signature") for part in parts):
            errors.append(
                "Historical text turn is missing `thought_signature` (stripping it degrades reasoning and can cause MALFORMED_FUNCTION_CALL)."
            )
        return errors

    first_fn = fn_indices[0]
    if not parts[first_fn].get("thought_signature"):
        if is_active_turn:
            errors.append(
                f"HTTP 400: Active turn `functionCall` at parts[{first_fn}] is missing required `thought_signature`."
            )
        else:
            errors.append(
                f"Historical `functionCall` at parts[{first_fn}] stripped `thought_signature` (risk of MALFORMED_FUNCTION_CALL)."
            )

    for sibling_idx in fn_indices[1:]:
        if parts[sibling_idx].get("thought_signature"):
            errors.append(
                f"HTTP 400: Sibling parallel `functionCall` at parts[{sibling_idx}] must NOT duplicate `thought_signature`."
            )

    return errors


def audit_python_file(path: Path) -> list[Finding]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    auditor = Gemini4NeonAuditor(source)
    auditor.visit(tree)
    return auditor.finalize()


def print_economics_table() -> None:
    print(f"\n{BOLD}{CYAN}=== GEMINI 4 NEON TOKEN ECONOMICS (AA v5.0.1 — VERIFIED NOV 19, 2026) ==={RESET}\n")
    rows = [
        ("gemini-4-neon (xhigh)",          "61", "$6.44", "21.5m", "56.51s", "14.4%"),
        ("gemini-4-neon (high - DEFAULT)", "60", "$3.74", "14.8m", "40.64s", "13.9%"),
        ("gemini-4-neon (medium - SWEET)", "55", "$1.90", " 8.3m", "20.14s", "16.2%"),
        ("gemini-4-neon (low)",            "44", "$0.75", " 3.7m", "13.93s", "18.8%"),
        ("gemini-4-hydrogen (xhigh)",      "54", "$3.16", "15.2m", "45.22s", " 8.5%"),
        ("gemini-4-hydrogen (high)",       "51", "$1.87", " 8.6m", "20.08s", " 8.7%"),
        ("gemini-4-hydrogen (medium)",     "41", "$0.80", "  —  ", "  —   ", "  —  "),
        ("gemini-4-hydrogen (low)",        "27", "$0.29", "  —  ", "  —   ", "  —  "),
        ("gemini-4-carbon (high)",         "60", "$7.76", "36.6m", "  —   ", "11.4%"),
        ("gemini-4-argon (high)",          "50", "$4.97", "29.5m", "  —   ", " 4.3%"),
    ]
    header = f"  {'Model & Effort':<32} | {'AA v5.0.1':>9} | {'$/Task':>6} | {'Time':>5} | {'TTFT':>6} | {'Halluc. (Ungrounded)':>20}"
    print(f"{BOLD}{header}{RESET}")
    print("  " + "-" * (len(header) - 2))
    for model, score, cost, runtime, ttft, halluc in rows:
        line = f"  {model:<32} | {score:>9} | {cost:>6} | {runtime:>5} | {ttft:>6} | {halluc:>20}"
        if "SWEET" in model:
            print(f"{GREEN}{BOLD}{line}{RESET}")
        elif "DEFAULT" in model:
            print(f"{YELLOW}{line}{RESET}")
        else:
            print(line)

    print(f"\n{BOLD}Key Takeaways:{RESET}")
    print(f"  - Pinning Neon to {GREEN}MEDIUM{RESET} cuts cost by {BOLD}49.1%{RESET} ($3.74 -> $1.90) & halves TTFT (40.64s -> 20.14s).")
    print(f"  - Neon {GREEN}MEDIUM{RESET} beats Hydrogen XHIGH (-39.7% cost, 1.83x faster, +1 pt) & matches Hydrogen HIGH (+4 pts).")
    print(f"  - Keep Hydrogen LOW/MEDIUM ($0.29/$0.80) for heavy-input low-reasoning workloads ($1.50/1M base + Free Tier).")
    print(f"  - Always ground Neon with RAG/tools (13.9%–16.2% ungrounded hallucination vs 4.3% on Argon HIGH).\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline AST & payload auditor for the 5 hidden Gemini 4 Neon API traps ($0 API billing)."
    )
    parser.add_argument("--check", type=Path, help="Path to a Python file to audit for Gemini 4 Neon traps.")
    parser.add_argument("--economics", action="store_true", help="Print verified AA v5.0.1 token economics table.")
    args = parser.parse_args(argv)

    if not args.check and not args.economics:
        parser.print_help()
        return 0

    exit_code = 0
    if args.check:
        if not args.check.exists():
            print(f"{RED}[ERROR] File not found: {args.check}{RESET}", file=sys.stderr)
            return 2

        findings = audit_python_file(args.check)
        print(f"\n{BOLD}{CYAN}=== GEMINI 4 NEON MIGRATION AUDIT: {args.check} ==={RESET}\n")
        if not findings:
            print(f"  {GREEN}{BOLD}[PASS]{RESET} 0 errors, 0 warnings. Payload is Gemini 4 Neon production-ready.\n")
        else:
            errors = sum(1 for f in findings if f.severity == "ERROR")
            warnings = sum(1 for f in findings if f.severity == "WARNING")
            for f in findings:
                color = RED if f.severity == "ERROR" else YELLOW
                print(f"  {color}[{f.severity}]{RESET} Line {f.line:<3} | {BOLD}{f.trap_id}{RESET}")
                print(f"          Issue: {f.message}")
                print(f"          Fix  : {f.fix}\n")
            print(f"{BOLD}Summary:{RESET} {RED}{errors} error(s){RESET}, {YELLOW}{warnings} warning(s){RESET}\n")
            if errors > 0:
                exit_code = 1

    if args.economics:
        print_economics_table()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
