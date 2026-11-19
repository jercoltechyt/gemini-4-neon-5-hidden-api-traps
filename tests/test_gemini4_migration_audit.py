# -*- coding: utf-8 -*-
"""Deterministic offline unit tests for `gemini4_migration_audit.py` ($0 API billing, zero dependencies)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gemini4_migration_audit import (
    audit_python_file,
    main,
    validate_parallel_tool_turn,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestGemini4MigrationAudit(unittest.TestCase):
    def test_broken_legacy_agent_detects_traps_1_to_4(self) -> None:
        findings = audit_python_file(REPO_ROOT / "examples" / "broken_legacy_agent.py")
        trap_ids = {f.trap_id for f in findings}
        self.assertIn("TRAP_01_THINKING_CONFIG", trap_ids)
        self.assertIn("TRAP_02_THOUGHT_SIGNATURE", trap_ids)
        self.assertIn("TRAP_03_EXPLICIT_CACHING", trap_ids)
        self.assertIn("TRAP_03_MODEL_PREFILL", trap_ids)
        self.assertIn("TRAP_04_SAMPLING_AND_WINDOW", trap_ids)

    def test_fixed_neon_agent_passes_cleanly(self) -> None:
        findings = audit_python_file(REPO_ROOT / "examples" / "fixed_neon_agent.py")
        self.assertEqual(findings, [])

    def test_trap_5_silent_default_high_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "implicit_high.py"
            sample.write_text(
                'client.models.generate_content(model="gemini-4-neon", contents="hi")\n',
                encoding="utf-8",
            )
            findings = audit_python_file(sample)
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0].trap_id, "TRAP_05_DEFAULT_HIGH_BILLING")
            self.assertEqual(findings[0].severity, "WARNING")

    def test_casing_mismatch_generate_content_vs_interactions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "casing_bug.py"
            sample.write_text(
                'types.ThinkingConfig(thinking_level="medium")\n'
                'client.interactions.create(model="gemini-4-neon", thinking_level="MEDIUM")\n',
                encoding="utf-8",
            )
            findings = audit_python_file(sample)
            self.assertEqual(len(findings), 2)
            self.assertTrue(
                all(f.trap_id == "TRAP_01_THINKING_CONFIG" and f.severity == "ERROR" for f in findings)
            )

    def test_parallel_thought_signature_runtime_validator(self) -> None:
        valid_parallel = [
            {"functionCall": {"name": "get_price"}, "thought_signature": "opaque_sig_01"},
            {"functionCall": {"name": "get_stock"}},
        ]
        self.assertEqual(validate_parallel_tool_turn(valid_parallel, is_active_turn=True), [])

        duplicated_parallel = [
            {"functionCall": {"name": "get_price"}, "thought_signature": "opaque_sig_01"},
            {"functionCall": {"name": "get_stock"}, "thought_signature": "opaque_sig_01"},
        ]
        dup_errors = validate_parallel_tool_turn(duplicated_parallel, is_active_turn=True)
        self.assertEqual(len(dup_errors), 1)
        self.assertIn("HTTP 400", dup_errors[0])

        missing_active = [
            {"functionCall": {"name": "get_price"}},
            {"functionCall": {"name": "get_stock"}},
        ]
        miss_errors = validate_parallel_tool_turn(missing_active, is_active_turn=True)
        self.assertEqual(len(miss_errors), 1)
        self.assertIn("HTTP 400", miss_errors[0])

        stripped_history = [{"text": "Final answer without signature"}]
        hist_errors = validate_parallel_tool_turn(stripped_history, is_active_turn=False)
        self.assertEqual(len(hist_errors), 1)
        self.assertIn("MALFORMED_FUNCTION_CALL", hist_errors[0])

    def test_cli_exit_codes(self) -> None:
        self.assertEqual(main(["--check", str(REPO_ROOT / "examples" / "broken_legacy_agent.py")]), 1)
        self.assertEqual(main(["--check", str(REPO_ROOT / "examples" / "fixed_neon_agent.py")]), 0)
        self.assertEqual(main(["--economics"]), 0)


if __name__ == "__main__":
    unittest.main()
