"""Regression tests for the verification guards.

Every test here encodes a hole an external review actually found — the
reviewer's counterexamples are kept as permanent regressions:

  round 5: dirty-tree check had to exempt derived paths;
           empty/subset reproduction runs reported success.
  round 6: '1.3' was "grounded" by longitude 121.36 (substring bypass);
           stale narrative fields could launder numbers into new reports.

Run from the repo root:
    python -m unittest discover -s tests
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


brs = load("brs", "scripts/build_results_summary.py")
lr = load("lr", "scripts/llm_report.py")


class GitDirtyDerivedExemption(unittest.TestCase):
    """git_dirty() must flag ONLY non-derived paths (round 5)."""

    CASES = [
        ("", False),
        (" M outputs/results_summary.json\n", False),
        (" M outputs/reproduction_report.json\n", False),
        (" M outputs/audit_archive/audit_eq2026053.json\n"
         "?? outputs/audit_archive/report_eq2026053.md\n", False),
        (" M docs/audit.json\n M vercel/audit.json\n", False),
        (" M docs/index.html\n M vercel/index.html\n", False),
        (" M outputs/results_summary.json\n M scripts/llm_report.py\n",
         True),
        (" M src/edgequake/live/engine.py\n", True),
        ("R  old.py -> outputs/results_summary.json\n", False),
        ("R  old.py -> src/new.py\n", True),
        ('?? "outputs/audit_archive/weird name.json"\n', False),
    ]

    def test_cases(self):
        orig = brs.subprocess.check_output
        try:
            for out, want in self.CASES:
                brs.subprocess.check_output = \
                    lambda *a, _o=out, **k: _o.encode()
                self.assertEqual(brs.git_dirty(), want, repr(out))
        finally:
            brs.subprocess.check_output = orig


class ReproductionRunGuards(unittest.TestCase):
    """verify_replay_reproduction.py must refuse degenerate runs
    (round 5: empty/subset runs used to report success)."""

    SCRIPT = str(ROOT / "scripts" / "verify_replay_reproduction.py")

    def run_args(self, *args):
        return subprocess.run([sys.executable, self.SCRIPT, *args],
                              capture_output=True, text=True, timeout=60)

    def test_empty_events_refused(self):
        r = self.run_args("--events")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("empty event list", r.stdout + r.stderr)

    def test_subset_requires_out(self):
        r = self.run_args("--events", "dapu")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--out", r.stdout + r.stderr)

    def test_subset_cannot_overwrite_official(self):
        r = self.run_args("--events", "dapu", "--out",
                          str(ROOT / "outputs" / "reproduction_report.json"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("may not overwrite", r.stdout + r.stderr)

    def test_unknown_event_refused(self):
        r = self.run_args("--events", "bogus")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("unknown event", r.stdout + r.stderr)


class DecimalGrounding(unittest.TestCase):
    """llm_report.validate numeric grounding (rounds 4-6)."""

    def test_reviewer_substring_bypass(self):
        # round-6 reviewer counterexample, verbatim: '1.3' must NOT be
        # grounded by longitude 121.36
        problems = lr.validate("1.3", {"x": 121.36})
        self.assertTrue(any("1.3" in p for p in problems), problems)

    def test_exact_token_is_grounded(self):
        self.assertFalse(
            [p for p in lr.validate("121.36", {"x": 121.36})
             if "121.36" in p])

    def test_invented_decimal_rejected(self):
        rec = {"final_mag": 4.81}
        problems = lr.validate("M4.81 深度 33.5 公里", rec)
        self.assertTrue(any("33.5" in p for p in problems), problems)

    def test_narrative_fields_do_not_launder(self):
        # a stale report / provenance note containing 6.2 must not make
        # an invented 6.2 legal
        rec = {"final_mag": 4.81,
               "report_md": "old report said M6.2",
               "provenance": {"original_run_note": "log said M6.2/43km"},
               "gate_note": "gates 6.2 something"}
        toks = lr.allowed_number_tokens(rec)
        self.assertNotIn("6.2", toks)
        self.assertIn("4.81", toks)

    def test_core_numbers_required(self):
        rec = {"t_first_loc_s": 4.9, "t_eew_s": None, "final_mag": 4.81,
               "cwa": {"mag": 4.7}}
        problems = lr.validate("完全沒有數字的報告", rec)
        self.assertTrue(any("4.81" in p for p in problems), problems)

    def test_forbidden_terms(self):
        problems = lr.validate("EdgeQuake 優於官方系統", {})
        self.assertTrue(any("優於" in p for p in problems), problems)


class ManifestRequiredKeys(unittest.TestCase):
    """The required-key constants must stay coherent (round 6)."""

    def test_replays_covered_by_required_hashes(self):
        for rel in brs.REPLAYS.values():
            self.assertIn(rel, brs.REQUIRED_FILE_HASHES)

    def test_original_log_pinned(self):
        self.assertIn(
            "outputs/audit_archive/"
            "audit_202607310058-EQ2026053-Waveform.txt",
            brs.REQUIRED_FILE_HASHES)

    def test_cross_check_fields(self):
        self.assertEqual(
            sorted(brs.CROSS_CHECK_FIELDS),
            sorted(("t_first_loc_s", "first_mag", "final_mag",
                    "final_err_km", "t_eew_s", "eew_fired",
                    "alert_fired")))

    def test_loader_is_a_tracked_source(self):
        self.assertIn("scripts/demo_convergence.py", brs.SOURCE_FILES)


if __name__ == "__main__":
    unittest.main()
