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


class LocalDateGrounding(unittest.TestCase):
    """round 9: a validator-clean report stated the UTC date 7月30日 in
    Chinese while English correctly said July 31 — dates are integers,
    outside the decimal grounding, so they get a dedicated check."""

    REC = {"origin_utc": "2026-07-30T16:58:36.000000Z"}

    def test_utc_date_rejected(self):
        problems = lr.date_problems(
            "2026 年 7 月30日深夜 12 時 58 分發生地震", self.REC)
        self.assertTrue(any("7月31日" in p for p in problems), problems)

    def test_local_date_accepted(self):
        self.assertEqual(
            lr.date_problems("台灣時間 7月31日凌晨 0 時 58 分", self.REC),
            [])

    def test_spaced_local_date_accepted(self):
        # round-9 field bug: Gemini writes "7 月 31 日" with spaces —
        # three correct drafts were rejected by an exact-string match
        self.assertEqual(
            lr.date_problems("2026 年 7 月 31 日凌晨 0 時 58 分",
                             self.REC), [])

    def test_spaced_utc_date_still_rejected(self):
        problems = lr.date_problems(
            "2026 年 7 月 30 日深夜，另於 7 月 31 日回顧", self.REC)
        self.assertTrue(any("presented" in p for p in problems),
                        problems)

    def test_same_day_no_false_positive(self):
        rec = {"origin_utc": "2026-07-30T02:00:00Z"}   # local 30th too
        self.assertEqual(
            lr.date_problems("台灣時間 7月30日上午 10 時", rec), [])


def _frame(t, mag, i, obs, gate_ok=True, pws=None):
    f = {"t": t, "mag": mag, "obs": obs, "gate_ok": gate_ok,
         "cty": [{"i": i}]}
    if pws is not None:
        f["pws"] = pws
    return f


def _payload(frames, origin_rel=0.0):
    return {"frames": frames, "origin_rel": origin_rel}


class PwsEvidence(unittest.TestCase):
    """rounds 7-8: the PWS 'reason' must be computed, not narrated, and
    the computation must be PER-FRAME equivalent to the actual rule.
    The two round-8 reviewer counterexamples are kept verbatim."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "src"))
        from edgequake.location.replay_sim import pws_evidence
        cls.pws_evidence = staticmethod(pws_evidence)

    def test_taitung_shape_rule_never_met(self):
        # M crossed 5.0 but intensity stayed at 3 — neither rule branch
        # ever held
        ev = self.pws_evidence(_payload([
            _frame(1.0, 5.88, 3, 30.0), _frame(2.0, 4.81, 3, 37.3)]))
        self.assertFalse(ev["fired"])
        self.assertEqual(ev["blockers"],
                         ["magnitude_intensity_rule_never_met"])

    def test_reviewer_counterexample_m65_branch(self):
        # round-8 counterexample 1: M6.6 / I3 / 10 gal — the SECOND rule
        # branch (M>=6.5 & I>=3) already holds; the true blocker is PGA,
        # and the old code wrongly blamed intensity
        ev = self.pws_evidence(_payload([_frame(1.0, 6.6, 3, 10.0)]))
        self.assertFalse(ev["fired"])
        self.assertEqual(ev["n_frames_rule_met"], 1)
        self.assertTrue(any(b.startswith("observed_pga_ge_")
                            for b in ev["blockers"]), ev["blockers"])
        self.assertNotIn("magnitude_intensity_rule_never_met",
                         ev["blockers"])

    def test_reviewer_counterexample_cross_frame(self):
        # round-8 counterexample 2: one frame I4/0 gal, another I3/30
        # gal — nothing holds simultaneously; the old code produced an
        # EMPTY blocker list
        ev = self.pws_evidence(_payload([
            _frame(1.0, 5.5, 4, 0.0), _frame(2.0, 5.5, 3, 30.0)]))
        self.assertFalse(ev["fired"])
        self.assertTrue(ev["blockers"], "blockers must not be empty")
        self.assertTrue(any(b.startswith("observed_pga_ge_")
                            for b in ev["blockers"]), ev["blockers"])

    def test_conditions_never_simultaneous(self):
        # within rule-met frames: one has shaking but no gate, the
        # other has gate but weak shaking — only the same-frame check
        # catches this
        ev = self.pws_evidence(_payload([
            _frame(1.0, 5.5, 4, 30.0, gate_ok=False),
            _frame(2.0, 5.5, 4, 10.0, gate_ok=True)]))
        self.assertFalse(ev["fired"])
        self.assertEqual(ev["blockers"],
                         ["all_conditions_held_but_never_in_same_frame"])

    def test_fired_time_is_origin_relative(self):
        # round-8 P2: frame t counts from the first P pick; the fired
        # time must be reported origin-relative
        ev = self.pws_evidence(_payload(
            [_frame(3.0, 6.1, 4, 40.0)], origin_rel=-1.18))
        self.assertTrue(ev["fired"])
        self.assertEqual(ev["first_fired_t_after_origin_s"], 4.2)

    def test_reviewer_rounding_magnitude_boundary(self):
        # round-9 counterexample 1: raw M4.996 displays as 5.00 — the
        # actual alert did NOT fire; evidence must follow the raw
        # decision components in frame["pws"], not the displayed value
        ev = self.pws_evidence(_payload([_frame(
            1.0, 5.0, 4, 30.0,
            pws={"rule": False, "obs_ok": True, "gate": True})]))
        self.assertFalse(ev["fired"])
        self.assertEqual(ev["blockers"],
                         ["magnitude_intensity_rule_never_met"])

    def test_reviewer_rounding_pga_boundary(self):
        # round-9 counterexample 2: raw PGA 24.96 displays as 25.0 —
        # the actual alert did NOT fire
        ev = self.pws_evidence(_payload([_frame(
            1.0, 5.5, 4, 25.0,
            pws={"rule": True, "obs_ok": False, "gate": True})]))
        self.assertFalse(ev["fired"])
        self.assertTrue(any(b.startswith("observed_pga_ge_")
                            for b in ev["blockers"]), ev["blockers"])

    def test_raw_components_beat_displayed_values(self):
        # displayed values say "should not fire" but the raw components
        # say it DID (M5.004 displays as 5.0? -> inverse direction):
        # evidence must follow pws, in both directions
        ev = self.pws_evidence(_payload([_frame(
            2.0, 4.99, 4, 24.9,
            pws={"rule": True, "obs_ok": True, "gate": True})]))
        self.assertTrue(ev["fired"])


class PwsMandatorySentences(unittest.TestCase):
    """The report's PWS wording is machine-built from evidence."""

    REC = {"pws_evidence": {
        "fired": False, "max_mag": 5.94,
        "max_predicted_county_intensity": 3,
        "max_observed_pga_gal": 37.3,
        "n_frames_rule_met": 0,
        "blockers": ["magnitude_intensity_rule_never_met"],
    }}

    def test_sentences_state_computed_reason(self):
        s = lr.pws_sentences(self.REC)
        self.assertEqual(len(s), 2)
        self.assertIn("M5.94", s[0])
        self.assertIn("3 級", s[0])
        self.assertNotIn("未達 M5.0", s[0])  # the invented wrong reason

    def test_sentences_for_pga_blocker(self):
        rec = {"pws_evidence": {
            "fired": False, "max_mag": 6.6,
            "max_predicted_county_intensity": 3,
            "max_observed_pga_gal": 10.0, "n_frames_rule_met": 1,
            "while_rule_met": {"max_observed_pga_gal": 10.0,
                               "quality_gate_ok_any": True},
            "blockers": ["observed_pga_ge_25_never_met_while_rule_met"],
        }}
        s = lr.pws_sentences(rec)
        self.assertIn("10.0 gal", s[0])
        self.assertNotIn("4 級）未達", s[0])

    def test_sentences_are_mandatory(self):
        s = lr.pws_sentences(self.REC)
        problems = lr.validate("報告沒有提到 PWS。" + "x" * 600
                               + "---" + "y" * 300, self.REC)
        self.assertTrue(any(s[0] in p for p in problems), problems)


class SummaryQuoteSelfConsistency(unittest.TestCase):
    """round 7: the summary cannot hash itself — verify rebuilds the
    quote list from the summary's own values instead."""

    def mini(self):
        def evrow(m):
            return {"site_corrected": {"final_mag": m},
                    "audit_record": {"t_first_loc_s": 4.9,
                                     "t_eew_s": 9.2}}
        return {"events": {"0403": evrow(7.08), "dapu": evrow(6.38),
                           "eq2026053": evrow(4.81)},
                "mean_abs_dmag": {"raw": 0.17, "site_corrected": 0.08}}

    def test_quotes_match_own_values(self):
        s = self.mini()
        q = brs.quotes_from(s)
        self.assertIn("M7.08", q)
        self.assertIn("origin+9.2", q)

    def test_tampered_event_value_changes_quotes(self):
        s = self.mini()
        q1 = brs.quotes_from(s)
        s["events"]["0403"]["site_corrected"]["final_mag"] = 7.20
        self.assertNotEqual(brs.quotes_from(s), q1)


class ManifestRequiredKeys(unittest.TestCase):
    """The required-key constants must stay coherent (round 6)."""

    def test_replays_covered_by_required_hashes(self):
        req = brs.required_file_hashes()
        for rel in brs.REPLAYS.values():
            self.assertIn(rel, req)

    def test_public_derived_artifacts_pinned(self):
        # round 7: every ancestor-diff-exempt public artifact must be
        # pinned by hash instead
        req = brs.required_file_hashes()
        for p in ("docs/audit.json", "vercel/audit.json",
                  "docs/index.html", "vercel/index.html"):
            self.assertIn(p, req)

    def test_required_hashes_are_dynamic(self):
        # round 8: EVERY file currently in the audit archive must be
        # required — new event files are pinned automatically, and
        # verify fails until a regenerated summary pins them
        req = set(brs.required_file_hashes())
        arch = brs.ROOT / "outputs" / "audit_archive"
        files = [p for p in arch.glob("*") if p.is_file()]
        self.assertTrue(files, "archive unexpectedly empty")
        for p in files:
            self.assertIn(p.relative_to(brs.ROOT).as_posix(), req)

    def test_new_archive_file_becomes_required(self):
        arch = brs.ROOT / "outputs" / "audit_archive"
        tmp = arch / "audit_eq_test_dynamic.json"
        tmp.write_text("{}", encoding="utf-8")
        try:
            self.assertIn(
                tmp.relative_to(brs.ROOT).as_posix(),
                brs.required_file_hashes())
        finally:
            tmp.unlink()

    def test_git_ignored_zip_not_required(self):
        # round 9: the downloaded waveform zip is git-ignored — pinning
        # it made CI verify pass while every clean clone failed
        arch = brs.ROOT / "outputs" / "audit_archive"
        tmp = arch / "cwawave_test_dynamic.zip"
        tmp.write_bytes(b"zip")
        try:
            self.assertNotIn(
                tmp.relative_to(brs.ROOT).as_posix(),
                brs.required_file_hashes())
        finally:
            tmp.unlink()

    def test_environment_keys_defined(self):
        self.assertEqual(sorted(brs.ENV_KEYS),
                         sorted(("python", "platform", "numpy", "scipy",
                                 "pandas", "obspy")))

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
