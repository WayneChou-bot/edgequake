"""Phase 8: LLM-written event reports for the shadow-audit log.

Turns each machine-readable audit record (outputs/audit_archive/
audit_eq*.json) into a short bilingual (zh-TW + EN) narrative report via
the Gemini API, embeds it into the record as "report_md" (so it flows
into docs/audit.json and the console's 稽核紀錄 tab), and saves a copy as
report_eq<ID>.md next to the record.

Design rules:
  - The LLM only NARRATES numbers we computed; it is told to invent
    nothing and to keep the research-prototype disclaimer.
  - Idempotent: --missing (default) only processes records without a
    report, so the GitHub Actions step re-runs safely.
  - No GEMINI_API_KEY -> prints a notice and exits 0 (the workflow must
    not fail just because the secret isn't set).

Usage:
    export GEMINI_API_KEY='AIza...'            # aistudio.google.com, free
    python scripts/llm_report.py               # all records missing one
    python scripts/llm_report.py --id eq2026053 --force
    python scripts/llm_report.py --model gemini-3.5-flash
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://generativelanguage.googleapis.com/v1beta"

PROMPT = """You are the reporting module of EdgeQuake, a research-prototype
earthquake early-warning system in Taiwan (NOT an official warning service).
Below is one machine-generated shadow-audit record: the CWA (Central Weather
Administration) published post-event strong-motion waveforms, and EdgeQuake
performed a POST-HOC ARRIVAL-TIME REPLAY — picks extracted offline enter the
estimation chain causally at their arrival times. Picker windowing, compute
and data-transport latency are NOT modeled, so every reported time is a
lower-bound estimate, not measured live performance.

Write a concise event report, first in Traditional Chinese (Taiwan usage),
then an English version, separated by a line containing only '---'.
Each version: 120-180 words, 2-3 short paragraphs, no headings, no lists.

Cover: what the event was (magnitude, region from the coordinates, depth);
how the engine did on the replayed timeline (first location, first
magnitude and how it evolved to the final value vs the CWA catalog value,
epicenter error); the instant the EEW issuance criteria were met — stated
ONLY via the mandatory sentences given below, NEVER compared with official
issuance times; the public-alert (PWS) outcome, stated ONLY via the
mandatory PWS sentences given below — NEVER invent, restate or
paraphrase a cause for the PWS decision (the sentences carry the
machine-derived reason; round-7 lesson: a narrated cause was wrong). If an
"exposure" field exists, phrase it strictly as an estimate: 「預估約 N 人
可能感受到震度3以上搖晃」 / "an estimated ~N people may have felt
intensity-3+ shaking"; NEVER 受災/波及/affected, never as a definite
fact; cite pop_version and the point-source/average-site assumption. If a
"similar" field exists, close with one sentence naming the most similar
historical event (zh name if present; note USGS magnitudes are Mw-class,
slightly different from CWA ML). Terminology: EEW 的中文為「強震即時警報」.

FORBIDDEN words/claims (report is machine-rejected if any appear):
優於、較快、更快、領先、時效、效率、受災、波及、完全正確、
faster, outperform, compares favorably, better than, superior,
high efficiency, correct decision, correctly, affected.
Never rank EdgeQuake against CWA or any official system in any way.
Time zone: origin_utc is UTC — ALWAYS present the event date/time in
Taiwan local time (UTC+8; e.g. origin_utc 07-30T16:58 is 台灣時間 7月31日
凌晨0時58分 — write 凌晨0時58分, NEVER 深夜12時58分 and NEVER the UTC
date 7月30日), matching how CWA reports it. A validator checks that the
Chinese version contains the correct LOCAL date (X月Y日) and rejects the
report if the UTC date appears instead.

Hard rules: use ONLY numbers present in the record — never invent data.
Every decimal number you write must appear VERBATIM in the record (a
validator rejects the report otherwise — do not re-round or convert).
State times as 發震後 X 秒 / origin+Xs. This audit is a POST-HOC
arrival-time replay: picker/compute latency is not modeled, so describe
detection/EEW times as lower-bound estimates (理論下界), never as
measured real-time performance. End each language version with one
sentence noting this is an automated research-prototype report, not
official information. Do not exaggerate.
"""


def pws_sentences(rec: dict) -> list[str]:
    """Machine-built verbatim sentences describing the PWS outcome.

    Round-7 review: the report claimed 'PWS did not fire because the
    magnitude stayed below M5.0' while quoting first magnitude 5.88 —
    an invented causal claim. The reason now comes from the machine-
    derived pws_evidence block, and the report must carry it verbatim.
    """
    evd = rec.get("pws_evidence")
    if not evd:
        return []
    if evd.get("fired"):
        t = evd.get("first_fired_t_after_origin_s")
        return [
            f"重播時間軸上，PWS 國家級警報條件於發震後 {t} 秒達成"
            "（理論下界，未含系統延遲）。",
            f"On the replayed timeline the PWS criteria were met at "
            f"origin+{t} s (a lower bound; system latency excluded).",
        ]
    mm = evd.get("max_mag")
    mi = evd.get("max_predicted_county_intensity")
    blockers = evd.get("blockers") or []
    if "magnitude_intensity_rule_never_met" in blockers:
        return [
            f"PWS 國家級警報未觸發：規模與預估震度組合從未達到發布"
            f"條件（M≥5.0 且縣市預估震度≥4 級，或 M≥6.5 且≥3 級）；"
            f"重播期間規模估計最高 M{mm}、預估縣市震度最高 {mi} 級"
            f"（兩極值未必同時出現）。",
            f"The PWS public alert did not trigger: the magnitude/"
            f"intensity rule ((M>=5.0 & county intensity>=4) or "
            f"(M>=6.5 & >=3)) was never met in any single instant — "
            f"the magnitude estimate peaked at M{mm} and the highest "
            f"predicted county intensity was {mi} (these extremes need "
            f"not be simultaneous).",
        ]
    w = evd.get("while_rule_met") or {}
    wo = w.get("max_observed_pga_gal")
    if "all_conditions_held_but_never_in_same_frame" in blockers:
        return [
            f"PWS 國家級警報未觸發：各項發布條件雖曾個別達成，但從未"
            f"在同一時刻同時成立（重播期間規模最高 M{mm}、預估縣市"
            f"震度最高 {mi} 級）。",
            f"The PWS public alert did not trigger: each criterion was "
            f"met at some point but never simultaneously (magnitude "
            f"peaked at M{mm}; highest predicted county intensity "
            f"{mi}).",
        ]
    zh_parts, en_parts = [], []
    if any(b.startswith("observed_pga_ge_") for b in blockers):
        zh_parts.append(f"該時段內實測 PGA 最高 {wo} gal，未達 25 gal "
                        "門檻")
        en_parts.append(f"the highest observed PGA in those frames was "
                        f"{wo} gal, below the 25 gal threshold")
    if "quality_gate_never_met_while_rule_met" in blockers:
        zh_parts.append("解的品質閘門（站數／誤差橢圓）未同時通過")
        en_parts.append("the solution-quality gate (station count / "
                        "error ellipse) was not met in those frames")
    return [
        f"PWS 國家級警報未觸發：規模與預估震度條件曾於重播中達成"
        f"（規模最高 M{mm}），但" + "、且".join(zh_parts) + "。",
        f"The PWS public alert did not trigger: the magnitude/"
        f"intensity rule was met during the replay (magnitude peaked "
        f"at M{mm}), but " + " and ".join(en_parts) + ".",
    ]


def mandatory_sentences(rec: dict) -> list[str]:
    """Fixed sentences the report MUST contain verbatim (validated)."""
    out = []
    t = rec.get("t_eew_s")
    if t is not None:
        out += [
            f"{t} 秒為未納入系統延遲的理論下界，不可與官方發布時間"
            "直接比較。",
            f"The {t} s figure is a theoretical lower bound that "
            "excludes system latency and must not be compared directly "
            "with official issuance times.",
        ]
    out += pws_sentences(rec)
    return out


FORBIDDEN = ["優於", "較快", "更快", "領先", "時效", "效率", "受災",
             "波及", "完全正確", "faster", "outperform",
             "compares favorably", "better than", "superior",
             "high efficiency", "correct decision", "correctly",
             "affected"]


# fields whose CONTENT is narrative, not data — excluded from grounding
# so a stale report or a provenance prose note can never launder numbers
# into a new report (round-6 review), plus any "*note*" key
NARRATIVE_KEYS = {"report_md", "report_model"}


def allowed_number_tokens(rec: dict) -> set[str]:
    """Exact decimal tokens a report may quote: string forms of the
    record's numeric leaves, plus decimal tokens inside non-narrative
    string leaves (e.g. timestamps). Round-6 review: the previous
    implementation was a substring test against the serialized record,
    so an invented '1.3' was 'grounded' by longitude 121.36."""
    toks: set[str] = set()

    def walk(key: str, v) -> None:
        if key in NARRATIVE_KEYS or "note" in key:
            return
        if isinstance(v, dict):
            for kk, vv in v.items():
                walk(kk, vv)
        elif isinstance(v, list):
            for vv in v:
                walk(key, vv)
        elif isinstance(v, bool) or v is None:
            pass
        elif isinstance(v, (int, float)):
            toks.add(str(v))
        elif isinstance(v, str):
            toks.update(re.findall(r"\d+\.\d+", v))

    walk("", rec)
    return toks


def validate(text: str, rec: dict) -> list[str]:
    """Return a list of violations; empty = accept.

    Numeric checks are "core-number + decimal-token grounding", NOT full
    semantic verification: the record's core numbers must appear, and
    every decimal token the model wrote must EXACTLY equal one of the
    record's number tokens (set membership — no substring matching, and
    narrative fields are excluded as sources). Integers and semantic
    ROLE are still not covered — a known limit, documented in README.
    """
    problems = []
    low = text.lower()
    for w in FORBIDDEN:
        if w.lower() in low:
            problems.append(f"forbidden term present: {w!r}")
    for s in mandatory_sentences(rec):
        if s not in text:
            problems.append(f"mandatory sentence missing: {s!r}")
    # forward grounding: the record's core numbers must be quoted
    core = [rec.get("t_first_loc_s"), rec.get("t_eew_s"),
            rec.get("final_mag"), (rec.get("cwa") or {}).get("mag")]
    for v in core:
        if v is not None and str(v) not in text:
            problems.append(f"core number missing from report: {v}")
    # reverse grounding: every decimal token must be a record token
    allowed = allowed_number_tokens(rec)
    for tok in sorted(set(re.findall(r"\d+\.\d+", text))):
        if tok not in allowed:
            problems.append(f"decimal number not present in the audit "
                            f"record: {tok}")
    problems += date_problems(text, rec)
    return problems


MONTH_EN = [None, "January", "February", "March", "April", "May",
            "June", "July", "August", "September", "October",
            "November", "December"]


def date_problems(text: str, rec: dict) -> list[str]:
    """Bilingual Taiwan-local datetime grounding.

    Round-9 review: a validator-clean report stated the UTC date
    7月30日 in Chinese while the English half correctly said July 31 —
    dates are integers, which the numeric grounding does not cover.
    Round-10 review widened the check: the first version only matched
    the Chinese month/day anywhere in the report, so a wrong YEAR, a
    wrong minute, or a wrong ENGLISH date all passed. Now the report is
    split at '---' and each half is checked in its own language: the
    Chinese half must carry the local year, month/day and minute (and
    not the UTC month/day when they differ); the English half must
    carry the local month-name date (and not the UTC one). Semantic
    placement is still not verified, and a similar-event date equal to
    the UTC month/day inside the same half could false-positive — both
    documented limits."""
    import datetime as _dt
    raw = rec.get("origin_utc")
    if not raw:
        return []
    try:
        utc = _dt.datetime.fromisoformat(
            str(raw).replace("Z", "+00:00"))
    except ValueError:
        return []
    local = utc + _dt.timedelta(hours=8)
    zh, _, en = text.partition("---")
    out = []

    # tolerate spacing variants: 7月31日 / 7 月 31 日 / 7月 31日 ...
    def zh_pat(month: int, day: int) -> str:
        return rf"{month}\s*月\s*{day}\s*日"

    def en_pat(month: int, day: int) -> str:
        name = MONTH_EN[month]
        return rf"(?:{name}\s+{day}\b|\b{day}(?:st|nd|rd|th)?\s+{name})"

    want = f"{local.month}月{local.day}日"
    if not re.search(zh_pat(local.month, local.day), zh):
        out.append(f"Chinese version must contain the Taiwan-local "
                   f"date {want!r} (origin_utc {raw} + 8h)")
    if not re.search(rf"{local.year}\s*年", zh):
        out.append(f"Chinese version must contain the local year "
                   f"{local.year}年")
    if local.minute and not (
            re.search(rf"{local.minute}\s*分", zh)
            or f"{local.hour:02d}:{local.minute:02d}" in zh):
        out.append(f"Chinese version must contain the local minute "
                   f"({local.minute}分 or "
                   f"{local.hour:02d}:{local.minute:02d})")
    if (utc.month, utc.day) != (local.month, local.day):
        if re.search(zh_pat(utc.month, utc.day), zh):
            out.append(f"UTC date {utc.month}月{utc.day}日 presented "
                       f"as if local — the Taiwan-local date is "
                       f"{want!r}")
        if en and re.search(en_pat(utc.month, utc.day), en):
            out.append(f"UTC date {MONTH_EN[utc.month]} {utc.day} "
                       f"presented as if local in the English version "
                       f"— the local date is {MONTH_EN[local.month]} "
                       f"{local.day}")
    if en and not re.search(en_pat(local.month, local.day), en):
        out.append(f"English version must contain the Taiwan-local "
                   f"date {MONTH_EN[local.month]} {local.day}")
    return out


def build_prompt(rec: dict) -> str:
    req = mandatory_sentences(rec)
    extra = ""
    if req:
        extra = ("\nMANDATORY: include these sentences VERBATIM (the zh "
                 "one in the Chinese version, the English one in the "
                 "English version):\n"
                 + "\n".join(f"  {s}" for s in req) + "\n")
    return PROMPT + extra + "\nAudit record JSON:\n" + json.dumps(
        rec, ensure_ascii=False, indent=1)


def gemini(model: str, key: str, text: str) -> str:
    # thinkingBudget 0: flash models "think" by default and those tokens
    # count against maxOutputTokens — with a small cap the visible report
    # comes out truncated (lesson: first live run produced 298 chars).
    gc = {"temperature": 0.4, "maxOutputTokens": 8192,
          "thinkingConfig": {"thinkingBudget": 0}}
    for attempt in (0, 1):
        body = json.dumps({
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": gc,
        }).encode()
        req = urllib.request.Request(
            f"{API}/models/{model}:generateContent",
            data=body, headers={"Content-Type": "application/json",
                                "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            break
        except urllib.error.HTTPError as e:
            # some models reject thinkingConfig -> retry once without it
            if e.code == 400 and attempt == 0 and "thinkingConfig" in gc:
                gc = {k: v for k, v in gc.items() if k != "thinkingConfig"}
                continue
            raise
    return d["candidates"][0]["content"]["parts"][0]["text"].strip()


def looks_complete(text: str) -> bool:
    """Bilingual report sanity: both halves present, not truncated."""
    parts = text.split("---")
    return len(text) >= 500 and len(parts) >= 2 and len(parts[-1]) >= 200


def pick_fallback_model(key: str) -> str | None:
    """If the default model 404s, ask the API what flash models exist."""
    req = urllib.request.Request(f"{API}/models?pageSize=50",
                                 headers={"x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
        names = [m["name"].split("/")[-1] for m in d.get("models", [])
                 if "flash" in m["name"] and
                 "generateContent" in m.get("supportedGenerationMethods", [])]
        names = [n for n in names if "lite" not in n] + names
        return names[0] if names else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--id", default=None,
                    help="process only this event id (e.g. eq2026053)")
    ap.add_argument("--force", action="store_true",
                    help="regenerate even if report_md already exists")
    args = ap.parse_args()

    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        print("[report] GEMINI_API_KEY not set — skipping (this is fine; "
              "reports are optional)")
        return

    arch = ROOT / "outputs" / "audit_archive"
    files = sorted(arch.glob("audit_eq*.json"))
    if args.id:
        files = [f for f in files if f.stem == f"audit_{args.id}"]
    todo = []
    for f in files:
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[report] skip {f.name}: {e}", file=sys.stderr)
            continue
        if rec.get("report_md") and not args.force:
            continue
        todo.append((f, rec))
    if not todo:
        print("[report] nothing to do (all records already have reports)")
        return

    model = args.model
    for f, rec in todo:
        base_payload = build_prompt(rec)
        text = None
        feedback = ""
        for attempt in range(3):   # generate + up to 2 validated retries
            payload = base_payload + feedback
            try:
                cand = gemini(model, key, payload)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    fb = pick_fallback_model(key)
                    if fb and fb != model:
                        print(f"[report] {model} not available -> {fb}")
                        model = fb
                        cand = gemini(model, key, payload)
                    else:
                        print(f"[report] no usable model ({e}) — aborting")
                        return
                elif e.code in (429, 503):
                    print(f"[report] rate-limited ({e.code}), "
                          "retrying in 30s")
                    time.sleep(30)
                    cand = gemini(model, key, payload)
                else:
                    print(f"[report] API error {e.code}: "
                          f"{e.read()[:300]}", file=sys.stderr)
                    return
            problems = validate(cand, rec)
            if not looks_complete(cand):
                problems.append(f"output truncated / single-language "
                                f"({len(cand)} chars)")
            if not problems:
                text = cand
                break
            print(f"[report] {f.stem}: attempt {attempt + 1} rejected — "
                  + "; ".join(problems))
            feedback = ("\n\nYour previous draft was REJECTED by the "
                        "validator for these violations:\n"
                        + "\n".join(f"- {p}" for p in problems)
                        + "\nRegenerate the full bilingual report "
                        "following EVERY rule above.")
            time.sleep(3)
        if text is None:
            print(f"[report] {f.stem}: REJECTED after 3 attempts — no "
                  "report saved (record keeps its numbers; rerun with "
                  "--force after inspecting)")
            continue
        rec["report_md"] = text
        rec["report_model"] = model
        f.write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                     encoding="utf-8")
        md = arch / f"report_{rec.get('id', f.stem)}.md"
        md.write_text(text + "\n", encoding="utf-8")
        print(f"[report] {f.stem}: {len(text)} chars ({model}), "
              f"validator-clean -> {md.name}")
        time.sleep(2)   # stay far under free-tier RPM


if __name__ == "__main__":
    main()
