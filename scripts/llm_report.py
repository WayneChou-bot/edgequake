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
Administration) published post-event strong-motion waveforms, and the
EdgeQuake engine re-ran them blind to measure how it WOULD have performed
had it been live.

Write a concise event report, first in Traditional Chinese (Taiwan usage),
then an English version, separated by a line containing only '---'.
Each version: 120-180 words, 2-3 short paragraphs, no headings, no lists.

Cover: what the event was (magnitude, region from the coordinates, depth);
how the engine did on the true timeline (first location, first magnitude
and how it evolved to the final value vs the CWA catalog value, epicenter
error); the EEW-criteria instant versus CWA's official 10-20 s issuance
performance; whether the public-alert (PWS) gate fired and why that was
the correct decision for this event size. If an "exposure" field exists,
mention the estimated population in intensity-4+ (and 3+) shaking as an
order-of-magnitude figure, citing pop_version, and note it assumes a point
source and average site conditions. If a "similar" field exists, close
with one sentence comparing to the most similar historical event (use its
zh name if present; note USGS magnitudes are Mw-class, slightly different
from CWA ML). Terminology: EEW 的中文為「強震即時警報」.
Time zone: origin_utc is UTC — ALWAYS present the event date/time in
Taiwan local time (UTC+8; e.g. origin_utc 07-30T16:58 is 台灣時間 7月31日
凌晨0時58分), matching how CWA reports it. Never show the UTC date as if
it were the local date.

Hard rules: use ONLY numbers present in the record — never invent data.
State times as 發震後 X 秒 / origin+Xs. If alert_fired is false for a
moderate event, frame the silence as correct restraint, not failure. End
each language version with one sentence noting this is an automated
research-prototype report, not official information. Do not exaggerate.

Audit record JSON:
"""


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
        payload = PROMPT + json.dumps(rec, ensure_ascii=False, indent=1)
        try:
            text = gemini(model, key, payload)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                fb = pick_fallback_model(key)
                if fb and fb != model:
                    print(f"[report] {model} not available -> {fb}")
                    model = fb
                    text = gemini(model, key, payload)
                else:
                    print(f"[report] no usable model ({e}) — aborting")
                    return
            elif e.code in (429, 503):
                print(f"[report] rate-limited ({e.code}), retrying in 30s")
                time.sleep(30)
                text = gemini(model, key, payload)
            else:
                print(f"[report] API error {e.code}: "
                      f"{e.read()[:300]}", file=sys.stderr)
                return
        if not looks_complete(text):
            print(f"[report] {f.stem}: output looks truncated "
                  f"({len(text)} chars) — retrying once")
            time.sleep(3)
            text = gemini(model, key, payload + "\n\nIMPORTANT: output BOTH "
                          "language versions in full, Traditional Chinese "
                          "first, then '---', then English.")
            if not looks_complete(text):
                print(f"[report] {f.stem}: still short — keeping anyway, "
                      "regenerate later with --force")
        rec["report_md"] = text
        rec["report_model"] = model
        f.write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                     encoding="utf-8")
        md = arch / f"report_{rec.get('id', f.stem)}.md"
        md.write_text(text + "\n", encoding="utf-8")
        print(f"[report] {f.stem}: {len(text)} chars ({model}) -> "
              f"{md.name}")
        time.sleep(2)   # stay far under free-tier RPM


if __name__ == "__main__":
    main()
