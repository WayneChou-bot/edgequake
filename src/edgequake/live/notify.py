"""Alert notification channels for the live engine (Phase 5).

Channels (both free):
  * Email via Gmail SMTP  — env: EQ_SMTP_USER, EQ_SMTP_PASS (App Password,
    NOT the account password: Google Account -> Security -> 2-Step
    Verification -> App passwords), EQ_MAIL_TO (comma-separated)
  * Telegram Bot          — env: EQ_TG_TOKEN (from @BotFather),
    EQ_TG_CHAT (your chat id; message the bot once, then read
    https://api.telegram.org/bot<TOKEN>/getUpdates)

Design rules:
  * sends run in daemon threads — a slow SMTP handshake must never block
    the picking loop
  * the ENGINE decides when to notify (quality-gated PWS flag, once per
    event); this module only delivers
  * every message carries the research-prototype disclaimer — this is a
    personal system, not an official CWA alert

Test the channels standalone:
    python -m edgequake.live.notify --test
"""
from __future__ import annotations

import json
import os
import smtplib
import ssl
import threading
import urllib.parse
import urllib.request
from email.mime.text import MIMEText

DISCLAIMER = ("此為 EdgeQuake 個人研究原型自動發送，非中央氣象署官方警報。 "
              "Research prototype — NOT an official CWA alert.")


class Notifier:
    def __init__(self, email_cfg=None, telegram_cfg=None):
        self.email_cfg = email_cfg
        self.telegram_cfg = telegram_cfg

    @classmethod
    def from_env(cls) -> "Notifier":
        email_cfg = telegram_cfg = None
        if os.environ.get("EQ_SMTP_USER") and os.environ.get("EQ_SMTP_PASS"):
            email_cfg = dict(
                user=os.environ["EQ_SMTP_USER"],
                password=os.environ["EQ_SMTP_PASS"],
                to=[a.strip() for a in
                    os.environ.get("EQ_MAIL_TO",
                                   os.environ["EQ_SMTP_USER"]).split(",")],
                host=os.environ.get("EQ_SMTP_HOST", "smtp.gmail.com"),
                port=int(os.environ.get("EQ_SMTP_PORT", "465")),
            )
        if os.environ.get("EQ_TG_TOKEN") and os.environ.get("EQ_TG_CHAT"):
            telegram_cfg = dict(token=os.environ["EQ_TG_TOKEN"],
                                chat=os.environ["EQ_TG_CHAT"])
        return cls(email_cfg, telegram_cfg)

    @property
    def channels(self) -> list[str]:
        out = []
        if self.email_cfg:
            out.append("email")
        if self.telegram_cfg:
            out.append("telegram")
        return out

    # ------------------------------------------------------------- sending
    def send(self, subject: str, body: str, block: bool = False) -> None:
        """Fire-and-forget on daemon threads (block=True only for --test)."""
        body = f"{body}\n\n{DISCLAIMER}"
        threads = []
        if self.email_cfg:
            threads.append(threading.Thread(
                target=self._email, args=(subject, body), daemon=True))
        if self.telegram_cfg:
            threads.append(threading.Thread(
                target=self._telegram, args=(subject, body), daemon=True))
        for t in threads:
            t.start()
        if block:
            for t in threads:
                t.join(timeout=20)

    def _email(self, subject, body):
        cfg = self.email_cfg
        try:
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = cfg["user"]
            msg["To"] = ", ".join(cfg["to"])
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=ctx,
                                  timeout=15) as s:
                s.login(cfg["user"], cfg["password"])
                s.sendmail(cfg["user"], cfg["to"], msg.as_string())
            print(f"[notify] email sent to {len(cfg['to'])} recipient(s)")
        except Exception as e:  # never crash the engine over a notification
            print(f"[notify] email FAILED: {e}")

    def _telegram(self, subject, body):
        cfg = self.telegram_cfg
        try:
            data = urllib.parse.urlencode({
                "chat_id": cfg["chat"],
                "text": f"{subject}\n{body}",
            }).encode()
            url = f"https://api.telegram.org/bot{cfg['token']}/sendMessage"
            with urllib.request.urlopen(
                    urllib.request.Request(url, data=data), timeout=10) as r:
                ok = json.load(r).get("ok")
            print(f"[notify] telegram sent: ok={ok}")
        except Exception as e:
            print(f"[notify] telegram FAILED: {e}")


def format_alert(ev: dict, mode_label: str) -> tuple[str, str]:
    """Build (subject, body) from a live-engine event dict."""
    mag = ev.get("mag")
    mtxt = f"M{mag:.1f}" if mag is not None else "M --"
    cty = ev.get("cty") or []
    alerts = [c["name"] for c in cty if c.get("alert")]
    imax = max((c.get("i", 0) for c in cty), default=0)

    def fmt_i(i):
        return {5: "5弱", 5.5: "5強", 6: "6弱", 6.5: "6強"}.get(i, str(i))

    etas = {c["name"]: c.get("eta") for c in cty}
    eta_lines = " · ".join(
        f"{n} S波 {etas[n]:.0f}s" for n in ("Taipei", "Taichung", "Kaohsiung")
        if etas.get(n) is not None and etas[n] > 0)
    subject = f"⚠ EdgeQuake {mtxt} 地震警報（研究原型）"
    body = (
        f"{mtxt} est · 震央 {ev.get('lat', 0):.2f}N {ev.get('lon', 0):.2f}E "
        f"· 深度 {ev.get('depth', 0):.0f} km\n"
        f"定位測站 {ev.get('k', 0)} · 最大預測震度 {fmt_i(imax)}\n"
        f"達 PWS 門檻縣市 ({len(alerts)}): {', '.join(alerts[:8])}"
        f"{'…' if len(alerts) > 8 else ''}\n"
        f"{eta_lines}\n"
        f"來源: {mode_label}"
    )
    return subject, body


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    n = Notifier.from_env()
    print(f"channels configured: {n.channels or 'NONE (set env vars)'}")
    if args.test and n.channels:
        n.send("EdgeQuake 通知測試", "如果你看到這封，通知管道就通了。",
               block=True)
