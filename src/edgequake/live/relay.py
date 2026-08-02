"""State relay: push engine state to Upstash Redis so the public console
(Vercel /api/state) can show live detection to anyone, anywhere.

Setup:
  1. Vercel dashboard -> Storage / Marketplace -> Upstash Redis (free tier)
     -> create database; Vercel auto-injects UPSTASH_REDIS_REST_URL/TOKEN
     into the deployment (used by vercel/api/state.js).
  2. On the engine VM, set the SAME credentials:
       export EQ_REDIS_URL='https://xxxx.upstash.io'
       export EQ_REDIS_TOKEN='AX...'
  3. run_live.py picks them up automatically.

Cadence (decided by run_live): heartbeat every ~30 s when idle, every ~2 s
during an active event — ~90k commands/month, well inside the free tier.
The key expires after 90 s, so a dead engine disappears from the public
page instead of showing stale data.
"""
from __future__ import annotations

import json
import threading
import urllib.request


class UpstashPusher:
    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.token = token
        self._latest = None
        self._kick = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        self.ok = 0
        self.fail = 0

    def push(self, state: dict) -> None:
        """Queue the latest state (mailbox of one; never blocks the loop)."""
        self._latest = state
        self._kick.set()

    def _run(self):
        while True:
            self._kick.wait()
            self._kick.clear()
            state = self._latest
            if state is None:
                continue
            try:
                body = json.dumps(
                    ["SET", "eq_state", json.dumps(state), "EX", "90"]
                ).encode()
                req = urllib.request.Request(
                    self.url, data=body,
                    headers={"Authorization": f"Bearer {self.token}",
                             "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    r.read(64)
                self.ok += 1
            except Exception as e:
                self.fail += 1
                if self.fail % 20 == 1:
                    print(f"[relay] push failed ({e})")
