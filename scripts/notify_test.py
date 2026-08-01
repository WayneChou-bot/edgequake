"""Test the notification channels (email / Telegram) from env vars.

Usage (after setting the env vars below in the SAME terminal):
    python scripts/notify_test.py

Windows (cmd):
    set EQ_SMTP_USER=you@gmail.com
    set EQ_SMTP_PASS=xxxxxxxxxxxxxxxx     <- Gmail App Password (16 chars)
    set EQ_MAIL_TO=you@gmail.com
    set EQ_TG_TOKEN=123456:ABC-...        <- from @BotFather
    set EQ_TG_CHAT=123456789              <- your chat id
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgequake.live.notify import Notifier  # noqa: E402

n = Notifier.from_env()
print("channels configured:", n.channels or "NONE — 環境變數還沒設（見檔頭說明）")
if n.channels:
    n.send("EdgeQuake 通知測試", "如果你看到這封，通知管道就通了。", block=True)
    print("done — 檢查你的信箱 / Telegram")
