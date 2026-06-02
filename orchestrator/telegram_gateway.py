"""
telegram_gateway.py — Bidirectional robot ↔ Telegram communication.

Polls Telegram for user commands, injects them into the LLM conversation loop.
Relays robot's say() output and camera images back to Telegram.
"""
import os, time, threading, queue, json
import requests


class TelegramGateway:
    """Poll Telegram for messages, forward to LLM, and relay replies back."""

    def __init__(self, token: str, chat_id: str, message_queue: queue.Queue,
                 poll_interval: float = 1.0):
        self._token = token
        self._chat_id = chat_id
        self._queue = message_queue
        self._interval = poll_interval
        self._last_update_id = 0
        self._bot_url = f"https://api.telegram.org/bot{token}"
        self._running = False

    # ── Outgoing (robot → Telegram) ─────────────────────────────────────────

    def send_message(self, text: str):
        """Send a text message to the configured Telegram chat."""
        if not self._token:
            return
        try:
            requests.post(f"{self._bot_url}/sendMessage", json={
                "chat_id": self._chat_id,
                "text": text[:4000],  # Telegram 4096 char limit
            }, timeout=5)
        except Exception as e:
            print(f"  [Telegram] send error: {e}")

    def send_photo(self, image_path: str):
        """Send an annotated camera frame to Telegram."""
        if not self._token or not os.path.exists(image_path):
            return
        try:
            with open(image_path, "rb") as f:
                requests.post(f"{self._bot_url}/sendPhoto",
                              data={"chat_id": self._chat_id},
                              files={"photo": f},
                              timeout=10)
        except Exception as e:
            print(f"  [Telegram] photo error: {e}")

    def send_status(self, status_text: str):
        """Send a formatted status update."""
        self.send_message(f"🤖 {status_text}")

    # ── Incoming polling (Telegram → LLM) ──────────────────────────────────

    def start_polling(self):
        """Start the background polling thread."""
        if self._running or not self._token:
            return
        self._running = True
        t = threading.Thread(target=self._poll_loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def _poll_loop(self):
        """Background thread: poll Telegram for new messages."""
        while self._running:
            try:
                resp = requests.get(
                    f"{self._bot_url}/getUpdates",
                    params={
                        "offset": self._last_update_id + 1,
                        "timeout": 10,
                    },
                    timeout=15,
                )
                if not resp.ok:
                    time.sleep(self._interval)
                    continue

                for update in resp.json().get("result", []):
                    uid = update["update_id"]
                    self._last_update_id = uid

                    msg = update.get("message", {})
                    text = msg.get("text", "").strip()
                    chat_id = str(msg.get("chat", {}).get("id", ""))

                    # Only accept messages from our configured chat
                    if chat_id != self._chat_id:
                        continue

                    if text:
                        self._queue.put(("[telegram]", text))

            except requests.exceptions.Timeout:
                pass  # long poll timeout, just retry
            except Exception as e:
                print(f"  [Telegram] poll error: {e}")
                time.sleep(5)  # back off on error

            time.sleep(self._interval)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def test_connection(self) -> bool:
        """Verify the bot token and chat access."""
        if not self._token:
            print("  [Telegram] No token configured. Skipping Telegram.")
            return False
        try:
            resp = requests.get(f"{self._bot_url}/getMe", timeout=5)
            if not resp.ok:
                print(f"  [Telegram] Invalid token: {resp.status_code}")
                return False
            bot_name = resp.json().get("result", {}).get("first_name", "unknown")
            print(f"  [Telegram] Connected as @{bot_name}")

            # Send a test message
            self.send_message("🤖 PhysicalAI orchestrator online.")
            return True
        except Exception as e:
            print(f"  [Telegram] Connection test failed: {e}")
            return False
