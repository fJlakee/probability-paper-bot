from __future__ import annotations

import httpx


class Notifier:
    def __init__(self, enabled: bool, token: str, chat_id: str):
        self.enabled, self.token, self.chat_id = enabled, token, chat_id

    def send(self, text: str) -> None:
        print(text, flush=True)
        if not self.enabled:
            return
        if not self.token or not self.chat_id:
            raise RuntimeError("Telegram enabled, but token/chat_id is empty")
        response = httpx.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                              json={"chat_id": self.chat_id, "text": text}, timeout=20)
        response.raise_for_status()

