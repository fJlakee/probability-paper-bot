from __future__ import annotations

import httpx
import pandas as pd
import time


class BinancePublicClient:
    def __init__(self, base_url: str, timeout: float = 20.0):
        self.client = httpx.Client(base_url=base_url, timeout=timeout, headers={"User-Agent": "probability-paper-bot/0.1"})

    def close(self) -> None:
        self.client.close()

    def _get(self, path: str, params: dict | None = None):
        for attempt in range(6):
            response = self.client.get(path, params=params)
            if response.status_code not in (418, 429) and response.status_code < 500:
                response.raise_for_status()
                return response.json()
            delay = min(30.0, 2 ** attempt)
            retry_after = response.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after else delay)
        response.raise_for_status()

    def top_symbols(self, quote_asset: str, limit: int, allowlist: list[str], denylist: list[str]) -> list[str]:
        info = self._get("/fapi/v1/exchangeInfo")
        tradable = {
            item["symbol"] for item in info["symbols"]
            if item.get("quoteAsset") == quote_asset
            and item.get("contractType") == "PERPETUAL"
            and item.get("status") == "TRADING"
        }
        if allowlist:
            return [s for s in allowlist if s in tradable and s not in denylist][:limit]
        tickers = self._get("/fapi/v1/ticker/24hr")
        ranked = sorted(
            (x for x in tickers if x["symbol"] in tradable and x["symbol"] not in denylist),
            key=lambda x: float(x.get("quoteVolume", 0)),
            reverse=True,
        )
        return [x["symbol"] for x in ranked[:limit]]

    def klines(self, symbol: str, interval: str, limit: int, start_time: int | None = None) -> pd.DataFrame:
        params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1500)}
        if start_time is not None:
            params["startTime"] = start_time
        rows = self._get("/fapi/v1/klines", params)
        columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
        df = pd.DataFrame(rows, columns=columns)
        numeric = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_base", "taker_quote"]
        df[numeric] = df[numeric].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        return df

    def historical_klines(self, symbol: str, interval: str, bars: int) -> pd.DataFrame:
        interval_ms = _interval_ms(interval)
        start = int(time.time() * 1000) - bars * interval_ms
        chunks = []
        remaining = bars
        while remaining > 0:
            requested = min(1500, remaining)
            chunk = self.klines(symbol, interval, requested, start)
            if chunk.empty:
                break
            chunks.append(chunk)
            next_start = int(chunk.open_time.iloc[-1].timestamp() * 1000) + interval_ms
            if next_start <= start:
                break
            start = next_start
            remaining -= len(chunk)
            if len(chunk) < requested:
                break
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True).drop_duplicates("open_time").tail(bars).reset_index(drop=True)

    def klines_since(self, symbol: str, interval: str, start_time: int) -> pd.DataFrame:
        interval_ms = _interval_ms(interval)
        chunks = []
        cursor = start_time
        now_ms = int(time.time() * 1000)
        while cursor < now_ms:
            chunk = self.klines(symbol, interval, 1500, cursor)
            if chunk.empty:
                break
            chunks.append(chunk)
            next_cursor = int(chunk.open_time.iloc[-1].timestamp() * 1000) + interval_ms
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(chunk) < 1500:
                break
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True).drop_duplicates("open_time").reset_index(drop=True)


def _interval_ms(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    factors = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    if unit not in factors:
        raise ValueError(f"Unsupported interval: {interval}")
    return value * factors[unit]
