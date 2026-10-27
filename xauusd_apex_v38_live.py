"""
================================================================================
XAUUSD APEX V38.0 — LIVE PRODUCTION ENGINE
Institutional quant risk + real broker execution (Paper / OANDA).

SAFETY DEFAULTS
---------------
- TRADING_MODE=paper (default). Live butuh CONFIRM_LIVE env.
- Semua API call punya timeout + retry + backoff.
- State persisten via Gist (fallback file).
- Kill-switch persisten multi-trigger.
- Reconciliation on startup; orphan -> HALT.
================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import argparse
import hashlib
import tempfile
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from typing import Tuple, Dict, Any, List, Optional, Iterator

import requests
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier, VotingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from scipy.stats import norm


# ==============================================================================
# 0. LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("V38_LIVE")


# ==============================================================================
# 1. CONFIG (env-driven, safe defaults)
# ==============================================================================
def _env(k: str, d: str, cast=str):
    v = os.getenv(k, d)
    try:
        return cast(v)
    except Exception:
        return d


class Cfg:
    MODE: str = _env("TRADING_MODE", "paper").lower()  # paper | oanda-practice | oanda-live
    CONFIRM_LIVE: str = _env("CONFIRM_LIVE", "")
    SYMBOL: str = _env("SYMBOL", "XAU_USD")
    TIMEFRAME: str = "H1"
    BARS_FOR_TRAIN: int = _env("BARS_FOR_TRAIN", "5000", int)
    MIN_DATA_BARS: int = _env("MIN_DATA_BARS", "2000", int)

    # Broker credentials (OANDA)
    OANDA_TOKEN: str = _env("OANDA_API_TOKEN", "")
    OANDA_ACCOUNT: str = _env("OANDA_ACCOUNT_ID", "")
    OANDA_HOST_PRACTICE: str = "https://api-fxpractice.oanda.com"
    OANDA_HOST_LIVE: str = "https://api-fxtrade.oanda.com"

    # State
    GIST_ID: str = _env("GIST_ID", "")
    GIST_TOKEN: str = _env("GIST_TOKEN", "")
    GIST_FILENAME: str = _env("GIST_FILENAME", "xauusd_v38_state.json")
    LOCAL_STATE: str = _env("LOCAL_STATE", "state/xauusd_v38_state.json")

    # Alerts
    TELEGRAM_TOKEN: str = _env("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID: str = _env("TELEGRAM_CHAT_ID", "")

    # Risk
    MAX_RISK_PER_TRADE: float = _env("MAX_RISK_PER_TRADE", "0.01", float)
    MAX_POSITION_FRAC: float = _env("MAX_POSITION_FRAC", "0.5", float)
    MAX_DRAWDOWN_KILL: float = _env("MAX_DRAWDOWN_KILL", "0.10", float)
    MAX_DAILY_LOSS_KILL: float = _env("MAX_DAILY_LOSS_KILL", "0.03", float)
    MAX_CONSECUTIVE_LOSSES: int = _env("MAX_CONSECUTIVE_LOSSES", "5", int)
    MIN_CONFIDENCE: float = _env("MIN_CONFIDENCE", "0.55", float)
    EQUITY_START: float = _env("EQUITY_START", "100000", float)

    # Labels
    PT_MULTIPLIER: float = _env("PT_MULTIPLIER", "2.0", float)
    SL_MULTIPLIER: float = _env("SL_MULTIPLIER", "1.0", float)
    MAX_VERTICAL_BARS: int = _env("MAX_VERTICAL_BARS", "12", int)

    # Costs (fallback estimate; broker spread API overrides)
    SPREAD_USD_DEFAULT: float = _env("SPREAD_USD_DEFAULT", "0.20", float)
    SLIPPAGE_USD: float = _env("SLIPPAGE_USD", "0.10", float)
    MAX_SPREAD_USD: float = _env("MAX_SPREAD_USD", "1.00", float)

    # Market guards
    CLOSE_BLACKOUT_MIN: int = _env("CLOSE_BLACKOUT_MIN", "30", int)
    OPEN_BLACKOUT_MIN: int = _env("OPEN_BLACKOUT_MIN", "5", int)

    # Heartbeat
    MAX_HEARTBEAT_AGE_SEC: int = _env("MAX_HEARTBEAT_AGE_SEC", "7200", int)  # 2h

    # Orphan handling
    ORPHAN_ACTION: str = _env("ORPHAN_ACTION", "halt")  # halt | adopt | close

    # ML
    RANDOM_STATE: int = 42
    N_STRATEGY_TRIALS: int = _env("N_STRATEGY_TRIALS", "1", int)

    @classmethod
    def validate(cls) -> None:
        assert cls.MODE in ("paper", "oanda-practice", "oanda-live")
        assert 0 < cls.MAX_RISK_PER_TRADE <= 0.05
        assert 0 < cls.MAX_POSITION_FRAC <= 1.0
        assert 0 < cls.MAX_DRAWDOWN_KILL <= 0.5
        assert 0 < cls.MAX_DAILY_LOSS_KILL <= 0.5
        assert 0.5 <= cls.MIN_CONFIDENCE < 1.0
        assert cls.PT_MULTIPLIER > 0 and cls.SL_MULTIPLIER > 0
        assert cls.MAX_VERTICAL_BARS > 0
        assert cls.ORPHAN_ACTION in ("halt", "adopt", "close")

        if cls.MODE == "oanda-live":
            if cls.CONFIRM_LIVE != "I_UNDERSTAND_THE_RISK":
                raise RuntimeError(
                    "MODE=oanda-live membutuhkan CONFIRM_LIVE=I_UNDERSTAND_THE_RISK"
                )
            if not cls.OANDA_TOKEN or not cls.OANDA_ACCOUNT:
                raise RuntimeError("OANDA credentials wajib untuk live.")
        if cls.MODE == "oanda-practice":
            if not cls.OANDA_TOKEN or not cls.OANDA_ACCOUNT:
                raise RuntimeError("OANDA credentials wajib untuk practice.")


def seed_everything(seed: int) -> None:
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    return obj


# ==============================================================================
# 2. STATE STORE (Gist + local file, atomic, ETag)
# ==============================================================================
class StateStore:
    def __init__(self):
        self._etag: Optional[str] = None

    @property
    def gist_enabled(self) -> bool:
        return bool(Cfg.GIST_ID and Cfg.GIST_TOKEN)

    def _local_load(self) -> Dict[str, Any]:
        try:
            with open(Cfg.LOCAL_STATE) as f:
                return json.load(f)
        except Exception:
            return {}

    def _local_save(self, data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(Cfg.LOCAL_STATE) or ".", exist_ok=True)
        tmp = Cfg.LOCAL_STATE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(json_safe(data), f, indent=2, sort_keys=True, allow_nan=False)
        os.replace(tmp, Cfg.LOCAL_STATE)

    def _gist_headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {Cfg.GIST_TOKEN}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def load(self) -> Dict[str, Any]:
        # Prefer Gist (cross-runner), fallback local
        if self.gist_enabled:
            try:
                r = requests.get(
                    f"https://api.github.com/gists/{Cfg.GIST_ID}",
                    headers=self._gist_headers(),
                    timeout=15,
                )
                if r.status_code == 404:
                    return self._local_load()
                r.raise_for_status()
                self._etag = r.headers.get("ETag")
                content = r.json().get("files", {}).get(Cfg.GIST_FILENAME, {}).get("content")
                if content:
                    return json.loads(content)
            except Exception as e:
                logger.warning(f"[STATE] Gist load gagal ({e}); fallback local.")
        return self._local_load()

    def save(self, data: Dict[str, Any]) -> None:
        self._local_save(data)
        if self.gist_enabled:
            try:
                payload = {
                    "files": {
                        Cfg.GIST_FILENAME: {
                            "content": json.dumps(json_safe(data), indent=2, sort_keys=True)
                        }
                    }
                }
                headers = self._gist_headers()
                if self._etag:
                    headers["If-Match"] = self._etag
                r = requests.patch(
                    f"https://api.github.com/gists/{Cfg.GIST_ID}",
                    headers=headers, json=payload, timeout=15,
                )
                if r.status_code == 412:
                    logger.warning("[STATE] Gist ETag conflict; refresh lalu coba lagi.")
                    # Refresh and retry once
                    self.load()
                    headers = self._gist_headers()
                    if self._etag:
                        headers["If-Match"] = self._etag
                    r = requests.patch(
                        f"https://api.github.com/gists/{Cfg.GIST_ID}",
                        headers=headers, json=payload, timeout=15,
                    )
                r.raise_for_status()
                self._etag = r.headers.get("ETag")
            except Exception as e:
                logger.error(f"[STATE] Gist save gagal ({e}); state lokal tetap tersimpan.")


def default_state() -> Dict[str, Any]:
    return {
        "version": "38.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "heartbeat_at": None,
        "equity": Cfg.EQUITY_START,
        "peak_equity": Cfg.EQUITY_START,
        "day_start_equity": Cfg.EQUITY_START,
        "day_start_date": None,
        "consecutive_losses": 0,
        "kill_switch": {"tripped": False, "reason": None, "tripped_at": None},
        "positions": {},          # dict[str, dict] keyed by position_id
        "orders_pending": {},     # dict[str, dict]
        "last_signal_key": None,
        "last_bar_ts": None,
        "trades": [],             # append-only
        "events": [],             # append-only
    }


# ==============================================================================
# 3. TELEGRAM
# ==============================================================================
class Telegram:
    def __init__(self):
        self.enabled = bool(Cfg.TELEGRAM_TOKEN and Cfg.TELEGRAM_CHAT_ID)

    def send(self, text: str) -> bool:
        if not self.enabled:
            logger.info("[TG] disabled.")
            return False
        url = f"https://api.telegram.org/bot{Cfg.TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": Cfg.TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        for attempt in range(3):
            try:
                r = requests.post(url, json=payload, timeout=10)
                if r.status_code == 200 and r.json().get("ok"):
                    return True
                if r.status_code == 429:
                    time.sleep(int(r.json().get("parameters", {}).get("retry_after", 5)) + 1)
                    continue
                if 500 <= r.status_code < 600:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                r.raise_for_status()
            except requests.Timeout:
                # JANGAN retry: bisa duplikat
                logger.error("[TG] timeout; tidak di-retry.")
                return False
            except requests.RequestException as e:
                logger.warning(f"[TG] attempt {attempt+1} gagal: {e}")
                time.sleep(min(2 ** attempt, 30))
        return False


# ==============================================================================
# 4. BROKER ABSTRACTION
# ==============================================================================
@dataclass
class Position:
    position_id: str
    symbol: str
    side: str           # "long" | "short"
    units: float
    entry_price: float
    sl: float
    tp: float
    bar_ts: str
    opened_at: str
    broker_id: Optional[str] = None
    unrealized_pnl: float = 0.0


@dataclass
class OrderResult:
    ok: bool
    broker_id: Optional[str]
    fill_price: Optional[float]
    units: Optional[float]
    error: Optional[str] = None


class Broker:
    def get_account(self) -> Dict[str, Any]: ...
    def get_candles(self, count: int) -> pd.DataFrame: ...
    def get_price(self) -> Tuple[float, float]:  # bid, ask
        ...
    def get_positions(self) -> List[Position]: ...
    def open_position(self, side: str, units: float, sl: float, tp: float,
                      client_id: str, bar_ts: str) -> OrderResult: ...
    def close_position(self, position_id: str) -> bool: ...
    def is_market_open(self) -> bool: ...


# ---------- Paper broker ----------
class PaperBroker(Broker):
    """Simulated fill using last bar close + fixed slippage. State in memory; engine persists."""

    def __init__(self, state: Dict[str, Any], data: pd.DataFrame, spread: float = None):
        self._state = state
        self._data = data
        self._spread = spread if spread is not None else Cfg.SPREAD_USD_DEFAULT
        self._next_id = 1

    def get_account(self) -> Dict[str, Any]:
        return {
            "equity": self._state["equity"],
            "balance": self._state["equity"],
            "currency": "USD",
        }

    def get_candles(self, count: int) -> pd.DataFrame:
        return self._data.tail(count).copy()

    def get_price(self) -> Tuple[float, float]:
        mid = float(self._data["close"].iloc[-1])
        half = self._spread / 2.0
        return mid - half, mid + half

    def get_positions(self) -> List[Position]:
        return [Position(**p) for p in self._state["positions"].values()]

    def open_position(self, side: str, units: float, sl: float, tp: float,
                      client_id: str, bar_ts: str) -> OrderResult:
        if client_id in {p["broker_id"] for p in self._state["positions"].values()}:
            return OrderResult(False, None, None, None, "duplicate client_id")

        bid, ask = self.get_price()
        if side == "long":
            fill = ask + Cfg.SLIPPAGE_USD
        else:
            fill = bid - Cfg.SLIPPAGE_USD

        pid = f"paper_{self._next_id}_{bar_ts}"
        self._next_id += 1
        pos = Position(
            position_id=pid, symbol=Cfg.SYMBOL, side=side, units=abs(units),
            entry_price=fill, sl=sl, tp=tp, bar_ts=bar_ts,
            opened_at=datetime.now(timezone.utc).isoformat(),
            broker_id=client_id,
        )
        self._state["positions"][pid] = asdict(pos)
        return OrderResult(True, pid, fill, abs(units))

    def close_position(self, position_id: str) -> bool:
        pos = self._state["positions"].get(position_id)
        if not pos:
            return False
        bid, ask = self.get_price()
        exit_price = bid if pos["side"] == "long" else ask
        pnl = (exit_price - pos["entry_price"]) * pos["units"] * (
            1 if pos["side"] == "long" else -1
        )
        pnl -= (Cfg.SPREAD_USD_DEFAULT + Cfg.SLIPPAGE_USD) * pos["units"]
        self._state["equity"] += pnl
        self._state["peak_equity"] = max(self._state["peak_equity"], self._state["equity"])
        self._state["consecutive_losses"] = (
            self._state["consecutive_losses"] + 1 if pnl < 0 else 0
        )
        self._state["trades"].append({
            "position_id": position_id,
            "side": pos["side"], "units": pos["units"],
            "entry_price": pos["entry_price"], "exit_price": exit_price,
            "pnl": pnl, "bar_ts": pos["bar_ts"],
            "closed_at": datetime.now(timezone.utc).isoformat(),
        })
        del self._state["positions"][position_id]
        return True

    def is_market_open(self) -> bool:
        now = datetime.now(timezone.utc)
        return now.weekday() < 5  # Mon-Fri UTC (paper approximation)


# ---------- OANDA broker ----------
class OandaBroker(Broker):
    """
    OANDA v20 REST. Practice default; live butuh CONFIRM_LIVE.
    Docs: https://developer.oanda.com/rest-live-v20/
    """

    def __init__(self):
        host = (
            Cfg.OANDA_HOST_LIVE if Cfg.MODE == "oanda-live"
            else Cfg.OANDA_HOST_PRACTICE
        )
        self.base = host
        self.account = Cfg.OANDA_ACCOUNT
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {Cfg.OANDA_TOKEN}",
            "Content-Type": "application/json",
        })

    def _req(self, method: str, path: str, **kw) -> Dict[str, Any]:
        url = f"{self.base}{path}"
        for attempt in range(4):
            try:
                r = self.session.request(method, url, timeout=15, **kw)
                if r.status_code == 429:
                    time.sleep(int(r.headers.get("Retry-After", "5")) + 1)
                    continue
                if 500 <= r.status_code < 600:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                r.raise_for_status()
                return r.json()
            except requests.Timeout:
                raise
            except requests.RequestException as e:
                if attempt == 3:
                    raise
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError("OANDA request gagal setelah retries.")

    def get_account(self) -> Dict[str, Any]:
        j = self._req("GET", f"/v3/accounts/{self.account}")
        a = j["account"]
        return {
            "equity": float(a["NAV"]),
            "balance": float(a["balance"]),
            "currency": a["currency"],
            "margin_used": float(a.get("marginUsed", 0)),
            "margin_available": float(a.get("marginAvailable", 0)),
        }

    def get_candles(self, count: int) -> pd.DataFrame:
        j = self._req(
            "GET",
            f"/v3/instruments/{Cfg.SYMBOL}/candles",
            params={"granularity": Cfg.TIMEFRAME, "count": count, "price": "M"},
        )
        rows = []
        for c in j.get("candles", []):
            if not c.get("complete", False):
                continue  # hanya closed bar
            m = c["mid"]
            rows.append({
                "timestamp": pd.Timestamp(c["time"]).tz_convert("UTC"),
                "open": float(m["o"]),
                "high": float(m["h"]),
                "low": float(m["l"]),
                "close": float(m["c"]),
                "volume": int(c["volume"]),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("Tidak ada candle closed dari OANDA.")
        return df

    def get_price(self) -> Tuple[float, float]:
        j = self._req(
            "GET",
            f"/v3/accounts/{self.account}/pricing",
            params={"instruments": Cfg.SYMBOL},
        )
        p = j["prices"][0]
        bid = float(p["bids"][0]["price"])
        ask = float(p["asks"][0]["price"])
        return bid, ask

    def get_positions(self) -> List[Position]:
        j = self._req("GET", f"/v3/accounts/{self.account}/positions")
        out: List[Position] = []
        for p in j.get("positions", []):
            instrument = p["instrument"]
            if instrument != Cfg.SYMBOL:
                continue
            long_units = float(p["long"]["units"])
            short_units = float(p["short"]["units"])
            if long_units == 0 and short_units == 0:
                continue
            side = "long" if long_units > 0 else "short"
            units = abs(long_units if side == "long" else short_units)
            side_data = p[side]
            entry = float(side_data["averagePrice"])
            sl, tp = None, None
            # Fetch trade details for SL/TP
            try:
                trade_id = side_data.get("tradeIDs", [None])[0]
                if trade_id:
                    tj = self._req("GET", f"/v3/accounts/{self.account}/trades/{trade_id}")
                    t = tj["trade"]
                    sl = float(t["stopLossOrder"]["price"]) if t.get("stopLossOrder") else None
                    tp = float(t["takeProfitOrder"]["price"]) if t.get("takeProfitOrder") else None
                    client_id = t.get("clientExtensions", {}).get("id", trade_id)
                    bar_ts = t.get("clientExtensions", {}).get("tag", "")
                else:
                    client_id, bar_ts = p.get("id"), ""
            except Exception:
                client_id, bar_ts = p.get("id"), ""
            unrealized = float(side_data.get("unrealizedPL", 0))
            out.append(Position(
                position_id=p["id"],
                symbol=instrument,
                side=side,
                units=units,
                entry_price=entry,
                sl=sl if sl is not None else 0.0,
                tp=tp if tp is not None else 0.0,
                bar_ts=bar_ts,
                opened_at=datetime.now(timezone.utc).isoformat(),
                broker_id=client_id,
                unrealized_pnl=unrealized,
            ))
        return out

    def open_position(self, side: str, units: float, sl: float, tp: float,
                      client_id: str, bar_ts: str) -> OrderResult:
        signed_units = units if side == "long" else -units
        payload = {
            "order": {
                "type": "MARKET",
                "instrument": Cfg.SYMBOL,
                "units": str(int(round(signed_units))),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
                "clientExtensions": {"id": client_id, "tag": bar_ts},
                "stopLossOnFill": {
                    "price": f"{sl:.2f}",
                    "timeInForce": "GTC",
                },
                "takeProfitOnFill": {
                    "price": f"{tp:.2f}",
                    "timeInForce": "GTC",
                },
            }
        }
        try:
            j = self._req("POST", f"/v3/accounts/{self.account}/orders", json=payload)
        except requests.HTTPError as e:
            body = ""
            try:
                body = e.response.text[:300]
            except Exception:
                pass
            return OrderResult(False, None, None, None, f"HTTP {e.response.status_code}: {body}")
        except Exception as e:
            return OrderResult(False, None, None, None, str(e))

        fill = j.get("orderFillTransaction")
        if not fill:
            return OrderResult(False, None, None, None, f"no fill: {json.dumps(j)[:300]}")
        pid = fill.get("id")
        price = float(fill.get("price", 0))
        filled_units = abs(float(fill.get("units", 0)))
        return OrderResult(True, pid, price, filled_units)

    def close_position(self, position_id: str) -> bool:
        try:
            self._req(
                "PUT",
                f"/v3/accounts/{self.account}/positions/{Cfg.SYMBOL}/close",
                json={"longUnits": "ALL", "shortUnits": "ALL"},
            )
            return True
        except Exception as e:
            logger.error(f"[OANDA] close_position gagal: {e}")
            return False

    def is_market_open(self) -> bool:
        now = datetime.now(timezone.utc)
        # Emas: Senin 00:00 UTC s/d Jumat 21:00 UTC (approx)
        if now.weekday() >= 5:
            return False
        if now.weekday() == 4 and now.hour >= 21:
            return False
        return True


def build_broker(state: Dict[str, Any], data: pd.DataFrame) -> Broker:
    if Cfg.MODE == "paper":
        return PaperBroker(state, data)
    if Cfg.MODE in ("oanda-practice", "oanda-live"):
        return OandaBroker()
    raise ValueError(f"Mode tidak dikenal: {Cfg.MODE}")


# ==============================================================================
# 5. RISK
# ==============================================================================
class PositionSizer:
    @staticmethod
    def size(equity: float, price: float, sl: float, confidence: float) -> float:
        """
        Risk-based sizing:
          risk_capital = equity * MAX_RISK_PER_TRADE
          sl_distance = |price - sl|
          units = risk_capital / sl_distance   (untuk XAU: 1 unit = 1 oz)
          dibatasi oleh MAX_POSITION_FRAC * equity / price (leverage guard)
        """
        if equity <= 0 or price <= 0:
            return 0.0
        sl_dist = abs(price - sl)
        if sl_dist <= 0:
            return 0.0

        risk_capital = equity * Cfg.MAX_RISK_PER_TRADE
        units_by_risk = risk_capital / sl_dist

        notional_cap = (Cfg.MAX_POSITION_FRAC * equity) / price
        units = min(units_by_risk, notional_cap)

        conf_scale = float(np.clip(
            (confidence - Cfg.MIN_CONFIDENCE) / (1 - Cfg.MIN_CONFIDENCE),
            0.0, 1.0,
        ))
        units *= conf_scale
        return max(0.0, float(units))


class KillSwitch:
    @staticmethod
    def check(state: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        ks = state.get("kill_switch", {})
        if ks.get("tripped"):
            return True, ks.get("reason", "PREVIOUSLY_TRIPPED")

        equity = float(state.get("equity", 0))
        peak = float(state.get("peak_equity", 0))
        day_start = float(state.get("day_start_equity", 0))
        consec = int(state.get("consecutive_losses", 0))

        if peak > 0 and (equity / peak - 1) <= -Cfg.MAX_DRAWDOWN_KILL:
            return True, f"MAX_DRAWDOWN ({equity/peak-1:.2%})"
        if day_start > 0 and (equity / day_start - 1) <= -Cfg.MAX_DAILY_LOSS_KILL:
            return True, f"MAX_DAILY_LOSS ({equity/day_start-1:.2%})"
        if consec >= Cfg.MAX_CONSECUTIVE_LOSSES:
            return True, f"CONSECUTIVE_LOSSES ({consec})"

        # Manual trigger via file
        if os.path.exists("/tmp/KILL_SWITCH"):
            return True, "MANUAL_FILE"

        # Heartbeat staleness (if we've run before)
        hb = state.get("heartbeat_at")
        if hb:
            try:
                age = (datetime.now(timezone.utc)
                       - datetime.fromisoformat(hb).replace(tzinfo=timezone.utc)).total_seconds()
                if age > Cfg.MAX_HEARTBEAT_AGE_SEC:
                    return True, f"HEARTBEAT_STALE ({age:.0f}s)"
            except Exception:
                pass

        return False, None

    @staticmethod
    def trip(state: Dict[str, Any], reason: str) -> None:
        state["kill_switch"] = {
            "tripped": True,
            "reason": reason,
            "tripped_at": datetime.now(timezone.utc).isoformat(),
        }


class RiskGovernor:
    @staticmethod
    def pre_trade(
        broker: Broker,
        state: Dict[str, Any],
        signal_key: str,
    ) -> Tuple[bool, str]:
        # 1. Market open?
        if not broker.is_market_open():
            return False, "MARKET_CLOSED"

        # 2. Kill switch?
        tripped, reason = KillSwitch.check(state)
        if tripped:
            return False, f"KILL_SWITCH: {reason}"

        # 3. Duplicate signal?
        if state.get("last_signal_key") == signal_key:
            return False, "DUPLICATE_SIGNAL"

        # 4. Max concurrent positions (1 for now)
        if len(state.get("positions", {})) >= 1:
            return False, "POSITION_ALREADY_OPEN"

        # 5. Spread guard
        try:
            bid, ask = broker.get_price()
            spread = ask - bid
            if spread > Cfg.MAX_SPREAD_USD:
                return False, f"SPREAD_TOO_WIDE ({spread:.2f})"
        except Exception as e:
            return False, f"PRICE_FETCH_FAILED ({e})"

        return True, "OK"


# ==============================================================================
# 6. ML PIPELINE (closed-bar + purged CV)
# ==============================================================================
class Features:
    @staticmethod
    def compute_shannon_entropy(x: np.ndarray, bins: int = 10) -> float:
        x = x[np.isfinite(x)]
        if x.size < bins:
            return np.nan
        counts, _ = np.histogram(x, bins=bins)
        total = counts.sum()
        if total == 0:
            return np.nan
        p = counts.astype(float) / total
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    @staticmethod
    def compute_hurst(prices: np.ndarray, lag_min: int = 2, lag_max: int = 19) -> float:
        prices = prices[np.isfinite(prices)]
        n = prices.size
        if n < lag_max * 2:
            return np.nan
        lags = np.arange(lag_min, lag_max + 1)
        tau = np.full(lags.size, np.nan, dtype=float)
        for k, lag in enumerate(lags):
            diff = prices[lag:] - prices[:-lag]
            s = float(np.std(diff))
            if np.isfinite(s) and s > 0:
                tau[k] = s
        valid = np.isfinite(tau) & (tau > 0)
        if valid.sum() < 2:
            return np.nan
        try:
            slope = np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)[0]
            return float(np.clip(slope, 0.0, 1.0))
        except Exception:
            return np.nan

    @classmethod
    def transform(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["_orig_idx"] = np.arange(len(df))
        close, high, low = df["close"], df["high"], df["low"]

        for lag in (1, 3, 5, 10):
            df[f"return_{lag}"] = close.pct_change(lag)
        for period in (9, 21, 50, 200):
            df[f"ema_{period}"] = close.ewm(span=period, adjust=False, min_periods=period).mean()

        df["ema_spread_9_21"] = (df["ema_9"] - df["ema_21"]) / close
        df["ema_slope_50"] = df["ema_50"].diff(3) / close

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-12))

        tr = pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        df["volatility"] = close.pct_change().rolling(20).std()

        roll_mean = close.rolling(20).mean()
        roll_std = close.rolling(20).std() + 1e-12
        df["zscore"] = (close - roll_mean) / roll_std

        rets = close.pct_change()
        df["entropy"] = rets.rolling(100).apply(
            lambda w: cls.compute_shannon_entropy(w), raw=True
        )
        df["hurst"] = close.rolling(100).apply(
            lambda w: cls.compute_hurst(w), raw=True
        )
        df = df.replace([np.inf, -np.inf], np.nan)
        return df


class Labels:
    @staticmethod
    def triple_barrier(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        atr = df["atr"].to_numpy(dtype=float)
        orig_idx = df["_orig_idx"].to_numpy(dtype=int)
        n = len(df)

        labels = np.full(n, np.nan)
        label_end_orig = np.full(n, -1, dtype=int)
        horizon = Cfg.MAX_VERTICAL_BARS

        for i in range(n - horizon):
            if not np.isfinite(atr[i]) or atr[i] <= 0:
                continue
            entry = close[i]
            upper = entry + Cfg.PT_MULTIPLIER * atr[i]
            lower = entry - Cfg.SL_MULTIPLIER * atr[i]
            end_j = min(i + horizon, n - 1)
            lab = 0
            for j in range(i + 1, end_j + 1):
                hu = high[j] >= upper
                hd = low[j] <= lower
                if hu and hd:
                    lab, end_j = np.nan, j
                    break
                if hu:
                    lab, end_j = 1, j
                    break
                if hd:
                    lab, end_j = -1, j
                    break
            labels[i] = lab
            label_end_orig[i] = int(orig_idx[end_j])

        df = df.copy()
        df["target"] = labels
        df["label_end_orig_idx"] = label_end_orig
        return df


class PurgedCV:
    def __init__(self, n_splits: int = 5, min_train_frac: float = 0.3,
                 embargo: int = 12):
        self.n_splits = n_splits
        self.min_train_frac = min_train_frac
        self.embargo = max(embargo, Cfg.MAX_VERTICAL_BARS)

    def split(self, n: int, label_end_orig_idx: np.ndarray,
              orig_idx: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        fold_size = int((n * (1 - self.min_train_frac)) // self.n_splits)
        if fold_size <= 0:
            raise ValueError("n_splits terlalu besar.")
        initial = n - fold_size * self.n_splits
        positions = np.arange(n, dtype=int)
        for i in range(self.n_splits):
            test_start = initial + i * fold_size
            test_end = n if i == self.n_splits - 1 else initial + (i + 1) * fold_size
            orig_test_start = int(orig_idx[test_start])
            purge_cutoff = orig_test_start - self.embargo
            train_mask = (positions < test_start) & (label_end_orig_idx < purge_cutoff)
            train = positions[train_mask]
            if len(train) < 100:
                continue
            yield train, positions[test_start:test_end]


def build_ensemble():
    rs = Cfg.RANDOM_STATE
    return VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(
                n_estimators=200, max_depth=6, class_weight="balanced",
                random_state=rs, n_jobs=-1)),
            ("et", ExtraTreesClassifier(
                n_estimators=200, max_depth=6, class_weight="balanced",
                random_state=rs, n_jobs=-1)),
            ("gb", HistGradientBoostingClassifier(
                max_iter=100, max_depth=4, class_weight="balanced",
                random_state=rs)),
        ],
        voting="soft", n_jobs=-1,
    )


# ==============================================================================
# 7. LIVE ENGINE
# ==============================================================================
class LiveEngine:
    def __init__(self):
        self.store = StateStore()
        self.tg = Telegram()
        self.state = self.store.load()
        self._merge_defaults()

    def _merge_defaults(self):
        d = default_state()
        for k, v in d.items():
            self.state.setdefault(k, v)

    def _event(self, level: str, kind: str, data: Dict[str, Any]):
        self.state["events"].append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level, "kind": kind, "data": json_safe(data),
        })
        # Cap audit log
        if len(self.state["events"]) > 1000:
            self.state["events"] = self.state["events"][-1000:]

    def run(self):
        Cfg.validate()
        seed_everything(Cfg.RANDOM_STATE)

        logger.info(f"[BOOT] mode={Cfg.MODE} symbol={Cfg.SYMBOL}")
        self.state["heartbeat_at"] = datetime.now(timezone.utc).isoformat()

        # ---------- 1. Fetch data ----------
        broker_for_data = build_broker(self.state, pd.DataFrame())
        try:
            df_raw = broker_for_data.get_candles(Cfg.BARS_FOR_TRAIN)
        except Exception as e:
            logger.error(f"[DATA] gagal fetch candle: {e}")
            self._event("ERROR", "data_fetch", {"err": str(e)})
            self.store.save(self.state)
            sys.exit(1)

        if len(df_raw) < Cfg.MIN_DATA_BARS:
            logger.error(f"[DATA] terlalu pendek: {len(df_raw)}")
            self._event("ERROR", "data_short", {"rows": len(df_raw)})
            self.store.save(self.state)
            sys.exit(1)

        self._event("INFO", "data_loaded", {"rows": len(df_raw),
                                             "last": str(df_raw["timestamp"].iloc[-1])})

        # ---------- 2. Reconcile broker vs state ----------
        try:
            self._reconcile(df_raw)
        except SystemExit:
            raise
        except Exception as e:
            logger.exception(f"[RECONCILE] crash: {e}")
            self._event("ERROR", "reconcile", {"err": str(e)})
            self.store.save(self.state)
            sys.exit(1)

        # ---------- 3. Kill switch check ----------
        tripped, reason = KillSwitch.check(self.state)
        if tripped:
            logger.error(f"[KILL] {reason}")
            self._event("ERROR", "kill_switch", {"reason": reason})
            self._handle_kill(tripped_reason=reason)
            self.store.save(self.state)
            sys.exit(0)

        # ---------- 4. Market hours ----------
        broker = build_broker(self.state, df_raw)
        if not broker.is_market_open():
            logger.info("[MARKET] closed — skip trading.")
            self._event("INFO", "market_closed", {})
            self.store.save(self.state)
            sys.exit(0)

        # ---------- 5. Build ML pipeline ----------
        df_full = Features.transform(df_raw)
        df = Labels.triple_barrier(df_full)
        feature_cols = [c for c in df.columns if c not in {
            "timestamp", "open", "high", "low", "close", "volume",
            "target", "_orig_idx", "label_end_orig_idx",
        }]
        df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
        if len(df) < 500:
            logger.error(f"[ML] terlalu sedikit baris trainable: {len(df)}")
            self._event("ERROR", "ml_short", {"rows": len(df)})
            self.store.save(self.state)
            sys.exit(1)

        X = df[feature_cols].to_numpy(dtype=float)
        y = (df["target"].to_numpy() + 1).astype(int)
        orig_idx = df["_orig_idx"].to_numpy(dtype=int)
        label_end_orig = df["label_end_orig_idx"].to_numpy(dtype=int)

        cv = PurgedCV()
        splits = list(cv.split(len(df), label_end_orig, orig_idx))
        if not splits:
            logger.error("[ML] tidak ada CV split valid.")
            self._event("ERROR", "cv_empty", {})
            self.store.save(self.state)
            sys.exit(1)

        # OOF predictions
        oof_idx: List[int] = []
        oof_proba: List[np.ndarray] = []
        for fold, (tr, te) in enumerate(splits):
            scaler = StandardScaler().fit(X[tr])
            model = build_ensemble()
            model.fit(scaler.transform(X[tr]), y[tr])
            proba = model.predict_proba(scaler.transform(X[te]))
            oof_idx.extend(te.tolist())
            oof_proba.extend(proba)
        order = np.argsort(np.array(oof_idx))
        oof_idx = [oof_idx[k] for k in order]
        oof_proba = [oof_proba[k] for k in order]

        # ---------- 6. Train final model on all data, predict last closed bar ----------
        last_row = df_full.iloc[[-1]]
        if last_row[feature_cols].isna().any().any():
            logger.warning("[LIVE] bar terakhir fitur NaN; abort.")
            self._event("INFO", "live_skip_nan", {})
            self.store.save(self.state)
            sys.exit(0)

        last_ts = str(last_row["timestamp"].iat[0])
        last_price = float(last_row["close"].iat[0])
        last_atr = float(last_row["atr"].iat[0])
        last_ema200 = float(last_row["ema_200"].iat[0])

        # Staleness guard
        age = datetime.now(timezone.utc) - last_row["timestamp"].iat[0]
        if age > timedelta(hours=4):
            logger.warning(f"[LIVE] data stale: {age}")
            self._event("INFO", "live_skip_stale", {"age_sec": age.total_seconds()})
            self.store.save(self.state)
            sys.exit(0)

        scaler_full = StandardScaler().fit(X)
        model_full = build_ensemble()
        model_full.fit(scaler_full.transform(X), y)
        proba_last = model_full.predict_proba(
            scaler_full.transform(last_row[feature_cols].to_numpy(dtype=float))
        )[0]

        max_p = float(proba_last.max())
        cls = int(proba_last.argmax()) - 1  # -1, 0, 1

        logger.info(
            f"[LIVE] bar={last_ts} close={last_price:.2f} "
            f"ema200={last_ema200:.2f} atr={last_atr:.2f} "
            f"proba={proba_last.round(3).tolist()} cls={cls} conf={max_p:.3f}"
        )

        # ---------- 7. Signal decision ----------
        if max_p < Cfg.MIN_CONFIDENCE or cls == 0:
            logger.info(f"[SIGNAL] NEUTRAL (conf={max_p:.3f} < {Cfg.MIN_CONFIDENCE} atau cls=0)")
            self._event("INFO", "signal_neutral", {
                "conf": max_p, "cls": cls, "bar_ts": last_ts,
            })
            self.state["last_signal_key"] = f"{last_ts}|neutral"
            self.state["last_bar_ts"] = last_ts
            self.store.save(self.state)
            sys.exit(0)

        # EMA filter
        if cls == 1 and last_price < last_ema200:
            logger.info("[SIGNAL] LONG difilter EMA200.")
            self._event("INFO", "signal_ema_filtered", {"side": "long"})
            self.state["last_signal_key"] = f"{last_ts}|filtered_long"
            self.store.save(self.state)
            sys.exit(0)
        if cls == -1 and last_price > last_ema200:
            logger.info("[SIGNAL] SHORT difilter EMA200.")
            self._event("INFO", "signal_ema_filtered", {"side": "short"})
            self.state["last_signal_key"] = f"{last_ts}|filtered_short"
            self.store.save(self.state)
            sys.exit(0)

        side = "long" if cls == 1 else "short"
        sl = (last_price - Cfg.SL_MULTIPLIER * last_atr) if side == "long" \
            else (last_price + Cfg.SL_MULTIPLIER * last_atr)
        tp = (last_price + Cfg.PT_MULTIPLIER * last_atr) if side == "long" \
            else (last_price - Cfg.PT_MULTIPLIER * last_atr)

        signal_key = f"{last_ts}|{side}|{round(last_price, 2)}"

        # ---------- 8. Pre-trade risk ----------
        ok, why = RiskGovernor.pre_trade(broker, self.state, signal_key)
        if not ok:
            logger.info(f"[RISK] ditolak: {why}")
            self._event("INFO", "pre_trade_reject", {"reason": why})
            self.store.save(self.state)
            sys.exit(0)

        # ---------- 9. Size & place order ----------
        account = broker.get_account()
        equity = float(account["equity"])
        units = PositionSizer.size(equity, last_price, sl, max_p)
        if units < 0.01:
            logger.info(f"[SIZE] units terlalu kecil ({units:.4f}); skip.")
            self._event("INFO", "size_too_small", {"units": units})
            self.store.save(self.state)
            sys.exit(0)

        client_id = f"v38_{int(pd.Timestamp(last_ts).timestamp())}_{side}"
        logger.info(
            f"[ORDER] {side} units={units:.2f} entry~{last_price:.2f} "
            f"sl={sl:.2f} tp={tp:.2f} client_id={client_id}"
        )

        res = broker.open_position(side, units, sl, tp, client_id, last_ts)
        if not res.ok:
            logger.error(f"[ORDER] gagal: {res.error}")
            self._event("ERROR", "order_fail", {"err": res.error})
            self.tg.send(f"⚠️ <b>ORDER GAGAL</b>\n{side} {units:.2f} @ {last_price:.2f}\nErr: {res.error}")
            self.store.save(self.state)
            sys.exit(1)

        logger.info(f"[ORDER] filled @ {res.fill_price:.2f} units={res.units:.2f} id={res.broker_id}")
        self._event("INFO", "order_filled", {
            "side": side, "units": res.units,
            "fill_price": res.fill_price, "broker_id": res.broker_id,
            "sl": sl, "tp": tp, "client_id": client_id,
        })

        pos = Position(
            position_id=res.broker_id or client_id,
            symbol=Cfg.SYMBOL, side=side, units=float(res.units or units),
            entry_price=float(res.fill_price or last_price),
            sl=sl, tp=tp, bar_ts=last_ts,
            opened_at=datetime.now(timezone.utc).isoformat(),
            broker_id=client_id,
        )
        self.state["positions"][pos.position_id] = asdict(pos)
        self.state["last_signal_key"] = signal_key
        self.state["last_bar_ts"] = last_ts

        # Alert
        self.tg.send(
            f"<b>🚨 XAUUSD {side.upper()}</b>\n"
            f"Entry: {pos.entry_price:.2f}\n"
            f"Units: {pos.units:.2f}\n"
            f"SL: {sl:.2f} | TP: {tp:.2f}\n"
            f"Conf: {max_p*100:.1f}% | ATR: {last_atr:.2f}\n"
            f"Bar: {last_ts}\n"
            f"Mode: <b>{Cfg.MODE}</b>"
        )

        # ---------- 10. Persist ----------
        self.store.save(self.state)
        logger.info("[DONE] siklus selesai.")
        sys.exit(0)

    def _reconcile(self, df_raw: pd.DataFrame) -> None:
        """Bandingkan posisi broker vs state lokal."""
        broker = build_broker(self.state, df_raw)
        try:
            broker_positions = broker.get_positions()
        except Exception as e:
            logger.warning(f"[RECONCILE] gagal fetch posisi broker: {e}")
            self._event("WARN", "reconcile_skip", {"err": str(e)})
            return

        broker_ids = {p.position_id for p in broker_positions}
        state_ids = set(self.state["positions"].keys())

        # Broker has, state doesn't: orphan
        orphans = broker_ids - state_ids
        if orphans:
            logger.warning(f"[RECONCILE] {len(orphans)} posisi orphan di broker.")
            for pid in orphans:
                p = next(x for x in broker_positions if x.position_id == pid)
                self._event("WARN", "orphan_detected", asdict(p))
                if Cfg.ORPHAN_ACTION == "halt":
                    KillSwitch.trip(self.state, f"ORPHAN_POSITION_{pid}")
                    logger.error("[RECONCILE] ORPHAN_ACTION=halt -> kill switch.")
                    self.tg.send(f"⚠️ <b>ORPHAN POSITION</b>\n{pid} {p.side} {p.units}\nTrading HALTED.")
                    self.store.save(self.state)
                    sys.exit(0)
                elif Cfg.ORPHAN_ACTION == "adopt":
                    self.state["positions"][pid] = asdict(p)
                    logger.info(f"[RECONCILE] adopted orphan {pid}")
                elif Cfg.ORPHAN_ACTION == "close":
                    ok = broker.close_position(pid)
                    logger.info(f"[RECONCILE] closed orphan {pid} ok={ok}")

        # State has, broker doesn't: posisi tutup di sisi broker (SL/TP hit)
        stale = state_ids - broker_ids
        if stale:
            logger.info(f"[RECONCILE] {len(stale)} posisi state tidak ada di broker (kemungkinan SL/TP hit).")
            for pid in stale:
                pos = self.state["positions"].pop(pid, None)
                if pos:
                    self._event("INFO", "position_closed_remotely", pos)
                    # Equity akan diperbarui dari account (jika OANDA)
            try:
                acc = broker.get_account()
                self.state["equity"] = float(acc["equity"])
                self.state["peak_equity"] = max(
                    float(acc["equity"]), float(self.state["peak_equity"])
                )
            except Exception:
                pass

        # Sync posisi yang masih ada
        for p in broker_positions:
            if p.position_id in self.state["positions"]:
                self.state["positions"][p.position_id]["unrealized_pnl"] = p.unrealized_pnl

        # Daily reset
        today = datetime.now(timezone.utc).date().isoformat()
        if self.state.get("day_start_date") != today:
            self.state["day_start_date"] = today
            self.state["day_start_equity"] = float(self.state["equity"])

        # Update equity dari broker untuk OANDA (sumber kebenaran)
        if Cfg.MODE in ("oanda-practice", "oanda-live"):
            try:
                acc = broker.get_account()
                self.state["equity"] = float(acc["equity"])
                self.state["peak_equity"] = max(
                    float(acc["equity"]), float(self.state["peak_equity"])
                )
            except Exception:
                pass

        self._event("INFO", "reconciled", {
            "broker_positions": len(broker_positions),
            "state_positions": len(self.state["positions"]),
        })

    def _handle_kill(self, tripped_reason: str):
        # Close all positions bila broker bisa
        try:
            broker = build_broker(self.state, pd.DataFrame())
            for pid in list(self.state["positions"].keys()):
                logger.warning(f"[KILL] menutup posisi {pid}")
                broker.close_position(pid)
        except Exception as e:
            logger.error(f"[KILL] close gagal: {e}")
        self.tg.send(f"🛑 <b>KILL SWITCH TRIPPED</b>\nReason: {tripped_reason}\nTrading halted.")


# ==============================================================================
# 8. MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Load state & data saja, tanpa order.")
    args = parser.parse_args()

    if args.dry_run:
        Cfg.validate()
        eng = LiveEngine()
        logger.info(f"[DRY] state keys: {list(eng.state.keys())}")
        logger.info(f"[DRY] equity={eng.state['equity']:.2f} positions={len(eng.state['positions'])}")
        sys.exit(0)

    LiveEngine().run()


if __name__ == "__main__":
    main()
