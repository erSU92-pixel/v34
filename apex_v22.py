#!/usr/bin/env python3
"""
APEX PREDATOR v22.0 — Production-Ready XAUUSD Trading Engine
=============================================================

Arsitektur (Clean OOP):
    MarketDataHandler     → OHLCV fetch + validation + resampling (no look-ahead)
    SignalGenerator       → H1 trend + M15 momentum + M5 EMA/ATR zone entry
    RiskManagementEngine  → ATR-based SL/TP + dynamic sizing + Daily DD Lock
    NewsSafetyFilter      → ForexFactory 30-min blackout window
    TelegramNotifier      → MT5-mobile-ready signal formatting
    TradingBotOrchestrator→ Main coordination loop

Cara pakai:
    python apex_v22.py run          # one-shot (cocok untuk cron / GH Actions)
    python apex_v22.py run --loop   # daemon (loop 60 detik)
    python apex_v22.py smoke        # cek koneksi + sinyal saat ini
    python apex_v22.py validate     # walk-forward backtest
    python apex_v22.py test-tg      # test kirim pesan Telegram
    python apex_v22.py selftest     # verifikasi instalasi
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

VERSION = "22.0"
UTC = timezone.utc


# ═══════════════════════════════════════════════════════════════
# 1. CONFIG
# ═══════════════════════════════════════════════════════════════
@dataclass
class Config:
    # Instrument
    symbol: str = "XAUUSD"
    contract_size: float = 100.0        # 1 lot = 100 oz
    price_decimals: int = 2

    # Account
    account_balance: float = 1000.0
    min_balance_for_trading: float = 50.0
    risk_pct_per_trade: float = 1.0      # 1% risk per trade
    max_lot: float = 5.0
    min_lot: float = 0.01
    hard_leverage_cap: float = 30.0

    # Daily DD Protection
    max_daily_loss_pct: float = 3.0      # lock kalau daily DD ≥ 3%
    daily_dd_pause_hours: int = 24

    # Signal parameters (EMA/ATR)
    ema_fast: int = 20                   # M15 momentum fast
    ema_slow: int = 50                   # M15 momentum slow
    ema_trend_fast: int = 50             # H1 trend fast
    ema_trend_slow: int = 200            # H1 trend slow
    ema_entry_fast: int = 9              # M5 entry fast
    ema_entry_slow: int = 21             # M5 entry slow
    ema_zone_atr: float = 0.5            # zona entry = EMA21 ± 0.5·ATR

    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp1_mult: float = 1.5
    atr_tp2_mult: float = 3.0
    atr_tp3_mult: float = 4.5
    atr_pct_min: float = 0.15
    atr_pct_max: float = 0.95
    atr_pct_lookback: int = 200

    # Data
    min_bars_m5: int = 100
    min_bars_m15: int = 100
    min_bars_h1: int = 250
    m5_fetch_size: int = 800
    h1_fetch_size: int = 800
    max_stale_mult: float = 3.5
    price_min: float = 500.0
    price_max: float = 10000.0

    # News
    news_blackout_min: int = 30
    news_cache_sec: int = 1800
    news_fail_open: bool = True          # True = fail-open (jangan block kalau API mati)

    # Loop
    loop_interval_sec: int = 60
    cooldown_minutes: int = 60
    max_pending: int = 2

    # Killzone (UTC): London AM 07-10, NY PM 12-16
    skip_killzone: bool = False
    killzones: Tuple[Tuple[int,int], ...] = ((7,10), (12,16))

    # HTTP
    http_timeout: Tuple[int, int] = (3, 10)
    max_retry: int = 3

    # Safety
    dry_run: bool = True
    kill_switch: bool = False

    # State persistence
    state_file: str = "apex_state_v22.json"
    gist_id: str = ""
    gist_token: str = ""
    require_state_persistence: bool = True

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        for name in cls.__dataclass_fields__:
            key = f"APEX_{name.upper()}"
            raw = os.getenv(key)
            if raw is None or not str(raw).strip():
                continue
            cur = getattr(cfg, name)
            try:
                if isinstance(cur, bool):
                    setattr(cfg, name, str(raw).strip().lower() in ("1","true","yes","on"))
                elif isinstance(cur, int):
                    setattr(cfg, name, int(raw))
                elif isinstance(cur, float):
                    setattr(cfg, name, float(raw))
                elif isinstance(cur, str):
                    setattr(cfg, name, str(raw).strip())
            except (ValueError, TypeError):
                pass
        cfg.gist_id = os.getenv("GIST_ID", "").strip()
        cfg.gist_token = os.getenv("GIST_TOKEN", "").strip()
        if os.getenv("APEX_KILL_SWITCH", "").lower() in ("1","true","yes"):
            cfg.kill_switch = True
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.risk_pct_per_trade <= 0 or self.risk_pct_per_trade > 5:
            raise ValueError("risk_pct_per_trade must be (0, 5]")
        if not (self.max_lot >= self.min_lot > 0):
            raise ValueError("invalid lot bounds")
        if not (0 < self.atr_tp1_mult < self.atr_tp2_mult < self.atr_tp3_mult):
            raise ValueError("TP multipliers must be increasing")
        if not (0 <= self.atr_pct_min < self.atr_pct_max <= 1.0):
            raise ValueError("invalid ATR percentile range")
        if self.max_daily_loss_pct <= 0 or self.max_daily_loss_pct >= 50:
            raise ValueError("max_daily_loss_pct must be (0, 50)")


# ═══════════════════════════════════════════════════════════════
# 2. LOGGING + UTIL
# ═══════════════════════════════════════════════════════════════
def setup_logging(level=logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    root.addHandler(h)


def redact(text: str) -> str:
    s = str(text).replace("\n", " ").replace("\r", " ")
    s = re.sub(r"(token|apikey|api_key|bearer|secret)\s*[=:]\s*[^\s&'\"]+",
               r"\1=***", s, flags=re.I)
    return s[:500]


class DataFetchError(Exception):
    pass


class StateError(Exception):
    pass


# ═══════════════════════════════════════════════════════════════
# 3. HTTP CLIENT — retry + timeout
# ═══════════════════════════════════════════════════════════════
class HTTPClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", self.cfg.http_timeout)
        last_err = "unknown"
        for attempt in range(self.cfg.max_retry):
            try:
                r = requests.request(method, url, **kwargs)
                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {r.status_code}"
                    self._backoff(attempt)
                    continue
                r.raise_for_status()
                return r
            except requests.RequestException as e:
                last_err = redact(str(e))
                self._backoff(attempt)
        raise DataFetchError(f"retry exhausted: {last_err}")

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(0.5 * (2 ** attempt) + random.uniform(0, 0.3))


# ═══════════════════════════════════════════════════════════════
# 4. STATE STORE — local file atau GitHub Gist
# ═══════════════════════════════════════════════════════════════
class StateStore:
    def __init__(self, cfg: Config, http: HTTPClient):
        self.cfg = cfg
        self.http = http
        self.use_gist = bool(cfg.gist_id and cfg.gist_token)
        self._cache: Optional[dict] = None
        if cfg.require_state_persistence and not self.use_gist and not cfg.dry_run:
            raise StateError("Production mode requires GIST_ID + GIST_TOKEN (or set "
                             "APEX_REQUIRE_STATE_PERSISTENCE=false to use local file)")

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.cfg.gist_token}",
            "Accept": "application/vnd.github+json",
        }

    def load(self) -> dict:
        if self._cache is not None:
            return self._cache
        if self.use_gist:
            try:
                r = self.http.request(
                    "GET", f"https://api.github.com/gists/{self.cfg.gist_id}",
                    headers=self._headers(),
                )
                files = r.json().get("files", {})
                content = files.get(self.cfg.state_file, {}).get("content", "").strip()
                self._cache = json.loads(content) if content else {}
            except Exception as e:
                logging.warning(f"Gist load failed ({redact(str(e))}), using empty state")
                self._cache = {}
        else:
            p = Path(self.cfg.state_file)
            try:
                self._cache = json.loads(p.read_text()) if p.exists() else {}
            except Exception as e:
                logging.warning(f"Local load failed ({e}), using empty state")
                self._cache = {}
        if not isinstance(self._cache, dict):
            self._cache = {}
        return self._cache

    def save(self, data: dict) -> bool:
        self._cache = data
        if self.use_gist:
            try:
                body = {"files": {self.cfg.state_file: {
                    "content": json.dumps(data, default=str, sort_keys=True)
                }}}
                self.http.request(
                    "PATCH", f"https://api.github.com/gists/{self.cfg.gist_id}",
                    headers=self._headers(), json=body,
                )
                return True
            except Exception as e:
                logging.error(f"Gist save failed: {redact(str(e))}")
                return False
        else:
            try:
                tmp = Path(self.cfg.state_file).with_suffix(".tmp")
                tmp.write_text(json.dumps(data, default=str, sort_keys=True, indent=2))
                tmp.replace(self.cfg.state_file)
                return True
            except Exception as e:
                logging.error(f"Local save failed: {e}")
                return False

    def get(self, key: str, default=None):
        return self.load().get(key, default)

    def set(self, key: str, value) -> bool:
        data = dict(self.load())
        data[key] = value
        return self.save(data)

    def force_reload(self) -> None:
        self._cache = None
        self.load()


# ═══════════════════════════════════════════════════════════════
# 5. INDICATORS — akurat, no look-ahead
# ═══════════════════════════════════════════════════════════════
class Indicators:
    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Wilder's ATR (EMA with alpha=1/period)."""
        h, l, c = df["high"], df["low"], df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([
            h - l,
            (h - prev_c).abs(),
            (l - prev_c).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / period, adjust=False).mean()

    @staticmethod
    def atr_last(df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < 2:
            return 0.0
        v = Indicators.atr_series(df, period).iloc[-1]
        return 0.0 if pd.isna(v) or v <= 0 else float(v)

    @staticmethod
    def atr_percentile(df: pd.DataFrame, period: int, lookback: int) -> float:
        if len(df) < lookback + period:
            return 0.5
        atr = Indicators.atr_series(df, period)
        cur = atr.iloc[-1]
        if pd.isna(cur):
            return 0.5
        window = atr.iloc[-lookback:]
        lo, hi = float(window.min()), float(window.max())
        if hi - lo < 1e-9:
            return 0.5
        return float(np.clip((cur - lo) / (hi - lo), 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════
# 6. MARKET DATA HANDLER — fetch + validate + resample
# ═══════════════════════════════════════════════════════════════
class MarketDataHandler:
    TD_URL = "https://api.twelvedata.com/time_series"

    def __init__(self, cfg: Config, http: HTTPClient, store: StateStore):
        self.cfg = cfg
        self.http = http
        self.store = store
        self._last_call = 0.0

    def _validate_df(self, df: pd.DataFrame, name: str, min_bars: int) -> None:
        if df is None or df.empty:
            raise DataFetchError(f"{name}: empty dataframe")
        if len(df) < min_bars:
            raise DataFetchError(f"{name}: only {len(df)} bars (need {min_bars})")
        if df["datetime"].duplicated().any():
            raise DataFetchError(f"{name}: duplicated datetimes")
        if (df["high"] < df["low"]).any():
            raise DataFetchError(f"{name}: high < low detected")
        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            raise DataFetchError(f"{name}: non-positive prices")
        last = float(df["close"].iloc[-1])
        if not (self.cfg.price_min <= last <= self.cfg.price_max):
            raise DataFetchError(f"{name}: last close {last} outside sanity range")

    def _check_staleness(self, df: pd.DataFrame, tf_minutes: int, name: str) -> None:
        if df.empty:
            return
        last_ts = df["datetime"].iloc[-1]
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=UTC)
        age_min = (datetime.now(UTC) - last_ts.to_pydatetime()).total_seconds() / 60.0
        if age_min > self.cfg.max_stale_mult * tf_minutes:
            raise DataFetchError(f"{name}: stale ({age_min:.0f} min old)")

    def _fetch_td(self, interval: str, outputsize: int) -> pd.DataFrame:
        key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
        if not key:
            raise DataFetchError("TWELVE_DATA_API_KEY not set")
        # rate limit
        elapsed = time.time() - self._last_call
        if elapsed < 1.5:
            time.sleep(1.5 - elapsed)
        self._last_call = time.time()

        r = self.http.request(
            "GET", self.TD_URL,
            params={
                "symbol": "XAU/USD",
                "interval": interval,
                "outputsize": outputsize,
                "apikey": key,
                "timezone": "UTC",
                "order": "ASC",
            },
        )
        payload = r.json()
        if "values" not in payload or not payload["values"]:
            raise DataFetchError(f"TD {interval}: {payload.get('message', 'no values')}")
        df = pd.DataFrame(payload["values"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume", 100.0), errors="coerce").fillna(100.0)
        df = df.dropna(subset=["open","high","low","close"]).sort_values("datetime").reset_index(drop=True)
        return df[["datetime","open","high","low","close","volume"]]

    @staticmethod
    def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        return (
            df.set_index("datetime")
              .resample(rule, label="left", closed="left")
              .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
              .dropna()
              .reset_index()
        )

    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        """Fetch M5, H1, H4, derive M15. Returns dict keyed by TF."""
        m5 = self._fetch_td("5min", self.cfg.m5_fetch_size)
        h1 = self._fetch_td("1h", self.cfg.h1_fetch_size)
        h4 = self._fetch_td("4h", self.cfg.h1_fetch_size)

        self._validate_df(m5, "M5", self.cfg.min_bars_m5)
        self._validate_df(h1, "H1", self.cfg.min_bars_h1)
        self._validate_df(h4, "H4", 50)

        self._check_staleness(m5, 5, "M5")
        self._check_staleness(h1, 60, "H1")

        m15 = self._resample(m5, "15min")
        self._validate_df(m15, "M15", self.cfg.min_bars_m15)

        return {"M5": m5, "M15": m15, "H1": h1, "H4": h4}


# ═══════════════════════════════════════════════════════════════
# 7. SIGNAL GENERATOR — EMA/ATR zones, no look-ahead
# ═══════════════════════════════════════════════════════════════
@dataclass
class Signal:
    direction: str      # "BUY" or "SELL"
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    atr: float
    atr_pct: float
    trend_h1: str
    reason: str


class SignalGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def generate(
        self,
        m5: pd.DataFrame,
        m15: pd.DataFrame,
        h1: pd.DataFrame,
    ) -> Optional[Signal]:
        """
        Rule (semua timeframe harus aligned):

        [1] H1 TREND FILTER
            BUY  : close > EMA200_H1  AND  EMA50_H1 > EMA200_H1
            SELL : close < EMA200_H1  AND  EMA50_H1 < EMA200_H1

        [2] M15 MOMENTUM
            BUY  : EMA20_M15 > EMA50_M15
            SELL : EMA20_M15 < EMA50_M15

        [3] ATR REGIME FILTER
            ATR percentile ∈ [atr_pct_min, atr_pct_max]
            (menghindari pasar mati / panik)

        [4] M5 ENTRY ZONE
            price berada dalam zona EMA21_M5 ± (ema_zone_atr · ATR_M5)
            BUY  : EMA9_M5 > EMA21_M5
            SELL : EMA9_M5 < EMA21_M5
        """
        cfg = self.cfg

        # ---- Guard: minimal bars termasuk bar closed ----
        if len(m5) < cfg.min_bars_m5 + 1:
            return None
        if len(m15) < cfg.min_bars_m15 + 1:
            return None
        if len(h1) < cfg.min_bars_h1 + 1:
            return None

        # ---- Drop bar terakhir (belum closed) → no look-ahead ----
        m5 = m5.iloc[:-1].reset_index(drop=True)
        m15 = m15.iloc[:-1].reset_index(drop=True)
        h1 = h1.iloc[:-1].reset_index(drop=True)

        # ---- [1] H1 Trend Filter ----
        ema50_h1 = Indicators.ema(h1["close"], cfg.ema_trend_fast)
        ema200_h1 = Indicators.ema(h1["close"], cfg.ema_trend_slow)
        p_h1 = float(h1["close"].iloc[-1])

        if p_h1 > ema200_h1.iloc[-1] and ema50_h1.iloc[-1] > ema200_h1.iloc[-1]:
            trend = "BUY"
        elif p_h1 < ema200_h1.iloc[-1] and ema50_h1.iloc[-1] < ema200_h1.iloc[-1]:
            trend = "SELL"
        else:
            return None

        # ---- [2] M15 Momentum ----
        ema20_m15 = float(Indicators.ema(m15["close"], cfg.ema_fast).iloc[-1])
        ema50_m15 = float(Indicators.ema(m15["close"], cfg.ema_slow).iloc[-1])
        if trend == "BUY" and ema20_m15 <= ema50_m15:
            return None
        if trend == "SELL" and ema20_m15 >= ema50_m15:
            return None

        # ---- [3] ATR Regime ----
        atr_now = Indicators.atr_last(m5, cfg.atr_period)
        if atr_now <= 0:
            return None
        atr_pct = Indicators.atr_percentile(m5, cfg.atr_period, cfg.atr_pct_lookback)
        if not (cfg.atr_pct_min <= atr_pct <= cfg.atr_pct_max):
            return None

        # ---- [4] M5 Entry Zone ----
        ema9_m5 = float(Indicators.ema(m5["close"], cfg.ema_entry_fast).iloc[-1])
        ema21_m5 = float(Indicators.ema(m5["close"], cfg.ema_entry_slow).iloc[-1])
        price = float(m5["close"].iloc[-1])

        zone_lo = ema21_m5 - atr_now * cfg.ema_zone_atr
        zone_hi = ema21_m5 + atr_now * cfg.ema_zone_atr
        if not (zone_lo <= price <= zone_hi):
            return None

        if trend == "BUY":
            if ema9_m5 <= ema21_m5:
                return None
            direction = "BUY"
            sl = price - atr_now * cfg.atr_sl_mult
            tp1 = price + atr_now * cfg.atr_tp1_mult
            tp2 = price + atr_now * cfg.atr_tp2_mult
            tp3 = price + atr_now * cfg.atr_tp3_mult
        else:
            if ema9_m5 >= ema21_m5:
                return None
            direction = "SELL"
            sl = price + atr_now * cfg.atr_sl_mult
            tp1 = price - atr_now * cfg.atr_tp1_mult
            tp2 = price - atr_now * cfg.atr_tp2_mult
            tp3 = price - atr_now * cfg.atr_tp3_mult

        p = cfg.price_decimals
        return Signal(
            direction=direction,
            entry=round(price, p),
            sl=round(sl, p),
            tp1=round(tp1, p),
            tp2=round(tp2, p),
            tp3=round(tp3, p),
            atr=atr_now,
            atr_pct=atr_pct,
            trend_h1=trend,
            reason=f"H1={trend} | M15 aligned | ATR%={atr_pct:.0%}",
        )


# ═══════════════════════════════════════════════════════════════
# 8. RISK MANAGEMENT ENGINE
# ═══════════════════════════════════════════════════════════════
class RiskManagementEngine:
    def __init__(self, cfg: Config, store: StateStore):
        self.cfg = cfg
        self.store = store

    # ---------- Balance tracking ----------
    def current_balance(self) -> float:
        state = self.store.load()
        if "current_balance" in state:
            try:
                v = float(state["current_balance"])
                if v > 0:
                    return v
            except (ValueError, TypeError):
                pass
        return float(self.cfg.account_balance)

    def _ensure_balance_initialized(self) -> None:
        state = self.store.load()
        if "current_balance" not in state:
            state["current_balance"] = float(self.cfg.account_balance)
            self.store.save(state)

    def apply_trade_result(self, r_multiple: float, risk_amount_usd: float) -> None:
        """Update balance setelah trade close."""
        state = self.store.load()
        bal = float(state.get("current_balance", self.cfg.account_balance))
        pnl = risk_amount_usd * r_multiple
        state["current_balance"] = bal + pnl
        self.store.save(state)

    # ---------- Daily DD Lock ----------
    def daily_dd_check(self) -> Tuple[bool, str]:
        """Return (is_locked, reason)."""
        self._ensure_balance_initialized()
        state = self.store.load()
        balance = self.current_balance()
        today = datetime.now(UTC).date().isoformat()

        if state.get("daily_date") != today:
            state["daily_date"] = today
            state["daily_start_balance"] = balance
            state["daily_locked"] = False
            self.store.save(state)

        if state.get("daily_locked"):
            return True, "daily_dd_already_locked"

        start = float(state.get("daily_start_balance", balance))
        if start <= 0:
            return False, ""
        dd_pct = (start - balance) / start * 100.0
        if dd_pct >= self.cfg.max_daily_loss_pct:
            state["daily_locked"] = True
            self.store.save(state)
            logging.warning(
                f"🛑 DAILY DD LOCK ENGAGED: -{dd_pct:.2f}% (start=${start:.2f}, now=${balance:.2f})"
            )
            return True, f"daily_dd_{dd_pct:.2f}%"
        return False, ""

    # ---------- Position sizing ----------
    def compute_lot(self, balance: float, entry: float, sl: float) -> Optional[float]:
        if balance < self.cfg.min_balance_for_trading:
            logging.warning(f"balance {balance:.2f} < min {self.cfg.min_balance_for_trading}")
            return None
        sl_dist = abs(entry - sl)
        if sl_dist <= 1e-6:
            return None

        risk_usd = balance * (self.cfg.risk_pct_per_trade / 100.0)
        lot_raw = risk_usd / (sl_dist * self.cfg.contract_size)
        lot = max(self.cfg.min_lot, min(self.cfg.max_lot, round(lot_raw, 2)))

        # Leverage cap
        notional = lot * self.cfg.contract_size * entry
        if notional > self.cfg.hard_leverage_cap * balance:
            max_lot = self.cfg.hard_leverage_cap * balance / (self.cfg.contract_size * entry)
            lot = max(self.cfg.min_lot, round(max_lot, 2))

        # Safety: min-lot risk harus ≤ 2% balance
        min_risk = self.cfg.min_lot * self.cfg.contract_size * sl_dist
        if min_risk > balance * 0.02:
            logging.warning(f"min-lot risk ${min_risk:.2f} > 2% of ${balance:.2f}")
            return None

        return lot


# ═══════════════════════════════════════════════════════════════
# 9. NEWS SAFETY FILTER
# ═══════════════════════════════════════════════════════════════
class NewsSafetyFilter:
    FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

    def __init__(self, cfg: Config, http: HTTPClient, store: StateStore):
        self.cfg = cfg
        self.http = http
        self.store = store

    def is_blackout(self) -> Tuple[bool, str]:
        if self.cfg.news_blackout_min <= 0:
            return False, ""

        cached = self.store.get("news_cache", {}) or {}
        now_ts = time.time()
        if now_ts - float(cached.get("ts", 0)) < self.cfg.news_cache_sec:
            return bool(cached.get("blackout")), cached.get("title", "")

        try:
            r = self.http.request("GET", self.FF_URL, headers={"User-Agent": "Mozilla/5.0"})
            events = r.json()
        except Exception as e:
            logging.warning(f"News fetch failed: {redact(str(e))}")
            if self.cfg.news_fail_open:
                return False, ""
            return True, "news_api_down"

        now = datetime.now(UTC)
        blackout_min = self.cfg.news_blackout_min

        for ev in events:
            if str(ev.get("country", "")).upper() != "USD":
                continue
            if str(ev.get("impact", "")).lower() != "high":
                continue
            try:
                evt = datetime.fromisoformat(str(ev["date"]).replace("Z", "+00:00"))
                if evt.tzinfo is None:
                    evt = evt.replace(tzinfo=UTC)
            except Exception:
                continue
            delta_min = abs((evt - now).total_seconds()) / 60.0
            if delta_min <= blackout_min:
                title = str(ev.get("title", "High-impact event"))
                self.store.set("news_cache", {"ts": now_ts, "blackout": True, "title": title})
                return True, title

        self.store.set("news_cache", {"ts": now_ts, "blackout": False, "title": ""})
        return False, ""


# ═══════════════════════════════════════════════════════════════
# 10. TELEGRAM NOTIFIER
# ═══════════════════════════════════════════════════════════════
class TelegramNotifier:
    def __init__(self, cfg: Config, http: HTTPClient, store: StateStore):
        self.cfg = cfg
        self.http = http
        self.store = store
        self.token = os.getenv("TELEGRAM_TOKEN", "").strip()
        self.chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    def send(self, text: str) -> bool:
        if self.cfg.dry_run:
            text = "🧪 <b>[DRY RUN]</b>\n" + text
        if not (self.token and self.chat):
            if self.cfg.dry_run:
                logging.info(f"[DRY-TG] {redact(text)}")
                return True
            logging.error("Telegram token/chat not configured")
            return False

        # Dedup berdasarkan hash
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        now = datetime.now(UTC)
        hashes = self.store.get("tg_hashes", {}) or {}
        hashes = {k: v for k, v in hashes.items()
                  if self._recent(v, now, 3600)}
        if digest in hashes:
            return True

        try:
            r = self.http.request(
                "POST",
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat, "text": text, "parse_mode": "HTML"},
            )
            if r.json().get("ok"):
                hashes[digest] = now.isoformat()
                self.store.set("tg_hashes", hashes)
                return True
        except Exception as e:
            logging.error(f"Telegram send failed: {redact(str(e))}")
        return False

    @staticmethod
    def _recent(iso: str, now: datetime, sec: int) -> bool:
        try:
            return (now - datetime.fromisoformat(iso)).total_seconds() < sec
        except Exception:
            return False

    # ---------- Formatting ----------
    def format_signal(self, sig: Signal, lot: float, balance: float, risk_usd: float) -> str:
        p = self.cfg.price_decimals
        sl_dist = abs(sig.entry - sig.sl)
        emoji = "🟢" if sig.direction == "BUY" else "🔴"
        return (
            f"{emoji} <b>{self.cfg.symbol} — {sig.direction}</b> (M5)\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 <b>Entry:</b> <code>{sig.entry:.{p}f}</code>\n"
            f"🛑 <b>SL   :</b> <code>{sig.sl:.{p}f}</code>  (<i>-{sl_dist:.{p}f}</i>)\n"
            f"🎯 <b>TP1  :</b> <code>{sig.tp1:.{p}f}</code>  (<i>+{abs(sig.tp1-sig.entry):.{p}f}</i>)\n"
            f"🎯 <b>TP2  :</b> <code>{sig.tp2:.{p}f}</code>  (<i>+{abs(sig.tp2-sig.entry):.{p}f}</i>)\n"
            f"🎯 <b>TP3  :</b> <code>{sig.tp3:.{p}f}</code>  (<i>+{abs(sig.tp3-sig.entry):.{p}f}</i>)\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📦 <b>Lot:</b> <code>{lot}</code>\n"
            f"💰 <b>Risk:</b> ${risk_usd:.2f} ({self.cfg.risk_pct_per_trade:.1f}% of ${balance:.2f})\n"
            f"📊 ATR: {sig.atr:.2f} (pct {sig.atr_pct:.0%})\n"
            f"🧠 {sig.reason}"
        )


# ═══════════════════════════════════════════════════════════════
# 11. OUTCOME TRACKER — update balance + journal
# ═══════════════════════════════════════════════════════════════
class OutcomeTracker:
    def __init__(self, cfg: Config, store: StateStore, risk: RiskManagementEngine):
        self.cfg = cfg
        self.store = store
        self.risk = risk

    def update(self, m5: pd.DataFrame) -> List[dict]:
        """Walk pending journal entries → close those whose SL/TP hit."""
        state = self.store.load()
        journal: List[dict] = state.get("journal", []) or []
        closed_events: List[dict] = []
        changed = False
        now = datetime.now(UTC)

        for entry in journal:
            if entry.get("outcome") not in ("PENDING", None):
                continue

            try:
                ts = pd.Timestamp(entry.get("signal_ts", entry.get("timestamp")), tz="UTC")
            except Exception:
                continue

            # Expire stale pending
            if (now - ts.to_pydatetime()) > timedelta(hours=24):
                entry["outcome"] = "STALE"
                entry["r_achieved"] = 0.0
                changed = True
                continue

            side = entry["signal"]
            ep = float(entry["entry"])
            sl = float(entry["sl"])
            tp1 = float(entry["tp1"])
            tp3 = float(entry["tp3"])
            risk_usd = float(entry.get("risk_amount_usd", 0.0))

            bars = m5[m5["datetime"] > ts].sort_values("datetime")
            if bars.empty:
                continue

            # Fill detection
            if not entry.get("filled"):
                fill_idx = None
                for i, b in enumerate(bars.itertuples(index=False)):
                    if side == "BUY" and b.low <= ep:
                        fill_idx = i
                        break
                    if side == "SELL" and b.high >= ep:
                        fill_idx = i
                        break
                if fill_idx is None:
                    continue
                entry["filled"] = 1
                changed = True
                walk = bars.iloc[fill_idx:]
            else:
                walk = bars

            # Walk to outcome
            hit_tp1 = bool(entry.get("hit_tp1"))
            outcome = "PENDING"
            r_val = 0.0

            for b in walk.itertuples(index=False):
                # Move SL to BE after TP1
                active_sl = ep if hit_tp1 else sl

                if side == "BUY":
                    if b.low <= active_sl:
                        outcome = "WIN_BE" if hit_tp1 else "LOSS"
                        r_val = 0.0 if hit_tp1 else -1.0
                        break
                    if not hit_tp1 and b.high >= tp1:
                        hit_tp1 = True
                    if b.high >= tp3:
                        outcome = "WIN_TP3"
                        r_val = (tp3 - ep) / abs(ep - sl)
                        break
                else:
                    if b.high >= active_sl:
                        outcome = "WIN_BE" if hit_tp1 else "LOSS"
                        r_val = 0.0 if hit_tp1 else -1.0
                        break
                    if not hit_tp1 and b.low <= tp1:
                        hit_tp1 = True
                    if b.low <= tp3:
                        outcome = "WIN_TP3"
                        r_val = (ep - tp3) / abs(ep - sl)
                        break

            entry["hit_tp1"] = int(hit_tp1)
            if outcome != "PENDING":
                entry["outcome"] = outcome
                entry["r_achieved"] = round(r_val, 3)
                entry["closed_ts"] = now.isoformat()
                self.risk.apply_trade_result(r_val, risk_usd)
                closed_events.append(dict(entry))
                changed = True

        if changed:
            state["journal"] = journal
            self.store.save(state)
        return closed_events


# ═══════════════════════════════════════════════════════════════
# 12. ORCHESTRATOR — loop utama
# ═══════════════════════════════════════════════════════════════
class TradingBotOrchestrator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.http = HTTPClient(cfg)
        self.store = StateStore(cfg, self.http)
        self.data = MarketDataHandler(cfg, self.http, self.store)
        self.signal = SignalGenerator(cfg)
        self.risk = RiskManagementEngine(cfg, self.store)
        self.news = NewsSafetyFilter(cfg, self.http, self.store)
        self.tg = TelegramNotifier(cfg, self.http, self.store)
        self.tracker = OutcomeTracker(cfg, self.store, self.risk)

    # ---------- Helpers ----------
    def _in_killzone(self, now: datetime) -> bool:
        if self.cfg.skip_killzone:
            return True
        h = now.hour + now.minute / 60.0
        for start, end in self.cfg.killzones:
            if start <= h < end:
                return True
        return False

    def _cooldown_active(self) -> bool:
        last = self.store.get("last_signal_ts", "")
        if not last:
            return False
        try:
            t = datetime.fromisoformat(last)
            if t.tzinfo is None:
                t = t.replace(tzinfo=UTC)
            return (datetime.now(UTC) - t) < timedelta(minutes=self.cfg.cooldown_minutes)
        except Exception:
            return False

    def _count_pending(self) -> int:
        journal = self.store.get("journal", []) or []
        return sum(1 for j in journal if j.get("outcome") == "PENDING")

    # ---------- Single tick ----------
    def tick(self) -> None:
        now = datetime.now(UTC)

        # --- Global gates ---
        if self.cfg.kill_switch:
            logging.info("KILL switch engaged → abort")
            return
        if now.weekday() >= 5:
            logging.info("Weekend → skip")
            return
        if not self._in_killzone(now):
            logging.info(f"Outside killzone ({now.strftime('%H:%M')} UTC) → skip")
            return

        # --- Daily DD lock ---
        locked, reason = self.risk.daily_dd_check()
        if locked:
            logging.warning(f"Daily DD locked: {reason}")
            return

        # --- News blackout ---
        blackout, title = self.news.is_blackout()
        if blackout:
            logging.info(f"News blackout: {title}")
            return

        # --- Fetch data ---
        try:
            dfs = self.data.fetch_all()
        except Exception as e:
            logging.error(f"Data fetch failed: {redact(str(e))}")
            return

        # --- Update pending trades first ---
        try:
            closed = self.tracker.update(dfs["M5"])
            for ev in closed:
                emoji = "🏆" if ev["outcome"].startswith("WIN") else (
                    "⚖️" if ev["outcome"] == "WIN_BE" else "💀"
                )
                msg = (
                    f"{emoji} <b>Trade Closed</b> — {ev['signal']} {self.cfg.symbol}\n"
                    f"Outcome: <b>{ev['outcome']}</b>  ({ev['r_achieved']:+.2f}R)\n"
                    f"Entry: {ev['entry']}  SL: {ev['sl']}"
                )
                self.tg.send(msg)
        except Exception as e:
            logging.error(f"Tracker error: {redact(str(e))}")

        # --- Cooldown + pending limit ---
        if self._cooldown_active():
            logging.info("Cooldown active → skip new signal")
            return
        if self._count_pending() >= self.cfg.max_pending:
            logging.info(f"Max pending reached ({self.cfg.max_pending})")
            return

        # --- Generate signal ---
        sig = self.signal.generate(dfs["M5"], dfs["M15"], dfs["H1"])
        if sig is None:
            logging.info("No signal")
            return

        # --- Position sizing ---
        balance = self.risk.current_balance()
        lot = self.risk.compute_lot(balance, sig.entry, sig.sl)
        if lot is None:
            logging.warning("Position sizing rejected")
            return
        risk_usd = lot * self.cfg.contract_size * abs(sig.entry - sig.sl)

        # --- Telegram ---
        msg = self.tg.format_signal(sig, lot, balance, risk_usd)
        if not self.tg.send(msg):
            logging.error("Telegram send failed → tidak mencatat trade")
            return

        # --- Persist journal ---
        state = self.store.load()
        journal = state.get("journal", []) or []
        journal.append({
            "id": f"{now.strftime('%Y%m%dT%H%M%S')}_{sig.direction}",
            "timestamp": now.isoformat(),
            "signal_ts": dfs["M5"].iloc[-1]["datetime"].isoformat(),
            "symbol": self.cfg.symbol,
            "signal": sig.direction,
            "entry": sig.entry,
            "sl": sig.sl,
            "tp1": sig.tp1,
            "tp2": sig.tp2,
            "tp3": sig.tp3,
            "lot": lot,
            "risk_amount_usd": round(risk_usd, 2),
            "atr": sig.atr,
            "outcome": "PENDING",
            "filled": 0,
            "r_achieved": 0.0,
        })
        state["journal"] = journal[-500:]
        state["last_signal_ts"] = now.isoformat()
        self.store.save(state)
        logging.info(f"✅ Signal dispatched: {sig.direction} lot={lot} entry={sig.entry} sl={sig.sl}")

    # ---------- Loop ----------
    def run_once(self) -> None:
        try:
            self.tick()
        except Exception:
            logging.exception("Unhandled error in tick")

    def run_loop(self) -> None:
        logging.info(f"🚀 APEX v{VERSION} starting loop (every {self.cfg.loop_interval_sec}s)")
        while not self.cfg.kill_switch:
            self.run_once()
            time.sleep(self.cfg.loop_interval_sec)


# ═══════════════════════════════════════════════════════════════
# 13. CLI COMMANDS
# ═══════════════════════════════════════════════════════════════
def cmd_smoke(cfg: Config) -> None:
    http = HTTPClient(cfg)
    store = StateStore(cfg, http)
    data = MarketDataHandler(cfg, http, store)
    sig_gen = SignalGenerator(cfg)
    print("─── SMOKE TEST ───")
    dfs = data.fetch_all()
    for tf, df in dfs.items():
        print(f"  {tf}: {len(df)} bars | last={df['close'].iloc[-1]:.2f} @ {df['datetime'].iloc[-1]}")
    sig = sig_gen.generate(dfs["M5"], dfs["M15"], dfs["H1"])
    if sig:
        print(f"\n✅ SIGNAL: {sig.direction}")
        print(f"   entry={sig.entry}  sl={sig.sl}  tp1={sig.tp1}  tp2={sig.tp2}  tp3={sig.tp3}")
        print(f"   {sig.reason}")
    else:
        print("\n⊘ No signal (market conditions not met)")


def cmd_validate(cfg: Config) -> None:
    print("─── WALK-FORWARD VALIDATION ───")
    http = HTTPClient(cfg)
    store = StateStore(cfg, http)
    data = MarketDataHandler(cfg, http, store)
    sig_gen = SignalGenerator(cfg)

    dfs = data.fetch_all()
    m5 = dfs["M5"].reset_index(drop=True)
    m15 = dfs["M15"].reset_index(drop=True)
    h1 = dfs["H1"].reset_index(drop=True)

    n = len(m5)
    lookback = 250
    step = 3
    forward = 60
    results = []

    for i in range(lookback, n - forward, step):
        cur_ts = m5.iloc[i]["datetime"]
        m5w = m5.iloc[:i]
        m15w = m15[m15["datetime"] <= cur_ts]
        h1w = h1[h1["datetime"] <= cur_ts]
        if len(m15w) < 100 or len(h1w) < 250:
            continue

        sig = sig_gen.generate(m5w, m15w, h1w)
        if sig is None:
            continue

        fut = m5.iloc[i:i + forward]
        fill_idx = None
        for j, b in enumerate(fut.itertuples(index=False)):
            if sig.direction == "BUY" and b.low <= sig.entry:
                fill_idx = j
                break
            if sig.direction == "SELL" and b.high >= sig.entry:
                fill_idx = j
                break
        if fill_idx is None:
            continue

        post = fut.iloc[fill_idx:]
        rr = abs(sig.entry - sig.sl)
        outcome = "OPEN"
        r_val = 0.0
        hit_tp1 = False
        for b in post.itertuples(index=False):
            active_sl = sig.entry if hit_tp1 else sig.sl
            if sig.direction == "BUY":
                if b.low <= active_sl:
                    outcome = "WIN_BE" if hit_tp1 else "LOSS"
                    r_val = 0.0 if hit_tp1 else -1.0
                    break
                if not hit_tp1 and b.high >= sig.tp1:
                    hit_tp1 = True
                if b.high >= sig.tp3:
                    outcome = "WIN_TP3"
                    r_val = (sig.tp3 - sig.entry) / rr
                    break
            else:
                if b.high >= active_sl:
                    outcome = "WIN_BE" if hit_tp1 else "LOSS"
                    r_val = 0.0 if hit_tp1 else -1.0
                    break
                if not hit_tp1 and b.low <= sig.tp1:
                    hit_tp1 = True
                if b.low <= sig.tp3:
                    outcome = "WIN_TP3"
                    r_val = (sig.entry - sig.tp3) / rr
                    break
        if outcome != "OPEN":
            results.append({"outcome": outcome, "r": r_val})

    if not results:
        print("⊘ No trades executed in backtest.")
        return

    wins = sum(1 for r in results if r["outcome"].startswith("WIN"))
    total_r = sum(r["r"] for r in results)
    wr = wins / len(results) * 100
    exp_r = total_r / len(results)

    print(f"\nTrades: {len(results)}")
    print(f"Win rate: {wr:.1f}%")
    print(f"Total R: {total_r:+.2f}")
    print(f"Expectancy: {exp_r:+.3f} R/trade")
    verdict = "✅ EDGE POSITIVE" if exp_r > 0.05 else "❌ EDGE NEGATIVE/WEAK"
    print(verdict)


def cmd_test_tg(cfg: Config) -> None:
    http = HTTPClient(cfg)
    store = StateStore(cfg, http)
    tg = TelegramNotifier(cfg, http, store)
    dummy = Signal(
        direction="BUY", entry=2045.50, sl=2038.20,
        tp1=2052.80, tp2=2060.10, tp3=2067.40,
        atr=8.10, atr_pct=0.55, trend_h1="BUY", reason="SELF-TEST",
    )
    msg = tg.format_signal(dummy, lot=0.10, balance=1000.0, risk_usd=10.0)
    ok = tg.send("🔔 <b>APEX v" + VERSION + "</b> — Telegram OK\n\n" + msg)
    print("SENT" if ok else "FAILED")


def cmd_selftest(cfg: Config) -> None:
    # Check imports
    try:
        import numpy, pandas, requests
        print(f"✓ numpy {numpy.__version__}")
        print(f"✓ pandas {pandas.__version__}")
        print(f"✓ requests {requests.__version__}")
    except ImportError as e:
        print(f"✗ missing dependency: {e}")
        sys.exit(1)

    # Config sanity
    print(f"✓ Config loaded: symbol={cfg.symbol} risk={cfg.risk_pct_per_trade}%")
    print(f"✓ ATR: period={cfg.atr_period} sl={cfg.atr_sl_mult}x tp={cfg.atr_tp1_mult}/{cfg.atr_tp2_mult}/{cfg.atr_tp3_mult}x")
    print(f"✓ Daily DD lock: {cfg.max_daily_loss_pct}%")
    print(f"✓ News blackout: ±{cfg.news_blackout_min} min")
    print(f"\n✅ SELFTEST PASSED (v{VERSION})")


# ═══════════════════════════════════════════════════════════════
# 14. MAIN
# ═══════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description=f"APEX v{VERSION}")
    parser.add_argument("command", nargs="?", default="run",
                        choices=["run","smoke","validate","test-tg","selftest"])
    parser.add_argument("--loop", action="store_true", help="Daemon mode")
    args = parser.parse_args()

    setup_logging()

    try:
        cfg = Config.from_env()
    except Exception as e:
        print(f"CONFIG ERROR: {e}")
        sys.exit(2)

    if args.command == "selftest":
        cmd_selftest(cfg); return
    if args.command == "smoke":
        cmd_smoke(cfg); return
    if args.command == "validate":
        cmd_validate(cfg); return
    if args.command == "test-tg":
        cmd_test_tg(cfg); return

    # run
    try:
        bot = TradingBotOrchestrator(cfg)
        if args.loop:
            bot.run_loop()
        else:
            bot.run_once()
    except Exception:
        logging.exception("FATAL")
        sys.exit(1)


if __name__ == "__main__":
    main()
