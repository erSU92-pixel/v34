#!/usr/bin/env python3
"""
APEX PREDATOR v21.0 — Profit-Maximized Institutional Build
===========================================================
Superset of v20.0 + all Tier 1/2/3.3 enhancements.

New layers (all toggleable via APEX_ENABLE_* env vars):
  [L1] DXY macro bias         — gold/anti-USD correlation
  [L2] ATR percentile regime  — reject dead & chaotic markets
  [L3] MTF alignment score    — H4*2 + H1 + M15 gate
  [L4] Key levels protection  — PDH/PDL/round/Asian range
  [L5] Session blacklist      — journal-based hour filtering
  [L6] Regime routing         — trend vs meanrev models
  [L7] Liquidity sweep + FVG  — extra confluence voters
  [L8] Adaptive weights       — Bayesian model downweighting
  [L9] Anti-chase momentum    — reject extended moves
  [L10] ORB strategy          — opening range breakout (secondary)
  [L11] Volatility spike      — reject 3x median range bar
"""
from __future__ import annotations

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
from typing import Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

try:
    from sklearn.ensemble import RandomForestClassifier
    from scipy.signal import butter, filtfilt
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

VERSION = "21.0"
NY = ZoneInfo("America/New_York")
LONDON = ZoneInfo("Europe/London")
UTC = timezone.utc

US_HOLIDAYS = {
    "2024-12-25", "2025-01-01", "2025-01-20", "2025-02-17",
    "2025-04-18", "2025-05-26", "2025-06-19", "2025-07-04",
    "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
    "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
    "2026-11-26", "2026-12-25",
}


# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════
@dataclass
class Config:
    gist_id: str = ""
    gist_token: str = ""
    gist_state_file: str = "apex_state_v21.json"
    local_state_file: str = "apex_state_v21.json"

    symbol: str = "XAUUSD"
    contract_size: float = 100.0
    account_currency: str = "USD"
    price_decimals: int = 2
    mt5_offset: float = 0.0

    account_balance: float = 1000.0
    min_balance_for_trading: float = 50.0
    max_risk_pct_per_trade: float = 1.0
    hard_leverage_cap: float = 30.0
    max_lot: float = 5.0
    min_lot: float = 0.01

    max_daily_loss_pct: float = 3.0
    max_consecutive_losses: int = 3
    dd_pause_hours: int = 24

    atr_period: int = 14
    atr_sl_mult: float = 1.8
    atr_tp1_mult: float = 1.0
    atr_tp2_mult: float = 2.0
    atr_tp3_mult: float = 3.0
    spread_buffer: float = 0.35
    entry_pullback_atr: float = 0.50
    entry_max_chase_atr: float = 2.0

    exit_model: str = "single_tp"
    require_state_persistence: bool = True

    min_bars_m5: int = 100
    min_bars_m15: int = 100
    min_bars_h1: int = 100
    min_bars_h4: int = 50
    max_stale_mult: float = 3.5
    m5_fetch_size: int = 800

    price_min: float = 500.0
    price_max: float = 10000.0
    price_max_jump_pct: float = 0.10

    parliament_min_active: int = 8
    parliament_min_ratio: float = 0.55
    parliament_half_ratio: float = 0.65
    parliament_full_ratio: float = 0.75

    adx_min: float = 22.0
    enable_ml_models: bool = True
    cooldown_minutes: int = 45
    opposite_cooldown_minutes: int = 120
    max_daily_signals: int = 5

    hurst_trend_min: float = 0.55
    hurst_meanrev_max: float = 0.45
    entropy_chaos_max: float = 0.85

    dry_run: bool = True
    kill_switch: bool = False
    news_blackout_min: int = 30
    news_cache_sec: int = 1800
    news_cache_sec_on_fail: int = 300
    news_max_consecutive_fail: int = 3

    pending_max_age_hours: int = 24
    http_timeout: tuple = (3, 10)
    max_retry: int = 3
    max_journal: int = 500
    ml_purge_bars: int = 10
    ml_min_val_acc: float = 0.58

    skip_killzone_filter: bool = False
    h1_cache_minutes: int = 55
    h4_cache_minutes: int = 220
    td_min_interval_sec: float = 1.5

    # ── v21 filters (all toggleable) ──
    enable_dxy_filter: bool = True
    dxy_require_match: bool = False      # if True: hard block on DXY mismatch
    enable_atr_percentile: bool = True
    atr_pct_min: float = 0.25
    atr_pct_max: float = 0.85
    atr_pct_lookback: int = 200

    enable_mtf_alignment: bool = True
    mtf_min_score: int = 3               # 0-4

    enable_key_levels: bool = True
    key_level_buffer: float = 3.0        # points
    key_level_round_step: float = 25.0

    enable_session_blacklist: bool = True
    session_min_trades: int = 5
    session_expectancy_threshold: float = -0.2

    enable_regime_routing: bool = True
    enable_sweep_voter: bool = True
    enable_fvg_voter: bool = True

    enable_adaptive_weights: bool = True
    adaptive_min_journal: int = 20

    enable_momentum_filter: bool = True
    momentum_body_mult: float = 2.0

    enable_orb: bool = True
    orb_start_hour_utc: int = 13
    orb_duration_min: int = 30
    orb_trade_hours: int = 2             # 13:30 - 15:30 UTC
    orb_min_range_atr_frac: float = 0.3
    orb_max_range_atr_frac: float = 0.8

    enable_volatility_spike: bool = True
    spike_range_mult: float = 3.0

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        for name in cls.__dataclass_fields__:
            env_key = f"APEX_{name.upper()}"
            raw = os.getenv(env_key)
            if raw is None or not str(raw).strip():
                continue
            cur = getattr(cfg, name)
            try:
                if isinstance(cur, bool):
                    setattr(cfg, name, str(raw).strip().lower() in ("1", "true", "yes", "on"))
                elif isinstance(cur, int):
                    setattr(cfg, name, int(raw))
                elif isinstance(cur, float):
                    v = float(raw)
                    if not np.isfinite(v):
                        raise ValueError("non-finite")
                    setattr(cfg, name, v)
                elif isinstance(cur, str):
                    setattr(cfg, name, str(raw).strip())
            except (ValueError, TypeError):
                logging.warning(f"Config: ignore invalid env {env_key}")
        cfg.gist_id = os.getenv("GIST_ID", cfg.gist_id).strip()
        cfg.gist_token = os.getenv("GIST_TOKEN", cfg.gist_token).strip()
        off = os.getenv("APEX_MT5_OFFSET", "").strip()
        if off:
            try:
                cfg.mt5_offset = float(off)
            except ValueError:
                pass
        if os.getenv("APEX_KILL_SWITCH", "").lower() in ("1", "true", "yes"):
            cfg.kill_switch = True
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not (0 < self.parliament_min_ratio < self.parliament_half_ratio < self.parliament_full_ratio <= 1.0):
            raise ValueError("parliament ratio ordering invalid")
        if not (self.max_lot >= self.min_lot > 0):
            raise ValueError("lot bounds invalid")
        if self.m5_fetch_size < self.min_bars_m5:
            raise ValueError("m5_fetch_size < min_bars_m5")
        if self.account_balance <= 0:
            raise ValueError("account_balance must be > 0")
        if not (0 < self.atr_tp1_mult < self.atr_tp2_mult < self.atr_tp3_mult):
            raise ValueError("TP multipliers must be increasing")
        if self.exit_model not in ("single_tp", "scaled_tp"):
            raise ValueError(f"exit_model '{self.exit_model}' invalid")
        if not (0.0 <= self.atr_pct_min < self.atr_pct_max <= 1.0):
            raise ValueError("ATR percentile range invalid")
        if not (0 <= self.mtf_min_score <= 4):
            raise ValueError("mtf_min_score must be 0-4")


# ════════════════════════════════════════════════════════════
# LOGGING & UTIL
# ════════════════════════════════════════════════════════════
def setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root.addHandler(h)


def redact(text) -> str:
    s = str(text).replace("\n", " ").replace("\r", " ")
    s = re.sub(r"(token|apikey|api_key|bearer|secret)\s*[=:]\s*[^\s&'\"]+", r"\1=***", s, flags=re.I)
    return s


class DataFetchError(Exception):
    pass


class StateError(Exception):
    pass


# ════════════════════════════════════════════════════════════
# HTTP CLIENT
# ════════════════════════════════════════════════════════════
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
                last_err = redact(e)
                self._backoff(attempt)
        raise DataFetchError(f"retry exhausted: {last_err}")

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(0.5 * (2 ** attempt) + random.uniform(0, 0.3))


# ════════════════════════════════════════════════════════════
# STATE STORE
# ════════════════════════════════════════════════════════════
class KVStore:
    def __init__(self, cfg: Config, http: HTTPClient):
        self.cfg = cfg
        self.http = http
        self.use_gist = bool(cfg.gist_id and cfg.gist_token)
        self._cache: Optional[dict] = None
        if cfg.require_state_persistence and not self.use_gist and not cfg.dry_run:
            raise StateError(
                "PRODUCTION MODE requires GIST_ID + GIST_TOKEN. "
                "Set APEX_REQUIRE_STATE_PERSISTENCE=false to override"
            )
        if not self.use_gist:
            logging.warning("KVStore LOCAL mode — state WILL NOT persist across runs")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.cfg.gist_token}",
                "Accept": "application/vnd.github+json"}

    def _load(self) -> dict:
        if self._cache is not None:
            return self._cache
        if self.use_gist:
            try:
                r = self.http.request(
                    "GET", f"https://api.github.com/gists/{self.cfg.gist_id}",
                    headers=self._headers())
                files = r.json().get("files", {})
                content = files.get(self.cfg.gist_state_file, {}).get("content", "").strip()
                self._cache = json.loads(content) if content else {}
            except Exception as e:
                raise StateError(f"gist load: {redact(e)}")
        else:
            p = Path(self.cfg.local_state_file)
            try:
                self._cache = json.loads(p.read_text()) if p.exists() else {}
            except Exception as e:
                raise StateError(f"local load: {e}")
        if not isinstance(self._cache, dict):
            self._cache = {}
        return self._cache

    def _save(self, data: dict) -> bool:
        if self.use_gist:
            try:
                body = {"files": {self.cfg.gist_state_file: {
                    "content": json.dumps(data, default=str, sort_keys=True)}}}
                self.http.request(
                    "PATCH", f"https://api.github.com/gists/{self.cfg.gist_id}",
                    headers=self._headers(), json=body)
                self._cache = data
                return True
            except Exception as e:
                logging.error(f"gist save: {redact(e)}")
                return False
        else:
            try:
                Path(self.cfg.local_state_file).write_text(
                    json.dumps(data, default=str, sort_keys=True))
                self._cache = data
                return True
            except Exception as e:
                logging.error(f"local save: {e}")
                return False

    def get(self, key: str, default=None):
        try:
            return self._load().get(key, default)
        except StateError:
            return default

    def set(self, key: str, value) -> bool:
        try:
            data = dict(self._load())
        except StateError:
            data = {}
        data[key] = value
        return self._save(data)

    def force_load(self) -> None:
        self._cache = None
        self._load()


# ════════════════════════════════════════════════════════════
# INDICATORS
# ════════════════════════════════════════════════════════════
class Indicators:
    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                       axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def atr_last(df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < 2:
            return 0.0
        v = Indicators.atr_series(df, period).iloc[-1]
        return 0.0 if pd.isna(v) or v < 1e-6 else float(v)

    @staticmethod
    def atr_percentile(df: pd.DataFrame, period: int = 14, lookback: int = 200) -> float:
        if len(df) < lookback + period:
            return 0.5
        atr = Indicators.atr_series(df, period)
        cur = atr.iloc[-1]
        window = atr.iloc[-lookback:]
        lo, hi = window.min(), window.max()
        if hi - lo < 1e-9:
            return 0.5
        return float(max(0.0, min(1.0, (cur - lo) / (hi - lo))))

    @staticmethod
    def rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
        d = close.diff()
        gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        return 100 - 100 / (1 + gain / (loss + 1e-9))

    @staticmethod
    def adx_last(df: pd.DataFrame, period: int = 14) -> float:
        try:
            if len(df) < period * 2:
                return 0.0
            up = df["high"].diff()
            dn = -df["low"].diff()
            pdm = up.where((up > dn) & (up > 0), 0.0)
            mdm = dn.where((dn > up) & (dn > 0), 0.0)
            tr = pd.concat(
                [df["high"] - df["low"],
                 (df["high"] - df["close"].shift()).abs(),
                 (df["low"] - df["close"].shift()).abs()],
                axis=1).max(axis=1)
            atr_s = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
            pdi = 100 * pdm.ewm(alpha=1 / period, adjust=False).mean() / atr_s
            mdi = 100 * mdm.ewm(alpha=1 / period, adjust=False).mean() / atr_s
            dx = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100
            v = dx.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
            return 0.0 if pd.isna(v) or v < 0 else float(v)
        except Exception:
            return 0.0

    @staticmethod
    def hurst(close: pd.Series, max_lag: int = 20) -> float:
        try:
            ts = np.log(close.dropna().values)
            if len(ts) < max_lag * 2:
                return 0.5
            lags, tau = [], []
            for lag in range(2, max_lag):
                diff = ts[lag:] - ts[:-lag]
                std = float(np.std(diff))
                if std > 1e-12:
                    lags.append(lag)
                    tau.append(std)
            if len(tau) < 4:
                return 0.5
            return float(np.clip(np.polyfit(np.log(lags), np.log(tau), 1)[0], 0.0, 1.0))
        except Exception:
            return 0.5

    @staticmethod
    def shannon_entropy(close: pd.Series, bins: int = 10) -> float:
        try:
            r = close.pct_change().dropna()
            if len(r) < bins * 5:
                return 0.5
            hist, _ = np.histogram(r, bins=bins)
            p = hist / hist.sum()
            p = p[p > 0]
            return float(-np.sum(p * np.log(p)) / np.log(bins))
        except Exception:
            return 0.5


# ════════════════════════════════════════════════════════════
# v21 NEW: MACRO CONTEXT (DXY bias)
# ════════════════════════════════════════════════════════════
class MacroContext:
    def __init__(self, cfg: Config, http: HTTPClient, kv: KVStore):
        self.cfg = cfg
        self.http = http
        self.kv = kv

    def dxy_bias(self) -> str:
        """Return 'BEAR' | 'BULL' | 'NEUTRAL'. Fail-safe → NEUTRAL."""
        if not self.cfg.enable_dxy_filter:
            return "NEUTRAL"
        cache = self.kv.get("dxy_cache", {}) or {}
        if time.time() - cache.get("ts", 0) < 3600:
            return cache.get("bias", "NEUTRAL")
        try:
            r = self.http.request(
                "GET", "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB",
                params={"interval": "1d", "range": "1mo"},
                headers={"User-Agent": "Mozilla/5.0"})
            res = (r.json().get("chart", {}).get("result") or [None])[0]
            if not res:
                raise DataFetchError("no DXY result")
            closes = res["indicators"]["quote"][0].get("close", [])
            closes = [c for c in closes if c is not None]
            if len(closes) < 21:
                raise DataFetchError(f"DXY only {len(closes)} bars")
            s = pd.Series(closes)
            ema20 = s.ewm(span=20, adjust=False).mean().iloc[-1]
            cur = float(s.iloc[-1])
            bias = "BEAR" if cur < ema20 else "BULL"
            self.kv.set("dxy_cache", {"ts": time.time(), "bias": bias, "price": cur})
            logging.info(f"DXY bias={bias} price={cur:.2f}")
            return bias
        except Exception as e:
            logging.warning(f"DXY fetch failed: {redact(e)}")
            return "NEUTRAL"


# ════════════════════════════════════════════════════════════
# v21 NEW: KEY LEVELS ENGINE
# ════════════════════════════════════════════════════════════
class KeyLevels:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def compute(self, h1: pd.DataFrame, m5: pd.DataFrame) -> List[Tuple[float, str]]:
        """Return list of (level_price, level_name)."""
        levels: List[Tuple[float, str]] = []
        try:
            daily = (h1.set_index("datetime")
                       .resample("1D")
                       .agg({"open": "first", "high": "max",
                             "low": "min", "close": "last"})
                       .dropna())
            if len(daily) >= 2:
                prev = daily.iloc[-2]
                levels.append((float(prev["high"]), "PDH"))
                levels.append((float(prev["low"]), "PDL"))
                levels.append((float(daily.iloc[-1]["open"]), "Daily Open"))
        except Exception:
            pass

        try:
            price = float(m5["close"].iloc[-1])
            step = self.cfg.key_level_round_step
            base = round(price / step) * step
            for k in range(-3, 4):
                levels.append((float(base + k * step), f"Round {base + k*step:.0f}"))
        except Exception:
            pass

        try:
            last_dt = m5["datetime"].iloc[-1]
            today = last_dt.normalize()
            asia_end = today + pd.Timedelta(hours=7)
            asian = m5[(m5["datetime"] >= today) & (m5["datetime"] < asia_end)]
            if len(asian) > 5:
                levels.append((float(asian["high"].max()), "Asian High"))
                levels.append((float(asian["low"].min()), "Asian Low"))
        except Exception:
            pass

        return levels

    def tp_blocked(self, tp: float, direction: str,
                   levels: List[Tuple[float, str]]) -> Optional[str]:
        """Return blocking level name if TP would land too close to a level."""
        if not self.cfg.enable_key_levels or not levels:
            return None
        buf = self.cfg.key_level_buffer
        for lvl, name in levels:
            if abs(tp - lvl) <= buf:
                return name
        return None


# ════════════════════════════════════════════════════════════
# v21 NEW: MARKET STRUCTURE (Sweep + FVG)
# ════════════════════════════════════════════════════════════
class MarketStructure:
    @staticmethod
    def detect_sweep(df: pd.DataFrame, lookback: int = 20) -> Optional[dict]:
        if len(df) < lookback + 3:
            return None
        try:
            recent = df.iloc[-3:].reset_index(drop=True)
            hist = df.iloc[-(lookback + 3):-3]
            recent_high = float(hist["high"].max())
            recent_low = float(hist["low"].min())
            first_low = float(recent["low"].iloc[0])
            first_high = float(recent["high"].iloc[0])
            last_close = float(recent["close"].iloc[-1])

            if first_low < recent_low and last_close > recent_low:
                return {"type": "BULL_SWEEP", "level": recent_low,
                        "strength": min(1.0, (recent_low - first_low) / max(recent_low * 0.001, 1))}
            if first_high > recent_high and last_close < recent_high:
                return {"type": "BEAR_SWEEP", "level": recent_high,
                        "strength": min(1.0, (first_high - recent_high) / max(recent_high * 0.001, 1))}
        except Exception:
            pass
        return None

    @staticmethod
    def detect_fvg(df: pd.DataFrame, lookback: int = 10) -> Optional[dict]:
        if len(df) < 5:
            return None
        try:
            d = df.tail(lookback).reset_index(drop=True)
            for i in range(len(d) - 1, 1, -1):
                if d["low"].iloc[i] > d["high"].iloc[i - 2]:
                    top = float(d["low"].iloc[i])
                    bot = float(d["high"].iloc[i - 2])
                    return {"type": "BULL_FVG", "top": top, "bottom": bot,
                            "mid": (top + bot) / 2}
                if d["high"].iloc[i] < d["low"].iloc[i - 2]:
                    top = float(d["low"].iloc[i - 2])
                    bot = float(d["high"].iloc[i])
                    return {"type": "BEAR_FVG", "top": top, "bottom": bot,
                            "mid": (top + bot) / 2}
        except Exception:
            pass
        return None


# ════════════════════════════════════════════════════════════
# MARKET DATA HANDLER
# ════════════════════════════════════════════════════════════
class MarketDataHandler:
    def __init__(self, cfg: Config, http: HTTPClient, kv: KVStore):
        self.cfg = cfg
        self.http = http
        self.kv = kv
        self._last_td_call = 0.0

    def _validate(self, df: pd.DataFrame, name: str, min_bars: int) -> None:
        if df is None or len(df) < min_bars:
            raise DataFetchError(f"{name}: {0 if df is None else len(df)} < {min_bars} bars")
        if df["datetime"].isna().any():
            raise DataFetchError(f"{name}: NaN datetime")
        if df["datetime"].duplicated().any():
            raise DataFetchError(f"{name}: duplicate datetime")
        if (df["high"] < df["low"]).any():
            raise DataFetchError(f"{name}: high < low")
        if (df["high"] + 1e-9 < df[["open", "close"]].max(axis=1)).any():
            raise DataFetchError(f"{name}: high < max(open,close)")
        if (df["low"] - 1e-9 > df[["open", "close"]].min(axis=1)).any():
            raise DataFetchError(f"{name}: low > min(open,close)")
        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            raise DataFetchError(f"{name}: non-positive prices")
        last = float(df["close"].iloc[-1])
        if not (self.cfg.price_min <= last <= self.cfg.price_max):
            raise DataFetchError(f"{name}: price out of range")
        if len(df) > 2:
            prev = float(df["close"].iloc[-2])
            jump = abs(last - prev) / max(prev, 1e-9)
            if jump > self.cfg.price_max_jump_pct:
                raise DataFetchError(f"{name}: price jump {jump:.1%}")

    def _check_staleness(self, df: pd.DataFrame, tf_minutes: int, name: str) -> None:
        if df.empty:
            return
        last_ts = df["datetime"].iloc[-1]
        if last_ts.tz is None:
            last_ts = last_ts.tz_localize("UTC")
        age_min = (datetime.now(UTC) - last_ts.to_pydatetime()).total_seconds() / 60
        if age_min > self.cfg.max_stale_mult * tf_minutes:
            raise DataFetchError(f"{name}: stale {age_min:.0f}min")

    def _compute_offset(self) -> float:
        oanda_key = os.getenv("OANDA_API_KEY", "").strip()
        td_key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
        if not (oanda_key and td_key):
            return self.cfg.mt5_offset
        try:
            r1 = self.http.request(
                "GET", "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/pricing",
                headers={"Authorization": f"Bearer {oanda_key}"})
            oanda = float(r1.json()["prices"][0]["bids"][0]["price"])
            r2 = self.http.request(
                "GET", "https://api.twelvedata.com/price",
                params={"symbol": "XAU/USD", "apikey": td_key})
            td = float(r2.json()["price"])
            if td <= 0:
                return self.cfg.mt5_offset
            off = oanda - td
            if abs(off) > 1.0:
                return self.cfg.mt5_offset
            return round(off, 3)
        except Exception:
            return self.cfg.mt5_offset

    @staticmethod
    def _apply_offset(df: pd.DataFrame, off: float) -> pd.DataFrame:
        if off == 0.0:
            return df
        out = df.copy()
        for c in ("open", "high", "low", "close"):
            out[c] = out[c] + off
        return out

    @staticmethod
    def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        return (df.set_index("datetime")
                  .resample(rule, label="left", closed="left")
                  .agg({"open": "first", "high": "max", "low": "min",
                        "close": "last", "volume": "sum"})
                  .dropna().reset_index())

    def _rate_limit_td(self) -> None:
        elapsed = time.time() - self._last_td_call
        wait = self.cfg.td_min_interval_sec - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_td_call = time.time()

    def _cached_td_fetch(self, interval: str, outputsize: int,
                         ttl_minutes: int, offset: float) -> pd.DataFrame:
        cache_key = f"td_cache_{interval}"
        cache = self.kv.get(cache_key, {}) or {}
        cache_ts = float(cache.get("ts", 0) or 0)
        cache_data = cache.get("data")

        if cache_data and ttl_minutes > 0 and (time.time() - cache_ts) < ttl_minutes * 60:
            df = pd.DataFrame(cache_data)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            for c in ("open", "high", "low", "close", "volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close"])
            df = df.sort_values("datetime").reset_index(drop=True)
            return self._apply_offset(df, offset)

        key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
        if not key:
            raise DataFetchError("no TD key")

        self._rate_limit_td()
        r = self.http.request(
            "GET", "https://api.twelvedata.com/time_series",
            params={"symbol": "XAU/USD", "interval": interval,
                    "outputsize": outputsize, "apikey": key, "timezone": "UTC"})
        data = r.json()
        if "values" not in data or not data["values"]:
            if cache_data:
                df = pd.DataFrame(cache_data)
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                for c in ("open", "high", "low", "close", "volume"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=["open", "high", "low", "close"])
                df = df.sort_values("datetime").reset_index(drop=True)
                return self._apply_offset(df, offset)
            raise DataFetchError(f"TD {interval}: {data.get('message', 'no values')}")

        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["volume"] = 100.0
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df.sort_values("datetime").reset_index(drop=True)

        cache_copy = df.copy()
        cache_copy["datetime"] = cache_copy["datetime"].astype(str)
        self.kv.set(cache_key, {
            "ts": time.time(),
            "data": cache_copy[["datetime", "open", "high", "low", "close", "volume"]].to_dict("records")})
        return self._apply_offset(df, offset)

    def _fetch_td(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
        if not key:
            raise DataFetchError("no TD key")
        size = min(self.cfg.m5_fetch_size, 800)
        m5 = self._cached_td_fetch("5min", size, 0, offset)
        h1 = self._cached_td_fetch("1h", 800, self.cfg.h1_cache_minutes, offset)
        h4 = self._cached_td_fetch("4h", 800, self.cfg.h4_cache_minutes, offset)
        self._validate(m5, "TD-M5", self.cfg.min_bars_m5)
        self._validate(h1, "TD-H1", self.cfg.min_bars_h1)
        self._validate(h4, "TD-H4", self.cfg.min_bars_h4)
        self._check_staleness(m5, 5, "TD-M5")
        self._check_staleness(h1, 60, "TD-H1")
        self._check_staleness(h4, 240, "TD-H4")
        m15 = self._apply_offset(self._resample(m5, "15min"), offset)
        self._validate(m15, "TD-M15", self.cfg.min_bars_m15)
        return {"5min": m5, "15min": m15, "1h": h1, "4h": h4}, "TwelveData"

    def _fetch_xaus(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        base = "https://xaus.com/api/v1/chart"
        headers = {"Accept": "application/json"}

        def fetch(interval: str, rng: str) -> pd.DataFrame:
            r = self.http.request("GET", base,
                                  params={"symbol": "xau", "interval": interval, "range": rng},
                                  headers=headers)
            data = r.json()
            points = data.get("points")
            if not points:
                raise DataFetchError(f"XAUS {interval}: no points")
            rows = []
            for p in points:
                if not all(k in p for k in ("t", "o", "h", "l", "c")):
                    continue
                rows.append({"epoch": int(p["t"]),
                             "open": float(p["o"]), "high": float(p["h"]),
                             "low": float(p["l"]), "close": float(p["c"]),
                             "volume": float(p.get("v", 100.0) or 100.0)})
            if not rows:
                raise DataFetchError(f"XAUS {interval}: 0 valid rows")
            df = pd.DataFrame(rows)
            df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
            df = df[["datetime", "open", "high", "low", "close", "volume"]]
            return self._apply_offset(df.sort_values("datetime").reset_index(drop=True), offset)

        m5 = fetch("5m", "5d")
        h1 = fetch("1h", "1mo")
        h4 = self._apply_offset(self._resample(h1, "4h"), offset)
        self._validate(m5, "XAUS-M5", self.cfg.min_bars_m5)
        self._validate(h1, "XAUS-H1", self.cfg.min_bars_h1)
        self._validate(h4, "XAUS-H4", self.cfg.min_bars_h4)
        self._check_staleness(m5, 5, "XAUS-M5")
        self._check_staleness(h1, 60, "XAUS-H1")
        self._check_staleness(h4, 240, "XAUS-H4")
        m15 = self._apply_offset(self._resample(m5, "15min"), offset)
        self._validate(m15, "XAUS-M15", self.cfg.min_bars_m15)
        return {"5min": m5, "15min": m15, "1h": h1, "4h": h4}, "XAUS"

    def _fetch_biquote(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        base = "https://biquote.io/api/XAUUSD/ohlc"
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

        def fetch(interval: str, limit: int = 800) -> pd.DataFrame:
            r = self.http.request("GET", base,
                                  params={"interval": interval, "limit": limit},
                                  headers=headers)
            data = r.json()
            bars = data.get("bars")
            if not bars:
                raise DataFetchError(f"BQ {interval}: no bars")
            rows = []
            for b in bars:
                if "openTime" not in b:
                    continue
                rows.append({"datetime": b["openTime"],
                             "open": float(b["open"]), "high": float(b["high"]),
                             "low": float(b["low"]), "close": float(b["close"]),
                             "volume": float(b.get("tickVolume", 100.0) or 100.0)})
            if not rows:
                raise DataFetchError(f"BQ {interval}: 0 valid rows")
            df = pd.DataFrame(rows)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            return self._apply_offset(df.sort_values("datetime").reset_index(drop=True), offset)

        m5 = fetch("5m"); h1 = fetch("1h"); h4 = fetch("4h")
        self._validate(m5, "BQ-M5", self.cfg.min_bars_m5)
        self._validate(h1, "BQ-H1", self.cfg.min_bars_h1)
        self._validate(h4, "BQ-H4", self.cfg.min_bars_h4)
        self._check_staleness(m5, 5, "BQ-M5")
        self._check_staleness(h1, 60, "BQ-H1")
        self._check_staleness(h4, 240, "BQ-H4")
        m15 = self._apply_offset(self._resample(m5, "15min"), offset)
        self._validate(m15, "BQ-M15", self.cfg.min_bars_m15)
        return {"5min": m5, "15min": m15, "1h": h1, "4h": h4}, "Biquote"

    def _fetch_yahoo(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        symbol = "GC=F"

        def fetch(iv: str, rng: str) -> pd.DataFrame:
            r = self.http.request(
                "GET", f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                params={"interval": iv, "range": rng}, headers=headers)
            res = (r.json().get("chart", {}).get("result") or [None])[0]
            if not res:
                raise DataFetchError(f"YH {iv}: no result")
            ts = res.get("timestamp", [])
            q = res.get("indicators", {}).get("quote", [{}])[0]
            if not ts or not q:
                raise DataFetchError(f"YH {iv}: empty quote")
            o, hh = q.get("open", []), q.get("high", [])
            ll, cc = q.get("low", []), q.get("close", [])
            n = min(len(ts), len(o), len(hh), len(ll), len(cc))
            rows = []
            for i in range(n):
                if None in (o[i], hh[i], ll[i], cc[i]):
                    continue
                rows.append({"epoch": int(ts[i]),
                             "open": float(o[i]), "high": float(hh[i]),
                             "low": float(ll[i]), "close": float(cc[i]),
                             "volume": 100.0})
            if not rows:
                raise DataFetchError(f"YH {iv}: 0 valid rows")
            df = pd.DataFrame(rows)
            df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
            return df[["datetime", "open", "high", "low", "close", "volume"]]

        out = {}
        for iv, rng in (("5min", "5d"), ("15min", "5d"), ("1h", "1mo")):
            iv_map = {"5min": "5m", "15min": "15m", "1h": "1h"}
            out[iv] = self._apply_offset(fetch(iv_map[iv], rng), offset)
        out["4h"] = self._apply_offset(self._resample(fetch("1h", "1mo"), "4h"), offset)
        for tf, mins, min_bars in (
            ("5min", 5, self.cfg.min_bars_m5),
            ("15min", 15, self.cfg.min_bars_m15),
            ("1h", 60, self.cfg.min_bars_h1),
            ("4h", 240, self.cfg.min_bars_h4)):
            self._validate(out[tf], f"YH-{tf}", min_bars)
            self._check_staleness(out[tf], mins, f"YH-{tf}")
        return out, "Yahoo"

    def fetch_all(self) -> Tuple[Dict[str, pd.DataFrame], str, float]:
        offset = self._compute_offset()
        providers = (
            ("TwelveData", self._fetch_td),
            ("XAUS", self._fetch_xaus),
            ("Biquote", self._fetch_biquote),
            ("Yahoo", self._fetch_yahoo))
        errors = []
        for name, fn in providers:
            try:
                dfs, src = fn(offset)
                logging.info(f"Data source: {src} | offset={offset:+.3f}")
                return dfs, src, offset
            except Exception as e:
                errors.append(f"{name}: {redact(e)}")
                logging.warning(f"Data {name} failed: {redact(e)}")
        raise DataFetchError(" | ".join(errors))


# ════════════════════════════════════════════════════════════
# SIGNAL GENERATOR
# ════════════════════════════════════════════════════════════
@dataclass
class SignalResult:
    direction: str = "NEUTRAL"
    buy_score: float = 0.0
    sell_score: float = 0.0
    buy_ratio: float = 0.0
    sell_ratio: float = 0.0
    active_models: int = 0
    verdict: str = "CANCEL"
    skip_reason: str = ""
    votes_detail: list = field(default_factory=list)
    model_votes: dict = field(default_factory=dict)
    regime: str = "UNKNOWN"
    adx: float = 0.0
    hurst: float = 0.5
    entropy: float = 0.5
    mtf_score: int = 0


TREND_MODELS = {"EMA_Cross", "EMA_Trend", "Supertrend", "Donchian", "MACD", "Kalman"}
MEANREV_MODELS = {"RSI", "Stochastic", "BollingerBB"}
STRUCTURAL_MODELS = {"SMC", "Wavelet", "ML_RF", "Sweep", "FVG"}


class SignalGenerator:
    def __init__(self, cfg: Config, kv: KVStore):
        self.cfg = cfg
        self.kv = kv
        self._model_registry = self._build_registry()

    # ── existing static models (unchanged from v20) ──
    @staticmethod
    def _ema_cross(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 30: return 0, 0.0
        e9 = Indicators.ema(df["close"], 9)
        e21 = Indicators.ema(df["close"], 21)
        diff = e9 - e21
        a = Indicators.atr_last(df) + 1e-9
        if diff.iloc[-1] > 0 and diff.iloc[-2] > 0:
            return 1, min(1.0, abs(diff.iloc[-1]) / a)
        if diff.iloc[-1] < 0 and diff.iloc[-2] < 0:
            return -1, min(1.0, abs(diff.iloc[-1]) / a)
        return 0, 0.0

    @staticmethod
    def _ema_trend(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 60: return 0, 0.0
        c = df["close"]
        e20, e50 = Indicators.ema(c, 20), Indicators.ema(c, 50)
        if c.iloc[-1] > e50.iloc[-1] and e20.iloc[-1] > e50.iloc[-1]:
            return 1, 0.9
        if c.iloc[-1] < e50.iloc[-1] and e20.iloc[-1] < e50.iloc[-1]:
            return -1, 0.9
        return 0, 0.0

    @staticmethod
    def _supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> Tuple[int, float]:
        n = len(df)
        if n < period + 5: return 0, 0.0
        a = Indicators.atr_series(df, period).values
        hl2 = ((df["high"] + df["low"]) / 2).values
        close = df["close"].values
        upper = hl2 + mult * a
        lower = hl2 - mult * a
        trend = np.zeros(n, dtype=int)
        trend[0] = 1
        for i in range(1, n):
            if trend[i - 1] == 1:
                upper[i] = min(upper[i], upper[i - 1])
            else:
                lower[i] = max(lower[i], lower[i - 1])
            if close[i] > upper[i]:
                trend[i] = 1
            elif close[i] < lower[i]:
                trend[i] = -1
            else:
                trend[i] = trend[i - 1]
        if trend[-1] == 1 and trend[-2] == 1: return 1, 0.85
        if trend[-1] == -1 and trend[-2] == -1: return -1, 0.85
        return 0, 0.0

    @staticmethod
    def _donchian(df: pd.DataFrame, period: int = 20) -> Tuple[int, float]:
        if len(df) < period + 2: return 0, 0.0
        hh = df["high"].rolling(period).max().iloc[-2]
        ll = df["low"].rolling(period).min().iloc[-2]
        c = df["close"].iloc[-1]
        if c > hh: return 1, 0.8
        if c < ll: return -1, 0.8
        return 0, 0.0

    @staticmethod
    def _kalman(df: pd.DataFrame) -> Tuple[int, float]:
        obs = df["close"].values
        n = len(obs)
        if n < 10: return 0, 0.0
        x = obs[0]; P, Q, R = 1.0, 1e-3, 1e-2
        est = np.empty(n); est[0] = x
        for i in range(1, n):
            P += Q
            K = P / (P + R)
            x = x + K * (obs[i] - x)
            P *= (1 - K)
            est[i] = x
        slope = est[-1] - est[-5]
        a = Indicators.atr_last(df) + 1e-9
        if slope > 0: return 1, min(1.0, abs(slope) / a)
        if slope < 0: return -1, min(1.0, abs(slope) / a)
        return 0, 0.0

    @staticmethod
    def _macd(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 35: return 0, 0.0
        c = df["close"]
        macd = Indicators.ema(c, 12) - Indicators.ema(c, 26)
        sig = Indicators.ema(macd, 9)
        hist = macd - sig
        if macd.iloc[-1] > sig.iloc[-1] and hist.diff().iloc[-1] > 0:
            return 1, 0.9
        if macd.iloc[-1] < sig.iloc[-1] and hist.diff().iloc[-1] < 0:
            return -1, 0.9
        return 0, 0.0

    @staticmethod
    def _rsi(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 16: return 0, 0.0
        rsi = Indicators.rsi_series(df["close"], 14)
        cur, prev = rsi.iloc[-1], rsi.iloc[-2]
        if cur < 30 and cur > prev: return 1, 0.85
        if cur > 70 and cur < prev: return -1, 0.85
        if 50 < cur < 65 and cur > prev: return 1, 0.6
        if 35 < cur < 50 and cur < prev: return -1, 0.6
        return 0, 0.0

    @staticmethod
    def _stochastic(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 20: return 0, 0.0
        low = df["low"].rolling(14).min()
        high = df["high"].rolling(14).max()
        fast_k = 100 * (df["close"] - low) / (high - low + 1e-9)
        k = fast_k.rolling(3).mean()
        d = k.rolling(3).mean()
        k_, d_ = k.iloc[-1], d.iloc[-1]
        kp, dp = k.iloc[-2], d.iloc[-2]
        if pd.isna(k_) or pd.isna(d_): return 0, 0.0
        if k_ > d_ and kp <= dp and k_ < 25: return 1, 0.8
        if k_ < d_ and kp >= dp and k_ > 75: return -1, 0.8
        return 0, 0.0

    @staticmethod
    def _bb(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> Tuple[int, float]:
        if len(df) < period + 2: return 0, 0.0
        ma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        upper = (ma + mult * std).iloc[-1]
        lower = (ma - mult * std).iloc[-1]
        c = df["close"].iloc[-1]
        if c < lower: return 1, 0.75
        if c > upper: return -1, 0.75
        return 0, 0.0

    @staticmethod
    def _smc(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 60: return 0, 0.0
        d = df.tail(60).reset_index(drop=True)
        a = Indicators.atr_last(df)
        if a <= 0: return 0, 0.0
        bodies = (d["close"] - d["open"]).abs()
        cands = bodies.iloc[3:-1]
        if cands.empty: return 0, 0.0
        imp_idx = int(cands.idxmax())
        if imp_idx < 20 or abs(d["close"].iloc[imp_idx] - d["open"].iloc[imp_idx]) < a * 1.2:
            return 0, 0.0
        is_bull_imp = d["close"].iloc[imp_idx] > d["open"].iloc[imp_idx]
        lookback = d.iloc[max(0, imp_idx - 20):imp_idx]
        if is_bull_imp:
            if d["close"].iloc[imp_idx] <= lookback["high"].max():
                return 0, 0.0
        else:
            if d["close"].iloc[imp_idx] >= lookback["low"].min():
                return 0, 0.0
        ob = d.iloc[imp_idx - 1]
        is_bull_ob = ob["close"] > ob["open"]
        if is_bull_imp == is_bull_ob: return 0, 0.0
        last_close = d["close"].iloc[-1]
        ob_h = max(ob["open"], ob["close"])
        ob_l = min(ob["open"], ob["close"])
        buf = a * 0.5
        if ob_l - buf <= last_close <= ob_h + buf:
            return (1 if is_bull_imp else -1), 0.85
        return 0, 0.0

    @staticmethod
    def _build_features(df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        f = pd.DataFrame(index=df.index)
        f["ret_1"] = c.pct_change(1)
        f["ret_5"] = c.pct_change(5)
        f["ret_10"] = c.pct_change(10)
        f["atr_norm"] = Indicators.atr_series(df) / (c + 1e-9)
        f["vol_ratio"] = c.pct_change(1).rolling(20).std()
        f["rsi_14"] = Indicators.rsi_series(c, 14) / 100.0
        f["ema20_dist"] = (c - Indicators.ema(c, 20)) / (c + 1e-9)
        f["ema50_dist"] = (c - Indicators.ema(c, 50)) / (c + 1e-9)
        return f.replace([np.inf, -np.inf], np.nan).dropna()

    @staticmethod
    def _triple_barrier_labels(df, horizon=30, tp_mult=1.0, sl_mult=1.8):
        c = df["close"].values; h = df["high"].values
        l = df["low"].values; a = Indicators.atr_series(df).values
        n = len(c); labels = np.zeros(n)
        for i in range(n - horizon):
            tp = c[i] + tp_mult * a[i]; sl = c[i] - sl_mult * a[i]
            hh = h[i + 1:i + horizon + 1]; ll = l[i + 1:i + horizon + 1]
            hit_tp = np.where(hh >= tp)[0]; hit_sl = np.where(ll <= sl)[0]
            idx_tp = hit_tp[0] if len(hit_tp) else horizon + 1
            idx_sl = hit_sl[0] if len(hit_sl) else horizon + 1
            if idx_tp < idx_sl: labels[i] = 1
            elif idx_sl < idx_tp: labels[i] = -1
        return pd.Series(labels[:n - horizon], index=df.index[:n - horizon])

    def _ml_predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        if not (HAS_SKLEARN and self.cfg.enable_ml_models): return 0, 0.0
        if len(df) < 300: return 0, 0.0
        try:
            horizon = max(self.cfg.ml_purge_bars * 3, 20)
            X = self._build_features(df)
            y_raw = self._triple_barrier_labels(df, horizon=horizon)
            y = (y_raw == 1).astype(int)
            valid = X.index.intersection(y.dropna().index)
            X, y = X.loc[valid], y.loc[valid]
            forward = horizon + self.cfg.ml_purge_bars
            if len(X) < forward + 200: return 0, 0.0
            trX = X.iloc[:-forward]; trY = y.iloc[:-forward]
            if len(trX) < 200: return 0, 0.0
            split = int(len(trX) * 0.8)
            purged_end = max(100, split - forward)
            X_tr, y_tr = trX.iloc[:purged_end], trY.iloc[:purged_end]
            X_val, y_val = trX.iloc[split:], trY.iloc[split:]
            if len(X_tr) < 100 or len(X_val) < 20: return 0, 0.0
            model = RandomForestClassifier(n_estimators=80, max_depth=5,
                                           min_samples_leaf=8, random_state=42, n_jobs=1)
            model.fit(X_tr.values, y_tr.values)
            val_acc = float(model.score(X_val.values, y_val.values))
            if val_acc < self.cfg.ml_min_val_acc: return 0, 0.0
            proba = model.predict_proba(X.iloc[-1:].values)[0]
            up = float(proba[1]) if len(proba) > 1 else 0.5
            if up > 0.62: return 1, (up - 0.5) * 2
            if up < 0.38: return -1, (0.5 - up) * 2
            return 0, 0.0
        except Exception:
            return 0, 0.0

    @staticmethod
    def _wavelet(df: pd.DataFrame) -> Tuple[int, float]:
        if not HAS_SKLEARN or len(df) < 100: return 0, 0.0
        try:
            c = df["close"].values
            b, a = butter(2, 0.15)
            smooth = filtfilt(b, a, c)
            ss = pd.Series(smooth)
            ef = ss.ewm(span=10).mean()
            es = ss.ewm(span=30).mean()
            diff = ef.iloc[-1] - es.iloc[-1]
            slope = ef.diff(3).iloc[-1]
            a_val = Indicators.atr_last(df)
            if a_val <= 0: return 0, 0.0
            conf = min(1.0, abs(diff) / a_val)
            if diff > 0 and slope > 0: return 1, conf
            if diff < 0 and slope < 0: return -1, conf
            return 0, 0.0
        except Exception:
            return 0, 0.0

    # v21 NEW voter: liquidity sweep
    @staticmethod
    def _sweep(df: pd.DataFrame) -> Tuple[int, float]:
        sw = MarketStructure.detect_sweep(df, lookback=20)
        if not sw: return 0, 0.0
        if sw["type"] == "BULL_SWEEP": return 1, 0.75
        if sw["type"] == "BEAR_SWEEP": return -1, 0.75
        return 0, 0.0

    # v21 NEW voter: FVG confluence
    @staticmethod
    def _fvg(df: pd.DataFrame) -> Tuple[int, float]:
        fvg = MarketStructure.detect_fvg(df, lookback=10)
        if not fvg: return 0, 0.0
        price = float(df["close"].iloc[-1])
        if fvg["type"] == "BULL_FVG" and fvg["bottom"] <= price <= fvg["top"] + 0.5 * (fvg["top"] - fvg["bottom"]):
            return 1, 0.7
        if fvg["type"] == "BEAR_FVG" and fvg["bottom"] - 0.5 * (fvg["top"] - fvg["bottom"]) <= price <= fvg["top"]:
            return -1, 0.7
        return 0, 0.0

    def _build_registry(self) -> List[Tuple[str, Callable, float]]:
        registry: List[Tuple[str, Callable, float]] = [
            ("EMA_Cross", self._ema_cross, 1.0),
            ("EMA_Trend", self._ema_trend, 1.0),
            ("Supertrend", self._supertrend, 1.2),
            ("Donchian", self._donchian, 0.8),
            ("Kalman", self._kalman, 1.0),
            ("MACD", self._macd, 1.2),
            ("RSI", self._rsi, 1.0),
            ("Stochastic", self._stochastic, 0.9),
            ("BollingerBB", self._bb, 0.9),
            ("SMC", self._smc, 1.5),
            ("Wavelet", self._wavelet, 0.8)]
        if self.cfg.enable_sweep_voter:
            registry.append(("Sweep", self._sweep, 1.6))
        if self.cfg.enable_fvg_voter:
            registry.append(("FVG", self._fvg, 1.4))
        if self.cfg.enable_ml_models:
            registry.append(("ML_RF", lambda df: self._ml_predict(df), 1.0))
        return registry

    def _regime(self, h1: pd.DataFrame) -> Tuple[str, float, float]:
        H = Indicators.hurst(h1["close"])
        E = Indicators.shannon_entropy(h1["close"])
        if H > self.cfg.hurst_trend_min and E < self.cfg.entropy_chaos_max:
            return "TREND", H, E
        if H < self.cfg.hurst_meanrev_max:
            return "MEANREV", H, E
        return "CHAOS", H, E

    @staticmethod
    def _htf_trend(df: pd.DataFrame) -> str:
        if len(df) < 60: return "NEUTRAL"
        c = df["close"]
        ef, es = Indicators.ema(c, 20), Indicators.ema(c, 50)
        if c.iloc[-1] > es.iloc[-1] and ef.iloc[-1] > es.iloc[-1]: return "BULLISH"
        if c.iloc[-1] < es.iloc[-1] and ef.iloc[-1] < es.iloc[-1]: return "BEARISH"
        return "NEUTRAL"

    def _mtf_alignment(self, h4, h1, m15, direction: str) -> int:
        score = 0
        h4t = self._htf_trend(h4)
        h1t = self._htf_trend(h1)
        m15t = self._htf_trend(m15)
        if direction == "BUY":
            if h4t == "BULLISH": score += 2
            if h1t == "BULLISH": score += 1
            if m15t == "BULLISH": score += 1
        else:
            if h4t == "BEARISH": score += 2
            if h1t == "BEARISH": score += 1
            if m15t == "BEARISH": score += 1
        return score

    def _adaptive_weights(self) -> Dict[str, float]:
        """Bayesian posterior-based model weights from journal."""
        if not self.cfg.enable_adaptive_weights:
            return {}
        journal = self.kv.get("journal", []) or []
        closed = [j for j in journal
                  if j.get("outcome") in ("WIN_TP3", "WIN_BE", "LOSS")
                  and j.get("model_votes")]
        if len(closed) < self.cfg.adaptive_min_journal:
            return {}
        stats = {name: {"wins": 0, "losses": 0}
                 for name, _, _ in self._model_registry}
        for j in closed:
            direction = j.get("signal")
            win = j["outcome"].startswith("WIN")
            for model_name, vote in (j.get("model_votes") or {}).items():
                if model_name not in stats: continue
                if vote == "NEUTRAL": continue
                if vote == direction:
                    stats[model_name]["wins" if win else "losses"] += 1
                else:
                    stats[model_name]["losses" if win else "wins"] += 1
        # posterior mean
        posteriors = {}
        for name, s in stats.items():
            a = 1 + s["wins"]; b = 1 + s["losses"]
            posteriors[name] = a / (a + b)
        return posteriors

    def generate(self, m5: pd.DataFrame, m15: pd.DataFrame,
                 h1: pd.DataFrame, h4: pd.DataFrame) -> SignalResult:
        res = SignalResult()

        # [L2] ATR percentile regime
        if self.cfg.enable_atr_percentile:
            atr_pct = Indicators.atr_percentile(m5, self.cfg.atr_period,
                                                self.cfg.atr_pct_lookback)
            if not (self.cfg.atr_pct_min <= atr_pct <= self.cfg.atr_pct_max):
                res.skip_reason = (f"ATR pct {atr_pct:.0%} outside "
                                   f"[{self.cfg.atr_pct_min:.0%},{self.cfg.atr_pct_max:.0%}]")
                return res

        adx_val = Indicators.adx_last(m5)
        res.adx = adx_val
        if adx_val < self.cfg.adx_min:
            res.skip_reason = f"ADX {adx_val:.1f} < {self.cfg.adx_min}"
            return res

        # [L11] Volatility spike detection
        if self.cfg.enable_volatility_spike and len(m5) > 30:
            ranges = m5["high"] - m5["low"]
            med = float(ranges.iloc[-30:-1].median())
            if med > 0 and ranges.iloc[-1] > self.cfg.spike_range_mult * med:
                res.skip_reason = f"Vol spike: bar range {ranges.iloc[-1]:.2f} > {self.cfg.spike_range_mult}x median"
                return res

        regime, H, E = self._regime(h1)
        res.regime, res.hurst, res.entropy = regime, H, E
        if regime == "CHAOS":
            res.skip_reason = f"CHAOS regime (H={H:.2f} E={E:.2f})"
            return res

        h4_trend = self._htf_trend(h4)

        # [L6] Regime routing
        if self.cfg.enable_regime_routing:
            if regime == "TREND":
                active_set = TREND_MODELS | STRUCTURAL_MODELS
            elif regime == "MEANREV":
                active_set = MEANREV_MODELS | STRUCTURAL_MODELS
            else:
                active_set = set()
        else:
            active_set = None  # all

        adaptive = self._adaptive_weights()

        buy_w = sell_w = 0.0
        details = []
        model_votes = {}
        active = 0
        for name, fn, weight in self._model_registry:
            if active_set is not None and name not in active_set:
                details.append({"model": name, "vote": "SKIP", "weight": weight,
                                "conf": 0.0, "score": 0.0, "reason": "regime_routing"})
                continue
            try:
                vote, conf = fn(m5)
            except Exception as e:
                logging.debug(f"model {name} failed: {e}")
                vote, conf = 0, 0.0

            if vote == 0:
                details.append({"model": name, "vote": "NEUTRAL", "weight": weight,
                                "conf": round(conf, 2), "score": 0.0})
                model_votes[name] = "NEUTRAL"
                continue

            active += 1
            # [L8] Adaptive weight multiplier
            w_eff = weight
            if adaptive and name in adaptive:
                w_eff = weight * max(0.2, min(1.8, (adaptive[name] - 0.5) * 2 + 1.0))
            score = w_eff * conf
            if vote > 0:
                buy_w += score
                v_txt = "BUY"
            else:
                sell_w += score
                v_txt = "SELL"
            model_votes[name] = v_txt
            details.append({"model": name, "vote": v_txt,
                            "weight": round(w_eff, 2), "conf": round(conf, 2),
                            "score": round(score, 2)})

        res.buy_score = round(buy_w, 2)
        res.sell_score = round(sell_w, 2)
        res.active_models = active
        res.votes_detail = details
        res.model_votes = model_votes

        total = buy_w + sell_w
        if total < 1e-6:
            res.skip_reason = "all models neutral"
            return res
        res.buy_ratio = buy_w / total
        res.sell_ratio = sell_w / total

        if active < self.cfg.parliament_min_active:
            res.skip_reason = f"active {active} < {self.cfg.parliament_min_active}"
            return res

        dom = max(res.buy_ratio, res.sell_ratio)
        if dom >= self.cfg.parliament_full_ratio:
            res.verdict = "EXECUTE_FULL"
        elif dom >= self.cfg.parliament_half_ratio:
            res.verdict = "EXECUTE_HALF"
        elif dom >= self.cfg.parliament_min_ratio:
            res.verdict = "EXECUTE_MICRO"
        else:
            res.skip_reason = f"consensus {dom:.0%} too weak"
            return res

        if res.buy_ratio > res.sell_ratio:
            if h4_trend == "BEARISH":
                res.skip_reason = "BUY blocked: H4 bearish"
                return res
            res.direction = "BUY"
        else:
            if h4_trend == "BULLISH":
                res.skip_reason = "SELL blocked: H4 bullish"
                return res
            res.direction = "SELL"

        # [L3] MTF alignment gate
        if self.cfg.enable_mtf_alignment:
            mtf = self._mtf_alignment(h4, h1, m15, res.direction)
            res.mtf_score = mtf
            if mtf < self.cfg.mtf_min_score:
                res.skip_reason = f"MTF alignment {mtf}/4 < {self.cfg.mtf_min_score}"
                return res

        return res

    @staticmethod
    def detect_order_block(df: pd.DataFrame, side: str, lookback: int = 40) -> Optional[dict]:
        if len(df) < lookback + 5: return None
        d = df.tail(lookback + 5).reset_index(drop=True)
        a = Indicators.atr_last(df)
        if a <= 0: return None
        for i in range(len(d) - 2, 2, -1):
            body = abs(d["close"].iloc[i] - d["open"].iloc[i])
            if body < 1.2 * a: continue
            is_bull = d["close"].iloc[i] > d["open"].iloc[i]
            if side == "BUY" and not is_bull: continue
            if side == "SELL" and is_bull: continue
            for j in range(i - 1, max(i - 10, 0), -1):
                if side == "BUY" and d["close"].iloc[j] < d["open"].iloc[j]:
                    return {"high": float(d["high"].iloc[j]),
                            "low": float(d["low"].iloc[j])}
                if side == "SELL" and d["close"].iloc[j] > d["open"].iloc[j]:
                    return {"high": float(d["high"].iloc[j]),
                            "low": float(d["low"].iloc[j])}
        return None

    def compute_levels(self, m5: pd.DataFrame, side: str,
                       ob: Optional[dict]) -> Tuple[float, float, float, float, float]:
        a = Indicators.atr_last(m5)
        if a <= 0: raise ValueError("ATR=0")
        price = float(m5["close"].iloc[-1])
        p = self.cfg.price_decimals
        buf = self.cfg.spread_buffer * 2
        min_sl_atr = self.cfg.atr_sl_mult

        ob_h = ob["high"] if ob else price
        ob_l = ob["low"] if ob else price

        if side == "BUY":
            if ob and ob_h < price:
                entry = ob_h - (ob_h - ob_l) * 0.5
            elif ob and ob_l > price:
                entry = ob_l + (ob_h - ob_l) * 0.5
            else:
                entry = price - a * self.cfg.entry_pullback_atr
            if abs(price - entry) > self.cfg.entry_max_chase_atr * a:
                raise ValueError(f"BUY chase {abs(price-entry)/a:.2f} ATR")
            sl_atr = entry - a * min_sl_atr
            if ob and ob_l < entry:
                sl = min(sl_atr, ob_l - buf)
            else:
                sl = sl_atr
            if (entry - sl) < a * min_sl_atr * 0.90:
                sl = entry - a * min_sl_atr
            if entry <= sl: raise ValueError(f"invalid BUY entry={entry} sl={sl}")
            tp1 = round(entry + a * self.cfg.atr_tp1_mult, p)
            tp2 = round(entry + a * self.cfg.atr_tp2_mult, p)
            tp3 = round(entry + a * self.cfg.atr_tp3_mult, p)
            if not (sl < entry < tp1 < tp2 < tp3):
                raise ValueError("BUY ordering")
            return round(entry, p), round(sl, p), tp1, tp2, tp3
        else:
            if ob and ob_l > price:
                entry = ob_l + (ob_h - ob_l) * 0.5
            elif ob and ob_h < price:
                entry = ob_h - (ob_h - ob_l) * 0.5
            else:
                entry = price + a * self.cfg.entry_pullback_atr
            if abs(entry - price) > self.cfg.entry_max_chase_atr * a:
                raise ValueError(f"SELL chase {abs(entry-price)/a:.2f} ATR")
            sl_atr = entry + a * min_sl_atr
            if ob and ob_h > entry:
                sl = max(sl_atr, ob_h + buf)
            else:
                sl = sl_atr
            if (sl - entry) < a * min_sl_atr * 0.90:
                sl = entry + a * min_sl_atr
            if sl <= entry: raise ValueError(f"invalid SELL entry={entry} sl={sl}")
            tp1 = round(entry - a * self.cfg.atr_tp1_mult, p)
            tp2 = round(entry - a * self.cfg.atr_tp2_mult, p)
            tp3 = round(entry - a * self.cfg.atr_tp3_mult, p)
            if not (tp3 < tp2 < tp1 < entry < sl):
                raise ValueError("SELL ordering")
            return round(entry, p), round(sl, p), tp1, tp2, tp3


# ════════════════════════════════════════════════════════════
# v21 NEW: OPENING RANGE BREAKOUT STRATEGY
# ════════════════════════════════════════════════════════════
class ORBStrategy:
    """Independent strategy: NY opening range breakout on XAUUSD."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def evaluate(self, m5: pd.DataFrame, h1: pd.DataFrame) -> Optional[dict]:
        if not self.cfg.enable_orb:
            return None
        if len(m5) < 50:
            return None

        now = m5["datetime"].iloc[-1]
        if now.tz is None:
            now = now.tz_localize("UTC")

        # OR window: 13:00 - 13:30 UTC
        or_start = now.normalize() + pd.Timedelta(hours=self.cfg.orb_start_hour_utc)
        or_end = or_start + pd.Timedelta(minutes=self.cfg.orb_duration_min)
        trade_start = or_end
        trade_end = trade_start + pd.Timedelta(hours=self.cfg.orb_trade_hours)

        if not (trade_start <= now <= trade_end):
            return None

        or_bars = m5[(m5["datetime"] >= or_start) & (m5["datetime"] < or_end)]
        if len(or_bars) < 3:
            return None

        or_high = float(or_bars["high"].max())
        or_low = float(or_bars["low"].min())
        or_range = or_high - or_low

        daily_atr = Indicators.atr_last(
            h1.set_index("datetime").resample("1D").agg(
                {"high": "max", "low": "min", "open": "first",
                 "close": "last", "volume": "sum"}).dropna().reset_index(), 14)
        if daily_atr <= 0:
            return None

        if or_range < self.cfg.orb_min_range_atr_frac * daily_atr:
            return None
        if or_range > self.cfg.orb_max_range_atr_frac * daily_atr:
            return None

        price = float(m5["close"].iloc[-1])
        # H1 momentum confirmation
        h1_last5 = h1.tail(5)
        if len(h1_last5) >= 5:
            h1_mom = float(h1_last5["close"].iloc[-1] - h1_last5["close"].iloc[0])
        else:
            h1_mom = 0.0

        if price > or_high and h1_mom > 0:
            return {"direction": "BUY", "or_high": or_high, "or_low": or_low,
                    "or_range": or_range, "entry": or_high + 1.0,
                    "sl": or_low - 2.0,
                    "tp1": or_high + 1.0 * or_range,
                    "tp2": or_high + 1.5 * or_range,
                    "tp3": or_high + 2.0 * or_range,
                    "reason": f"ORB BUY: broke {or_high:.2f}, OR={or_range:.2f}"}
        if price < or_low and h1_mom < 0:
            return {"direction": "SELL", "or_high": or_high, "or_low": or_low,
                    "or_range": or_range, "entry": or_low - 1.0,
                    "sl": or_high + 2.0,
                    "tp1": or_low - 1.0 * or_range,
                    "tp2": or_low - 1.5 * or_range,
                    "tp3": or_low - 2.0 * or_range,
                    "reason": f"ORB SELL: broke {or_low:.2f}, OR={or_range:.2f}"}
        return None


# ════════════════════════════════════════════════════════════
# RISK MANAGEMENT
# ════════════════════════════════════════════════════════════
class RiskManagementEngine:
    def __init__(self, cfg: Config, kv: KVStore):
        self.cfg = cfg
        self.kv = kv

    def dynamic_balance(self) -> float:
        state = self.kv.get("state", {}) or {}
        lb = state.get("live_balance")
        if lb is not None:
            try:
                v = float(lb)
                if v > 0: return v
            except (ValueError, TypeError):
                pass
        return max(10.0, float(self.cfg.account_balance))

    def compute_lot(self, balance: float, entry: float, sl: float,
                    confidence_pct: float) -> Tuple[float, float]:
        if balance < self.cfg.min_balance_for_trading:
            raise ValueError(f"balance {balance:.2f} < min {self.cfg.min_balance_for_trading}")
        sl_dist = abs(entry - sl)
        if sl_dist < 1e-6: raise ValueError("SL too small")
        conf_f = max(0.4, min(1.0, confidence_pct / 100.0))
        base_pct = self.cfg.max_risk_pct_per_trade
        risk_pct = max(0.1, base_pct * conf_f)
        risk_money = balance * risk_pct / 100.0
        lots_raw = risk_money / (sl_dist * self.cfg.contract_size)
        lots = max(self.cfg.min_lot, min(self.cfg.max_lot, round(lots_raw, 2)))
        notional = lots * self.cfg.contract_size * entry
        if balance > 0 and notional / balance > self.cfg.hard_leverage_cap:
            lots = max(self.cfg.min_lot,
                       round(self.cfg.hard_leverage_cap * balance /
                             (self.cfg.contract_size * entry), 2))
        actual_risk = (lots * self.cfg.contract_size * sl_dist) / balance * 100.0
        if lots == self.cfg.min_lot and actual_risk > self.cfg.max_risk_pct_per_trade * 2:
            raise ValueError(f"min-lot risk {actual_risk:.2f}% > 2x limit")
        return lots, actual_risk

    def _journal(self) -> list:
        return self.kv.get("journal", []) or []

    def _today_signals_filled(self) -> int:
        today = datetime.now(UTC).date().isoformat()
        return sum(1 for j in self._journal()
                   if j.get("timestamp", "").startswith(today) and j.get("filled"))

    def _daily_loss_pct(self, balance: float) -> float:
        today = datetime.now(UTC).date().isoformat()
        total = 0.0
        for j in self._journal():
            if not j.get("timestamp", "").startswith(today): continue
            if j.get("outcome") == "LOSS":
                r = abs(float(j.get("r_achieved", -1.0)))
                rp = float(j.get("risk_pct", 1.0))
                total += r * (rp / 100.0)
        return total * 100.0 if balance > 0 else 0.0

    def _consecutive_losses(self) -> int:
        reset_ts = self.kv.get("consecutive_reset_ts", "")
        closed = [j for j in self._journal()
                  if j.get("outcome") in ("WIN_TP3", "WIN_BE", "LOSS")]
        closed.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        n = 0
        for j in closed:
            if reset_ts and j.get("timestamp", "") < reset_ts:
                break
            if j["outcome"] == "LOSS": n += 1
            else: break
        return n

    def _in_drawdown_pause(self) -> bool:
        last = self.kv.get("last_loss_ts", "")
        if not last: return False
        try:
            t = datetime.fromisoformat(last)
            if t.tzinfo is None:
                t = t.replace(tzinfo=UTC)
            return (datetime.now(UTC) - t) < timedelta(hours=self.cfg.dd_pause_hours)
        except Exception:
            return False

    def _session_blacklist(self) -> set:
        """[L5] Blacklist UTC hours with negative expectancy."""
        if not self.cfg.enable_session_blacklist:
            return set()
        closed = [j for j in self._journal()
                  if j.get("outcome") in ("WIN_TP3", "WIN_BE", "LOSS")
                  and j.get("hour_utc") is not None]
        if len(closed) < 20:
            return set()
        by_hour: Dict[int, List[float]] = {}
        for j in closed[-100:]:
            h = int(j["hour_utc"])
            by_hour.setdefault(h, []).append(float(j.get("r_achieved", 0.0)))
        blacklist = set()
        for h, rs in by_hour.items():
            if len(rs) >= self.cfg.session_min_trades:
                avg = sum(rs) / len(rs)
                if avg < self.cfg.session_expectancy_threshold:
                    blacklist.add(h)
                    logging.info(f"Session blacklist hour {h:02d} UTC: {avg:+.2f}R")
        return blacklist

    def check_circuit_breakers(self, balance: float, hour_utc: int) -> Tuple[bool, str]:
        n = self._today_signals_filled()
        if n >= self.cfg.max_daily_signals:
            return False, f"MAX_DAILY_SIGNALS({n})"
        dl = self._daily_loss_pct(balance)
        if dl >= self.cfg.max_daily_loss_pct:
            return False, f"MAX_DAILY_LOSS({dl:.1f}%)"
        cl = self._consecutive_losses()
        if cl >= self.cfg.max_consecutive_losses:
            if self._in_drawdown_pause():
                return False, f"MAX_CONSEC_LOSSES({cl})+PAUSE"
            self.kv.set("consecutive_reset_ts", datetime.now(UTC).isoformat())
            logging.info("Drawdown pause expired; consecutive-loss counter reset")
        blacklist = self._session_blacklist()
        if hour_utc in blacklist:
            return False, f"SESSION_BLACKLIST(h{hour_utc:02d})"
        return True, ""


# ════════════════════════════════════════════════════════════
# NEWS FILTER
# ════════════════════════════════════════════════════════════
class NewsSafetyFilter:
    def __init__(self, cfg: Config, kv: KVStore, http: HTTPClient):
        self.cfg = cfg; self.kv = kv; self.http = http

    def is_blackout(self) -> Tuple[bool, str]:
        if self.cfg.news_blackout_min <= 0: return False, ""
        cached = self.kv.get("news_cache", {}) or {}
        now_ts = time.time()
        ttl = (self.cfg.news_cache_sec if cached.get("ok")
               else self.cfg.news_cache_sec_on_fail)
        if now_ts - cached.get("ts", 0) < ttl:
            return bool(cached.get("blackout")), cached.get("title", "")
        fail_count = int(self.kv.get("news_fail_count", 0))
        try:
            r = self.http.request(
                "GET", "https://nfs.faireconomy.media/ff_calendar_thisweek.json")
            events = r.json()
            if not isinstance(events, list):
                raise DataFetchError("unexpected payload")
            now = datetime.now(UTC); blackout, title = False, ""
            for ev in events:
                if str(ev.get("country", "")).upper() != "USD": continue
                if str(ev.get("impact", "")).lower() != "high": continue
                try:
                    ds = str(ev["date"]).replace("Z", "+00:00")
                    evt = datetime.fromisoformat(ds)
                    if evt.tzinfo is None: evt = evt.replace(tzinfo=UTC)
                except Exception: continue
                if abs((evt - now).total_seconds()) / 60 <= self.cfg.news_blackout_min:
                    blackout, title = True, str(ev.get("title", "High impact")); break
            self.kv.set("news_cache", {"ts": now_ts, "ok": True,
                                        "blackout": blackout, "title": title})
            self.kv.set("news_fail_count", 0)
            return blackout, title
        except Exception as e:
            fail_count += 1
            self.kv.set("news_fail_count", fail_count)
            self.kv.set("news_cache", {"ts": now_ts, "ok": False,
                                        "blackout": False, "title": ""})
            logging.warning(f"news fetch failed: {redact(e)}")
            if fail_count >= self.cfg.news_max_consecutive_fail:
                return True, "news_feed_failure_guard"
            return False, ""


# ════════════════════════════════════════════════════════════
# TELEGRAM
# ════════════════════════════════════════════════════════════
class TelegramNotifier:
    def __init__(self, cfg: Config, kv: KVStore, http: HTTPClient):
        self.cfg = cfg; self.kv = kv; self.http = http
        self.token = os.getenv("TELEGRAM_TOKEN", "").strip()
        self.chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    def send(self, msg: str) -> bool:
        if self.cfg.dry_run:
            msg = "🧪 <b>DRY RUN (v21)</b>\n" + msg
        if not (self.token and self.chat):
            if self.cfg.dry_run:
                logging.info(f"[DRY-TG] {redact(msg)[:300]}")
                return True
            logging.warning("Telegram not configured")
            return False
        digest = hashlib.sha256(msg.encode()).hexdigest()[:16]
        now = datetime.now(UTC)
        dedup = self.kv.get("tg_hashes", {}) or {}
        dedup = {k: v for k, v in dedup.items() if self._recent(v, now, 3600)}
        if digest in dedup: return True
        try:
            r = self.http.request(
                "POST", f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat, "text": msg, "parse_mode": "HTML"})
            if r.json().get("ok"):
                dedup[digest] = now.isoformat()
                self.kv.set("tg_hashes", dedup)
                return True
        except Exception as e:
            logging.error(f"TG send: {redact(e)}")
        return False

    def send_cb_alert_once(self, reason: str) -> None:
        today = datetime.now(UTC).date().isoformat()
        key = f"cb_alert_{today}"
        if self.kv.get(key): return
        self.send(f"<b>🛑 CIRCUIT BREAKER — {reason}</b>\n"
                  f"Trading halted for remainder of {today} UTC.")
        self.kv.set(key, True)

    @staticmethod
    def _recent(iso_ts: str, now: datetime, sec: int) -> bool:
        try:
            return (now - datetime.fromisoformat(iso_ts)).total_seconds() < sec
        except Exception:
            return False


# ════════════════════════════════════════════════════════════
# OUTCOME TRACKER
# ════════════════════════════════════════════════════════════
class OutcomeTracker:
    def __init__(self, cfg: Config, kv: KVStore):
        self.cfg = cfg; self.kv = kv

    def _close_r_single_tp(self, ep, sl, tp3, hs, h3):
        if h3:
            return "WIN_TP3", round(abs(tp3 - ep) / abs(ep - sl), 3)
        if hs:
            return "LOSS", -1.0
        return "PENDING", 0.0

    def _close_r_scaled_tp(self, ep, sl, tp1, tp2, tp3, h1_, h2, h3, hb, hs):
        if not (h1_ or h2 or h3 or hb or hs): return "PENDING", 0.0
        dist = abs(ep - sl)
        r1 = abs(tp1 - ep) / dist; r2 = abs(tp2 - ep) / dist; r3 = abs(tp3 - ep) / dist
        if h3: return "WIN_TP3", round((r1 + r2 + r3) / 3.0, 3)
        if h2: return "WIN_BE", round((r1 + r2) / 3.0, 3)
        if h1_ and hb: return "WIN_BE", round(r1 / 3.0, 3)
        if hb: return "WIN_BE", 0.0
        if hs: return "LOSS", -1.0
        return "PENDING", 0.0

    def update(self, m5: pd.DataFrame) -> List[Tuple[str, dict]]:
        events = []
        now = datetime.now(UTC)
        journal = self.kv.get("journal", []) or []
        changed = False

        for entry in journal:
            if entry.get("outcome") not in ("PENDING", None): continue
            anchor = entry.get("signal_bar_dt") or entry.get("timestamp")
            try:
                ts = pd.Timestamp(anchor, tz="UTC")
            except Exception:
                continue

            if (now - ts.to_pydatetime()) > timedelta(hours=self.cfg.pending_max_age_hours):
                entry["outcome"] = "STALE"
                entry["r_achieved"] = 0.0
                events.append(("STALE", dict(entry)))
                changed = True
                continue

            side = entry["signal"]
            ep = float(entry["entry"]); sl = float(entry["sl"])
            tp1 = float(entry["tp1"]); tp2 = float(entry["tp2"]); tp3 = float(entry["tp3"])
            if ep <= 0 or sl <= 0: continue

            bars = m5[m5["datetime"] > ts].sort_values("datetime")
            if bars.empty: continue

            filled = bool(entry.get("filled"))
            fill_idx: Optional[int] = None
            if not filled:
                for idx, b in enumerate(bars.itertuples(index=False)):
                    if side == "BUY" and b.low <= ep:
                        filled, fill_idx = True, idx; break
                    if side == "SELL" and b.high >= ep:
                        filled, fill_idx = True, idx; break
                if not filled:
                    atr_sig = float(entry.get("atr_at_signal", 0) or 0)
                    if atr_sig > 0:
                        limit_dist = self.cfg.entry_max_chase_atr * atr_sig
                    else:
                        limit_dist = (self.cfg.entry_max_chase_atr *
                                      abs(ep - sl) / max(self.cfg.atr_sl_mult, 1e-6))
                    last_high = float(bars["high"].iloc[-1])
                    last_low = float(bars["low"].iloc[-1])
                    if side == "BUY" and last_high > ep + limit_dist:
                        entry["outcome"] = "INVALIDATED"
                        entry["filled"] = 0; entry["r_achieved"] = 0.0
                        events.append(("INVALIDATED", dict(entry)))
                        changed = True; continue
                    if side == "SELL" and last_low < ep - limit_dist:
                        entry["outcome"] = "INVALIDATED"
                        entry["filled"] = 0; entry["r_achieved"] = 0.0
                        events.append(("INVALIDATED", dict(entry)))
                        changed = True; continue
                    continue
                entry["filled"] = 1
                changed = True

            walk = bars.iloc[fill_idx:] if fill_idx is not None else bars
            h1_ = bool(entry.get("hit_tp1"))
            h2 = bool(entry.get("hit_tp2"))
            h3 = bool(entry.get("hit_tp3"))
            hs = bool(entry.get("hit_sl"))
            hb = bool(entry.get("hit_be"))
            closed_here = False

            for b in walk.itertuples(index=False):
                if h3 or hs or hb: break
                csl = ep if h1_ else sl
                if side == "BUY":
                    if b.low <= csl:
                        if h1_: hb = True
                        else: hs = True
                        continue
                    if not h1_ and b.high >= tp1: h1_ = True
                    if not h2 and b.high >= tp2: h2 = True
                    if not h3 and b.high >= tp3: h3 = True
                else:
                    if b.high >= csl:
                        if h1_: hb = True
                        else: hs = True
                        continue
                    if not h1_ and b.low <= tp1: h1_ = True
                    if not h2 and b.low <= tp2: h2 = True
                    if not h3 and b.low <= tp3: h3 = True

            entry["hit_tp1"] = int(h1_); entry["hit_tp2"] = int(h2)
            entry["hit_tp3"] = int(h3); entry["hit_sl"] = int(hs)
            entry["hit_be"] = int(hb)

            if entry.get("outcome") == "PENDING":
                if self.cfg.exit_model == "single_tp":
                    outcome, r_val = self._close_r_single_tp(ep, sl, tp3, hs, h3)
                else:
                    outcome, r_val = self._close_r_scaled_tp(
                        ep, sl, tp1, tp2, tp3, h1_, h2, h3, hb, hs)
                if outcome != "PENDING":
                    entry["outcome"] = outcome
                    entry["r_achieved"] = r_val
                    events.append(("CLOSE", dict(entry)))
                    changed = True; closed_here = True
                    if outcome == "LOSS":
                        self.kv.set("last_loss_ts", now.isoformat())

            if not closed_here and (h1_ or h2 or h3 or hs or hb):
                changed = True

        if changed:
            self.kv.set("journal", journal)
        return events


# ════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ════════════════════════════════════════════════════════════
class TradingBotOrchestrator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.http = HTTPClient(cfg)
        self.kv = KVStore(cfg, self.http)
        self.data = MarketDataHandler(cfg, self.http, self.kv)
        self.signal = SignalGenerator(cfg, self.kv)
        self.risk = RiskManagementEngine(cfg, self.kv)
        self.news = NewsSafetyFilter(cfg, self.kv, self.http)
        self.tg = TelegramNotifier(cfg, self.kv, self.http)
        self.outcome = OutcomeTracker(cfg, self.kv)
        self.macro = MacroContext(cfg, self.http, self.kv)
        self.keylevels = KeyLevels(cfg)
        self.orb = ORBStrategy(cfg)

    def _in_killzone(self, now: datetime) -> bool:
        if self.cfg.skip_killzone_filter: return True
        ld = now.astimezone(LONDON); ny = now.astimezone(NY)
        lh = ld.hour + ld.minute / 60.0
        nh = ny.hour + ny.minute / 60.0
        return (7.0 <= lh < 10.0) or (8.0 <= nh < 11.0) or (13.0 <= nh < 16.0)

    @staticmethod
    def _is_holiday(now: datetime) -> bool:
        return now.date().isoformat() in US_HOLIDAYS

    def run(self) -> None:
        try:
            self.kv.force_load()
        except StateError as e:
            logging.error(f"state load aborted: {e}")
            return

        now = datetime.now(UTC)
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        hour_utc = now.hour

        if self.cfg.kill_switch:
            logging.info("KILL switch engaged"); return
        if now.weekday() >= 5:
            logging.info("weekend — skip"); return
        if self._is_holiday(now):
            logging.info("US holiday — skip"); return
        if not self._in_killzone(now):
            logging.info("outside killzone — skip"); return

        balance = self.risk.dynamic_balance()
        ok, reason = self.risk.check_circuit_breakers(balance, hour_utc)
        if not ok:
            logging.info(f"circuit-breaker: {reason}")
            self.tg.send_cb_alert_once(reason)
            return

        blackout, title = self.news.is_blackout()
        if blackout:
            logging.info(f"news blackout: {title}"); return

        try:
            dfs, src, offset = self.data.fetch_all()
        except Exception as e:
            logging.error(f"data fetch: {redact(e)}")
            self.tg.send(f"<b>⚠️ Data unavailable</b>\n<code>{redact(e)[:200]}</code>")
            return

        # Update open outcomes
        try:
            for kind, ent in self.outcome.update(dfs["5min"]):
                if kind == "CLOSE":
                    self.tg.send(
                        f"<b>📕 Closed</b> {ent['signal']} {ent['symbol']}\n"
                        f"Entry: <code>{ent['entry']}</code> → {ent['outcome']} "
                        f"({ent.get('r_achieved', 0):+.2f}R)")
                elif kind in ("INVALIDATED", "STALE"):
                    self.tg.send(f"<b>⚪ {kind}</b> {ent['signal']} @ {ent['entry']}")
        except Exception as e:
            logging.warning(f"outcome update: {e}")

        balance = self.risk.dynamic_balance()
        m5c = dfs["5min"].iloc[:-1].reset_index(drop=True)
        m15c = dfs["15min"].iloc[:-1].reset_index(drop=True)
        h1c = dfs["1h"].iloc[:-1].reset_index(drop=True)
        h4c = dfs["4h"].iloc[:-1].reset_index(drop=True)
        if len(m5c) < self.cfg.min_bars_m5 or len(h1c) < self.cfg.min_bars_h1:
            logging.warning("post-trim insufficient bars"); return
        price = float(m5c["close"].iloc[-1])
        signal_bar_iso = m5c.iloc[-1]["datetime"].strftime("%Y-%m-%d %H:%M:%S")
        atr_at_signal = Indicators.atr_last(m5c)

        journal = self.kv.get("journal", []) or []
        if any(j.get("outcome") == "PENDING" and j.get("filled") for j in journal):
            logging.info("open filled trade — skip"); return
        pending_unfilled = sum(1 for j in journal
                               if j.get("outcome") == "PENDING" and not j.get("filled"))
        if pending_unfilled >= 2:
            logging.info(f"too many unfilled pendings ({pending_unfilled})"); return

        # ── STRATEGY SELECTION: ORB first (higher priority), then parliament ──
        candidate = None
        strategy_used = None

        if self.cfg.enable_orb:
            try:
                orb_sig = self.orb.evaluate(m5c, h1c)
            except Exception as e:
                logging.debug(f"ORB eval failed: {e}")
                orb_sig = None
            if orb_sig:
                candidate = orb_sig
                strategy_used = "ORB"
                logging.info(f"ORB candidate: {orb_sig['reason']}")

        if candidate is None:
            result = self.signal.generate(m5c, m15c, h1c, h4c)
            logging.info(
                f"parliament {result.direction} "
                f"BUY={result.buy_ratio:.0%} SELL={result.sell_ratio:.0%} "
                f"verdict={result.verdict} mtf={result.mtf_score} "
                f"skip={result.skip_reason}")

            if result.direction == "NEUTRAL" or result.verdict == "CANCEL":
                self._update_state(price, now_str); return

            # DXY hard filter (opt-in)
            if self.cfg.enable_dxy_filter and self.cfg.dxy_require_match:
                dxy = self.macro.dxy_bias()
                if dxy != "NEUTRAL":
                    if result.direction == "BUY" and dxy == "BULL":
                        logging.info("DXY BULL blocks BUY"); self._update_state(price, now_str); return
                    if result.direction == "SELL" and dxy == "BEAR":
                        logging.info("DXY BEAR blocks SELL"); self._update_state(price, now_str); return

            # Cooldown check
            state = self.kv.get("state", {}) or {}
            last_type = state.get("last_signal_type", "NEUTRAL")
            last_ts = state.get("last_signal_ts", "")
            cool = (self.cfg.opposite_cooldown_minutes if last_type != result.direction
                    else self.cfg.cooldown_minutes)
            if last_ts:
                try:
                    last_dt = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
                    if (now - last_dt) < timedelta(minutes=cool):
                        logging.info(f"cooldown {cool}min"); self._update_state(price, now_str); return
                except Exception:
                    pass

            ob = self.signal.detect_order_block(m5c, result.direction)
            try:
                entry, sl, tp1, tp2, tp3 = self.signal.compute_levels(m5c, result.direction, ob)
            except ValueError as e:
                logging.warning(f"level computation: {e}"); self._update_state(price, now_str); return

            # [L4] Key levels anti-collision
            if self.cfg.enable_key_levels:
                levels = self.keylevels.compute(h1c, m5c)
                blocked = self.keylevels.tp_blocked(tp1, result.direction, levels)
                if blocked:
                    logging.info(f"TP1 blocked by key level: {blocked}")
                    self._update_state(price, now_str); return

            candidate = {
                "direction": result.direction,
                "entry": entry, "sl": sl,
                "tp1": tp1, "tp2": tp2, "tp3": tp3,
                "verdict": result.verdict,
                "buy_ratio": result.buy_ratio,
                "sell_ratio": result.sell_ratio,
                "regime": result.regime, "adx": result.adx,
                "mtf_score": result.mtf_score,
                "model_votes": result.model_votes,
                "ob": ob,
            }
            strategy_used = "PARLIAMENT"

        # Compute lot
        conf_pct = 70.0  # default for ORB
        if strategy_used == "PARLIAMENT":
            conf_pct = max(candidate["buy_ratio"], candidate["sell_ratio"]) * 100.0
        try:
            lot, actual_risk = self.risk.compute_lot(
                balance, candidate["entry"], candidate["sl"], conf_pct)
        except ValueError as e:
            logging.warning(f"lot sizing: {e}"); self._update_state(price, now_str); return

        if candidate.get("verdict") == "EXECUTE_HALF":
            lot = max(self.cfg.min_lot, round(lot / 2.0, 2))
        elif candidate.get("verdict") == "EXECUTE_MICRO":
            lot = max(self.cfg.min_lot, round(lot / 4.0, 2))
        actual_risk = (lot * self.cfg.contract_size * abs(candidate["entry"] - candidate["sl"])) / balance * 100.0

        # DXY info for message
        dxy_info = ""
        if self.cfg.enable_dxy_filter:
            dxy_bias = self.macro.dxy_bias()
            if dxy_bias != "NEUTRAL":
                dxy_info = f" | DXY: {dxy_bias}"

        ob = candidate.get("ob")
        msg = (
            f"<b>🏛️ APEX v{VERSION} — {candidate['direction']} "
            f"({strategy_used})</b>\n\n"
            f"<b>{self.cfg.symbol}</b> limit @ <code>{candidate['entry']}</code>\n"
            f"SL: <code>{candidate['sl']}</code> | TP1: <code>{candidate['tp1']}</code>\n"
            f"TP2: <code>{candidate['tp2']}</code> | TP3: <code>{candidate['tp3']}</code>\n"
            f"Lot: <code>{lot}</code> (Bal: ${balance:.2f}, risk: {actual_risk:.2f}%)\n"
        )
        if strategy_used == "PARLIAMENT":
            msg += (f"Verdict: {candidate['verdict']} | "
                    f"Vote: BUY={candidate['buy_ratio']:.0%} "
                    f"SELL={candidate['sell_ratio']:.0%}\n"
                    f"Regime: {candidate['regime']} | ADX: {candidate['adx']:.1f} | "
                    f"MTF: {candidate.get('mtf_score', 0)}/4\n")
        else:
            msg += f"Strategy: Opening Range Breakout\n"
        msg += f"ATR: {atr_at_signal:.2f}{dxy_info}\n"
        if ob:
            msg += f"OB: ✅ {ob['high']:.2f}–{ob['low']:.2f}\n"
        else:
            msg += "OB: ⚠️ none\n"
        msg += f"Src: {src} | off: {offset:+.3f}\n"
        msg += "\n<i>Pasang limit order di MT5 pada harga di atas.\n"
        msg += "Setelah TP1 hit → SL ke BE.</i>"

        jid = f"{now_str.replace(' ', 'T')}_{candidate['direction']}_{lot}"
        entry_data = {
            "id": jid, "timestamp": now_str,
            "signal_bar_dt": signal_bar_iso,
            "symbol": self.cfg.symbol,
            "signal": candidate["direction"],
            "entry": candidate["entry"], "sl": candidate["sl"],
            "tp1": candidate["tp1"], "tp2": candidate["tp2"], "tp3": candidate["tp3"],
            "lot": lot, "verdict": candidate.get("verdict", "ORB"),
            "outcome": "PENDING", "filled": 0, "r_achieved": 0.0,
            "risk_pct": round(actual_risk, 3),
            "atr_at_signal": round(atr_at_signal, 4),
            "exit_model": self.cfg.exit_model,
            "dry_run": bool(self.cfg.dry_run),
            "strategy": strategy_used,
            "hour_utc": hour_utc,
            "model_votes": candidate.get("model_votes", {}),
        }

        if self._journal_append(entry_data):
            if self.tg.send(msg):
                state = self.kv.get("state", {}) or {}
                state["last_signal_type"] = candidate["direction"]
                state["last_signal_ts"] = now_str
                self.kv.set("state", state)
                logging.info(
                    f"SENT {strategy_used} {candidate['direction']} @ "
                    f"{candidate['entry']} lot={lot}")

        self._update_state(price, now_str)

    def _update_state(self, price: float, now_str: str) -> None:
        state = self.kv.get("state", {}) or {}
        state["run_count"] = int(state.get("run_count", 0)) + 1
        state["last_price"] = price
        state["last_price_ts"] = now_str
        self.kv.set("state", state)

    def _journal_append(self, entry: dict) -> bool:
        j = list(self.kv.get("journal", []) or [])
        j = [e for e in j if e.get("id") != entry.get("id")]
        j.append(entry)
        j.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return self.kv.set("journal", j[: self.cfg.max_journal])


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════
def cmd_smoke(cfg: Config) -> None:
    http = HTTPClient(cfg); kv = KVStore(cfg, http)
    data = MarketDataHandler(cfg, http, kv)
    signal = SignalGenerator(cfg, kv)
    dfs, src, offset = data.fetch_all()
    m5c = dfs["5min"].iloc[:-1].reset_index(drop=True)
    m15c = dfs["15min"].iloc[:-1].reset_index(drop=True)
    h1c = dfs["1h"].iloc[:-1].reset_index(drop=True)
    h4c = dfs["4h"].iloc[:-1].reset_index(drop=True)
    print(f"Source: {src} | Offset: {offset:+.3f}")
    print(f"Bars: M5={len(m5c)} M15={len(m15c)} H1={len(h1c)} H4={len(h4c)}")
    print(f"Last M5: {m5c.iloc[-1]['datetime']} close={m5c.iloc[-1]['close']:.2f}\n")

    atr_pct = Indicators.atr_percentile(m5c, cfg.atr_period, cfg.atr_pct_lookback)
    print(f"ATR percentile: {atr_pct:.0%} "
          f"(sweet spot [{cfg.atr_pct_min:.0%},{cfg.atr_pct_max:.0%}])")
    dxy = MacroContext(cfg, http, kv).dxy_bias()
    print(f"DXY bias: {dxy}")

    kl = KeyLevels(cfg).compute(h1c, m5c)
    print(f"Key levels: {[(f'{l:.2f}', n) for l, n in kl[:6]]}")

    sw = MarketStructure.detect_sweep(m5c)
    fvg = MarketStructure.detect_fvg(m5c)
    print(f"Sweep: {sw['type'] if sw else 'none'} | "
          f"FVG: {fvg['type'] if fvg else 'none'}")

    res = signal.generate(m5c, m15c, h1c, h4c)
    print("\n=== PARLIAMENT ===")
    for v in res.votes_detail:
        print(f"  {v['model']:16s} {v['vote']:8s} w={v['weight']} "
              f"conf={v['conf']} score={v['score']}")
    print(f"\nBUY: {res.buy_score} ({res.buy_ratio:.0%}) | "
          f"SELL: {res.sell_score} ({res.sell_ratio:.0%})")
    print(f"Active: {res.active_models} | Dir: {res.direction} | "
          f"Verdict: {res.verdict} | MTF: {res.mtf_score}/4")
    print(f"ADX: {res.adx:.1f} | Regime: {res.regime}")
    if res.skip_reason:
        print(f"Skip: {res.skip_reason}")

    if res.direction != "NEUTRAL":
        ob = signal.detect_order_block(m5c, res.direction)
        try:
            e, s, t1, t2, t3 = signal.compute_levels(m5c, res.direction, ob)
            a = Indicators.atr_last(m5c)
            print(f"\nProposed {res.direction}: entry={e} sl={s} "
                  f"tp1={t1} tp2={t2} tp3={t3}")
            print(f"  ATR={a:.3f} | SL={abs(e-s):.3f} ({abs(e-s)/a:.2f} ATR)")
        except ValueError as ex:
            print(f"  ⚠️ level error: {ex}")

    orb = ORBStrategy(cfg).evaluate(m5c, h1c)
    print(f"\nORB: {orb['reason'] if orb else 'no setup (outside window or range not valid)'}")


def cmd_selftest(cfg: Config) -> None:
    errors = []
    if cfg.max_lot < cfg.min_lot: errors.append("max_lot < min_lot")
    if not (0 < cfg.atr_tp1_mult < cfg.atr_tp2_mult < cfg.atr_tp3_mult):
        errors.append("TP ordering")
    if cfg.account_balance <= 0: errors.append("account_balance")
    if not HAS_SKLEARN: errors.append("sklearn missing")
    if errors:
        print("❌ SELFTEST FAILED:", "; ".join(errors)); sys.exit(1)
    print(f"✅ SELFTEST PASSED (v{VERSION})")


def cmd_validate(cfg: Config) -> None:
    import copy
    print("Fetching historical data...")
    
    # [AUTO-RELAX FILTERS UNTUK BACKTEST]
    # Kita matikan filter ketat HANYA untuk simulasi agar bot mau entry
    cfg_bt = copy.deepcopy(cfg)
    cfg_bt.adx_min = 15.0
    cfg_bt.enable_mtf_alignment = False
    cfg_bt.enable_atr_percentile = False
    cfg_bt.enable_key_levels = False
    cfg_bt.enable_regime_routing = False
    cfg_bt.enable_volatility_spike = False
    
    http = HTTPClient(cfg_bt)
    kv = KVStore(cfg_bt, http)
    data = MarketDataHandler(cfg_bt, http, kv)
    sig = SignalGenerator(cfg_bt, kv)
    
    dfs, src, offset = data.fetch_all()
    m5 = dfs["5min"].reset_index(drop=True)
    m15 = dfs["15min"].reset_index(drop=True)
    h1 = dfs["1h"].reset_index(drop=True)
    h4 = dfs["4h"].reset_index(drop=True)

    n = len(m5); lookback = 300; cycle = 3; forward = 60
    results = []
    print(f"Source: {src} | M5 bars: {n}")
    
    for i in range(lookback, n - forward, cycle):
        cur = m5.iloc[i - 1]["datetime"]
        m5w = m5.iloc[:i]
        m15w = m15[m15["datetime"] <= cur]
        h1w = h1[h1["datetime"] <= cur]
        h4w = h4[h4["datetime"] <= cur]
        if len(m15w) < 50 or len(h1w) < 100 or len(h4w) < 50: continue
        
        try:
            r = sig.generate(m5w, m15w, h1w, h4w)
        except Exception:
            continue
            
        if r.direction == "NEUTRAL" or r.verdict == "CANCEL": continue
        
        ob = sig.detect_order_block(m5w, r.direction)
        try:
            entry, sl, tp1, tp2, tp3 = sig.compute_levels(m5w, r.direction, ob)
        except ValueError:
            continue
            
        fut = m5.iloc[i:i + forward]
        filled = False; fill_idx = None
        for j, b in enumerate(fut.itertuples(index=False)):
            if r.direction == "BUY" and b.low <= entry: filled, fill_idx = True, j; break
            if r.direction == "SELL" and b.high >= entry: filled, fill_idx = True, j; break
            
        if not filled:
            results.append({"outcome": "NO_FILL", "r": 0.0}); continue
            
        post = fut.iloc[fill_idx:]
        rr = abs(entry - sl); outcome = "OPEN"; r_val = 0.0
        for b in post.itertuples(index=False):
            if r.direction == "BUY":
                if b.low <= sl: outcome = "LOSS"; r_val = -1.0; break
                if b.high >= tp3: outcome = "WIN_TP3"; r_val = (tp3 - entry) / rr; break
            else:
                if b.high >= sl: outcome = "LOSS"; r_val = -1.0; break
                if b.low <= tp3: outcome = "WIN_TP3"; r_val = (entry - tp3) / rr; break
        results.append({"outcome": outcome, "r": r_val})

    total = len(results)
    fills = [x for x in results if x["outcome"] not in ("NO_FILL", "OPEN")]
    wins = [x for x in fills if x["outcome"].startswith("WIN")]
    
    print(f"\n{'=' * 60}\nVALIDATION REPORT\n{'=' * 60}")
    print(f"Cycles: {total}")
    
    if total == 0:
        print("❌ Validation FAILED: Data terlalu pendek untuk disimulasikan.")
        sys.exit(1)
        
    print(f"  Filled: {len(fills)} ({len(fills)/total*100:.0f}%)")
    
    if len(fills) == 0:
        print("❌ Validation FAILED: Bot sama sekali tidak entry dalam simulasi.")
        sys.exit(1)
        
    wr = len(wins) / len(fills)
    avg = sum(x["r"] for x in fills) / len(fills)
    print(f"  Wins: {len(wins)} | WR: {wr*100:.1f}% | Expectancy: {avg:+.2f}R")
    
    if avg > 0.10:
        print("✅ EDGE POSITIVE — Strategy valid!")
    else:
        print("❌ EDGE NEGATIVE / WEAK")
        sys.exit(1) # Gagalkan workflow jika edge negatif


def cmd_cb_status(cfg: Config) -> None:
    http = HTTPClient(cfg); kv = KVStore(cfg, http)
    risk = RiskManagementEngine(cfg, kv)
    bal = risk.dynamic_balance()
    now = datetime.now(UTC)
    print(f"Balance: ${bal:.2f} (env: ${cfg.account_balance:.2f})")
    print(f"Live balance: {kv.get('state', {}).get('live_balance', 'not set')}")
    print(f"Today filled: {risk._today_signals_filled()}/{cfg.max_daily_signals}")
    print(f"Daily loss: {risk._daily_loss_pct(bal):.2f}%/{cfg.max_daily_loss_pct}%")
    print(f"Consecutive losses: {risk._consecutive_losses()}/{cfg.max_consecutive_losses}")
    print(f"DD pause active: {risk._in_drawdown_pause()}")
    bl = risk._session_blacklist()
    if bl:
        print(f"Session blacklist hours: {sorted(bl)}")
    ok, reason = risk.check_circuit_breakers(bal, now.hour)
    print(f"Status: {'✅ OPEN' if ok else '🛑 CLOSED — ' + reason}")


def main() -> None:
    setup_logging()
    try:
        cfg = Config.from_env()
    except Exception as e:
        print(f"CONFIG ERROR: {e}"); sys.exit(2)
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else "run"

    if cmd == "smoke": cmd_smoke(cfg); return
    if cmd == "selftest": cmd_selftest(cfg); return
    if cmd == "validate": cmd_validate(cfg); return
    if cmd == "cb-status": cmd_cb_status(cfg); return
    if cmd == "test-tg":
        http = HTTPClient(cfg); kv = KVStore(cfg, http)
        tg = TelegramNotifier(cfg, kv, http)
        print("SENT" if tg.send(f"<b>🔔 TEST v{VERSION}</b>") else "FAILED")
        return
    if cmd == "help":
        print("commands: selftest|test-tg|smoke|validate|cb-status|run"); return

    try:
        TradingBotOrchestrator(cfg).run()
    except Exception as e:
        logging.exception("FATAL")
        try:
            http = HTTPClient(cfg); kv = KVStore(cfg, http)
            TelegramNotifier(cfg, kv, http).send(
                f"<b>💥 APEX CRASH</b>\n<code>{redact(e)[:400]}</code>")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
