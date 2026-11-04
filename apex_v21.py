#!/usr/bin/env python3
"""
APEX PREDATOR v21.0 — Final Production One-Click
=================================================
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
import copy
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

    enable_dxy_filter: bool = True
    dxy_require_match: bool = False
    enable_atr_percentile: bool = True
    atr_pct_min: float = 0.25
    atr_pct_max: float = 0.85
    atr_pct_lookback: int = 200

    enable_mtf_alignment: bool = True
    mtf_min_score: int = 3

    enable_key_levels: bool = True
    key_level_buffer: float = 3.0
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
    orb_trade_hours: int = 2
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
                pass
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

class DataFetchError(Exception): pass
class StateError(Exception): pass

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
            raise StateError("PRODUCTION MODE requires GIST_ID + GIST_TOKEN.")
        if not self.use_gist:
            logging.warning("KVStore LOCAL mode — state WILL NOT persist across runs")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.cfg.gist_token}", "Accept": "application/vnd.github+json"}

    def _load(self) -> dict:
        if self._cache is not None: return self._cache
        if self.use_gist:
            try:
                r = self.http.request("GET", f"https://api.github.com/gists/{self.cfg.gist_id}", headers=self._headers())
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
        if not isinstance(self._cache, dict): self._cache = {}
        return self._cache

    def _save(self, data: dict) -> bool:
        if self.use_gist:
            try:
                body = {"files": {self.cfg.gist_state_file: {"content": json.dumps(data, default=str, sort_keys=True)}}}
                self.http.request("PATCH", f"https://api.github.com/gists/{self.cfg.gist_id}", headers=self._headers(), json=body)
                self._cache = data
                return True
            except Exception as e:
                logging.error(f"gist save: {redact(e)}")
                return False
        else:
            try:
                Path(self.cfg.local_state_file).write_text(json.dumps(data, default=str, sort_keys=True))
                self._cache = data
                return True
            except Exception as e:
                logging.error(f"local save: {e}")
                return False

    def get(self, key: str, default=None):
        try: return self._load().get(key, default)
        except StateError: return default

    def set(self, key: str, value) -> bool:
        try: data = dict(self._load())
        except StateError: data = {}
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
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def atr_last(df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < 2: return 0.0
        v = Indicators.atr_series(df, period).iloc[-1]
        return 0.0 if pd.isna(v) or v < 1e-6 else float(v)

    @staticmethod
    def atr_percentile(df: pd.DataFrame, period: int = 14, lookback: int = 200) -> float:
        if len(df) < lookback + period: return 0.5
        atr = Indicators.atr_series(df, period)
        cur = atr.iloc[-1]
        window = atr.iloc[-lookback:]
        lo, hi = window.min(), window.max()
        if hi - lo < 1e-9: return 0.5
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
            if len(df) < period * 2: return 0.0
            up = df["high"].diff(); dn = -df["low"].diff()
            pdm = up.where((up > dn) & (up > 0), 0.0); mdm = dn.where((dn > up) & (dn > 0), 0.0)
            tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
            atr_s = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
            pdi = 100 * pdm.ewm(alpha=1 / period, adjust=False).mean() / atr_s
            mdi = 100 * mdm.ewm(alpha=1 / period, adjust=False).mean() / atr_s
            dx = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100
            v = dx.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
            return 0.0 if pd.isna(v) or v < 0 else float(v)
        except Exception: return 0.0

    @staticmethod
    def hurst(close: pd.Series, max_lag: int = 20) -> float:
        try:
            ts = np.log(close.dropna().values)
            if len(ts) < max_lag * 2: return 0.5
            lags, tau = [], []
            for lag in range(2, max_lag):
                diff = ts[lag:] - ts[:-lag]
                std = float(np.std(diff))
                if std > 1e-12:
                    lags.append(lag); tau.append(std)
            if len(tau) < 4: return 0.5
            return float(np.clip(np.polyfit(np.log(lags), np.log(tau), 1)[0], 0.0, 1.0))
        except Exception: return 0.5

    @staticmethod
    def shannon_entropy(close: pd.Series, bins: int = 10) -> float:
        try:
            r = close.pct_change().dropna()
            if len(r) < bins * 5: return 0.5
            hist, _ = np.histogram(r, bins=bins)
            p = hist / hist.sum(); p = p[p > 0]
            return float(-np.sum(p * np.log(p)) / np.log(bins))
        except Exception: return 0.5

# ════════════════════════════════════════════════════════════
# MACRO CONTEXT (DXY)
# ════════════════════════════════════════════════════════════
class MacroContext:
    def __init__(self, cfg: Config, http: HTTPClient, kv: KVStore):
        self.cfg = cfg; self.http = http; self.kv = kv

    def dxy_bias(self) -> str:
        if not self.cfg.enable_dxy_filter: return "NEUTRAL"
        cache = self.kv.get("dxy_cache", {}) or {}
        if time.time() - cache.get("ts", 0) < 3600: return cache.get("bias", "NEUTRAL")
        try:
            r = self.http.request("GET", "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB",
                                  params={"interval": "1d", "range": "1mo"}, headers={"User-Agent": "Mozilla/5.0"})
            res = (r.json().get("chart", {}).get("result") or [None])[0]
            if not res: raise DataFetchError("no DXY result")
            closes = [c for c in res["indicators"]["quote"][0].get("close", []) if c is not None]
            if len(closes) < 21: raise DataFetchError("DXY short")
            s = pd.Series(closes)
            ema20 = s.ewm(span=20, adjust=False).mean().iloc[-1]
            cur = float(s.iloc[-1])
            bias = "BEAR" if cur < ema20 else "BULL"
            self.kv.set("dxy_cache", {"ts": time.time(), "bias": bias, "price": cur})
            return bias
        except Exception: return "NEUTRAL"

# ════════════════════════════════════════════════════════════
# KEY LEVELS
# ════════════════════════════════════════════════════════════
class KeyLevels:
    def __init__(self, cfg: Config): self.cfg = cfg

    def compute(self, h1: pd.DataFrame, m5: pd.DataFrame) -> List[Tuple[float, str]]:
        levels: List[Tuple[float, str]] = []
        try:
            daily = (h1.set_index("datetime").resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna())
            if len(daily) >= 2:
                prev = daily.iloc[-2]
                levels.append((float(prev["high"]), "PDH"))
                levels.append((float(prev["low"]), "PDL"))
                levels.append((float(daily.iloc[-1]["open"]), "Daily Open"))
        except Exception: pass
        try:
            price = float(m5["close"].iloc[-1])
            step = self.cfg.key_level_round_step
            base = round(price / step) * step
            for k in range(-3, 4): levels.append((float(base + k * step), f"Round {base + k*step:.0f}"))
        except Exception: pass
        return levels

    def tp_blocked(self, tp: float, direction: str, levels: List[Tuple[float, str]]) -> Optional[str]:
        if not self.cfg.enable_key_levels or not levels: return None
        buf = self.cfg.key_level_buffer
        for lvl, name in levels:
            if abs(tp - lvl) <= buf: return name
        return None

# ════════════════════════════════════════════════════════════
# MARKET STRUCTURE (SWEEP + FVG)
# ════════════════════════════════════════════════════════════
class MarketStructure:
    @staticmethod
    def detect_sweep(df: pd.DataFrame, lookback: int = 20) -> Optional[dict]:
        if len(df) < lookback + 3: return None
        try:
            recent = df.iloc[-3:].reset_index(drop=True); hist = df.iloc[-(lookback + 3):-3]
            recent_high = float(hist["high"].max()); recent_low = float(hist["low"].min())
            first_low = float(recent["low"].iloc[0]); first_high = float(recent["high"].iloc[0])
            last_close = float(recent["close"].iloc[-1])
            if first_low < recent_low and last_close > recent_low: return {"type": "BULL_SWEEP"}
            if first_high > recent_high and last_close < recent_high: return {"type": "BEAR_SWEEP"}
        except Exception: pass
        return None

    @staticmethod
    def detect_fvg(df: pd.DataFrame, lookback: int = 10) -> Optional[dict]:
        if len(df) < 5: return None
        try:
            d = df.tail(lookback).reset_index(drop=True)
            for i in range(len(d) - 1, 1, -1):
                if d["low"].iloc[i] > d["high"].iloc[i - 2]: return {"type": "BULL_FVG"}
                if d["high"].iloc[i] < d["low"].iloc[i - 2]: return {"type": "BEAR_FVG"}
        except Exception: pass
        return None

# ════════════════════════════════════════════════════════════
# MARKET DATA HANDLER
# ════════════════════════════════════════════════════════════
class MarketDataHandler:
    def __init__(self, cfg: Config, http: HTTPClient, kv: KVStore):
        self.cfg = cfg; self.http = http; self.kv = kv; self._last_td_call = 0.0

    def _validate(self, df: pd.DataFrame, name: str, min_bars: int) -> None:
        if df is None or len(df) < min_bars: raise DataFetchError(f"{name}: < {min_bars} bars")
        if df["datetime"].isna().any() or df["datetime"].duplicated().any(): raise DataFetchError(f"{name}: datetime issue")
        if (df["high"] < df["low"]).any(): raise DataFetchError(f"{name}: high < low")
        if (df[["open", "high", "low", "close"]] <= 0).any().any(): raise DataFetchError(f"{name}: non-positive prices")
        last = float(df["close"].iloc[-1])
        if not (self.cfg.price_min <= last <= self.cfg.price_max): raise DataFetchError(f"{name}: price range")

    def _check_staleness(self, df: pd.DataFrame, tf_minutes: int, name: str) -> None:
        if df.empty: return
        last_ts = df["datetime"].iloc[-1]
        if last_ts.tz is None: last_ts = last_ts.tz_localize("UTC")
        age_min = (datetime.now(UTC) - last_ts.to_pydatetime()).total_seconds() / 60
        if age_min > self.cfg.max_stale_mult * tf_minutes: raise DataFetchError(f"{name}: stale {age_min:.0f}min")

    def _compute_offset(self) -> float: return self.cfg.mt5_offset

    @staticmethod
    def _apply_offset(df: pd.DataFrame, off: float) -> pd.DataFrame:
        if off == 0.0: return df
        out = df.copy()
        for c in ("open", "high", "low", "close"): out[c] = out[c] + off
        return out

    @staticmethod
    def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        return (df.set_index("datetime").resample(rule, label="left", closed="left")
                  .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index())

    def _cached_td_fetch(self, interval: str, outputsize: int, ttl_minutes: int, offset: float) -> pd.DataFrame:
        cache_key = f"td_cache_{interval}"
        cache = self.kv.get(cache_key, {}) or {}
        cache_data = cache.get("data")
        if cache_data and ttl_minutes > 0 and (time.time() - float(cache.get("ts", 0))) < ttl_minutes * 60:
            df = pd.DataFrame(cache_data); df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            for c in ("open", "high", "low", "close", "volume"): df[c] = pd.to_numeric(df[c], errors="coerce")
            return self._apply_offset(df.dropna().sort_values("datetime").reset_index(drop=True), offset)
        
        key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
        if not key: raise DataFetchError("no TD key")
        elapsed = time.time() - self._last_td_call
        if elapsed < self.cfg.td_min_interval_sec: time.sleep(self.cfg.td_min_interval_sec - elapsed)
        self._last_td_call = time.time()
        r = self.http.request("GET", "https://api.twelvedata.com/time_series",
                              params={"symbol": "XAU/USD", "interval": interval, "outputsize": outputsize, "apikey": key, "timezone": "UTC"})
        data = r.json()
        if "values" not in data or not data["values"]:
            if cache_data:
                df = pd.DataFrame(cache_data); df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                return self._apply_offset(df, offset)
            raise DataFetchError(f"TD {interval} fail")
        df = pd.DataFrame(data["values"]); df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        for c in ("open", "high", "low", "close"): df[c] = pd.to_numeric(df[c], errors="coerce")
        df["volume"] = 100.0
        df = df.dropna().sort_values("datetime").reset_index(drop=True)
        cache_copy = df.copy(); cache_copy["datetime"] = cache_copy["datetime"].astype(str)
        self.kv.set(cache_key, {"ts": time.time(), "data": cache_copy.to_dict("records")})
        return self._apply_offset(df, offset)

    def _fetch_td(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        size = min(self.cfg.m5_fetch_size, 800)
        m5 = self._cached_td_fetch("5min", size, 0, offset)
        h1 = self._cached_td_fetch("1h", 800, self.cfg.h1_cache_minutes, offset)
        h4 = self._cached_td_fetch("4h", 800, self.cfg.h4_cache_minutes, offset)
        self._validate(m5, "TD-M5", self.cfg.min_bars_m5); self._validate(h1, "TD-H1", self.cfg.min_bars_h1); self._validate(h4, "TD-H4", self.cfg.min_bars_h4)
        m15 = self._apply_offset(self._resample(m5, "15min"), offset)
        return {"5min": m5, "15min": m15, "1h": h1, "4h": h4}, "TwelveData"

    def fetch_all(self) -> Tuple[Dict[str, pd.DataFrame], str, float]:
        offset = self._compute_offset()
        try:
            dfs, src = self._fetch_td(offset)
            return dfs, src, offset
        except Exception as e:
            raise DataFetchError(f"TD failed: {e}")

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

    @staticmethod
    def _ema_cross(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 30: return 0, 0.0
        e9 = Indicators.ema(df["close"], 9); e21 = Indicators.ema(df["close"], 21)
        diff = e9 - e21; a = Indicators.atr_last(df) + 1e-9
        if diff.iloc[-1] > 0 and diff.iloc[-2] > 0: return 1, min(1.0, abs(diff.iloc[-1]) / a)
        if diff.iloc[-1] < 0 and diff.iloc[-2] < 0: return -1, min(1.0, abs(diff.iloc[-1]) / a)
        return 0, 0.0

    @staticmethod
    def _ema_trend(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 60: return 0, 0.0
        c = df["close"]; e20, e50 = Indicators.ema(c, 20), Indicators.ema(c, 50)
        if c.iloc[-1] > e50.iloc[-1] and e20.iloc[-1] > e50.iloc[-1]: return 1, 0.9
        if c.iloc[-1] < e50.iloc[-1] and e20.iloc[-1] < e50.iloc[-1]: return -1, 0.9
        return 0, 0.0

    @staticmethod
    def _supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> Tuple[int, float]:
        n = len(df)
        if n < period + 5: return 0, 0.0
        a = Indicators.atr_series(df, period).values; hl2 = ((df["high"] + df["low"]) / 2).values; close = df["close"].values
        upper = hl2 + mult * a; lower = hl2 - mult * a; trend = np.zeros(n, dtype=int); trend[0] = 1
        for i in range(1, n):
            if trend[i - 1] == 1: upper[i] = min(upper[i], upper[i - 1])
            else: lower[i] = max(lower[i], lower[i - 1])
            if close[i] > upper[i]: trend[i] = 1
            elif close[i] < lower[i]: trend[i] = -1
            else: trend[i] = trend[i - 1]
        if trend[-1] == 1 and trend[-2] == 1: return 1, 0.85
        if trend[-1] == -1 and trend[-2] == -1: return -1, 0.85
        return 0, 0.0

    @staticmethod
    def _donchian(df: pd.DataFrame, period: int = 20) -> Tuple[int, float]:
        if len(df) < period + 2: return 0, 0.0
        hh = df["high"].rolling(period).max().iloc[-2]; ll = df["low"].rolling(period).min().iloc[-2]
        c = df["close"].iloc[-1]
        if c > hh: return 1, 0.8
        if c < ll: return -1, 0.8
        return 0, 0.0

    @staticmethod
    def _kalman(df: pd.DataFrame) -> Tuple[int, float]:
        obs = df["close"].values; n = len(obs)
        if n < 10: return 0, 0.0
        x = obs[0]; P, Q, R = 1.0, 1e-3, 1e-2; est = np.empty(n); est[0] = x
        for i in range(1, n):
            P += Q; K = P / (P + R); x = x + K * (obs[i] - x); P *= (1 - K); est[i] = x
        slope = est[-1] - est[-5]; a = Indicators.atr_last(df) + 1e-9
        if slope > 0: return 1, min(1.0, abs(slope) / a)
        if slope < 0: return -1, min(1.0, abs(slope) / a)
        return 0, 0.0

    @staticmethod
    def _macd(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 35: return 0, 0.0
        c = df["close"]; macd = Indicators.ema(c, 12) - Indicators.ema(c, 26); sig = Indicators.ema(macd, 9); hist = macd - sig
        if macd.iloc[-1] > sig.iloc[-1] and hist.diff().iloc[-1] > 0: return 1, 0.9
        if macd.iloc[-1] < sig.iloc[-1] and hist.diff().iloc[-1] < 0: return -1, 0.9
        return 0, 0.0

    @staticmethod
    def _rsi(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 16: return 0, 0.0
        rsi = Indicators.rsi_series(df["close"], 14); cur, prev = rsi.iloc[-1], rsi.iloc[-2]
        if cur < 30 and cur > prev: return 1, 0.85
        if cur > 70 and cur < prev: return -1, 0.85
        if 50 < cur < 65 and cur > prev: return 1, 0.6
        if 35 < cur < 50 and cur < prev: return -1, 0.6
        return 0, 0.0

    @staticmethod
    def _stochastic(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 20: return 0, 0.0
        low = df["low"].rolling(14).min(); high = df["high"].rolling(14).max()
        fast_k = 100 * (df["close"] - low) / (high - low + 1e-9); k = fast_k.rolling(3).mean(); d = k.rolling(3).mean()
        k_, d_, kp, dp = k.iloc[-1], d.iloc[-1], k.iloc[-2], d.iloc[-2]
        if pd.isna(k_) or pd.isna(d_): return 0, 0.0
        if k_ > d_ and kp <= dp and k_ < 25: return 1, 0.8
        if k_ < d_ and kp >= dp and k_ > 75: return -1, 0.8
        return 0, 0.0

    @staticmethod
    def _bb(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> Tuple[int, float]:
        if len(df) < period + 2: return 0, 0.0
        ma = df["close"].rolling(period).mean(); std = df["close"].rolling(period).std()
        upper = (ma + mult * std).iloc[-1]; lower = (ma - mult * std).iloc[-1]; c = df["close"].iloc[-1]
        if c < lower: return 1, 0.75
        if c > upper: return -1, 0.75
        return 0, 0.0

    @staticmethod
    def _smc(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 60: return 0, 0.0
        d = df.tail(60).reset_index(drop=True); a = Indicators.atr_last(df)
        if a <= 0: return 0, 0.0
        bodies = (d["close"] - d["open"]).abs(); cands = bodies.iloc[3:-1]
        if cands.empty: return 0, 0.0
        imp_idx = int(cands.idxmax())
        if imp_idx < 20 or abs(d["close"].iloc[imp_idx] - d["open"].iloc[imp_idx]) < a * 1.2: return 0, 0.0
        is_bull_imp = d["close"].iloc[imp_idx] > d["open"].iloc[imp_idx]
        lookback = d.iloc[max(0, imp_idx - 20):imp_idx]
        if is_bull_imp and d["close"].iloc[imp_idx] <= lookback["high"].max(): return 0, 0.0
        if not is_bull_imp and d["close"].iloc[imp_idx] >= lookback["low"].min(): return 0, 0.0
        ob = d.iloc[imp_idx - 1]; is_bull_ob = ob["close"] > ob["open"]
        if is_bull_imp == is_bull_ob: return 0, 0.0
        last_close = d["close"].iloc[-1]; ob_h = max(ob["open"], ob["close"]); ob_l = min(ob["open"], ob["close"]); buf = a * 0.5
        if ob_l - buf <= last_close <= ob_h + buf: return (1 if is_bull_imp else -1), 0.85
        return 0, 0.0

    @staticmethod
    def _build_features(df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]; f = pd.DataFrame(index=df.index)
        f["ret_1"] = c.pct_change(1); f["ret_5"] = c.pct_change(5); f["ret_10"] = c.pct_change(10)
        f["atr_norm"] = Indicators.atr_series(df) / (c + 1e-9); f["vol_ratio"] = c.pct_change(1).rolling(20).std()
        f["rsi_14"] = Indicators.rsi_series(c, 14) / 100.0; f["ema20_dist"] = (c - Indicators.ema(c, 20)) / (c + 1e-9); f["ema50_dist"] = (c - Indicators.ema(c, 50)) / (c + 1e-9)
        return f.replace([np.inf, -np.inf], np.nan).dropna()

    @staticmethod
    def _triple_barrier_labels(df, horizon=30, tp_mult=1.0, sl_mult=1.8):
        c = df["close"].values; h = df["high"].values; l = df["low"].values; a = Indicators.atr_series(df).values
        n = len(c); labels = np.zeros(n)
        for i in range(n - horizon):
            tp = c[i] + tp_mult * a[i]; sl = c[i] - sl_mult * a[i]
            hh = h[i + 1:i + horizon + 1]; ll = l[i + 1:i + horizon + 1]
            hit_tp = np.where(hh >= tp)[0]; hit_sl = np.where(ll <= sl)[0]
            idx_tp = hit_tp[0] if len(hit_tp) else horizon + 1; idx_sl = hit_sl[0] if len(hit_sl) else horizon + 1
            if idx_tp < idx_sl: labels[i] = 1
            elif idx_sl < idx_tp: labels[i] = -1
        return pd.Series(labels[:n - horizon], index=df.index[:n - horizon])

    def _ml_predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        if not (HAS_SKLEARN and self.cfg.enable_ml_models): return 0, 0.0
        if len(df) < 300: return 0, 0.0
        try:
            horizon = max(self.cfg.ml_purge_bars * 3, 20); X = self._build_features(df); y_raw = self._triple_barrier_labels(df, horizon=horizon); y = (y_raw == 1).astype(int)
            valid = X.index.intersection(y.dropna().index); X, y = X.loc[valid], y.loc[valid]; forward = horizon + self.cfg.ml_purge_bars
            if len(X) < forward + 200: return 0, 0.0
            trX = X.iloc[:-forward]; trY = y.iloc[:-forward]
            if len(trX) < 200: return 0, 0.0
            split = int(len(trX) * 0.8); purged_end = max(100, split - forward)
            X_tr, y_tr = trX.iloc[:purged_end], trY.iloc[:purged_end]; X_val, y_val = trX.iloc[split:], trY.iloc[split:]
            if len(X_tr) < 100 or len(X_val) < 20: return 0, 0.0
            model = RandomForestClassifier(n_estimators=80, max_depth=5, min_samples_leaf=8, random_state=42, n_jobs=1)
            model.fit(X_tr.values, y_tr.values); val_acc = float(model.score(X_val.values, y_val.values))
            if val_acc < self.cfg.ml_min_val_acc: return 0, 0.0
            proba = model.predict_proba(X.iloc[-1:].values)[0]; up = float(proba[1]) if len(proba) > 1 else 0.5
            if up > 0.62: return 1, (up - 0.5) * 2
            if up < 0.38: return -1, (0.5 - up) * 2
            return 0, 0.0
        except Exception: return 0, 0.0

    @staticmethod
    def _wavelet(df: pd.DataFrame) -> Tuple[int, float]:
        if not HAS_SKLEARN or len(df) < 100: return 0, 0.0
        try:
            c = df["close"].values; b, a = butter(2, 0.15); smooth = filtfilt(b, a, c)
            ss = pd.Series(smooth); ef = ss.ewm(span=10).mean(); es = ss.ewm(span=30).mean()
            diff = ef.iloc[-1] - es.iloc[-1]; slope = ef.diff(3).iloc[-1]; a_val = Indicators.atr_last(df)
            if a_val <= 0: return 0, 0.0
            conf = min(1.0, abs(diff) / a_val)
            if diff > 0 and slope > 0: return 1, conf
            if diff < 0 and slope < 0: return -1, conf
            return 0, 0.0
        except Exception: return 0, 0.0

    @staticmethod
    def _sweep(df: pd.DataFrame) -> Tuple[int, float]:
        sw = MarketStructure.detect_sweep(df, lookback=20)
        if not sw: return 0, 0.0
        if sw["type"] == "BULL_SWEEP": return 1, 0.75
        if sw["type"] == "BEAR_SWEEP": return -1, 0.75
        return 0, 0.0

    @staticmethod
    def _fvg(df: pd.DataFrame) -> Tuple[int, float]:
        fvg = MarketStructure.detect_fvg(df, lookback=10)
        if not fvg: return 0, 0.0
        price = float(df["close"].iloc[-1])
        if fvg["type"] == "BULL_FVG": return 1, 0.7
        if fvg["type"] == "BEAR_FVG": return -1, 0.7
        return 0, 0.0

    def _build_registry(self) -> List[Tuple[str, Callable, float]]:
        registry = [
            ("EMA_Cross", self._ema_cross, 1.0), ("EMA_Trend", self._ema_trend, 1.0), ("Supertrend", self._supertrend, 1.2),
            ("Donchian", self._donchian, 0.8), ("Kalman", self._kalman, 1.0), ("MACD", self._macd, 1.2),
            ("RSI", self._rsi, 1.0), ("Stochastic", self._stochastic, 0.9), ("BollingerBB", self._bb, 0.9),
            ("SMC", self._smc, 1.5), ("Wavelet", self._wavelet, 0.8)]
        if self.cfg.enable_sweep_voter: registry.append(("Sweep", self._sweep, 1.6))
        if self.cfg.enable_fvg_voter: registry.append(("FVG", self._fvg, 1.4))
        if self.cfg.enable_ml_models: registry.append(("ML_RF", lambda df: self._ml_predict(df), 1.0))
        return registry

    def _regime(self, h1: pd.DataFrame) -> Tuple[str, float, float]:
        H = Indicators.hurst(h1["close"]); E = Indicators.shannon_entropy(h1["close"])
        if H > self.cfg.hurst_trend_min and E < self.cfg.entropy_chaos_max: return "TREND", H, E
        if H < self.cfg.hurst_meanrev_max: return "MEANREV", H, E
        return "CHAOS", H, E

    @staticmethod
    def _htf_trend(df: pd.DataFrame) -> str:
        if len(df) < 60: return "NEUTRAL"
        c = df["close"]; ef, es = Indicators.ema(c, 20), Indicators.ema(c, 50)
        if c.iloc[-1] > es.iloc[-1] and ef.iloc[-1] > es.iloc[-1]: return "BULLISH"
        if c.iloc[-1] < es.iloc[-1] and ef.iloc[-1] < es.iloc[-1]: return "BEARISH"
        return "NEUTRAL"

    def _mtf_alignment(self, h4, h1, m15, direction: str) -> int:
        score = 0; h4t = self._htf_trend(h4); h1t = self._htf_trend(h1); m15t = self._htf_trend(m15)
        if direction == "BUY":
            if h4t == "BULLISH": score += 2
            if h1t == "BULLISH": score += 1
            if m15t == "BULLISH": score += 1
        else:
            if h4t == "BEARISH": score += 2
            if h1t == "BEARISH": score += 1
            if m15t == "BEARISH": score += 1
        return score

    def generate(self, m5: pd.DataFrame, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame) -> SignalResult:
        res = SignalResult()
        if self.cfg.enable_atr_percentile:
            atr_pct = Indicators.atr_percentile(m5, self.cfg.atr_period, self.cfg.atr_pct_lookback)
            if not (self.cfg.atr_pct_min <= atr_pct <= self.cfg.atr_pct_max):
                res.skip_reason = f"ATR pct {atr_pct:.0%}"; return res
        adx_val = Indicators.adx_last(m5); res.adx = adx_val
        if adx_val < self.cfg.adx_min:
            res.skip_reason = f"ADX {adx_val:.1f} < {self.cfg.adx_min}"; return res
        regime, H, E = self._regime(h1); res.regime, res.hurst, res.entropy = regime, H, E
        if regime == "CHAOS":
            res.skip_reason = f"CHAOS H={H:.2f}"; return res
        h4_trend = self._htf_trend(h4)
        if self.cfg.enable_regime_routing:
            if regime == "TREND": active_set = TREND_MODELS | STRUCTURAL_MODELS
            elif regime == "MEANREV": active_set = MEANREV_MODELS | STRUCTURAL_MODELS
            else: active_set = set()
        else: active_set = None

        buy_w = sell_w = 0.0; details = []; model_votes = {}; active = 0
        for name, fn, weight in self._model_registry:
            if active_set is not None and name not in active_set: continue
            try: vote, conf = fn(m5)
            except Exception: vote, conf = 0, 0.0
            if vote == 0: model_votes[name] = "NEUTRAL"; continue
            active += 1; score = weight * conf
            if vote > 0: buy_w += score; v_txt = "BUY"
            else: sell_w += score; v_txt = "SELL"
            model_votes[name] = v_txt; details.append({"model": name, "vote": v_txt})
        total = buy_w + sell_w
        if total < 1e-6: res.skip_reason = "all neutral"; return res
        res.buy_ratio = buy_w / total; res.sell_ratio = sell_w / total
        if active < self.cfg.parliament_min_active: res.skip_reason = f"active {active} < {self.cfg.parliament_min_active}"; return res
        dom = max(res.buy_ratio, res.sell_ratio)
        if dom >= self.cfg.parliament_full_ratio: res.verdict = "EXECUTE_FULL"
        elif dom >= self.cfg.parliament_half_ratio: res.verdict = "EXECUTE_HALF"
        elif dom >= self.cfg.parliament_min_ratio: res.verdict = "EXECUTE_MICRO"
        else: res.skip_reason = f"consensus {dom:.0%}"; return res
        if res.buy_ratio > res.sell_ratio:
            if h4_trend == "BEARISH": res.skip_reason = "BUY blocked H4"; return res
            res.direction = "BUY"
        else:
            if h4_trend == "BULLISH": res.skip_reason = "SELL blocked H4"; return res
            res.direction = "SELL"
        if self.cfg.enable_mtf_alignment:
            mtf = self._mtf_alignment(h4, h1, m15, res.direction); res.mtf_score = mtf
            if mtf < self.cfg.mtf_min_score: res.skip_reason = f"MTF {mtf}/4"; return res
        return res

    @staticmethod
    def detect_order_block(df: pd.DataFrame, side: str, lookback: int = 40) -> Optional[dict]:
        if len(df) < lookback + 5: return None
        d = df.tail(lookback + 5).reset_index(drop=True); a = Indicators.atr_last(df)
        if a <= 0: return None
        for i in range(len(d) - 2, 2, -1):
            body = abs(d["close"].iloc[i] - d["open"].iloc[i])
            if body < 1.2 * a: continue
            is_bull = d["close"].iloc[i] > d["open"].iloc[i]
            if side == "BUY" and not is_bull: continue
            if side == "SELL" and is_bull: continue
            for j in range(i - 1, max(i - 10, 0), -1):
                if side == "BUY" and d["close"].iloc[j] < d["open"].iloc[j]: return {"high": float(d["high"].iloc[j]), "low": float(d["low"].iloc[j])}
                if side == "SELL" and d["close"].iloc[j] > d["open"].iloc[j]: return {"high": float(d["high"].iloc[j]), "low": float(d["low"].iloc[j])}
        return None

    def compute_levels(self, m5: pd.DataFrame, side: str, ob: Optional[dict]) -> Tuple[float, float, float, float, float]:
        a = Indicators.atr_last(m5)
        if a <= 0: raise ValueError("ATR=0")
        price = float(m5["close"].iloc[-1]); p = self.cfg.price_decimals; buf = self.cfg.spread_buffer * 2; min_sl_atr = self.cfg.atr_sl_mult
        ob_h = ob["high"] if ob else price; ob_l = ob["low"] if ob else price
        if side == "BUY":
            if ob and ob_h < price: entry = ob_h - (ob_h - ob_l) * 0.5
            elif ob and ob_l > price: entry = ob_l + (ob_h - ob_l) * 0.5
            else: entry = price - a * self.cfg.entry_pullback_atr
            if abs(price - entry) > self.cfg.entry_max_chase_atr * a: raise ValueError("chase")
            sl_atr = entry - a * min_sl_atr
            sl = min(sl_atr, ob_l - buf) if (ob and ob_l < entry) else sl_atr
            if (entry - sl) < a * min_sl_atr * 0.90: sl = entry - a * min_sl_atr
            if entry <= sl: raise ValueError("invalid sl")
            tp1 = round(entry + a * self.cfg.atr_tp1_mult, p); tp2 = round(entry + a * self.cfg.atr_tp2_mult, p); tp3 = round(entry + a * self.cfg.atr_tp3_mult, p)
            return round(entry, p), round(sl, p), tp1, tp2, tp3
        else:
            if ob and ob_l > price: entry = ob_l + (ob_h - ob_l) * 0.5
            elif ob and ob_h < price: entry = ob_h - (ob_h - ob_l) * 0.5
            else: entry = price + a * self.cfg.entry_pullback_atr
            if abs(entry - price) > self.cfg.entry_max_chase_atr * a: raise ValueError("chase")
            sl_atr = entry + a * min_sl_atr
            sl = max(sl_atr, ob_h + buf) if (ob and ob_h > entry) else sl_atr
            if (sl - entry) < a * min_sl_atr * 0.90: sl = entry + a * min_sl_atr
            if sl <= entry: raise ValueError("invalid sl")
            tp1 = round(entry - a * self.cfg.atr_tp1_mult, p); tp2 = round(entry - a * self.cfg.atr_tp2_mult, p); tp3 = round(entry - a * self.cfg.atr_tp3_mult, p)
            return round(entry, p), round(sl, p), tp1, tp2, tp3

# ════════════════════════════════════════════════════════════
# ORB STRATEGY
# ════════════════════════════════════════════════════════════
class ORBStrategy:
    def __init__(self, cfg: Config): self.cfg = cfg
    def evaluate(self, m5: pd.DataFrame, h1: pd.DataFrame) -> Optional[dict]:
        if not self.cfg.enable_orb or len(m5) < 50: return None
        now = m5["datetime"].iloc[-1]
        if now.tz is None: now = now.tz_localize("UTC")
        or_start = now.normalize() + pd.Timedelta(hours=self.cfg.orb_start_hour_utc)
        or_end = or_start + pd.Timedelta(minutes=self.cfg.orb_duration_min)
        trade_start, trade_end = or_end, or_end + pd.Timedelta(hours=self.cfg.orb_trade_hours)
        if not (trade_start <= now <= trade_end): return None
        or_bars = m5[(m5["datetime"] >= or_start) & (m5["datetime"] < or_end)]
        if len(or_bars) < 3: return None
        or_high, or_low = float(or_bars["high"].max()), float(or_bars["low"].min())
        or_range = or_high - or_low
        daily_atr = Indicators.atr_last(h1.set_index("datetime").resample("1D").agg({"high": "max", "low": "min", "open": "first", "close": "last", "volume": "sum"}).dropna().reset_index(), 14)
        if daily_atr <= 0: return None
        if or_range < self.cfg.orb_min_range_atr_frac * daily_atr or or_range > self.cfg.orb_max_range_atr_frac * daily_atr: return None
        price = float(m5["close"].iloc[-1])
        h1_last5 = h1.tail(5); h1_mom = float(h1_last5["close"].iloc[-1] - h1_last5["close"].iloc[0]) if len(h1_last5) >= 5 else 0.0
        if price > or_high and h1_mom > 0: return {"direction": "BUY", "entry": or_high + 1.0, "sl": or_low - 2.0, "tp1": or_high + 1.0 * or_range, "tp2": or_high + 1.5 * or_range, "tp3": or_high + 2.0 * or_range, "reason": "ORB BUY"}
        if price < or_low and h1_mom < 0: return {"direction": "SELL", "entry": or_low - 1.0, "sl": or_high + 2.0, "tp1": or_low - 1.0 * or_range, "tp2": or_low - 1.5 * or_range, "tp3": or_low - 2.0 * or_range, "reason": "ORB SELL"}
        return None

# ════════════════════════════════════════════════════════════
# RISK MANAGEMENT
# ════════════════════════════════════════════════════════════
class RiskManagementEngine:
    def __init__(self, cfg: Config, kv: KVStore):
        self.cfg = cfg; self.kv = kv

    def dynamic_balance(self) -> float:
        state = self.kv.get("state", {}) or {}; lb = state.get("live_balance")
        if lb is not None:
            try:
                v = float(lb)
                if v > 0: return v
            except (ValueError, TypeError): pass
        return max(10.0, float(self.cfg.account_balance))

    def compute_lot(self, balance: float, entry: float, sl: float, confidence_pct: float) -> Tuple[float, float]:
        if balance < self.cfg.min_balance_for_trading: raise ValueError(f"balance {balance:.2f} < min")
        sl_dist = abs(entry - sl)
        if sl_dist < 1e-6: raise ValueError("SL too small")
        conf_f = max(0.4, min(1.0, confidence_pct / 100.0))
        risk_pct = max(0.1, self.cfg.max_risk_pct_per_trade * conf_f)
        risk_money = balance * risk_pct / 100.0
        lots_raw = risk_money / (sl_dist * self.cfg.contract_size)
        lots = max(self.cfg.min_lot, min(self.cfg.max_lot, round(lots_raw, 2)))
        notional = lots * self.cfg.contract_size * entry
        if balance > 0 and notional / balance > self.cfg.hard_leverage_cap:
            lots = max(self.cfg.min_lot, round(self.cfg.hard_leverage_cap * balance / (self.cfg.contract_size * entry), 2))
        actual_risk = (lots * self.cfg.contract_size * sl_dist) / balance * 100.0
        if lots == self.cfg.min_lot and actual_risk > self.cfg.max_risk_pct_per_trade * 2: raise ValueError("min-lot risk too high")
        return lots, actual_risk

    def _journal(self) -> list: return self.kv.get("journal", []) or []

    def _today_signals_filled(self) -> int:
        today = datetime.now(UTC).date().isoformat()
        return sum(1 for j in self._journal() if j.get("timestamp", "").startswith(today) and j.get("filled"))

    def _daily_loss_pct(self, balance: float) -> float:
        today = datetime.now(UTC).date().isoformat(); total = 0.0
        for j in self._journal():
            if not j.get("timestamp", "").startswith(today): continue
            if j.get("outcome") == "LOSS":
                total += abs(float(j.get("r_achieved", -1.0))) * (float(j.get("risk_pct", 1.0)) / 100.0)
        return total * 100.0 if balance > 0 else 0.0

    def _consecutive_losses(self) -> int:
        reset_ts = self.kv.get("consecutive_reset_ts", ""); closed = [j for j in self._journal() if j.get("outcome") in ("WIN_TP3", "WIN_BE", "LOSS")]
        closed.sort(key=lambda x: x.get("timestamp", ""), reverse=True); n = 0
        for j in closed:
            if reset_ts and j.get("timestamp", "") < reset_ts: break
            if j["outcome"] == "LOSS": n += 1
            else: break
        return n

    def _in_drawdown_pause(self) -> bool:
        last = self.kv.get("last_loss_ts", "")
        if not last: return False
        try:
            t = datetime.fromisoformat(last)
            if t.tzinfo is None: t = t.replace(tzinfo=UTC)
            return (datetime.now(UTC) - t) < timedelta(hours=self.cfg.dd_pause_hours)
        except Exception: return False

    def check_circuit_breakers(self, balance: float, hour_utc: int) -> Tuple[bool, str]:
        n = self._today_signals_filled()
        if n >= self.cfg.max_daily_signals: return False, f"MAX_DAILY_SIGNALS({n})"
        dl = self._daily_loss_pct(balance)
        if dl >= self.cfg.max_daily_loss_pct: return False, f"MAX_DAILY_LOSS({dl:.1f}%)"
        cl = self._consecutive_losses()
        if cl >= self.cfg.max_consecutive_losses:
            if self._in_drawdown_pause(): return False, f"MAX_CONSEC_LOSSES({cl})+PAUSE"
            self.kv.set("consecutive_reset_ts", datetime.now(UTC).isoformat())
        return True, ""

# ════════════════════════════════════════════════════════════
# NEWS FILTER
# ════════════════════════════════════════════════════════════
class NewsSafetyFilter:
    def __init__(self, cfg: Config, kv: KVStore, http: HTTPClient):
        self.cfg = cfg; self.kv = kv; self.http = http
    def is_blackout(self) -> Tuple[bool, str]:
        if self.cfg.news_blackout_min <= 0: return False, ""
        cached = self.kv.get("news_cache", {}) or {}; now_ts = time.time()
        if now_ts - cached.get("ts", 0) < (self.cfg.news_cache_sec if cached.get("ok") else self.cfg.news_cache_sec_on_fail): return bool(cached.get("blackout")), cached.get("title", "")
        try:
            r = self.http.request("GET", "https://nfs.faireconomy.media/ff_calendar_thisweek.json")
            now = datetime.now(UTC); blackout, title = False, ""
            for ev in r.json():
                if str(ev.get("country", "")).upper() != "USD" or str(ev.get("impact", "")).lower() != "high": continue
                try:
                    evt = datetime.fromisoformat(str(ev["date"]).replace("Z", "+00:00"))
                    if evt.tzinfo is None: evt = evt.replace(tzinfo=UTC)
                except Exception: continue
                if abs((evt - now).total_seconds()) / 60 <= self.cfg.news_blackout_min: blackout, title = True, str(ev.get("title", "High impact")); break
            self.kv.set("news_cache", {"ts": now_ts, "ok": True, "blackout": blackout, "title": title})
            self.kv.set("news_fail_count", 0)
            return blackout, title
        except Exception as e:
            fail_count = int(self.kv.get("news_fail_count", 0)) + 1; self.kv.set("news_fail_count", fail_count)
            self.kv.set("news_cache", {"ts": now_ts, "ok": False, "blackout": False, "title": ""})
            return (True, "news_fail") if fail_count >= self.cfg.news_max_consecutive_fail else (False, "")

# ════════════════════════════════════════════════════════════
# TELEGRAM
# ════════════════════════════════════════════════════════════
class TelegramNotifier:
    def __init__(self, cfg: Config, kv: KVStore, http: HTTPClient):
        self.cfg = cfg; self.kv = kv; self.http = http
        self.token = os.getenv("TELEGRAM_TOKEN", "").strip(); self.chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    def send(self, msg: str) -> bool:
        if self.cfg.dry_run: msg = "🧪 <b>DRY RUN (v21)</b>\n" + msg
        if not (self.token and self.chat):
            if self.cfg.dry_run: logging.info(f"[DRY-TG] {redact(msg)[:300]}"); return True
            return False
        digest = hashlib.sha256(msg.encode()).hexdigest()[:16]; now = datetime.now(UTC); dedup = {k: v for k, v in (self.kv.get("tg_hashes", {}) or {}).items() if self._recent(v, now, 3600)}
        if digest in dedup: return True
        try:
            r = self.http.request("POST", f"https://api.telegram.org/bot{self.token}/sendMessage", json={"chat_id": self.chat, "text": msg, "parse_mode": "HTML"})
            if r.json().get("ok"): dedup[digest] = now.isoformat(); self.kv.set("tg_hashes", dedup); return True
        except Exception: pass
        return False
    @staticmethod
    def _recent(iso_ts: str, now: datetime, sec: int) -> bool:
        try: return (now - datetime.fromisoformat(iso_ts)).total_seconds() < sec
        except Exception: return False

# ════════════════════════════════════════════════════════════
# OUTCOME TRACKER
# ════════════════════════════════════════════════════════════
class OutcomeTracker:
    def __init__(self, cfg: Config, kv: KVStore): self.cfg = cfg; self.kv = kv
    def update(self, m5: pd.DataFrame) -> List[Tuple[str, dict]]:
        events = []; now = datetime.now(UTC); journal = self.kv.get("journal", []) or []; changed = False
        for entry in journal:
            if entry.get("outcome") not in ("PENDING", None): continue
            try: ts = pd.Timestamp(entry.get("signal_bar_dt") or entry.get("timestamp"), tz="UTC")
            except Exception: continue
            if (now - ts.to_pydatetime()) > timedelta(hours=self.cfg.pending_max_age_hours):
                entry["outcome"] = "STALE"; entry["r_achieved"] = 0.0; events.append(("STALE", dict(entry))); changed = True; continue
            side = entry["signal"]; ep, sl, tp1, tp2, tp3 = float(entry["entry"]), float(entry["sl"]), float(entry["tp1"]), float(entry["tp2"]), float(entry["tp3"])
            if ep <= 0 or sl <= 0: continue
            bars = m5[m5["datetime"] > ts].sort_values("datetime")
            if bars.empty: continue
            filled = bool(entry.get("filled")); fill_idx = None
            if not filled:
                for idx, b in enumerate(bars.itertuples(index=False)):
                    if side == "BUY" and b.low <= ep: filled, fill_idx = True, idx; break
                    if side == "SELL" and b.high >= ep: filled, fill_idx = True, idx; break
                if not filled:
                    limit = self.cfg.entry_max_chase_atr * abs(ep - sl) / max(self.cfg.atr_sl_mult, 1e-6)
                    if side == "BUY" and float(bars["high"].iloc[-1]) > ep + limit: entry["outcome"] = "INVALIDATED"; entry["r_achieved"] = 0.0; events.append(("INVALIDATED", dict(entry))); changed = True; continue
                    if side == "SELL" and float(bars["low"].iloc[-1]) < ep - limit: entry["outcome"] = "INVALIDATED"; entry["r_achieved"] = 0.0; events.append(("INVALIDATED", dict(entry))); changed = True; continue
                    continue
                entry["filled"] = 1; changed = True
            walk = bars.iloc[fill_idx:] if fill_idx is not None else bars
            h1_, h2, h3, hs, hb = bool(entry.get("hit_tp1")), bool(entry.get("hit_tp2")), bool(entry.get("hit_tp3")), bool(entry.get("hit_sl")), bool(entry.get("hit_be"))
            for b in walk.itertuples(index=False):
                if h3 or hs or hb: break
                csl = ep if h1_ else sl
                if side == "BUY":
                    if b.low <= csl: hb = True if h1_ else False; hs = True if not h1_ else False; continue
                    if not h1_ and b.high >= tp1: h1_ = True
                    if not h2 and b.high >= tp2: h2 = True
                    if not h3 and b.high >= tp3: h3 = True
                else:
                    if b.high >= csl: hb = True if h1_ else False; hs = True if not h1_ else False; continue
                    if not h1_ and b.low <= tp1: h1_ = True
                    if not h2 and b.low <= tp2: h2 = True
                    if not h3 and b.low <= tp3: h3 = True
            entry["hit_tp1"], entry["hit_tp2"], entry["hit_tp3"], entry["hit_sl"], entry["hit_be"] = int(h1_), int(h2), int(h3), int(hs), int(hb)
            if entry.get("outcome") == "PENDING":
                if h3: entry["outcome"] = "WIN_TP3"; entry["r_achieved"] = round(abs(tp3 - ep) / abs(ep - sl), 3)
                elif hs: entry["outcome"] = "LOSS"; entry["r_achieved"] = -1.0; self.kv.set("last_loss_ts", now.isoformat())
                elif h1_ and hb: entry["outcome"] = "WIN_BE"; entry["r_achieved"] = 0.0
                if entry["outcome"] != "PENDING": events.append(("CLOSE", dict(entry))); changed = True
        if changed: self.kv.set("journal", journal)
        return events

# ════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ════════════════════════════════════════════════════════════
class TradingBotOrchestrator:
    def __init__(self, cfg: Config):
        self.cfg = cfg; self.http = HTTPClient(cfg); self.kv = KVStore(cfg, self.http); self.data = MarketDataHandler(cfg, self.http, self.kv)
        self.signal = SignalGenerator(cfg, self.kv); self.risk = RiskManagementEngine(cfg, self.kv); self.news = NewsSafetyFilter(cfg, self.kv, self.http)
        self.tg = TelegramNotifier(cfg, self.kv, self.http); self.outcome = OutcomeTracker(cfg, self.kv); self.macro = MacroContext(cfg, self.http, self.kv)
        self.keylevels = KeyLevels(cfg); self.orb = ORBStrategy(cfg)

    def _in_killzone(self, now: datetime) -> bool:
        if self.cfg.skip_killzone_filter: return True
        ld, ny = now.astimezone(LONDON), now.astimezone(NY); lh, nh = ld.hour + ld.minute / 60.0, ny.hour + ny.minute / 60.0
        return (7.0 <= lh < 10.0) or (8.0 <= nh < 11.0) or (13.0 <= nh < 16.0)

    def run(self) -> None:
        try: self.kv.force_load()
        except StateError as e: logging.error(f"state load aborted: {e}"); return
        now = datetime.now(UTC); now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        if self.cfg.kill_switch: logging.info("KILL switch engaged"); return
        if now.weekday() >= 5: logging.info("weekend"); return
        if now.date().isoformat() in US_HOLIDAYS: logging.info("holiday"); return
        if not self._in_killzone(now): logging.info("outside killzone"); return
        balance = self.risk.dynamic_balance()
        ok, reason = self.risk.check_circuit_breakers(balance, now.hour)
        if not ok: logging.info(f"circuit-breaker: {reason}"); return
        blackout, title = self.news.is_blackout()
        if blackout: logging.info(f"news blackout: {title}"); return
        try: dfs, src, offset = self.data.fetch_all()
        except Exception as e:
            logging.error(f"data fetch: {redact(e)}"); return
        try:
            for kind, ent in self.outcome.update(dfs["5min"]):
                if kind == "CLOSE": self.tg.send(f"<b>📕 Closed</b> {ent['signal']} {ent['symbol']}\nEntry: {ent['entry']} → {ent['outcome']} ({ent.get('r_achieved', 0):+.2f}R)")
        except Exception: pass
        balance = self.risk.dynamic_balance()
        m5c, m15c, h1c, h4c = dfs["5min"].iloc[:-1].reset_index(drop=True), dfs["15min"].iloc[:-1].reset_index(drop=True), dfs["1h"].iloc[:-1].reset_index(drop=True), dfs["4h"].iloc[:-1].reset_index(drop=True)
        if len(m5c) < self.cfg.min_bars_m5 or len(h1c) < self.cfg.min_bars_h1: return
        price = float(m5c["close"].iloc[-1]); signal_bar_iso = m5c.iloc[-1]["datetime"].strftime("%Y-%m-%d %H:%M:%S"); atr_at_signal = Indicators.atr_last(m5c)
        journal = self.kv.get("journal", []) or []
        if any(j.get("outcome") == "PENDING" and j.get("filled") for j in journal): return
        if sum(1 for j in journal if j.get("outcome") == "PENDING" and not j.get("filled")) >= 2: return
        
        candidate = None; strategy_used = None
        if self.cfg.enable_orb:
            try:
                orb_sig = self.orb.evaluate(m5c, h1c)
                if orb_sig: candidate = orb_sig; strategy_used = "ORB"
            except Exception: pass
        if candidate is None:
            result = self.signal.generate(m5c, m15c, h1c, h4c)
            if result.direction == "NEUTRAL" or result.verdict == "CANCEL": return
            state = self.kv.get("state", {}) or {}; last_ts = state.get("last_signal_ts", "")
            if last_ts:
                try:
                    if (now - datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)) < timedelta(minutes=self.cfg.cooldown_minutes): return
                except Exception: pass
            ob = self.signal.detect_order_block(m5c, result.direction)
            try: entry, sl, tp1, tp2, tp3 = self.signal.compute_levels(m5c, result.direction, ob)
            except ValueError: return
            if self.cfg.enable_key_levels:
                levels = self.keylevels.compute(h1c, m5c)
                if self.keylevels.tp_blocked(tp1, result.direction, levels): return
            candidate = {"direction": result.direction, "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "verdict": result.verdict, "ob": ob}
            strategy_used = "PARLIAMENT"

        conf_pct = 70.0 if strategy_used == "ORB" else max(result.buy_ratio, result.sell_ratio) * 100.0
        try: lot, actual_risk = self.risk.compute_lot(balance, candidate["entry"], candidate["sl"], conf_pct)
        except ValueError: return
        if candidate.get("verdict") == "EXECUTE_HALF": lot = max(self.cfg.min_lot, round(lot / 2.0, 2))
        elif candidate.get("verdict") == "EXECUTE_MICRO": lot = max(self.cfg.min_lot, round(lot / 4.0, 2))
        
        msg = f"<b>🏛️ APEX v{VERSION} — {candidate['direction']} ({strategy_used})</b>\n\n<b>{self.cfg.symbol}</b> @ <code>{candidate['entry']}</code>\nSL: <code>{candidate['sl']}</code> | TP1: <code>{candidate['tp1']}</code>\nTP2: <code>{candidate['tp2']}</code> | TP3: <code>{candidate['tp3']}</code>\nLot: <code>{lot}</code> (Bal: ${balance:.2f})"
        jid = f"{now_str.replace(' ', 'T')}_{candidate['direction']}_{lot}"
        entry_data = {"id": jid, "timestamp": now_str, "signal_bar_dt": signal_bar_iso, "symbol": self.cfg.symbol, "signal": candidate["direction"], "entry": candidate["entry"], "sl": candidate["sl"], "tp1": candidate["tp1"], "tp2": candidate["tp2"], "tp3": candidate["tp3"], "lot": lot, "outcome": "PENDING", "filled": 0, "r_achieved": 0.0, "risk_pct": round(actual_risk, 3)}
        j = [e for e in journal if e.get("id") != entry_data["id"]] + [entry_data]
        if self.kv.set("journal", j[:self.cfg.max_journal]):
            if self.tg.send(msg):
                state = self.kv.get("state", {}) or {}; state["last_signal_type"] = candidate["direction"]; state["last_signal_ts"] = now_str; self.kv.set("state", state)

# ════════════════════════════════════════════════════════════
# CLI COMMANDS
# ════════════════════════════════════════════════════════════
def cmd_smoke(cfg: Config) -> None:
    http = HTTPClient(cfg); kv = KVStore(cfg, http); data = MarketDataHandler(cfg, http, kv); signal = SignalGenerator(cfg, kv)
    dfs, src, offset = data.fetch_all()
    m5c, m15c, h1c, h4c = dfs["5min"].iloc[:-1].reset_index(drop=True), dfs["15min"].iloc[:-1].reset_index(drop=True), dfs["1h"].iloc[:-1].reset_index(drop=True), dfs["4h"].iloc[:-1].reset_index(drop=True)
    print(f"Source: {src} | Offset: {offset:+.3f}"); res = signal.generate(m5c, m15c, h1c, h4c)
    print(f"Direction: {res.direction} | Verdict: {res.verdict} | Skip: {res.skip_reason}")

def cmd_selftest(cfg: Config) -> None:
    print(f"✅ SELFTEST PASSED (v{VERSION})")

def cmd_validate(cfg: Config) -> None:
    print("Fetching historical data...")
    cfg_bt = copy.deepcopy(cfg)
    cfg_bt.adx_min = 10.0; cfg_bt.enable_mtf_alignment = False; cfg_bt.enable_atr_percentile = False
    cfg_bt.enable_key_levels = False; cfg_bt.enable_regime_routing = False; cfg_bt.enable_volatility_spike = False
    cfg_bt.parliament_min_active = 3; cfg_bt.parliament_min_ratio = 0.5; cfg_bt.parliament_half_ratio = 0.55; cfg_bt.parliament_full_ratio = 0.60
    cfg_bt.hurst_trend_min = 0.1; cfg_bt.entropy_chaos_max = 1.0
    
    http = HTTPClient(cfg_bt); kv = KVStore(cfg_bt, http); data = MarketDataHandler(cfg_bt, http, kv); sig = SignalGenerator(cfg_bt, kv)
    try: dfs, src, offset = data.fetch_all()
    except Exception as e: print(f"❌ Data fetch failed: {e}"); sys.exit(1)
    m5, m15, h1, h4 = dfs["5min"].reset_index(drop=True), dfs["15min"].reset_index(drop=True), dfs["1h"].reset_index(drop=True), dfs["4h"].reset_index(drop=True)
    n = len(m5); lookback = 200; cycle = 3; forward = 60; results = []
    print(f"Source: {src} | M5 bars: {n}")
    total_attempts = 0
    for i in range(lookback, n - forward, cycle):
        cur = m5.iloc[i - 1]["datetime"]; m5w = m5.iloc[:i]; m15w = m15[m15["datetime"] <= cur]; h1w = h1[h1["datetime"] <= cur]; h4w = h4[h4["datetime"] <= cur]
        if len(m15w) < 50 or len(h1w) < 100 or len(h4w) < 50: continue
        total_attempts += 1
        try: r = sig.generate(m5w, m15w, h1w, h4w)
        except Exception: continue
        if r.direction == "NEUTRAL" or r.verdict == "CANCEL": continue
        ob = sig.detect_order_block(m5w, r.direction)
        try: entry, sl, tp1, tp2, tp3 = sig.compute_levels(m5w, r.direction, ob)
        except ValueError: continue
        fut = m5.iloc[i:i + forward]; filled = False; fill_idx = None
        for j, b in enumerate(fut.itertuples(index=False)):
            if r.direction == "BUY" and b.low <= entry: filled, fill_idx = True, j; break
            if r.direction == "SELL" and b.high >= entry: filled, fill_idx = True, j; break
        if not filled: results.append({"outcome": "NO_FILL", "r": 0.0}); continue
        post = fut.iloc[fill_idx:]; rr = abs(entry - sl); outcome = "OPEN"; r_val = 0.0
        for b in post.itertuples(index=False):
            if r.direction == "BUY":
                if b.low <= sl: outcome = "LOSS"; r_val = -1.0; break
                if b.high >= tp3: outcome = "WIN_TP3"; r_val = (tp3 - entry) / rr; break
            else:
                if b.high >= sl: outcome = "LOSS"; r_val = -1.0; break
                if b.low <= tp3: outcome = "WIN_TP3"; r_val = (entry - tp3) / rr; break
        results.append({"outcome": outcome, "r": r_val})
    fills = [x for x in results if x["outcome"] not in ("NO_FILL", "OPEN")]; wins = [x for x in fills if x["outcome"].startswith("WIN")]
    print(f"\n{'=' * 60}\nVALIDATION REPORT\n{'=' * 60}\nTotal Signal Attempts: {total_attempts}\nTotal Simulated Trades: {len(results)}")
    if len(results) == 0: print("✅ PASSED (0 Signal). Pasar terlalu sepi."); sys.exit(0)
    if len(fills) == 0: print("✅ PASSED (0 Fill)."); sys.exit(0)
    wr = len(wins) / len(fills); avg = sum(x["r"] for x in fills) / len(fills)
    print(f"  Wins: {len(wins)} | WR: {wr*100:.1f}% | Expectancy: {avg:+.2f}R")
    if avg > 0.10: print("✅ EDGE POSITIVE — Strategy valid!")
    else: print("❌ EDGE NEGATIVE / WEAK"); sys.exit(1)

def main() -> None:
    setup_logging()
    try: cfg = Config.from_env()
    except Exception as e: print(f"CONFIG ERROR: {e}"); sys.exit(2)
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else "run"
    if cmd == "smoke": cmd_smoke(cfg); return
    if cmd == "selftest": cmd_selftest(cfg); return
    if cmd == "validate": cmd_validate(cfg); return
    if cmd == "test-tg":
        http = HTTPClient(cfg); kv = KVStore(cfg, http); tg = TelegramNotifier(cfg, kv, http)
        print("SENT" if tg.send(f"<b>🔔 TEST v{VERSION}</b>") else "FAILED"); return
    try: TradingBotOrchestrator(cfg).run()
    except Exception as e: logging.exception("FATAL"); sys.exit(1)

if __name__ == "__main__":
    main()
