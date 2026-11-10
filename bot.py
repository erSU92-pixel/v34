"""
XAUUSD Quantitative Institutional Signal Bot — v7.0.4 FINAL
============================================================
FIX LOG:
  - FIX-1: TwelveData volume scalar
  - FIX-2: Weekend/holiday gap false positive
  - FIX-3: check_news_guard wrapper
  - FIX-4: params.json missing → blend returns 0 → SL/TP broken
  - FIX-5: HTML entity injection in notes → Telegram 400 Bad Request
  - FIX-6: Suppress WAIT notifications (NOTIFY_WAIT env, default OFF)
           WAIT still logged + stored (SQLite/Supabase), only Telegram skipped
"""
from __future__ import annotations

import html
import json
import logging
import os
import re
import sqlite3
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fuzzy_regime import blend_params, dominant, membership
from kelly import kelly_fraction, risk_pct_from_env
from news_calendar import NewsResult, check_news_guard as _live_news_guard
from supabase_client import SUPABASE, try_insert_signal

VERSION = "7.0.4"
RUN_ID = uuid.uuid4().hex[:8]
REGIMES = ("TRENDING", "WEAK_TREND", "RANGING")

DEFAULT_PARAMS_FALLBACK: Dict[str, float] = {
    "threshold": 70.0,
    "sl_mult": 1.5,
    "tp1_mult": 1.5,
    "tp2_mult": 2.5,
    "max_hold": 48.0,
}


# ============================================================================
# 1. CONFIG
# ============================================================================
@dataclass
class BotConfig:
    min_bars: int = 500
    retry_total: int = 3
    retry_backoff: float = 0.5
    http_timeout: int = 12
    telegram_timeout: int = 8
    account_size_usd: float = 10_000.0
    risk_pct: float = 1.0
    signal_cooldown_min: int = 30
    spot_cache_min: int = 2
    macro_cache_min: int = 30
    news_enabled: bool = True
    notify_wait: bool = False          # FIX-6: default OFF (anti-spam)
    twelve_data_api_key: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    csv_path: str = "data/xauusd_h1.csv"
    db_path: str = "data/quant_audit.db"

    @classmethod
    def from_env(cls) -> "BotConfig":
        return cls(
            account_size_usd=float(os.getenv("ACCOUNT_SIZE_USD", "10000")),
            risk_pct=float(os.getenv("RISK_PCT", "1.0")),
            news_enabled=os.getenv("NEWS_ENABLED", "1") == "1",
            notify_wait=os.getenv("NOTIFY_WAIT", "0") == "1",
            twelve_data_api_key=os.getenv("TWELVE_DATA_API_KEY"),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
            csv_path=os.getenv("OHLCV_CSV_PATH", "data/xauusd_h1.csv"),
            db_path=os.getenv("AUDIT_DB_PATH", "data/quant_audit.db"),
        )

    def warnings(self) -> List[str]:
        w = []
        if not self.twelve_data_api_key:
            w.append("TWELVE_DATA_API_KEY missing → Yahoo fallback")
        if not self.telegram_bot_token or not self.telegram_chat_id:
            w.append("Telegram creds missing → stdout only")
        if not SUPABASE.enabled:
            w.append("Supabase disabled → SQLite only")
        if not self.notify_wait:
            w.append("NOTIFY_WAIT=0 → WAIT suppressed in Telegram (log only)")
        return w


# ============================================================================
# 2. LOGGING
# ============================================================================
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        p = {"ts": datetime.now(timezone.utc).isoformat(), "run_id": RUN_ID,
             "level": record.levelname, "msg": record.getMessage()}
        extra = getattr(record, "extra_data", None)
        if isinstance(extra, dict):
            p.update(extra)
        if record.exc_info:
            p["exc"] = self.formatException(record.exc_info)
        return json.dumps(p, default=str)


def _get_logger() -> logging.Logger:
    lg = logging.getLogger("xauusd")
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(JsonFormatter())
        lg.addHandler(h)
        lg.propagate = False
    return lg


LOG = _get_logger()


def log_event(level: int = logging.INFO, event: str = "", **kw: Any) -> None:
    rec = LOG.makeRecord(LOG.name, level, "", 0, event, (), None)
    rec.extra_data = kw
    LOG.handle(rec)


# ============================================================================
# 3. MARKET HOURS
# ============================================================================
def is_market_open(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(timezone.utc)
    wd, h = now.weekday(), now.hour
    if wd == 5:
        return False
    if wd == 6:
        return h >= 22
    if wd == 4:
        return h < 21
    return True


# ============================================================================
# 4. INDICATORS
# ============================================================================
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def sma(s, p): return s.rolling(p, min_periods=p).mean()


def rsi(s, p=14):
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / p, adjust=False).mean()
    loss = -d.clip(upper=0).ewm(alpha=1 / p, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + gain / (loss + 1e-10))


def atr(df, p=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / p, adjust=False).mean()


def adx(df, p=14):
    up, dn = df["high"].diff(), -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = atr(df, p)
    p_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / p, adjust=False).mean() / (tr + 1e-10)
    m_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / p, adjust=False).mean() / (tr + 1e-10)
    dx = 100 * (p_di - m_di).abs() / (p_di + m_di + 1e-10)
    return dx.ewm(alpha=1 / p, adjust=False).mean(), p_di, m_di


def volume_profile_poc(df, bins=25):
    if df["volume"].sum() <= 0:
        return float(df["close"].median())
    hist, edges = np.histogram(df["close"], bins=bins, weights=df["volume"])
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2.0)


# ============================================================================
# 5. DATA QUALITY
# ============================================================================
def validate_ohlcv(df, min_bars, label, allow_gap=False):
    req = {"time", "open", "high", "low", "close", "volume"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"{label}: missing {miss}")
    if len(df) < min_bars:
        raise ValueError(f"{label}: {len(df)}<{min_bars}")
    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError(f"{label}: NaN OHLC")
    if (df["high"] < df["low"]).any():
        raise ValueError(f"{label}: high<low")
    if df["volume"].sum() <= 0:
        raise ValueError(f"{label}: zero volume")

    if not allow_gap and len(df) > 1:
        gaps = df["time"].diff()
        for i in range(1, len(df)):
            g = gaps.iloc[i]
            if g <= pd.Timedelta("4h"):
                continue
            prev_t = df["time"].iloc[i - 1]
            curr_t = df["time"].iloc[i]
            if g > pd.Timedelta("24h"):
                continue
            if prev_t.weekday() == 4 and prev_t.hour >= 20:
                continue
            if curr_t.weekday() in (6, 0):
                continue
            raise ValueError(f"{label}: internal gap {g} at {curr_t}")

    last_ts = pd.to_datetime(df["time"].iloc[-1])
    now = datetime.now(timezone.utc)
    age_h = (now.replace(tzinfo=None) - last_ts).total_seconds() / 3600
    max_age = 6 if is_market_open(now) else 72
    if age_h > max_age:
        raise ValueError(f"{label}: stale {age_h:.1f}h (max {max_age})")


# ============================================================================
# 6. SESSION, MACRO, NEWS
# ============================================================================
def get_session_profile(cfg):
    now = datetime.now(timezone.utc)
    if not is_market_open(now):
        return "WEEKEND", False, 999.0, 1.0
    h = now.hour
    if 0 <= h < 7:
        return "ASIA_RANGE", True, 65.0, 1.0
    if 7 <= h < 12:
        return "LONDON_TREND", True, 70.0, 1.3
    if 12 <= h < 16:
        return "NY_OVERLAP", True, 75.0, 1.5
    if 16 <= h < 21:
        return "NY_PM", True, 70.0, 1.3
    return "OFF_HOURS", False, 999.0, 1.0


_macro_cache: Dict[str, Tuple[datetime, Tuple[bool, str]]] = {}


def check_macro_correlation(cfg):
    now = datetime.now(timezone.utc)
    if "m" in _macro_cache:
        ts, val = _macro_cache["m"]
        if (now - ts).total_seconds() < cfg.macro_cache_min * 60:
            return val
    if yf is None:
        return True, "bypass"
    try:
        dxy = yf.Ticker("DX-Y.NYB").history(period="5d")["Close"]
        tnx = yf.Ticker("^TNX").history(period="5d")["Close"]
        notes, safe = [], True
        if len(dxy) >= 2:
            ch = (dxy.iloc[-1] - dxy.iloc[-2]) / dxy.iloc[-2] * 100
            if ch > 0.4:
                safe = False
                notes.append(f"DXY+{ch:.2f}%")
        if len(tnx) >= 2:
            ch = (tnx.iloc[-1] - tnx.iloc[-2]) / tnx.iloc[-2] * 100
            if ch > 0.5:
                safe = False
                notes.append(f"10Y+{ch:.2f}%")
        res = (True, "normal") if safe else (False, "veto " + ", ".join(notes))
        _macro_cache["m"] = (now, res)
        return res
    except Exception as e:
        res = (True, f"bypass ({str(e)[:30]})")
        _macro_cache["m"] = (now, res)
        return res


def check_news_guard(cfg: BotConfig) -> NewsResult:
    if not cfg.news_enabled:
        return NewsResult(allowed=True, reason="news disabled", source="bypass")
    return _live_news_guard()


# ============================================================================
# 7. DATA FETCHER
# ============================================================================
def _retry_session(retries, backoff):
    s = requests.Session()
    r = Retry(total=retries, backoff_factor=backoff,
              status_forcelist=[429, 500, 502, 503, 504],
              allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.headers.update({"User-Agent": f"XAUUSD-Quant/{VERSION}"})
    return s


@dataclass
class MarketData:
    h1: pd.DataFrame
    m15: pd.DataFrame
    source: str
    m15_synthetic: bool = False


class DataFetcher:
    def __init__(self, cfg):
        self.cfg = cfg
        self.session = _retry_session(cfg.retry_total, cfg.retry_backoff)
        self._spot_cache: Optional[Tuple[datetime, float]] = None

    def _twelve(self, interval, size):
        if not self.cfg.twelve_data_api_key:
            return None
        url = (
            f"https://api.twelvedata.com/time_series?symbol=XAU/USD"
            f"&interval={interval}&outputsize={size}"
            f"&apikey={self.cfg.twelve_data_api_key}"
        )
        r = self.session.get(url, timeout=self.cfg.http_timeout)
        r.raise_for_status()
        j = r.json()
        if j.get("status") == "error" or j.get("code", 200) >= 400:
            raise RuntimeError(j.get("message", "TwelveData error"))
        vals = j.get("values") or []
        if not vals:
            return None
        df = pd.DataFrame(vals).rename(columns={"datetime": "time"})
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1000.0)
        else:
            df["volume"] = 1000.0
        return df.sort_values("time").reset_index(drop=True)

    def _yf(self):
        if yf is None:
            raise RuntimeError("yfinance missing")
        tk = yf.Ticker("GC=F")
        h1_raw = tk.history(period="60d", interval="1h")
        m15_raw = tk.history(period="60d", interval="15m")
        if h1_raw.empty:
            raise RuntimeError("Yahoo H1 empty")

        def norm(df):
            df = df.reset_index()
            tcol = "Datetime" if "Datetime" in df.columns else (
                "Date" if "Date" in df.columns else df.columns[0])
            df = df.rename(columns={tcol: "time", "Open": "open", "High": "high",
                                    "Low": "low", "Close": "close", "Volume": "volume"})
            df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
            for c in ["open", "high", "low", "close"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1000.0)
            else:
                df["volume"] = 1000.0
            return df.dropna(subset=["open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)

        h1 = norm(h1_raw)
        if not m15_raw.empty:
            m15 = norm(m15_raw)
            if len(m15) >= 50:
                return MarketData(h1, m15, "yfinance", m15_synthetic=False)
        m15 = h1.set_index("time").resample("15min").ffill().reset_index()
        m15["volume"] /= 4.0
        return MarketData(h1, m15, "yfinance+synthetic", m15_synthetic=True)

    def _csv(self):
        p = self.cfg.csv_path
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            d = os.path.dirname(p)
            if d:
                os.makedirs(d, exist_ok=True)
            dummy = pd.date_range(
                end=datetime.now(timezone.utc).replace(tzinfo=None),
                periods=600, freq="h")
            pd.DataFrame({
                "time": dummy, "open": 2000.0, "high": 2001.0,
                "low": 1999.0, "close": 2000.0, "volume": 1000.0,
            }).to_csv(p, index=False)
            log_event(logging.WARNING, "csv_seeded_dummy", path=p)
        df = pd.read_csv(p)
        df.columns = [c.strip().lower() for c in df.columns]
        tcol = "datetime" if "datetime" in df.columns else (
            "time" if "time" in df.columns else df.columns[0])
        df["time"] = pd.to_datetime(df[tcol], utc=True).dt.tz_localize(None)
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1000.0)
        else:
            df["volume"] = 1000.0
        df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)
        m15 = df.set_index("time").resample("15min").ffill().reset_index()
        m15["volume"] /= 4.0
        return MarketData(df, m15, "csv", m15_synthetic=True)

    def fetch(self):
        try:
            h1 = self._twelve("1h", 500)
            m15 = self._twelve("15min", 200)
            if h1 is not None and len(h1) >= self.cfg.min_bars:
                validate_ohlcv(h1, self.cfg.min_bars, "12data-H1")
                if m15 is None or len(m15) < 50:
                    m15 = h1.set_index("time").resample("15min").ffill().reset_index()
                    m15["volume"] /= 4.0
                    synth = True
                else:
                    validate_ohlcv(m15, 50, "12data-M15")
                    synth = False
                log_event(event="data_ok", source="twelvedata", m15_synthetic=synth)
                return MarketData(h1, m15, "twelvedata", m15_synthetic=synth)
        except Exception as e:
            log_event(logging.WARNING, "twelvedata_fail", err=str(e)[:120])

        try:
            md = self._yf()
            validate_ohlcv(md.h1, self.cfg.min_bars, "yf-H1")
            validate_ohlcv(md.m15, 50, "yf-M15")
            log_event(event="data_ok", source=md.source, m15_synthetic=md.m15_synthetic)
            return md
        except Exception as e:
            log_event(logging.WARNING, "yfinance_fail", err=str(e)[:120])

        md = self._csv()
        validate_ohlcv(md.h1, self.cfg.min_bars, "csv-H1", allow_gap=True)
        return md

    def fetch_spot(self, h1):
        if self._spot_cache:
            ts, v = self._spot_cache
            if (datetime.now(timezone.utc) - ts).total_seconds() < self.cfg.spot_cache_min * 60:
                return v
        price = None
        if self.cfg.twelve_data_api_key:
            try:
                r = self.session.get(
                    f"https://api.twelvedata.com/price?symbol=XAU/USD&apikey={self.cfg.twelve_data_api_key}",
                    timeout=6)
                r.raise_for_status()
                p = float(r.json().get("price", 0) or 0)
                if p > 0:
                    price = p
            except Exception:
                pass
        if price is None and yf is not None:
            try:
                p = float(yf.Ticker("GC=F").fast_info["last_price"])
                if p > 0:
                    price = p
            except Exception:
                pass
        if price is None:
            price = float(h1["close"].iloc[-1])
        self._spot_cache = (datetime.now(timezone.utc), price)
        return price

    def get_basis(self, md):
        fut = float(md.h1["close"].iloc[-1])
        spot = self.fetch_spot(md.h1)
        basis = spot - fut
        if abs(basis) > 100:
            log_event(logging.WARNING, "basis_anomaly", basis=round(basis, 2))
            return 0.0
        return basis


# ============================================================================
# 8. ENSEMBLE
# ============================================================================
@dataclass
class EnsembleResult:
    action: str
    total_score: float
    voter_agreement: float
    forensic_score: float
    regime: str
    regime_weights: Dict[str, float]
    poc_level: float
    notes: List[str]
    atr_value: float
    atr_multiplier: float
    blended_params: Dict[str, float]
    using_fallback: bool = False
    voters_debug: Dict[str, Any] = field(default_factory=dict)


def run_21_voters(df):
    close = df["close"]
    c = float(close.iloc[-1])
    v: List[int] = []
    dbg: Dict[str, int] = {}

    sma_votes = [1 if c > sma(close, p).iloc[-1] else -1 for p in (5, 10, 20, 50, 100, 200)]
    v.extend(sma_votes)
    dbg["sma"] = sum(sma_votes)

    e1 = 1 if ema(close, 9).iloc[-1] > ema(close, 21).iloc[-1] else -1
    e2 = 1 if ema(close, 12).iloc[-1] > ema(close, 26).iloc[-1] else -1
    e3 = 1 if ema(close, 50).iloc[-1] > ema(close, 200).iloc[-1] else -1
    v.extend([e1, e2, e3])
    dbg["ema"] = e1 + e2 + e3

    r = float(rsi(close, 14).iloc[-1])
    r_v = [1 if r > 50 else -1, 1 if r < 70 else -1, 1 if r > 30 else -1]
    v.extend(r_v)
    dbg["rsi"] = sum(r_v)

    adx_v, p_di, m_di = adx(df, 14)
    m_v = [
        1 if p_di.iloc[-1] > m_di.iloc[-1] else -1,
        1 if adx_v.iloc[-1] > 20 else -1,
        1 if c > close.iloc[-2] else -1,
        1 if c > close.iloc[-5] else -1,
        1 if c > close.iloc[-10] else -1,
    ]
    v.extend(m_v)
    dbg["momentum"] = sum(m_v)

    mp = 0
    for n in (5, 10):
        hi_n, lo_n = df["high"].iloc[-n:].max(), df["low"].iloc[-n:].min()
        pos = (c - lo_n) / max(hi_n - lo_n, 1e-10)
        vote = 1 if pos > 0.5 else -1
        v.append(vote)
        mp += vote
    dbg["micro"] = mp

    vol_v = 1 if df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] else -1
    v.append(vol_v)
    dbg["volume"] = vol_v

    poc = volume_profile_poc(df)
    poc_v = 1 if c > poc else -1
    v.append(poc_v)
    dbg["poc"] = poc_v

    assert len(v) == 21
    bull = sum(1 for x in v if x == 1)
    bear = 21 - bull
    return max(bull, bear) / 21.0, bull, bear, dbg


def _load_params():
    path = os.getenv("PARAMS_FILE", "params.json")
    if not os.path.exists(path):
        return {}, 1
    try:
        with open(path) as f:
            j = json.load(f)
        schema = int(j.get("schema", 1))
        if schema >= 2:
            return j.get("per_regime", {}), schema
        return {"_flat": j.get("best_params", {})}, schema
    except Exception:
        return {}, 1


_PARAMS, _PARAMS_SCHEMA = _load_params()


def build_ensemble(md, cfg):
    df, m15 = md.h1, md.m15
    close = df["close"]
    c_last = float(close.iloc[-1])
    atr_val = float(atr(df, 14).iloc[-1])
    adx_val = float(adx(df, 14)[0].iloc[-1])

    weights = membership(adx_val)
    dom = dominant(weights)
    if _PARAMS_SCHEMA >= 2:
        regime_params = _PARAMS
    else:
        flat = _PARAMS.get("_flat", {})
        regime_params = {r: flat for r in REGIMES}
    blended = blend_params(weights, regime_params)

    using_fallback = False
    if blended.get("threshold", 0.0) <= 0.0:
        using_fallback = True
        log_event(logging.WARNING, "params_fallback",
                  note="params.json missing/empty → DEFAULT_PARAMS_FALLBACK",
                  schema=_PARAMS_SCHEMA)
        blended = dict(DEFAULT_PARAMS_FALLBACK)

    threshold = blended.get("threshold", DEFAULT_PARAMS_FALLBACK["threshold"])
    atr_mult = blended.get("sl_mult", DEFAULT_PARAMS_FALLBACK["sl_mult"])

    poc = volume_profile_poc(df)
    session, allowed, _, _ = get_session_profile(cfg)
    news = check_news_guard(cfg)
    corr_ok, corr_msg = check_macro_correlation(cfg)
    ema200 = float(ema(close, 200).iloc[-1])

    last, prev = df.iloc[-1], df.iloc[-21:-1]
    op, hi, lo, cl = last["open"], last["high"], last["low"], last["close"]
    rng = max(hi - lo, 0.01)
    wick_pct = (max(hi - max(op, cl), min(op, cl) - lo) / rng) * 100.0
    sweep = "NONE"
    if hi > prev["high"].max() and cl < prev["high"].max():
        sweep = "BEAR"
    elif lo < prev["low"].min() and cl > prev["low"].min():
        sweep = "BULL"
    h1_vr = last["volume"] / (df["volume"].iloc[-21:-1].mean() + 1e-10)
    m15_vr = (
        m15["volume"].iloc[-1] / (m15["volume"].iloc[-20:].mean() + 1e-10)
        if not md.m15_synthetic else 1.0
    )
    f_score = 30.0
    if sweep != "NONE":
        f_score += 35.0
    if wick_pct > 40.0:
        f_score += 20.0
    if h1_vr > 1.2 or m15_vr > 1.3:
        f_score += 15.0
    f_score = min(f_score, 100.0)

    m15_bull = ema(m15["close"], 9).iloc[-1] > ema(m15["close"], 21).iloc[-1]
    voter_agree, bull, bear, dbg = run_21_voters(df)
    combined = round((voter_agree * 0.6 + (f_score / 100.0) * 0.4) * 100, 1)
    intended = "BUY" if bull > bear else "SELL"
    macro_ok = (
        (intended == "BUY" and c_last > ema200)
        or (intended == "SELL" and c_last < ema200)
    )

    notes, action = [], "WAIT"
    if not news.allowed:
        notes.append(f"NEWS: {news.reason}")
    elif not allowed:
        notes.append(f"SESSION: {session}")
    elif not corr_ok:
        notes.append(f"MACRO: {corr_msg}")
    elif combined < threshold:
        notes.append(f"GATED[{dom}]: {combined}% under {threshold:.1f}%")
    elif not macro_ok:
        notes.append("VETO EMA200 H1")
    elif intended == "BUY" and m15_bull:
        action = "BUY"
        notes.append(f"OK[{dom}/{session}] {combined}%")
    elif intended == "SELL" and not m15_bull:
        action = "SELL"
        notes.append(f"OK[{dom}/{session}] {combined}%")
    else:
        notes.append("M15 misaligned")

    dbg["news_source"] = news.source
    return EnsembleResult(
        action=action, total_score=combined,
        voter_agreement=round(voter_agree, 4), forensic_score=round(f_score, 1),
        regime=dom, regime_weights=weights, poc_level=round(poc, 2),
        notes=notes, atr_value=round(atr_val, 4), atr_multiplier=round(atr_mult, 4),
        blended_params={k: round(v, 4) for k, v in blended.items()},
        using_fallback=using_fallback,
        voters_debug=dbg,
    )


# ============================================================================
# 9. RISK & KELLY
# ============================================================================
def calc_levels(sig, entry, atr_v, atr_mult, cfg, blended, kelly_risk_pct, kelly_reason):
    if sig not in ("BUY", "SELL"):
        return None
    sl_d = max(atr_v * atr_mult, 1.0)
    risk_usd = cfg.account_size_usd * kelly_risk_pct / 100.0
    lot = round(max(risk_usd / (sl_d * 100.0), 0.01), 2)
    actual_risk = lot * sl_d * 100.0
    tp1_r = blended.get("tp1_mult", 1.5) or 1.5
    tp2_r = blended.get("tp2_mult", 2.5) or 2.5
    if sig == "BUY":
        sl, tp1, tp2, be = entry - sl_d, entry + sl_d * tp1_r, entry + sl_d * tp2_r, entry + sl_d
    else:
        sl, tp1, tp2, be = entry + sl_d, entry - sl_d * tp1_r, entry - sl_d * tp2_r, entry - sl_d
    return {
        "entry_price": round(entry, 2), "sl_price": round(sl, 2),
        "tp1_price": round(tp1, 2), "tp2_price": round(tp2, 2),
        "break_even_trigger": round(be, 2),
        "lot_size": lot, "risk_amount_usd": round(risk_usd, 2),
        "actual_risk_usd": round(actual_risk, 2),
        "risk_pct": kelly_risk_pct, "kelly_reason": kelly_reason,
    }


# ============================================================================
# 10. AUDIT DB
# ============================================================================
class AuditDB:
    def __init__(self, path):
        self.ok = False
        try:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            self.conn = sqlite3.connect(path, check_same_thread=False)
            with self.conn:
                self.conn.execute("""CREATE TABLE IF NOT EXISTS signals(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT, action TEXT, spot REAL, score REAL,
                    regime TEXT, kelly_pct REAL, lot REAL, risk REAL, notes TEXT)""")
            self.ok = True
        except Exception as e:
            log_event(logging.WARNING, "db_init_fail", err=str(e)[:80])

    def record(self, e, spot, lv):
        if not self.ok:
            return
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO signals(ts,action,spot,score,regime,kelly_pct,lot,risk,notes) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (datetime.now(timezone.utc).isoformat(), e.action, spot, e.total_score,
                     e.regime, lv["risk_pct"] if lv else 0, lv["lot_size"] if lv else 0,
                     lv["risk_amount_usd"] if lv else 0, "; ".join(e.notes)))
        except Exception as ex:
            log_event(logging.WARNING, "db_write_fail", err=str(ex)[:80])

    def last_action(self):
        if not self.ok:
            return None
        try:
            row = self.conn.execute(
                "SELECT action, ts, spot FROM signals WHERE action IN ('BUY','SELL') "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                return row[0], datetime.fromisoformat(row[1]), row[2]
        except Exception:
            pass
        return None

    def is_dup(self, cooldown, action, spot, tol=1.0):
        last = self.last_action()
        if not last:
            return False
        a, ts, sp = last
        age = (datetime.now(timezone.utc) - ts).total_seconds() / 60
        return age < cooldown and a == action and abs(sp - spot) < tol


# ============================================================================
# 11. TELEGRAM
# ============================================================================
def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def _tg_post(cfg, payload):
    if not (cfg.telegram_bot_token and cfg.telegram_chat_id):
        return False
    if ":" not in cfg.telegram_bot_token:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage",
            json=payload, timeout=cfg.telegram_timeout)
        if r.status_code == 200:
            return True

        if r.status_code == 400 and "parse entities" in r.text:
            log_event(logging.WARNING, "telegram_html_retry", body=r.text[:200])
            plain = {
                "chat_id": payload["chat_id"],
                "text": _strip_html(payload["text"]),
                "disable_web_page_preview": True,
            }
            if "reply_markup" in payload:
                plain["reply_markup"] = payload["reply_markup"]
            r2 = requests.post(
                f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage",
                json=plain, timeout=cfg.telegram_timeout)
            if r2.status_code == 200:
                return True
            log_event(logging.ERROR, "telegram_retry_fail",
                      code=r2.status_code, body=r2.text[:200])
            return False

        log_event(logging.ERROR, "telegram_error",
                  code=r.status_code, body=r.text[:200])
        return False
    except Exception as e:
        log_event(logging.ERROR, "telegram_exc", err=str(e)[:120])
        return False


def send_signal(cfg, e, spot, levels):
    # FIX-6: suppress WAIT unless NOTIFY_WAIT=1
    if e.action == "WAIT" and not cfg.notify_wait:
        log_event(event="telegram_suppressed", action="WAIT",
                  score=e.total_score,
                  reason=e.notes[0] if e.notes else "")
        return False

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    kb = {"inline_keyboard": [[
        {"text": "🔄 Refresh", "callback_data": "refresh"},
        {"text": "📊 Macro", "callback_data": "macro"},
    ]]}
    w = e.regime_weights
    fuzzy_str = html.escape(
        f"{e.regime} (R:{w.get('RANGING', 0):.2f} "
        f"T:{w.get('TRANSITION', 0):.2f} Tr:{w.get('TRENDING', 0):.2f})"
    )
    safe_reason = html.escape(e.notes[0]) if e.notes else ""
    safe_kelly = html.escape(levels["kelly_reason"]) if levels else ""

    fallback_warn = ""
    if e.using_fallback and e.action in ("BUY", "SELL"):
        fallback_warn = "\n⚠️ <b>DEFAULT PARAMS</b> (params.json missing → run optimizer)"

    if e.action == "WAIT" or not levels:
        text = (f"🟡 <b>XAUUSD v{VERSION} — WAIT</b>\n"
                f"⏰ <code>{now}</code>\n"
                f"Score: <b>{e.total_score}%</b> | Regime: {fuzzy_str}\n"
                f"Reason: {safe_reason}{fallback_warn}")
    else:
        em = "🟢" if e.action == "BUY" else "🔴"
        text = (f"{em} <b>XAUUSD v{VERSION} — {e.action}</b>\n"
                f"⏰ <code>{now}</code>\n───────────────────\n"
                f"Score: <b>{e.total_score}%</b>\n"
                f"Regime: {fuzzy_str}\n"
                f"POC: ${e.poc_level:.2f} | ATR: ${e.atr_value:.2f}\n"
                f"<code>ENTRY : ${levels['entry_price']:.2f}\n"
                f"SL    : ${levels['sl_price']:.2f}\n"
                f"TP1   : ${levels['tp1_price']:.2f}\n"
                f"TP2   : ${levels['tp2_price']:.2f}\n"
                f"BE @  : ${levels['break_even_trigger']:.2f}</code>\n"
                f"Lot: <b>{levels['lot_size']}</b> | "
                f"Risk: ${levels['actual_risk_usd']:.0f} ({levels['risk_pct']:.2f}%)\n"
                f"<i>Kelly: {safe_kelly}</i>{fallback_warn}")
    return _tg_post(cfg, {
        "chat_id": cfg.telegram_chat_id, "text": text,
        "parse_mode": "HTML", "reply_markup": kb,
        "disable_web_page_preview": True,
    })


def send_failure(cfg, err):
    if not is_market_open():
        return
    _tg_post(cfg, {
        "chat_id": cfg.telegram_chat_id,
        "text": f"⚠️ <b>BOT ERROR</b> [<code>{RUN_ID}</code>]\n<code>{html.escape(err[:400])}</code>",
        "parse_mode": "HTML",
    })


# ============================================================================
# 12. MAIN
# ============================================================================
def main() -> int:
    cfg = BotConfig.from_env()
    log_event(event="bot_start", version=VERSION, params_schema=_PARAMS_SCHEMA,
              supabase_enabled=SUPABASE.enabled, notify_wait=cfg.notify_wait,
              warnings=cfg.warnings())

    if not is_market_open():
        log_event(event="market_closed", skip=True)
        print("ACTION: SKIP — market closed")
        return 0

    fetcher = DataFetcher(cfg)

    try:
        md = fetcher.fetch()
    except Exception as ex:
        err = f"{type(ex).__name__}: {ex}"
        log_event(logging.ERROR, "fatal_data", err=err)
        send_failure(cfg, err)
        print(f"ACTION: ERROR — {err}")
        return 1

    try:
        e = build_ensemble(md, cfg)
        basis = fetcher.get_basis(md)
        tradable = float(md.h1["close"].iloc[-1]) + basis

        frac, cap = risk_pct_from_env()
        R_history = SUPABASE.fetch_recent_r(50) if SUPABASE.enabled else None
        if R_history:
            kelly_pct, kelly_reason = kelly_fraction(R_history, fraction=frac, cap_pct=cap)
        else:
            kelly_pct, kelly_reason = cfg.risk_pct, "fallback (no R history)"

        levels = calc_levels(e.action, tradable, e.atr_value, e.atr_multiplier,
                             cfg, e.blended_params, kelly_pct, kelly_reason)

        db = AuditDB(cfg.db_path)
        if levels and db.is_dup(cfg.signal_cooldown_min, e.action, tradable):
            log_event(event="signal_deduped", action=e.action, spot=round(tradable, 2))
            print(f"ACTION: {e.action} (deduped) | Score: {e.total_score}%")
            return 0

        if levels:
            sb_ok = try_insert_signal(
                action=e.action, spot=tradable, score=e.total_score,
                regime=e.regime, kelly_frac=kelly_pct / 100.0,
                lot=levels["lot_size"], risk_usd=levels["actual_risk_usd"],
                sl=levels["sl_price"], tp1=levels["tp1_price"], tp2=levels["tp2_price"],
                notes="; ".join(e.notes))
        else:
            sb_ok = try_insert_signal(
                action="WAIT", spot=tradable, score=e.total_score,
                regime=e.regime, kelly_frac=kelly_pct / 100.0,
                lot=0, risk_usd=0, sl=0, tp1=0, tp2=0,
                notes="; ".join(e.notes))

        db.record(e, tradable, levels)
        sent = send_signal(cfg, e, tradable, levels)

        log_event(event="signal_emitted", action=e.action, score=e.total_score,
                  spot=round(tradable, 2), basis=round(basis, 2),
                  source=md.source, m15_synthetic=md.m15_synthetic,
                  regime=e.regime, regime_weights=e.regime_weights,
                  using_fallback=e.using_fallback,
                  kelly_pct=kelly_pct, kelly_reason=kelly_reason,
                  supabase_ok=sb_ok, voters=e.voters_debug, tg=sent)
        print(f"ACTION: {e.action} | Score: {e.total_score}% | Regime: {e.regime} | "
              f"Kelly: {kelly_pct:.2f}% | Spot: {tradable:.2f} | TG: {sent} | SB: {sb_ok}"
              f" | fallback={e.using_fallback}")
        return 0
    except Exception as ex:
        err = f"{type(ex).__name__}: {ex}"
        log_event(logging.ERROR, "fatal_engine", err=err)
        send_failure(cfg, err)
        print(f"ACTION: ERROR — {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
