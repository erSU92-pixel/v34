"""
XAUUSD QUANT BOT v10.0 — FINAL PRODUCTION
================================================================
Semua konfigurasi via env. TIDAK PERLU EDIT KODE.

Default (paling masuk akal secara teori):
    INSTRUMENT=XAU/USD  PRIMARY_TF=1d
    PRIMARY_MODE=trend  BARRIER_MODE=asym_2_1

Untuk eksperimen, ubah env:
    INSTRUMENT: XAU/USD, BTC/USD, EUR/USD
    PRIMARY_TF: 4h, 1d
    PRIMARY_MODE: trend, mean_rev, hybrid
    BARRIER_MODE: asym_2_1, sym_15, sym_20
================================================================
"""
from __future__ import annotations

import os, sys, time, json, math, logging
from io import StringIO
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
)
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("XAUUSD_BOT")

# =============================================================================
# ENV
# =============================================================================
TWELVE_DATA_KEY = os.getenv("TWELVEDATA_API_KEY")
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT   = os.getenv("TELEGRAM_CHAT_ID")
GIST_ID         = os.getenv("GIST_ID")
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN")

# =============================================================================
# INSTRUMENT CONFIG
# =============================================================================
INSTRUMENT = os.getenv("INSTRUMENT", "XAU/USD")
INSTRUMENTS = {
    "XAU/USD": dict(symbol="XAU/USD", contract_size=100, tick_size=0.01,
                    spread_buffer=0.30, slippage_buffer=0.15,
                    session_start=0, session_end=24,
                    min_lot=0.01, max_lot=5.0, ref_etf="GLD", ref_dxy="UUP"),
    "BTC/USD": dict(symbol="BTC/USD", contract_size=1, tick_size=0.01,
                    spread_buffer=20.0, slippage_buffer=10.0,
                    session_start=0, session_end=24,
                    min_lot=0.001, max_lot=1.0, ref_etf="IBIT", ref_dxy="UUP"),
    "EUR/USD": dict(symbol="EUR/USD", contract_size=100000, tick_size=0.00001,
                    spread_buffer=0.00010, slippage_buffer=0.00005,
                    session_start=7, session_end=16,
                    min_lot=0.01, max_lot=10.0, ref_etf=None, ref_dxy="UUP"),
}
if INSTRUMENT not in INSTRUMENTS:
    raise ValueError(f"Unknown instrument: {INSTRUMENT}")
CFG = INSTRUMENTS[INSTRUMENT]

# =============================================================================
# TIMEFRAME & MODE
# =============================================================================
MULTI_TF            = os.getenv("MULTI_TF", "1") == "1"
PRIMARY_TF          = os.getenv("PRIMARY_TF", "1d")
PRIMARY_OUTPUT_SIZE = int(os.getenv("PRIMARY_OUTPUT_SIZE", "2000"))
MTF_OUTPUT_SIZE     = int(os.getenv("MTF_OUTPUT_SIZE", "500"))

PRIMARY_MODE = os.getenv("PRIMARY_MODE", "trend")   # trend | mean_rev | hybrid
BARRIER_MODE = os.getenv("BARRIER_MODE", "asym_2_1") # asym_2_1 | sym_15 | sym_20

# Barrier presets
BARRIER_PRESETS = {
    "asym_2_1": dict(tp=2.0, sl=1.0, breakeven=1.0/(2.0+1.0)),   # 33.3%
    "sym_15":   dict(tp=1.5, sl=1.5, breakeven=0.5),               # 50.0%
    "sym_20":   dict(tp=2.0, sl=2.0, breakeven=0.5),               # 50.0%
}
if BARRIER_MODE not in BARRIER_PRESETS:
    raise ValueError(f"Unknown BARRIER_MODE: {BARRIER_MODE}")
BARRIER = BARRIER_PRESETS[BARRIER_MODE]

INTERVAL_MAP = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "45m": "45min", "1h": "1h", "2h": "2h", "4h": "4h", "8h": "8h",
    "1d": "1day", "1day": "1day", "1w": "1week", "1M": "1month",
}
def td_interval(tf: str) -> str:
    if tf in INTERVAL_MAP:
        return INTERVAL_MAP[tf]
    if tf in INTERVAL_MAP.values():
        return tf
    raise ValueError(f"Unknown interval: {tf}")

# =============================================================================
# ALT DATA
# =============================================================================
ENABLE_FRED     = os.getenv("ENABLE_FRED", "1") == "1"
ENABLE_ETF_FLOW = os.getenv("ENABLE_ETF_FLOW", "1") == "1"
ENABLE_DXY      = os.getenv("ENABLE_DXY", "1") == "1"
FRED_START      = os.getenv("FRED_START", "2023-01-01")

# =============================================================================
# BARRIER & RISK
# =============================================================================
MAX_HOLDING_BARS = int(os.getenv("MAX_HOLDING_BARS", "12"))
PURGE_EMBARGO    = int(os.getenv("PURGE_EMBARGO", "24"))
N_CV_SPLITS      = 5
MIN_PRIMARY_BARS = 60   # 1D data lebih sedikit, turunkan threshold

COST_BPS        = float(os.getenv("COST_BPS", "3.5"))
TRADE_MARGIN    = float(os.getenv("TRADE_MARGIN", "0.03"))
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "10000"))
RISK_PCT        = float(os.getenv("RISK_PCT", "0.01"))
ATR_MULT_SL     = 1.5
MAX_DAILY_LOSS_PCT     = 0.02
MAX_DRAWDOWN_PCT       = 0.10
MAX_CONSECUTIVE_LOSSES = 3
HURST_MIN              = float(os.getenv("HURST_MIN", "0.52"))
HURST_MODE             = os.getenv("HURST_MODE", "VETO")
NOTIFY_HOLD            = os.getenv("NOTIFY_HOLD", "0") == "1"

# =============================================================================
# RESILIENT HTTP
# =============================================================================
class ResilientSession:
    def __init__(self, min_interval_sec: float = 7.5):
        self.session = requests.Session()
        self.min_interval = min_interval_sec
        self._last = 0.0

    def _throttle(self):
        elapsed = time.time() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        for attempt in range(5):
            self._throttle()
            try:
                r = self.session.request(method, url, timeout=30, **kwargs)
                self._last = time.time()
                if r.status_code == 429:
                    ra = int(r.headers.get("Retry-After", 2 ** attempt))
                    logger.warning("HTTP 429 — sleep %ss", ra)
                    time.sleep(ra + np.random.uniform(0, 1))
                    continue
                if 500 <= r.status_code < 600:
                    time.sleep((2 ** attempt) + np.random.uniform(0, 1))
                    continue
                return r
            except requests.RequestException as e:
                logger.warning("HTTP err: %s", e)
                time.sleep((2 ** attempt) + np.random.uniform(0, 1))
        raise RuntimeError(f"HTTP failed: {method} {url}")

http = ResilientSession()

# =============================================================================
# DATA FETCH
# =============================================================================
def fetch_td(symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol, "interval": td_interval(interval),
        "outputsize": outputsize, "apikey": TWELVE_DATA_KEY,
        "format": "JSON", "timezone": "UTC",
    }
    resp = http.request("GET", url, params=params)
    data = resp.json()
    if data.get("status") == "error":
        raise RuntimeError(f"TD [{symbol} {interval}]: {data.get('message')}")
    values = data.get("values")
    if not values:
        raise RuntimeError(f"TD empty [{symbol} {interval}]")

    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    now = pd.Timestamp.now(tz="UTC")
    df = df[df.index <= now]
    if len(df) < 10:
        raise RuntimeError(f"TD [{symbol} {interval}] too few bars")
    return df.reset_index()


def fetch_multi_tf_ohlcv() -> Dict[str, pd.DataFrame]:
    result = {}
    primary = fetch_td(CFG["symbol"], PRIMARY_TF, PRIMARY_OUTPUT_SIZE)
    result[PRIMARY_TF] = primary
    logger.info("TD [%s %s]: %d bars | %s → %s",
                CFG["symbol"], PRIMARY_TF, len(primary),
                primary["datetime"].iloc[0], primary["datetime"].iloc[-1])
    if MULTI_TF:
        for tf in ["1h", "1d"]:
            if tf == PRIMARY_TF:
                continue
            try:
                df = fetch_td(CFG["symbol"], tf, MTF_OUTPUT_SIZE)
                result[tf] = df
                logger.info("TD [%s %s]: %d bars", CFG["symbol"], tf, len(df))
            except Exception as e:
                logger.warning("MTF [%s] failed: %s", tf, e)

    now = pd.Timestamp.now(tz="UTC")
    age_min = (now - primary["datetime"].iloc[-1]).total_seconds() / 60
    max_age = {"1h": 120, "4h": 300, "1d": 1800}.get(PRIMARY_TF, 300)
    if age_min > max_age:
        raise RuntimeError(f"Primary TF stale: {age_min:.0f}min > {max_age}")
    logger.info("Data age: %.0f min (max %d)", age_min, max_age)
    return result


def fetch_dxy() -> Optional[pd.DataFrame]:
    if not ENABLE_DXY:
        return None
    try:
        dxy = fetch_td(CFG["ref_dxy"], PRIMARY_TF, PRIMARY_OUTPUT_SIZE)
        out = dxy[["datetime", "close"]].rename(columns={"close": "dxy_close"})
        logger.info("DXY proxy [%s]: %d bars", CFG["ref_dxy"], len(out))
        return out
    except Exception as e:
        logger.warning("DXY proxy failed: %s", e)
        return None


def fetch_etf_flow() -> Optional[pd.DataFrame]:
    if not ENABLE_ETF_FLOW or not CFG["ref_etf"]:
        return None
    try:
        etf = fetch_td(CFG["ref_etf"], "1d", 400)
        out = etf[["datetime", "close", "volume"]].rename(
            columns={"close": "etf_close", "volume": "etf_volume"})
        logger.info("ETF [%s]: %d bars", CFG["ref_etf"], len(out))
        return out
    except Exception as e:
        logger.warning("ETF [%s] failed: %s", CFG["ref_etf"], e)
        return None


def _fetch_fred_csv(series_id: str, start: str) -> Optional[pd.Series]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    try:
        r = http.request("GET", url)
        if r.status_code != 200:
            return None
        df = pd.read_csv(StringIO(r.text))
        if df.shape[1] < 2:
            return None
        df.columns = ["datetime", "value"]
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        return df.set_index("datetime")["value"]
    except Exception as e:
        logger.warning("FRED [%s] failed: %s", series_id, e)
        return None


def fetch_fred() -> Optional[pd.DataFrame]:
    if not ENABLE_FRED:
        return None
    try:
        t10y = _fetch_fred_csv("DGS10", FRED_START)
        cpi = _fetch_fred_csv("CPIAUCSL", FRED_START)
        if t10y is None or cpi is None or len(t10y) < 30 or len(cpi) < 15:
            return None
        cpi_yoy = cpi.pct_change(12).dropna() * 100
        if len(cpi_yoy) < 5:
            return None
        t10y_df = t10y.reset_index(); t10y_df.columns = ["datetime", "t10y"]
        cpi_df = cpi_yoy.reset_index(); cpi_df.columns = ["datetime", "cpi_yoy"]
        t10y_df = t10y_df.sort_values("datetime")
        cpi_df = cpi_df.sort_values("datetime")
        merged = pd.merge_asof(t10y_df, cpi_df, on="datetime", direction="backward")
        merged["real_yield"] = merged["t10y"] - merged["cpi_yoy"]
        ry = merged[["datetime", "real_yield"]].dropna()
        if len(ry) < 30:
            return None
        logger.info("FRED real yield: %d obs | last %.2f",
                    len(ry), float(ry["real_yield"].iloc[-1]))
        return ry
    except Exception as e:
        logger.warning("FRED failed: %s", e)
        return None

# =============================================================================
# FEATURES
# =============================================================================
def _rolling_hurst(series: np.ndarray) -> float:
    if len(series) < 20: return 0.5
    lags = list(range(2, min(20, len(series) // 2)))
    if len(lags) < 2: return 0.5
    try:
        tau = [np.std(series[lag:] - series[:-lag]) for lag in lags]
        tau = [t for t in tau if t > 1e-12]
        if len(tau) < 2: return 0.5
        poly = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
        return float(np.clip(poly[0], 0.0, 1.0))
    except Exception:
        return 0.5


def _rolling_entropy(series: np.ndarray, bins: int = 10) -> float:
    if len(series) < 10: return 0.0
    rets = np.diff(np.log(np.clip(series, 1e-12, None)))
    if len(rets) == 0 or np.std(rets) < 1e-12: return 0.0
    hist, _ = np.histogram(rets, bins=bins)
    probs = hist / max(hist.sum(), 1)
    probs = probs[probs > 0]
    if len(probs) == 0: return 0.0
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _safe_int(s):
    return s.fillna(0).replace([np.inf, -np.inf], 0).astype(int)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for span in (9, 21, 50, 200):
        df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    roll_std = df["close"].rolling(20).std().replace(0, np.nan)
    df["zscore_20"] = (df["close"] - df["close"].rolling(20).mean()) / roll_std

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    low_5 = df["low"].rolling(5).min()
    high_5 = df["high"].rolling(5).max()
    rng = (high_5 - low_5).replace(0, np.nan)
    df["stoch_k"] = 100 * (df["close"] - low_5) / rng
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean().replace(0, np.nan).ffill()
    df["atr_pct"] = (df["atr_14"] / df["close"]).replace([np.inf, -np.inf], np.nan)

    df["donchian_high_20"] = df["high"].rolling(20).max()
    df["donchian_low_20"] = df["low"].rolling(20).min()
    dc = (df["donchian_high_20"] - df["donchian_low_20"]).replace(0, np.nan)
    df["donchian_pos"] = (df["close"] - df["donchian_low_20"]) / dc

    has_volume = False
    if "volume" in df.columns:
        v = df["volume"]
        has_volume = (v.notna().sum() > len(v) * 0.5) and (v.std() > 0)
    if has_volume:
        vmean = df["volume"].rolling(20).mean().replace(0, np.nan)
        df["volume_ratio"] = (df["volume"] / vmean).fillna(1.0)
        df["_volume_available"] = 1
        logger.info("Volume detected — filter ACTIVE")
    else:
        df["volume_ratio"] = 1.0
        df["_volume_available"] = 0
        logger.info("Volume NOT available — filter DISABLED")

    df["hurst_100"] = df["close"].rolling(100).apply(_rolling_hurst, raw=True)
    df["entropy_50"] = df["close"].rolling(50).apply(_rolling_entropy, raw=True)

    df["datetime_utc"] = pd.to_datetime(df["datetime"], utc=True)
    df["hour"] = df["datetime_utc"].dt.hour
    df["in_session"] = (
        (df["hour"] >= CFG["session_start"]) &
        (df["hour"] < CFG["session_end"])).astype(int)
    return df


def add_multi_tf_confluence(df, mtf) -> pd.DataFrame:
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    df["trend_primary"] = _safe_int(np.sign((df["close"] - df["sma_200"]).fillna(0)))

    def get_tf_trend(tf_df, name):
        t = tf_df.copy().sort_values("datetime").reset_index(drop=True)
        t["ema50"] = t["close"].ewm(span=50, adjust=False).mean()
        t["trend"] = np.sign((t["close"] - t["ema50"]).fillna(0))
        t = t[["datetime", "trend"]].rename(columns={"trend": f"trend_{name}"})
        merged = pd.merge_asof(df, t, on="datetime", direction="backward")
        df[f"trend_{name}"] = _safe_int(merged[f"trend_{name}"].fillna(0))

    if MULTI_TF:
        if "1d" in mtf: get_tf_trend(mtf["1d"], "1d")
        else: df["trend_1d"] = 0
        if "1h" in mtf: get_tf_trend(mtf["1h"], "1h")
        else: df["trend_1h"] = 0
    else:
        df["trend_1d"] = 0; df["trend_1h"] = 0

    df["htf_score"] = df["trend_primary"] + df["trend_1d"] + df["trend_1h"]
    df["htf_1d_aligned"] = _safe_int((df["trend_1d"] == df["trend_primary"]).astype(int))
    df["htf_1h_aligned"] = _safe_int((df["trend_1h"] == df["trend_primary"]).astype(int))
    return df


def add_alt_data(df, dxy, etf, fred) -> pd.DataFrame:
    df = df.copy().sort_values("datetime").reset_index(drop=True)

    dxy_cols = ["dxy_close", "dxy_corr_20", "dxy_above_sma50", "dxy_momentum_20"]
    if dxy is not None and len(dxy) > 20:
        try:
            m = pd.merge_asof(df, dxy.sort_values("datetime"),
                              on="datetime", direction="backward")
            df["dxy_close"] = m["dxy_close"].ffill()
            df["dxy_corr_20"] = df["close"].pct_change().rolling(20).corr(df["dxy_close"].pct_change())
            df["dxy_above_sma50"] = (df["dxy_close"] > df["dxy_close"].rolling(50).mean()).astype(int)
            df["dxy_momentum_20"] = df["dxy_close"].pct_change(20)
        except Exception as e:
            logger.warning("DXY merge failed: %s", e)
            for c in dxy_cols: df[c] = 0.0
    else:
        for c in dxy_cols: df[c] = 0.0

    etf_cols = ["etf_close", "etf_volume", "etf_flow_zscore", "etf_momentum_5d"]
    if etf is not None and len(etf) > 20:
        try:
            m = pd.merge_asof(df, etf.sort_values("datetime"),
                              on="datetime", direction="backward")
            df["etf_close"] = m["etf_close"].ffill()
            df["etf_volume"] = m["etf_volume"].ffill()
            vmean = df["etf_volume"].rolling(50).mean()
            vstd = df["etf_volume"].rolling(50).std().replace(0, np.nan)
            df["etf_flow_zscore"] = (df["etf_volume"] - vmean) / vstd
            df["etf_momentum_5d"] = df["etf_close"].pct_change(5)
        except Exception as e:
            logger.warning("ETF merge failed: %s", e)
            for c in etf_cols: df[c] = 0.0
    else:
        for c in etf_cols: df[c] = 0.0

    fred_cols = ["real_yield", "real_yield_change_5d", "real_yield_change_20d"]
    if fred is not None and len(fred) > 10:
        try:
            m = pd.merge_asof(df, fred.sort_values("datetime"),
                              on="datetime", direction="backward")
            df["real_yield"] = m["real_yield"].ffill()
            df["real_yield_change_5d"] = df["real_yield"].diff(5)
            df["real_yield_change_20d"] = df["real_yield"].diff(20)
        except Exception as e:
            logger.warning("FRED merge failed: %s", e)
            for c in fred_cols: df[c] = 0.0
    else:
        for c in fred_cols: df[c] = 0.0

    for c in dxy_cols + etf_cols + fred_cols:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return df


FEATURE_COLS = [
    "ema_9", "ema_21", "ema_50", "zscore_20",
    "rsi_14", "stoch_k", "stoch_d",
    "atr_pct", "donchian_pos", "volume_ratio",
    "hurst_100", "entropy_50",
    "in_session",
    "htf_score", "htf_1d_aligned", "htf_1h_aligned",
    "dxy_corr_20", "dxy_above_sma50", "dxy_momentum_20",
    "etf_flow_zscore", "etf_momentum_5d",
    "real_yield_change_5d", "real_yield_change_20d",
]

# =============================================================================
# PRIMARY SIGNAL — 3 MODES
# =============================================================================
def primary_signal(df: pd.DataFrame) -> pd.Series:
    rsi = df["rsi_14"].values
    ema9 = df["ema_9"].values
    ema21 = df["ema_21"].values
    ema50 = df["ema_50"].values
    trend = df["trend_primary"].values
    htf = df["htf_score"].values
    valid = (np.isfinite(rsi) & np.isfinite(ema9) &
             np.isfinite(ema21) & np.isfinite(ema50) &
             np.isfinite(trend) & np.isfinite(htf))

    logger.info("═══ PRIMARY MODE: %s ═══", PRIMARY_MODE.upper())
    primary = np.zeros(len(df), dtype=int)

    if PRIMARY_MODE == "trend":
        # Pure momentum: EMA alignment, no RSI
        long_cond = (valid & (trend == 1) & (ema9 > ema21) & (ema21 > ema50))
        short_cond = (valid & (trend == -1) & (ema9 < ema21) & (ema21 < ema50))
        primary[long_cond] = 1
        primary[short_cond] = -1

    elif PRIMARY_MODE == "mean_rev":
        # Original RSI counter-trend (untuk pembanding)
        long_cond = (valid & (trend == 1) & (rsi < 30))
        short_cond = (valid & (trend == -1) & (rsi > 70))
        primary[long_cond] = 1
        primary[short_cond] = -1

    elif PRIMARY_MODE == "hybrid":
        # Trend filter + RSI pullback (momentum + dip buying)
        long_cond = (valid & (trend == 1) & (ema9 > ema21) &
                     (rsi > 40) & (rsi < 60))
        short_cond = (valid & (trend == -1) & (ema9 < ema21) &
                      (rsi > 40) & (rsi < 60))
        primary[long_cond] = 1
        primary[short_cond] = -1

    s = pd.Series(primary, index=df.index, name="primary")
    n_l = int((s == 1).sum())
    n_s = int((s == -1).sum())
    logger.info("LONG=%d SHORT=%d TOTAL=%d (%.1f%%)",
                n_l, n_s, n_l + n_s, 100 * (n_l + n_s) / len(s))
    return s

# =============================================================================
# BARRIER & EV
# =============================================================================
def barrier_labels(df: pd.DataFrame, primary: pd.Series) -> pd.Series:
    close = df["close"].values; high = df["high"].values; low = df["low"].values
    vol = df["atr_pct"].values
    vol = np.where((vol < 1e-5) | ~np.isfinite(vol), 0.003, vol)
    pv = primary.values; n = len(df)
    labels = np.full(n, np.nan)
    tp_mult = BARRIER["tp"]; sl_mult = BARRIER["sl"]

    for i in range(n - 1):
        d = pv[i]
        if d == 0: continue
        entry = close[i]
        if d > 0:
            tp = entry * (1 + tp_mult * vol[i])
            sl = entry * (1 - sl_mult * vol[i])
        else:
            tp = entry * (1 - tp_mult * vol[i])
            sl = entry * (1 + sl_mult * vol[i])
        horizon = min(i + MAX_HOLDING_BARS, n - 1)
        label = 0
        for j in range(i + 1, horizon + 1):
            if d > 0:
                if low[j] <= sl: label = 0; break
                if high[j] >= tp: label = 1; break
            else:
                if high[j] >= sl: label = 0; break
                if low[j] <= tp: label = 1; break
        labels[i] = label
    return pd.Series(labels, index=df.index, name="label")


def _ev_stats(labels, vol):
    mask = ~np.isnan(labels.values)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "win": 0.0, "ev_gross": 0.0, "ev_net": -1.0}
    y = labels.values[mask]; v = vol[mask]
    pnl = np.where(y == 1, BARRIER["tp"] * v, -BARRIER["sl"] * v)
    win = float((y == 1).mean())
    ev_gross = float(pnl.mean())
    ev_net = ev_gross - COST_BPS / 10000.0
    return {"n": n, "win": win, "ev_gross": ev_gross, "ev_net": ev_net}


def ev_check_with_diagnostic(df, primary) -> Dict[str, float]:
    vol = df["atr_pct"].values
    vol = np.where((vol < 1e-5) | ~np.isfinite(vol), 0.003, vol)

    orig = _ev_stats(barrier_labels(df, primary), vol)
    inv = _ev_stats(barrier_labels(df, -primary), vol)
    be = BARRIER["breakeven"]

    logger.info("═══ EV (GATE 1) | BARRIER=%s | BE=%.1f%% ═══",
                BARRIER_MODE, be * 100)
    logger.info("")
    logger.info("ORIGINAL (%s):", PRIMARY_MODE)
    logger.info("  n=%d | win=%.2f%% | edge=%+.2f pp | EV net=%+.6f",
                orig["n"], orig["win"] * 100,
                (orig["win"] - be) * 100, orig["ev_net"])
    logger.info("INVERTED:")
    logger.info("  n=%d | win=%.2f%% | edge=%+.2f pp | EV net=%+.6f",
                inv["n"], inv["win"] * 100,
                (inv["win"] - be) * 100, inv["ev_net"])
    logger.info("")

    if orig["n"] == 0:
        logger.critical("❌ GATE 1 FAILED: 0 signals")
        return {"n_primary": 0, "win_rate": 0.0, "breakeven": be,
                "edge_pp": 0.0, "ev_gross": 0.0, "ev_net": -1.0,
                "inverted_ev_net": inv["ev_net"], "inverted_win": inv["win"]}

    if orig["ev_net"] <= 0:
        logger.critical("❌ GATE 1 FAILED")
        if inv["ev_net"] > orig["ev_net"] + 0.001:
            logger.warning("💡 INVERTED better: %.6f vs %.6f",
                           inv["ev_net"], orig["ev_net"])
            logger.warning("💡 Coba env: PRIMARY_MODE=%s tapi inverse logic manual",
                           PRIMARY_MODE)
    else:
        logger.info("✅ GATE 1 PASSED")

    return {"n_primary": orig["n"], "win_rate": orig["win"], "breakeven": be,
            "edge_pp": (orig["win"] - be) * 100,
            "ev_gross": orig["ev_gross"], "ev_net": orig["ev_net"],
            "inverted_ev_net": inv["ev_net"], "inverted_win": inv["win"]}

# =============================================================================
# PURGED CV + TRAIN
# =============================================================================
def purged_kfold_primary(idx, n_splits=N_CV_SPLITS, embargo=PURGE_EMBARGO):
    n = len(idx); pos = np.arange(n); fold = n // n_splits
    splits = []
    for k in range(n_splits):
        ts, te = k * fold, min((k + 1) * fold, n)
        test_pos = pos[ts:te]
        if len(test_pos) < 10: continue
        test_idx = idx[test_pos]
        tmin, tmax = test_idx.min(), test_idx.max()
        train_mask = np.ones(n, dtype=bool)
        for p in pos:
            if tmin - embargo <= idx[p] <= tmax + embargo:
                train_mask[p] = False
        train_pos = pos[train_mask]
        if len(train_pos) >= 20:
            splits.append((train_pos, test_pos))
    return splits


def train_meta_model(df, primary, labels):
    meta = {"baseline_acc": None, "random_acc": None, "model_weights": {},
            "cv_scores": {}, "feature_importance": {},
            "gate2_passed": False, "best_model_acc": None}
    valid_mask = (~np.isnan(labels.values) &
                  df[FEATURE_COLS].notna().all(axis=1).values)
    if valid_mask.sum() < MIN_PRIMARY_BARS:
        logger.warning("Insufficient: %d < %d", valid_mask.sum(), MIN_PRIMARY_BARS)
        return None, None, meta
    idx = np.where(valid_mask)[0]
    X = df[FEATURE_COLS].values[idx].astype(np.float64)
    y = labels.values[idx].astype(int)
    if len(np.unique(y)) < 2:
        logger.warning("Single class"); return None, None, meta
    splits = purged_kfold_primary(idx)
    if not splits:
        logger.warning("No splits"); return None, None, meta
    logger.info("Training: %d samples, %d splits", len(idx), len(splits))

    factories = {
        "rf": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=10,
            random_state=42, n_jobs=-1),
        "gb": lambda: GradientBoostingClassifier(
            n_estimators=150, max_depth=3, min_samples_leaf=10, random_state=42),
        "et": lambda: ExtraTreesClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=10,
            random_state=42, n_jobs=-1),
    }
    scores = {k: [] for k in factories}
    dummy_scores, random_scores, cv_preds = [], [], []

    for train_pos, test_pos in splits:
        X_tr, X_te = X[train_pos], X[test_pos]
        y_tr, y_te = y[train_pos], y[test_pos]
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
        dm = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
        dummy_scores.append(accuracy_score(y_te, dm.predict(X_te)))
        rd = DummyClassifier(strategy="stratified", random_state=42).fit(X_tr, y_tr)
        random_scores.append(accuracy_score(y_te, rd.predict(X_te)))
        fold_probs = []
        for name, factory in factories.items():
            m = factory(); m.fit(X_tr, y_tr)
            scores[name].append(accuracy_score(y_te, m.predict(X_te)))
            fold_probs.append(m.predict_proba(X_te)[:, 1])
        ens_prob = np.mean(fold_probs, axis=0)
        cv_preds.append((y_te, (ens_prob > 0.5).astype(int), ens_prob))

    meta["baseline_acc"] = float(np.mean(dummy_scores))
    meta["random_acc"] = float(np.mean(random_scores))
    meta["cv_scores"] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in scores.items()}

    logger.info("═══ CV (GATE 2) ═══")
    logger.info("Dummy: %.4f | Random: %.4f",
                meta["baseline_acc"], meta["random_acc"])
    for name, (m, s) in meta["cv_scores"].items():
        logger.info("  %s: %.4f ± %.4f | edge: %+.4f",
                    name.upper(), m, s, m - meta["random_acc"])

    raw_w = {n: max(0.0, m - meta["random_acc"])
             for n, (m, _) in meta["cv_scores"].items()}
    tw = sum(raw_w.values())
    meta["model_weights"] = ({k: v / tw for k, v in raw_w.items()} if tw > 0
                             else {k: 1.0 / len(raw_w) for k in raw_w})
    best = max(m for m, _ in meta["cv_scores"].values())
    meta["best_model_acc"] = best
    meta["gate2_passed"] = (best - meta["random_acc"]) > 0.02
    if not meta["gate2_passed"]:
        logger.critical("❌ GATE 2 FAILED: %.4f vs %.4f", best, meta["random_acc"])
    else:
        logger.info("✅ GATE 2 PASSED")

    sc_final = StandardScaler(); X_all = sc_final.fit_transform(X)
    final_models = {}
    for name, factory in factories.items():
        m = factory(); m.fit(X_all, y)
        final_models[name] = m
        if hasattr(m, "feature_importances_"):
            imp = m.feature_importances_
            top = np.argsort(imp)[-5:][::-1]
            meta["feature_importance"][name] = [
                (FEATURE_COLS[i], float(imp[i])) for i in top]
    meta["cv_predictions"] = cv_preds
    return final_models, sc_final, meta


def predict_profit_prob(df, models, scaler, meta):
    if models is None or scaler is None: return None
    row = df.iloc[-2]
    try:
        X = row[FEATURE_COLS].to_numpy(dtype=np.float64).reshape(1, -1)
    except (ValueError, TypeError):
        return None
    if not np.isfinite(X).all(): return None
    Xs = scaler.transform(X)
    weights = meta.get("model_weights", {})
    p = 0.0
    for name, m in models.items():
        w = weights.get(name, 1.0 / len(models))
        p += w * float(m.predict_proba(Xs)[0, 1])
    return p

# =============================================================================
# BACKTEST
# =============================================================================
def backtest(cv_preds, df, labels):
    if not cv_preds: return {}
    valid = ~np.isnan(labels.values)
    vol = df["atr_pct"].values
    vol = np.where((vol < 1e-5) | ~np.isfinite(vol), 0.003, vol)
    y_pred = np.concatenate([p[1] for p in cv_preds])
    n = len(y_pred)
    v = vol[valid][-n:]; lab = labels.values[valid][-n:]
    pnl_all = np.where(lab == 1, BARRIER["tp"] * v, -BARRIER["sl"] * v) - COST_BPS / 10000.0
    eq_p = np.cumprod(1 + pnl_all)
    primary_m = {"n": int(len(pnl_all)), "win": float((lab == 1).mean()),
                 "ret": float(eq_p[-1] - 1),
                 "sharpe": float(pnl_all.mean() / pnl_all.std()) if pnl_all.std() > 0 else 0.0}
    mask = y_pred == 1
    ml_m = {}
    if mask.sum() > 0:
        pnl_ml = pnl_all[mask]
        eq = np.cumprod(1 + pnl_ml)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        ml_m = {"n": int(mask.sum()), "win": float((pnl_ml > 0).mean()),
                "ret": float(eq[-1] - 1),
                "sharpe": float(pnl_ml.mean() / pnl_ml.std()) if pnl_ml.std() > 0 else 0.0,
                "max_dd": float(dd.min()),
                "pf": float(pnl_ml[pnl_ml > 0].sum() / abs(pnl_ml[pnl_ml < 0].sum()))
                    if (pnl_ml < 0).any() else float("inf")}
    logger.info("═══ BACKTEST ═══")
    logger.info("PRIMARY: n=%d | win=%.1f%% | ret=%+.2f%% | sharpe=%.2f",
                primary_m["n"], primary_m["win"] * 100,
                primary_m["ret"] * 100, primary_m["sharpe"])
    if ml_m:
        logger.info("ML-FILTERED: n=%d | win=%.1f%% | ret=%+.2f%% | dd=%.2f%% | sharpe=%.2f | pf=%.2f",
                    ml_m["n"], ml_m["win"] * 100, ml_m["ret"] * 100,
                    ml_m["max_dd"] * 100, ml_m["sharpe"], ml_m["pf"])
    return {"primary_only": primary_m, "ml_filtered": ml_m}

# =============================================================================
# RISK ENGINE
# =============================================================================
class RiskEngine:
    def __init__(self, balance, risk_pct, cfg):
        self.balance = balance; self.risk_pct = risk_pct; self.cfg = cfg
        self.peak_equity = balance; self.daily_pnl = 0.0
        self.consecutive_losses = 0; self.halted = False
        self._reset = datetime.now(timezone.utc).date()

    def _daily_reset(self):
        t = datetime.now(timezone.utc).date()
        if t != self._reset:
            self.daily_pnl = 0.0; self._reset = t

    def check_circuit_breaker(self, equity):
        self._daily_reset()
        self.peak_equity = max(self.peak_equity, equity)
        dd = (self.peak_equity - equity) / max(self.peak_equity, 1e-9)
        dl = -self.daily_pnl / self.balance if self.daily_pnl < 0 else 0.0
        if self.halted: return True
        if dd >= MAX_DRAWDOWN_PCT:
            logger.critical("CB: drawdown %.2f%%", dd * 100); self.halted = True
        elif dl >= MAX_DAILY_LOSS_PCT:
            logger.critical("CB: daily loss %.2f%%", dl * 100); self.halted = True
        elif self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            logger.critical("CB: %d consecutive losses", self.consecutive_losses); self.halted = True
        return self.halted

    def position_size(self, atr, hurst):
        risk_money = self.balance * self.risk_pct
        stop_dist = ATR_MULT_SL * max(atr, self.cfg["tick_size"] * 10)
        loss_per_lot = stop_dist * self.cfg["contract_size"]
        if loss_per_lot <= 0: return 0.0
        lot = math.floor((risk_money / loss_per_lot) / self.cfg["min_lot"]) * self.cfg["min_lot"]
        if HURST_MODE == "SCALE":
            if hurst < 0.45: lot *= 0.5
            elif hurst < 0.52: lot *= 0.75
        return max(min(lot, self.cfg["max_lot"]), 0.0)

    def compute_sl_tp(self, entry, direction, atr):
        sl_d = ATR_MULT_SL * atr
        tp_d = (BARRIER["tp"] / BARRIER["sl"]) * sl_d
        buf = self.cfg["spread_buffer"] + self.cfg["slippage_buffer"]
        if direction == 1:
            return round(entry - sl_d - buf, 4), round(entry + tp_d + buf, 4)
        return round(entry + sl_d + buf, 4), round(entry - tp_d - buf, 4)

# =============================================================================
# STATE & TELEGRAM
# =============================================================================
def load_state():
    url = f"https://api.github.com/gists/{GIST_ID}"
    h = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    r = http.request("GET", url, headers=h)
    if r.status_code != 200: return {"etag": None, "content": {}}
    etag = r.headers.get("ETag"); content = {}
    for name, f in r.json().get("files", {}).items():
        if name.endswith(".json"):
            try: content = json.loads(f.get("content", "{}")); break
            except json.JSONDecodeError: continue
    return {"etag": etag, "content": content}


def save_state(state, etag):
    url = f"https://api.github.com/gists/{GIST_ID}"
    h = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    if etag: h["If-Match"] = etag
    payload = {"files": {"bot_state.json": {"content": json.dumps(state, indent=2, default=str)}}}
    r = http.request("PATCH", url, headers=h, json=payload)
    return r.status_code in (200, 201)


def send_tg(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT: return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT, "text": msg, "parse_mode": "HTML"}
    for a in range(3):
        try: r = http.request("POST", url, json=payload)
        except RuntimeError: return False
        if r.status_code == 200: return True
        if r.status_code == 429:
            ra = 5
            try: ra = int(r.json().get("parameters", {}).get("retry_after", 5))
            except Exception: pass
            time.sleep(ra + 1); continue
        time.sleep(2 ** a)
    return False

# =============================================================================
# MAIN
# =============================================================================
def main():
    logger.info("═══ %s BOT v10.0 FINAL ═══", INSTRUMENT)
    logger.info("Config: INSTRUMENT=%s | PRIMARY_TF=%s | PRIMARY_MODE=%s | BARRIER=%s",
                INSTRUMENT, PRIMARY_TF, PRIMARY_MODE, BARRIER_MODE)

    missing = [n for n, v in [
        ("TWELVEDATA_API_KEY", TWELVE_DATA_KEY),
        ("TELEGRAM_TOKEN", TELEGRAM_TOKEN),
        ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT),
        ("GIST_ID", GIST_ID), ("GITHUB_TOKEN", GITHUB_TOKEN),
    ] if not v]
    if missing:
        logger.error("Env missing: %s", missing); sys.exit(1)

    state = load_state(); content = state.get("content") or {}
    risk = RiskEngine(
        balance=float(content.get("balance", ACCOUNT_BALANCE)),
        risk_pct=RISK_PCT, cfg=CFG)
    risk.halted = bool(content.get("halted", False))
    risk.daily_pnl = float(content.get("daily_pnl", 0.0))
    risk.consecutive_losses = int(content.get("consecutive_losses", 0))
    risk.peak_equity = float(content.get("peak_equity", risk.balance))

    if risk.check_circuit_breaker(risk.balance):
        send_tg("🚨 <b>CIRCUIT BREAKER</b>"); return

    try:
        mtf = fetch_multi_tf_ohlcv()
        primary_df = mtf[PRIMARY_TF].copy()
        dxy = fetch_dxy()
        etf = fetch_etf_flow()
        fred = fetch_fred()
    except Exception as e:
        logger.exception("Data pipeline failed")
        send_tg(f"⚠️ Data error: {e}"); return

    df = add_features(primary_df)
    df = add_multi_tf_confluence(df, mtf)
    df = add_alt_data(df, dxy, etf, fred)

    primary = primary_signal(df)
    ev = ev_check_with_diagnostic(df, primary)

    if ev["n_primary"] == 0:
        send_tg(
            f"🛑 <b>NO PRIMARY</b>\n"
            f"{INSTRUMENT} {PRIMARY_TF} {PRIMARY_MODE}\n"
            f"Signal tidak fire pada kondisi pasar ini.")
        return

    if ev["ev_net"] <= 0:
        inv_note = ""
        if ev.get("inverted_ev_net", -1) > 0.0001:
            inv_note = (
                f"\n\n💡 INVERTED EV positif: {ev['inverted_ev_net']:+.6f}\n"
                f"Win inverted: {ev['inverted_win']:.2%}\n"
                f"Artinya: pasar counter-momentum kuat.\n"
                f"Rekomendasi: coba PRIMARY_TF=1d atau INSTRUMENT=BTC/USD.")
        send_tg(
            f"🛑 <b>GATE 1 FAILED</b>\n"
            f"{INSTRUMENT} {PRIMARY_TF} {PRIMARY_MODE} {BARRIER_MODE}\n"
            f"Win: {ev['win_rate']:.2%} | BE: {ev['breakeven']:.2%}\n"
            f"Edge: {ev['edge_pp']:+.2f} pp\n"
            f"EV net: {ev['ev_net']:+.6f}\n"
            f"n={ev['n_primary']}{inv_note}")
        return

    labels = barrier_labels(df, primary)
    models, scaler, meta = train_meta_model(df, primary, labels)
    if models is None:
        send_tg("⚠️ Training failed"); return
    bt = backtest(meta.get("cv_predictions", []), df, labels)

    if not meta.get("gate2_passed", False):
        send_tg(
            f"🛑 <b>GATE 2 FAILED</b>\n"
            f"Best: {meta.get('best_model_acc', 0):.4f} | "
            f"Random: {meta.get('random_acc', 0):.4f}")
        return

    hurst_val = df["hurst_100"].iloc[-2]
    hurst = 0.5 if pd.isna(hurst_val) else float(hurst_val)
    if HURST_MODE == "VETO" and hurst < HURST_MIN:
        logger.info("VETO: Hurst %.3f < %.2f", hurst, HURST_MIN)
        send_tg(f"🚫 <b>VETO</b> — Hurst {hurst:.3f} < {HURST_MIN}")
        return

    if not int(df["in_session"].iloc[-2]):
        logger.info("Outside session"); return

    cur_p = int(primary.iloc[-2])
    if cur_p == 0:
        logger.info("No primary at current bar")
        if NOTIFY_HOLD: send_tg("⏸️ No primary")
        return

    p_profit = predict_profit_prob(df, models, scaler, meta)
    if p_profit is None:
        send_tg("⚠️ Prediction failed"); return

    cur_vol = float(df["atr_pct"].iloc[-2])
    if cur_vol < 1e-6 or not np.isfinite(cur_vol): cur_vol = 0.003
    cost_rel = (COST_BPS / 10000.0) / cur_vol
    threshold = BARRIER["sl"] / (BARRIER["tp"] + BARRIER["sl"]) + cost_rel / (BARRIER["tp"] + BARRIER["sl"]) + TRADE_MARGIN
    if threshold > 0.85:
        logger.warning("Threshold %.4f > 0.85 — vol too low", threshold); return

    logger.info("═══ DECISION ═══")
    logger.info("Primary=%+d | p=%.4f | thr=%.4f | Hurst=%.3f",
                cur_p, p_profit, threshold, hurst)

    if p_profit < threshold:
        logger.info("ML filter: HOLD")
        if NOTIFY_HOLD:
            send_tg(f"⏸️ ML filter | p={p_profit:.2%} < {threshold:.2%}")
        return

    atr = float(df["atr_14"].iloc[-2])
    lot = risk.position_size(atr, hurst)
    if lot < CFG["min_lot"]:
        send_tg("⚠️ Lot below minimum"); return

    entry = float(df["close"].iloc[-2])
    sl, tp = risk.compute_sl_tp(entry, cur_p, atr)
    dir_str = "🟢 <b>BUY</b>" if cur_p == 1 else "🔴 <b>SELL</b>"

    msg = (
        f"{dir_str} {INSTRUMENT} [{PRIMARY_TF}] {PRIMARY_MODE}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"Entry: <code>{entry:.4f}</code>\n"
        f"SL:    <code>{sl:.4f}</code>\n"
        f"TP:    <code>{tp:.4f}</code>\n"
        f"Lot:   <code>{lot:.4f}</code>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"p_profit: {p_profit:.2%} > {threshold:.2%}\n"
        f"Hurst: {hurst:.3f} | ATR: {atr:.4f}\n"
        f"EV net: {ev['ev_net']:+.6f}\n"
        f"Barrier: {BARRIER_MODE}")
    if bt.get("ml_filtered", {}).get("n", 0) > 0:
        m = bt["ml_filtered"]
        msg += (f"\n\n📊 Backtest (n={m['n']}):\n"
                f"Win {m['win']:.1%} | Ret {m['ret']:+.2%}\n"
                f"Sharpe {m['sharpe']:.2f} | PF {m['pf']:.2f}")
    send_tg(msg)

    new_state = {
        "instrument": INSTRUMENT, "primary_tf": PRIMARY_TF,
        "primary_mode": PRIMARY_MODE, "barrier_mode": BARRIER_MODE,
        "balance": risk.balance, "halted": risk.halted,
        "daily_pnl": risk.daily_pnl,
        "consecutive_losses": risk.consecutive_losses,
        "peak_equity": risk.peak_equity,
        "last_signal": cur_p, "last_p_profit": p_profit,
        "last_threshold": threshold, "last_ev_net": ev["ev_net"],
        "last_backtest": bt,
        "last_update": datetime.now(timezone.utc).isoformat()}
    for _ in range(2):
        if save_state(new_state, state.get("etag")): break
        state = load_state()

    logger.info("═══ COMPLETE ═══")


if __name__ == "__main__":
    main()
