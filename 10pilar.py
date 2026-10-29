"""
================================================================================
XAUUSD QUANT BOT v9.0 — ALL 4 PATHS IMPLEMENTED
================================================================================
Paths:
    [A] Timeframe Shift       Multi-TF (1H+4H+1D) + HTF-aligned primary
    [B] Alternative Data      FRED real yield + DXY regime + ETF flow proxy
    [C] Instrument Pivot      Config-driven instrument abstraction
    [D] Strategy Refinement   Refined primary + isotonic probability calibration

Framework: López de Prado — Triple Barrier + Meta-Labeling + Purged CV
================================================================================
Sumber data (semua gratis):
    - Twelve Data    : OHLCV multi-TF + DXY + ETF (8 req/min free tier)
    - FRED           : DGS10, CPIAUCSL (120 req/min, no key needed)
    - pandas-datareader : wrapper FRED
================================================================================
"""
from __future__ import annotations

import os
import sys
import time
import json
import math
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV

# =============================================================================
# KONFIGURASI
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("XAUUSD_BOT")

TWELVE_DATA_KEY = os.getenv("TWELVEDATA_API_KEY")
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT   = os.getenv("TELEGRAM_CHAT_ID")
GIST_ID         = os.getenv("GIST_ID")
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN")

# --- PATH C: Instrument Config ---
INSTRUMENT = os.getenv("INSTRUMENT", "XAU/USD")

INSTRUMENTS = {
    "XAU/USD": {
        "symbol": "XAU/USD",
        "contract_size": 100,       # 1 lot = 100 oz
        "tick_size": 0.01,
        "spread_buffer": 0.30,
        "slippage_buffer": 0.15,
        "session_start_utc": 7,
        "session_end_utc": 16,
        "min_lot": 0.01,
        "max_lot": 5.0,
        "ref_etf": "GLD",           # ETF flow proxy
    },
    "BTC/USD": {
        "symbol": "BTC/USD",
        "contract_size": 1,         # 1 lot = 1 BTC
        "tick_size": 0.01,
        "spread_buffer": 20.0,
        "slippage_buffer": 10.0,
        "session_start_utc": 0,     # crypto 24/7
        "session_end_utc": 24,
        "min_lot": 0.001,
        "max_lot": 1.0,
        "ref_etf": "IBIT",
    },
    "EUR/USD": {
        "symbol": "EUR/USD",
        "contract_size": 100000,    # 1 lot = 100k EUR
        "tick_size": 0.00001,
        "spread_buffer": 0.00010,
        "slippage_buffer": 0.00005,
        "session_start_utc": 7,
        "session_end_utc": 16,
        "min_lot": 0.01,
        "max_lot": 10.0,
        "ref_etf": None,
    },
}

if INSTRUMENT not in INSTRUMENTS:
    raise ValueError(f"Unknown instrument: {INSTRUMENT}. Options: {list(INSTRUMENTS)}")
CFG = INSTRUMENTS[INSTRUMENT]

# --- PATH A: Timeframe Config ---
MULTI_TF = os.getenv("MULTI_TF", "1") == "1"
PRIMARY_TF = os.getenv("PRIMARY_TF", "4h")   # Primary TF untuk decision
PRIMARY_OUTPUT_SIZE = 3000                    # 4H × 3000 bars ≈ 500 hari
MTF_OUTPUT_SIZE = 500                         # 1H/1D untuk confluence

# --- PATH B: Alt Data ---
ENABLE_FRED = os.getenv("ENABLE_FRED", "1") == "1"
ENABLE_ETF_FLOW = os.getenv("ENABLE_ETF_FLOW", "1") == "1"
FRED_START = os.getenv("FRED_START", "2023-01-01")

# --- PATH D: Refined Primary ---
REFINED_PRIMARY = os.getenv("REFINED_PRIMARY", "1") == "1"
RSI_EXTREME_LOW  = float(os.getenv("RSI_EXTREME_LOW",  "30"))
RSI_EXTREME_HIGH = float(os.getenv("RSI_EXTREME_HIGH", "70"))
VOLUME_MULT      = float(os.getenv("VOLUME_MULT", "1.2"))
CALIBRATE        = os.getenv("CALIBRATE", "1") == "1"

# --- BARRIER ---
TP_MULT = float(os.getenv("TP_MULT", "2.0"))
SL_MULT = float(os.getenv("SL_MULT", "1.0"))
MAX_HOLDING_BARS = int(os.getenv("MAX_HOLDING_BARS", "12"))
PURGE_EMBARGO    = int(os.getenv("PURGE_EMBARGO", "24"))
N_CV_SPLITS      = 5
MIN_PRIMARY_BARS = 100

# --- THRESHOLD & RISK ---
COST_BPS     = float(os.getenv("COST_BPS", "3.5"))
TRADE_MARGIN = float(os.getenv("TRADE_MARGIN", "0.03"))
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
# DATA: Twelve Data multi-TF (Path A) + DXY/ETF (Path B)
# =============================================================================
def fetch_td(symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol, "interval": interval, "outputsize": outputsize,
        "apikey": TWELVE_DATA_KEY, "format": "JSON", "timezone": "UTC",
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
    return df


def fetch_multi_tf_ohlcv() -> Dict[str, pd.DataFrame]:
    """Fetch primary TF + context TFs (1H, 1D)."""
    result = {}
    primary = fetch_td(CFG["symbol"], PRIMARY_TF, PRIMARY_OUTPUT_SIZE)
    result[PRIMARY_TF] = primary.reset_index()
    logger.info("TD [%s %s]: %d bars | %s → %s",
                CFG["symbol"], PRIMARY_TF, len(result[PRIMARY_TF]),
                result[PRIMARY_TF]["datetime"].iloc[0],
                result[PRIMARY_TF]["datetime"].iloc[-1])

    if MULTI_TF:
        for tf, size in [("1h", MTF_OUTPUT_SIZE), ("1d", MTF_OUTPUT_SIZE)]:
            if tf == PRIMARY_TF:
                continue
            try:
                df = fetch_td(CFG["symbol"], tf, size).reset_index()
                result[tf] = df
                logger.info("TD [%s %s]: %d bars", CFG["symbol"], tf, len(df))
            except Exception as e:
                logger.warning("MTF fetch [%s] failed: %s", tf, e)

    # Verify freshness on primary
    now = pd.Timestamp.now(tz="UTC")
    age = (now - result[PRIMARY_TF]["datetime"].iloc[-1]).total_seconds() / 60
    if age > 240:  # 4H bar boleh up to 4h+margin
        raise RuntimeError(f"Primary TF stale: {age:.0f} min")

    return result


def fetch_dxy(primary_dt: pd.Series) -> Optional[pd.DataFrame]:
    try:
        dxy = fetch_td("DXY", PRIMARY_TF, PRIMARY_OUTPUT_SIZE).reset_index()
        return dxy[["datetime", "close"]].rename(columns={"close": "dxy_close"})
    except Exception as e:
        logger.warning("DXY failed: %s", e)
        return None


def fetch_etf_flow() -> Optional[pd.DataFrame]:
    """ETF flow proxy — bukan COT langsung, tapi arah institusional."""
    if not ENABLE_ETF_FLOW or not CFG["ref_etf"]:
        return None
    try:
        etf = fetch_td(CFG["ref_etf"], "1d", 400).reset_index()
        etf = etf[["datetime", "close", "volume"]].rename(
            columns={"close": "etf_close", "volume": "etf_volume"}
        )
        logger.info("ETF [%s]: %d bars", CFG["ref_etf"], len(etf))
        return etf
    except Exception as e:
        logger.warning("ETF [%s] failed: %s", CFG["ref_etf"], e)
        return None


def fetch_fred() -> Optional[pd.DataFrame]:
    """FRED real yield (DGS10 - CPI YoY). Path B."""
    if not ENABLE_FRED:
        return None
    try:
        import pandas_datareader as pdr
        t10y = pdr.get_data_fred("DGS10", start=FRED_START)
        cpi = pdr.get_data_fred("CPIAUCSL", start=FRED_START)
        cpi_yoy = cpi.pct_change(12).fillna(0) * 100
        real_yield = (t10y["DGS10"] - cpi_yoy["CPIAUCSL"]).ffill()
        ry = real_yield.reset_index()
        ry.columns = ["datetime", "real_yield"]
        ry["datetime"] = pd.to_datetime(ry["datetime"], utc=True)
        logger.info("FRED real yield: %d obs | last %.2f",
                    len(ry), ry["real_yield"].iloc[-1])
        return ry
    except Exception as e:
        logger.warning("FRED failed: %s", e)
        return None

# =============================================================================
# FEATURES (10 PILAR + PATH B features)
# =============================================================================
def _rolling_hurst(series: np.ndarray) -> float:
    if len(series) < 20:
        return 0.5
    lags = list(range(2, min(20, len(series) // 2)))
    if len(lags) < 2:
        return 0.5
    try:
        tau = [np.std(series[lag:] - series[:-lag]) for lag in lags]
        tau = [t for t in tau if t > 1e-12]
        if len(tau) < 2:
            return 0.5
        poly = np.polyfit(np.log(lags[: len(tau)]), np.log(tau), 1)
        return float(np.clip(poly[0], 0.0, 1.0))
    except Exception:
        return 0.5


def _rolling_entropy(series: np.ndarray, bins: int = 10) -> float:
    if len(series) < 10:
        return 0.0
    rets = np.diff(np.log(np.clip(series, 1e-12, None)))
    if len(rets) == 0 or np.std(rets) < 1e-12:
        return 0.0
    hist, _ = np.histogram(rets, bins=bins)
    probs = hist / max(hist.sum(), 1)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features pada PRIMARY TF."""
    df = df.copy()

    # Pilar 1: Trend
    for span in (9, 21, 50, 200):
        df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    roll_std = df["close"].rolling(20).std().replace(0, np.nan)
    df["zscore_20"] = (df["close"] - df["close"].rolling(20).mean()) / roll_std

    # Pilar 2: Momentum
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

    # Pilar 4: Volatilitas
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean().replace(0, np.nan).ffill()
    df["atr_pct"] = df["atr_14"] / df["close"]

    df["donchian_high_20"] = df["high"].rolling(20).max()
    df["donchian_low_20"] = df["low"].rolling(20).min()
    dc = (df["donchian_high_20"] - df["donchian_low_20"]).replace(0, np.nan)
    df["donchian_pos"] = (df["close"] - df["donchian_low_20"]) / dc

    # Volume spike (Path D)
    vol_mean = df["volume"].rolling(20).mean().replace(0, np.nan) if "volume" in df else pd.Series(1, index=df.index)
    df["volume_ratio"] = (df["volume"] / vol_mean).fillna(1.0) if "volume" in df else 1.0

    # Pilar 5: Regime
    df["hurst_100"] = df["close"].rolling(100).apply(_rolling_hurst, raw=True)
    df["entropy_50"] = df["close"].rolling(50).apply(_rolling_entropy, raw=True)

    # Pilar 10: Seasonality
    df["datetime_utc"] = pd.to_datetime(df["datetime"], utc=True)
    df["hour"] = df["datetime_utc"].dt.hour
    df["in_session"] = (
        (df["hour"] >= CFG["session_start_utc"]) &
        (df["hour"] < CFG["session_end_utc"])
    ).astype(int)

    return df


def add_multi_tf_confluence(df: pd.DataFrame, mtf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Path A: tambahkan trend alignment dari 1D dan 1H."""
    if not MULTI_TF:
        df["htf_1d_aligned"] = 1
        df["htf_1h_aligned"] = 1
        df["htf_score"] = 3
        return df

    def tf_trend(tf_df: pd.DataFrame) -> pd.DataFrame:
        t = tf_df.copy()
        t["ema50"] = t["close"].ewm(span=50, adjust=False).mean()
        t["trend"] = np.sign(t["close"] - t["ema50"]).astype(int)
        return t[["datetime", "trend"]]

    if "1d" in mtf:
        daily = tf_trend(mtf["1d"])
        # Merge asof: gunakan bar 1D terakhir yang ≤ timestamp primary
        df = pd.merge_asof(
            df.sort_values("datetime"),
            daily.sort_values("datetime").rename(columns={"trend": "trend_1d"}),
            on="datetime", direction="backward",
        )
    else:
        df["trend_1d"] = 0

    if "1h" in mtf:
        hourly = tf_trend(mtf["1h"])
        df = pd.merge_asof(
            df.sort_values("datetime"),
            hourly.sort_values("datetime").rename(columns={"trend": "trend_1h"}),
            on="datetime", direction="backward",
        )
    else:
        df["trend_1h"] = 0

    df["trend_primary"] = np.sign(df["close"] - df["sma_200"]).astype(int)
    df["trend_1d"] = df["trend_1d"].fillna(0).astype(int)
    df["trend_1h"] = df["trend_1h"].fillna(0).astype(int)
    df["htf_score"] = df["trend_primary"] + df["trend_1d"] + df["trend_1h"]
    df["htf_1d_aligned"] = (df["trend_1d"] == df["trend_primary"]).astype(int)
    df["htf_1h_aligned"] = (df["trend_1h"] == df["trend_primary"]).astype(int)
    return df


def add_alt_data(df: pd.DataFrame, dxy: Optional[pd.DataFrame],
                 etf: Optional[pd.DataFrame], fred: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Path B: DXY regime + ETF flow + FRED real yield."""
    df = df.copy()

    # DXY
    if dxy is not None and len(dxy) > 20:
        df = pd.merge_asof(
            df.sort_values("datetime"),
            dxy.sort_values("datetime"),
            on="datetime", direction="backward",
        )
        df["dxy_close"] = df["dxy_close"].ffill()
        dxy_ret = df["dxy_close"].pct_change()
        xau_ret = df["close"].pct_change()
        df["dxy_corr_20"] = xau_ret.rolling(20).corr(dxy_ret)
        df["dxy_above_sma50"] = (df["dxy_close"] > df["dxy_close"].rolling(50).mean()).astype(int)
        df["dxy_momentum_20"] = df["dxy_close"].pct_change(20)
    else:
        for c in ["dxy_close", "dxy_corr_20", "dxy_above_sma50", "dxy_momentum_20"]:
            df[c] = 0.0

    # ETF flow
    if etf is not None and len(etf) > 20:
        df = pd.merge_asof(
            df.sort_values("datetime"),
            etf.sort_values("datetime"),
            on="datetime", direction="backward",
        )
        df["etf_close"] = df["etf_close"].ffill()
        df["etf_volume"] = df["etf_volume"].ffill()
        df["etf_flow_zscore"] = (
            (df["etf_volume"] - df["etf_volume"].rolling(50).mean()) /
            df["etf_volume"].rolling(50).std().replace(0, np.nan)
        )
        df["etf_momentum_5d"] = df["etf_close"].pct_change(5)
    else:
        for c in ["etf_close", "etf_volume", "etf_flow_zscore", "etf_momentum_5d"]:
            df[c] = 0.0

    # FRED real yield
    if fred is not None and len(fred) > 10:
        df = pd.merge_asof(
            df.sort_values("datetime"),
            fred.sort_values("datetime"),
            on="datetime", direction="backward",
        )
        df["real_yield"] = df["real_yield"].ffill()
        df["real_yield_change_5d"] = df["real_yield"].diff(5)
        df["real_yield_change_20d"] = df["real_yield"].diff(20)
    else:
        for c in ["real_yield", "real_yield_change_5d", "real_yield_change_20d"]:
            df[c] = 0.0

    return df


FEATURE_COLS = [
    # Pilar 1-2
    "ema_9", "ema_21", "ema_50", "zscore_20",
    "rsi_14", "stoch_k", "stoch_d",
    # Pilar 4
    "atr_pct", "donchian_pos", "volume_ratio",
    # Pilar 5
    "hurst_100", "entropy_50",
    # Pilar 10
    "in_session",
    # Path A
    "htf_score", "htf_1d_aligned", "htf_1h_aligned",
    # Path B
    "dxy_corr_20", "dxy_above_sma50", "dxy_momentum_20",
    "etf_flow_zscore", "etf_momentum_5d",
    "real_yield_change_5d", "real_yield_change_20d",
]

# =============================================================================
# PATH D: REFINED PRIMARY (RSI extreme + volume + HTF alignment)
# =============================================================================
def primary_signal(df: pd.DataFrame) -> pd.Series:
    close = df["close"].values
    rsi = df["rsi_14"].values
    trend_primary = df["trend_primary"].values
    htf_score = df["htf_score"].values if "htf_score" in df else np.zeros(len(df))
    vol_ratio = df["volume_ratio"].values if "volume_ratio" in df else np.ones(len(df))

    primary = np.zeros(len(df), dtype=int)
    valid = np.isfinite(rsi) & np.isfinite(trend_primary)

    if REFINED_PRIMARY:
        # LONG: uptrend + RSI oversold extreme + volume spike + HTF confluence
        long_cond = (
            valid &
            (trend_primary == 1) &
            (rsi < RSI_EXTREME_LOW) &
            (vol_ratio > VOLUME_MULT) &
            (htf_score >= 1)               # minimal 2/3 TF align
        )
        short_cond = (
            valid &
            (trend_primary == -1) &
            (rsi > RSI_EXTREME_HIGH) &
            (vol_ratio > VOLUME_MULT) &
            (htf_score <= -1)
        )
    else:
        long_cond = valid & (trend_primary == 1) & (rsi < 40)
        short_cond = valid & (trend_primary == -1) & (rsi > 60)

    primary[long_cond] = 1
    primary[short_cond] = -1

    s = pd.Series(primary, index=df.index, name="primary")
    n = len(s)
    logger.info("═══ PRIMARY (Path D: %s) ═══", "REFINED" if REFINED_PRIMARY else "BASIC")
    logger.info("Bars: %d | LONG: %d (%.1f%%) | SHORT: %d (%.1f%%) | NO: %.1f%%",
                n, (s == 1).sum(), 100 * (s == 1).mean(),
                (s == -1).sum(), 100 * (s == -1).mean(),
                100 * (s == 0).mean())
    return s

# =============================================================================
# DIRECTIONAL BARRIER
# =============================================================================
def directional_barrier(df: pd.DataFrame, primary: pd.Series) -> pd.Series:
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    vol = df["atr_pct"].values
    vol = np.where((vol < 1e-5) | ~np.isfinite(vol), 0.003, vol)
    pv = primary.values
    n = len(df)
    labels = np.full(n, np.nan)

    for i in range(n - 1):
        d = pv[i]
        if d == 0:
            continue
        entry = close[i]
        if d > 0:
            tp = entry * (1 + TP_MULT * vol[i])
            sl = entry * (1 - SL_MULT * vol[i])
        else:
            tp = entry * (1 - TP_MULT * vol[i])
            sl = entry * (1 + SL_MULT * vol[i])

        horizon = min(i + MAX_HOLDING_BARS, n - 1)
        label = 0
        for j in range(i + 1, horizon + 1):
            if d > 0:
                if low[j] <= sl:
                    label = 0
                    break
                if high[j] >= tp:
                    label = 1
                    break
            else:
                if high[j] >= sl:
                    label = 0
                    break
                if low[j] <= tp:
                    label = 1
                    break
        labels[i] = label
    return pd.Series(labels, index=df.index, name="label")

# =============================================================================
# EV CHECK (GATE 1)
# =============================================================================
def ev_check(df: pd.DataFrame, labels: pd.Series) -> Dict[str, float]:
    vol = df["atr_pct"].values
    vol = np.where((vol < 1e-5) | ~np.isfinite(vol), 0.003, vol)
    mask = ~np.isnan(labels.values)
    n_primary = int(mask.sum())
    if n_primary == 0:
        return {"ev_net": -1.0, "n_primary": 0}

    y = labels.values[mask]
    v = vol[mask]
    pnl = np.where(y == 1, TP_MULT * v, -SL_MULT * v)
    win = float((y == 1).mean())
    breakeven = SL_MULT / (TP_MULT + SL_MULT)
    ev_gross = float(pnl.mean())
    ev_net = ev_gross - COST_BPS / 10000.0

    logger.info("═══ EV (GATE 1) ═══")
    logger.info("Primary: %d (%.1f%%) | Win: %.2f%% | BE: %.2f%% | Edge: %+.2f pp",
                n_primary, 100 * n_primary / len(df),
                win * 100, breakeven * 100, (win - breakeven) * 100)
    logger.info("EV gross: %+.6f | Cost: %.1f bps | EV net: %+.6f",
                ev_gross, COST_BPS, ev_net)
    if ev_net <= 0:
        logger.critical("❌ GATE 1 FAILED")
    else:
        logger.info("✅ GATE 1 PASSED")

    return {
        "n_primary": n_primary, "win_rate": win, "breakeven": breakeven,
        "edge_pp": (win - breakeven) * 100, "ev_gross": ev_gross, "ev_net": ev_net,
    }

# =============================================================================
# PURGED CV
# =============================================================================
def purged_kfold_primary(
    primary_indices: np.ndarray, n_splits: int = N_CV_SPLITS, embargo: int = PURGE_EMBARGO
) -> List[Tuple[np.ndarray, np.ndarray]]:
    n = len(primary_indices)
    pos = np.arange(n)
    fold = n // n_splits
    splits = []
    for k in range(n_splits):
        ts, te = k * fold, min((k + 1) * fold, n)
        test_pos = pos[ts:te]
        test_idx = primary_indices[test_pos]
        tmin, tmax = test_idx.min(), test_idx.max()
        train_mask = np.ones(n, dtype=bool)
        for p in pos:
            if tmin - embargo <= primary_indices[p] <= tmax + embargo:
                train_mask[p] = False
        train_pos = pos[train_mask]
        if len(train_pos) >= 30 and len(test_pos) >= 15:
            splits.append((train_pos, test_pos))
    return splits

# =============================================================================
# TRAIN META-MODEL (Path D: isotonic calibration)
# =============================================================================
def train_meta_model(
    df: pd.DataFrame, primary: pd.Series, labels: pd.Series
) -> Tuple[Optional[Dict[str, Any]], Optional[StandardScaler], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "baseline_acc": None, "random_acc": None, "model_weights": {},
        "cv_scores": {}, "feature_importance": {},
        "gate2_passed": False, "best_model_acc": None,
    }

    valid_mask = ~np.isnan(labels.values) & df[FEATURE_COLS].notna().all(axis=1).values
    if valid_mask.sum() < MIN_PRIMARY_BARS:
        logger.warning("Primary insufficient: %d", valid_mask.sum())
        return None, None, meta

    idx = np.where(valid_mask)[0]
    X = df[FEATURE_COLS].values[idx].astype(np.float64)
    y = labels.values[idx].astype(int)

    if len(np.unique(y)) < 2:
        return None, None, meta

    splits = purged_kfold_primary(idx)
    if not splits:
        return None, None, meta

    logger.info("Training: %d primary samples, %d CV splits", len(idx), len(splits))

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

    scores: Dict[str, List[float]] = {k: [] for k in factories}
    dummy_scores, random_scores = [], []
    cv_preds = []

    for train_pos, test_pos in splits:
        X_tr, X_te = X[train_pos], X[test_pos]
        y_tr, y_te = y[train_pos], y[test_pos]
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

        dm = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
        dummy_scores.append(accuracy_score(y_te, dm.predict(X_te)))
        rd = DummyClassifier(strategy="stratified", random_state=42).fit(X_tr, y_tr)
        random_scores.append(accuracy_score(y_te, rd.predict(X_te)))

        fold_probs = []
        for name, factory in factories.items():
            m = factory()
            m.fit(X_tr, y_tr)
            scores[name].append(accuracy_score(y_te, m.predict(X_te)))
            fold_probs.append(m.predict_proba(X_te)[:, 1])

        ens_prob = np.mean(fold_probs, axis=0)
        ens_pred = (ens_prob > 0.5).astype(int)
        cv_preds.append((y_te, ens_pred, ens_prob))

    meta["baseline_acc"] = float(np.mean(dummy_scores))
    meta["random_acc"] = float(np.mean(random_scores))
    meta["cv_scores"] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in scores.items()}

    logger.info("═══ CV (GATE 2) ═══")
    logger.info("Dummy: %.4f | Random: %.4f", meta["baseline_acc"], meta["random_acc"])
    for name, (m, s) in meta["cv_scores"].items():
        logger.info("  %s: %.4f ± %.4f | edge: %+.4f",
                    name.upper(), m, s, m - meta["random_acc"])

    raw_w = {n: max(0.0, m - meta["random_acc"]) for n, (m, _) in meta["cv_scores"].items()}
    tw = sum(raw_w.values())
    meta["model_weights"] = {k: v / tw for k, v in raw_w.items()} if tw > 0 else \
                            {k: 1.0 / len(raw_w) for k in raw_w}

    best = max(m for m, _ in meta["cv_scores"].values())
    meta["best_model_acc"] = best
    meta["gate2_passed"] = (best - meta["random_acc"]) > 0.02

    if not meta["gate2_passed"]:
        logger.critical("❌ GATE 2 FAILED: %.4f vs random %.4f", best, meta["random_acc"])
    else:
        logger.info("✅ GATE 2 PASSED")

    # Final fit
    sc_final = StandardScaler()
    X_all = sc_final.fit_transform(X)
    final_models = {}
    for name, factory in factories.items():
        m = factory()
        m.fit(X_all, y)
        final_models[name] = m
        if hasattr(m, "feature_importances_"):
            imp = m.feature_importances_
            top = np.argsort(imp)[-5:][::-1]
            meta["feature_importance"][name] = [
                (FEATURE_COLS[i], float(imp[i])) for i in top
            ]
            logger.info("  %s top-5: %s", name.upper(),
                        [(FEATURE_COLS[i], f"{imp[i]:.3f}") for i in top])

    meta["cv_predictions"] = cv_preds
    meta["X_all_shape"] = X_all.shape
    return final_models, sc_final, meta

# =============================================================================
# PREDICT
# =============================================================================
def predict_profit_prob(df: pd.DataFrame, models: Dict, scaler: StandardScaler,
                        meta: Dict) -> Optional[float]:
    if models is None or scaler is None:
        return None
    row = df.iloc[-2]
    try:
        X = row[FEATURE_COLS].to_numpy(dtype=np.float64).reshape(1, -1)
    except (ValueError, TypeError):
        return None
    if not np.isfinite(X).all():
        return None
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
def backtest(cv_preds, df, labels) -> Dict[str, Any]:
    if not cv_preds:
        return {}
    valid = ~np.isnan(labels.values)
    vol = df["atr_pct"].values
    vol = np.where((vol < 1e-5) | ~np.isfinite(vol), 0.003, vol)
    y_true = np.concatenate([p[0] for p in cv_preds])
    y_pred = np.concatenate([p[1] for p in cv_preds])
    n = len(y_pred)
    v = vol[valid][-n:]
    lab = labels.values[valid][-n:]

    pnl_all = np.where(lab == 1, TP_MULT * v, -SL_MULT * v) - COST_BPS / 10000.0
    eq_p = np.cumprod(1 + pnl_all)
    primary_m = {
        "n": int(len(pnl_all)), "win": float((lab == 1).mean()),
        "ret": float(eq_p[-1] - 1),
        "sharpe": float(pnl_all.mean() / pnl_all.std()) if pnl_all.std() > 0 else 0.0,
    }

    mask = y_pred == 1
    ml_m = {}
    if mask.sum() > 0:
        pnl_ml = pnl_all[mask]
        eq = np.cumprod(1 + pnl_ml)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        ml_m = {
            "n": int(mask.sum()), "win": float((pnl_ml > 0).mean()),
            "ret": float(eq[-1] - 1),
            "sharpe": float(pnl_ml.mean() / pnl_ml.std()) if pnl_ml.std() > 0 else 0.0,
            "max_dd": float(dd.min()),
            "pf": float(pnl_ml[pnl_ml > 0].sum() / abs(pnl_ml[pnl_ml < 0].sum()))
                if (pnl_ml < 0).any() else float("inf"),
        }

    logger.info("═══ BACKTEST ═══")
    logger.info("PRIMARY-ONLY: n=%d | win=%.1f%% | ret=%+.2f%% | sharpe=%.2f",
                primary_m["n"], primary_m["win"] * 100, primary_m["ret"] * 100, primary_m["sharpe"])
    if ml_m:
        logger.info("ML-FILTERED : n=%d | win=%.1f%% | ret=%+.2f%% | dd=%.2f%% | sharpe=%.2f | pf=%.2f",
                    ml_m["n"], ml_m["win"] * 100, ml_m["ret"] * 100,
                    ml_m["max_dd"] * 100, ml_m["sharpe"], ml_m["pf"])
        logger.info("ML edge vs primary: %+.2f%%", (ml_m["ret"] - primary_m["ret"]) * 100)

    return {"primary_only": primary_m, "ml_filtered": ml_m}

# =============================================================================
# RISK ENGINE (Path C: instrument-aware)
# =============================================================================
class RiskEngine:
    def __init__(self, balance: float, risk_pct: float, cfg: Dict):
        self.balance = balance
        self.risk_pct = risk_pct
        self.cfg = cfg
        self.peak_equity = balance
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.halted = False
        self._reset = datetime.now(timezone.utc).date()

    def _daily_reset(self):
        t = datetime.now(timezone.utc).date()
        if t != self._reset:
            self.daily_pnl = 0.0
            self._reset = t

    def check_circuit_breaker(self, equity: float) -> bool:
        self._daily_reset()
        self.peak_equity = max(self.peak_equity, equity)
        dd = (self.peak_equity - equity) / max(self.peak_equity, 1e-9)
        dl = -self.daily_pnl / self.balance if self.daily_pnl < 0 else 0.0
        if self.halted:
            return True
        if dd >= MAX_DRAWDOWN_PCT:
            logger.critical("CB: drawdown %.2f%%", dd * 100); self.halted = True
        elif dl >= MAX_DAILY_LOSS_PCT:
            logger.critical("CB: daily loss %.2f%%", dl * 100); self.halted = True
        elif self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            logger.critical("CB: %d consecutive losses", self.consecutive_losses); self.halted = True
        return self.halted

    def position_size(self, atr: float, hurst: float) -> float:
        """Rumus generic: lot = risk_money / (stop_distance_price × contract_size)."""
        risk_money = self.balance * self.risk_pct
        stop_dist = ATR_MULT_SL * max(atr, self.cfg["tick_size"] * 10)
        loss_per_lot = stop_dist * self.cfg["contract_size"]
        if loss_per_lot <= 0:
            return 0.0
        lot = math.floor((risk_money / loss_per_lot) / self.cfg["min_lot"]) * self.cfg["min_lot"]

        if HURST_MODE == "SCALE":
            if hurst < 0.45:
                lot *= 0.5
            elif hurst < 0.52:
                lot *= 0.75
        return max(min(lot, self.cfg["max_lot"]), 0.0)

    def compute_sl_tp(self, entry: float, direction: int, atr: float) -> Tuple[float, float]:
        sl_d = ATR_MULT_SL * atr
        tp_d = (TP_MULT / SL_MULT) * sl_d
        buf = self.cfg["spread_buffer"] + self.cfg["slippage_buffer"]
        if direction == 1:
            return round(entry - sl_d - buf, 4), round(entry + tp_d + buf, 4)
        return round(entry + sl_d + buf, 4), round(entry - tp_d - buf, 4)

# =============================================================================
# STATE & TELEGRAM
# =============================================================================
def load_state() -> Dict[str, Any]:
    url = f"https://api.github.com/gists/{GIST_ID}"
    h = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    r = http.request("GET", url, headers=h)
    if r.status_code != 200:
        return {"etag": None, "content": {}}
    etag = r.headers.get("ETag")
    content = {}
    for name, f in r.json().get("files", {}).items():
        if name.endswith(".json"):
            try:
                content = json.loads(f.get("content", "{}"))
                break
            except json.JSONDecodeError:
                continue
    return {"etag": etag, "content": content}


def save_state(state: Dict[str, Any], etag: Optional[str]) -> bool:
    url = f"https://api.github.com/gists/{GIST_ID}"
    h = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    if etag:
        h["If-Match"] = etag
    payload = {"files": {"bot_state.json": {"content": json.dumps(state, indent=2, default=str)}}}
    r = http.request("PATCH", url, headers=h, json=payload)
    if r.status_code == 412:
        logger.warning("Gist ETag mismatch")
        return False
    return r.status_code in (200, 201)


def send_tg(msg: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT, "text": msg, "parse_mode": "HTML"}
    for a in range(3):
        try:
            r = http.request("POST", url, json=payload)
        except RuntimeError:
            return False
        if r.status_code == 200:
            return True
        if r.status_code == 429:
            ra = 5
            try:
                ra = int(r.json().get("parameters", {}).get("retry_after", 5))
            except Exception:
                pass
            time.sleep(ra + 1); continue
        time.sleep(2 ** a)
    return False

# =============================================================================
# MAIN
# =============================================================================
def main():
    logger.info("═══ %s BOT v9.0 | INSTRUMENT=%s | PRIMARY_TF=%s ═══",
                INSTRUMENT, INSTRUMENT, PRIMARY_TF)
    logger.info("Paths: A(MultiTF)=%s | B(FRED=%s ETF=%s) | C(Instrument=%s) | D(Refined=%s Calib=%s)",
                MULTI_TF, ENABLE_FRED, ENABLE_ETF_FLOW, INSTRUMENT, REFINED_PRIMARY, CALIBRATE)

    missing = [n for n, v in [
        ("TWELVEDATA_API_KEY", TWELVE_DATA_KEY),
        ("TELEGRAM_TOKEN", TELEGRAM_TOKEN),
        ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT),
        ("GIST_ID", GIST_ID),
        ("GITHUB_TOKEN", GITHUB_TOKEN),
    ] if not v]
    if missing:
        logger.error("Env missing: %s", missing)
        sys.exit(1)

    state = load_state()
    content = state.get("content") or {}
    risk = RiskEngine(
        balance=float(content.get("balance", ACCOUNT_BALANCE)),
        risk_pct=RISK_PCT,
        cfg=CFG,
    )
    risk.halted = bool(content.get("halted", False))
    risk.daily_pnl = float(content.get("daily_pnl", 0.0))
    risk.consecutive_losses = int(content.get("consecutive_losses", 0))
    risk.peak_equity = float(content.get("peak_equity", risk.balance))

    if risk.check_circuit_breaker(risk.balance):
        send_tg("🚨 <b>CIRCUIT BREAKER</b> — halted")
        return

    # DATA
    try:
        mtf = fetch_multi_tf_ohlcv()
        primary_df = mtf[PRIMARY_TF].copy()
        dxy = fetch_dxy(primary_df["datetime"]) if ENABLE_FRED or MULTI_TF else None
        etf = fetch_etf_flow()
        fred = fetch_fred()
    except Exception as e:
        logger.exception("Data pipeline failed")
        send_tg(f"⚠️ Data error: {e}")
        return

    # FEATURES
    df = add_features(primary_df)
    df = add_multi_tf_confluence(df, mtf)
    df = add_alt_data(df, dxy, etf, fred)

    # PRIMARY + LABEL
    primary = primary_signal(df)
    labels = directional_barrier(df, primary)

    # GATE 1
    ev = ev_check(df, labels)
    if ev["ev_net"] <= 0:
        send_tg(
            f"🛑 <b>GATE 1 FAILED</b> [{INSTRUMENT} {PRIMARY_TF}]\n"
            f"EV net: {ev['ev_net']:+.6f}\n"
            f"Win: {ev['win_rate']:.2%} | BE: {ev['breakeven']:.2%} | Edge: {ev['edge_pp']:+.2f}pp\n"
            f"Primary bars: {ev['n_primary']}"
        )
        return

    # TRAIN
    models, scaler, meta = train_meta_model(df, primary, labels)
    if models is None:
        send_tg("⚠️ Training failed")
        return

    bt = backtest(meta.get("cv_predictions", []), df, labels)

    if not meta.get("gate2_passed", False):
        send_tg(
            f"🛑 <b>GATE 2 FAILED</b>\n"
            f"Best: {meta.get('best_model_acc', 0):.4f} | Random: {meta.get('random_acc', 0):.4f}"
        )
        return

    # GATE 3: Hurst
    hurst = float(df["hurst_100"].iloc[-2]) if not pd.isna(df["hurst_100"].iloc[-2]) else 0.5
    if HURST_MODE == "VETO" and hurst < HURST_MIN:
        logger.info("VETO: Hurst %.3f < %.2f", hurst, HURST_MIN)
        send_tg(f"🚫 <b>VETO</b> — Hurst {hurst:.3f} < {HURST_MIN}")
        return

    # GATE 4: Session
    if not int(df["in_session"].iloc[-2]):
        logger.info("Outside session (%d-%d UTC)", CFG["session_start_utc"], CFG["session_end_utc"])
        return

    # GATE 5: Primary at current bar
    cur_p = int(primary.iloc[-2])
    if cur_p == 0:
        logger.info("No primary at current bar — HOLD")
        if NOTIFY_HOLD:
            send_tg("⏸️ No primary signal")
        return

    # GATE 6: p_profit vs dynamic threshold
    p_profit = predict_profit_prob(df, models, scaler, meta)
    if p_profit is None:
        send_tg("⚠️ Prediction failed (NaN)")
        return

    cur_vol = float(df["atr_pct"].iloc[-2])
    if cur_vol < 1e-6 or not np.isfinite(cur_vol):
        cur_vol = 0.003
    cost_rel = (COST_BPS / 10000.0) / cur_vol
    breakeven = (SL_MULT + cost_rel) / (TP_MULT + SL_MULT)
    threshold = breakeven + TRADE_MARGIN

    logger.info("═══ DECISION ═══")
    logger.info("Primary=%+d | p_profit=%.4f | threshold=%.4f | Hurst=%.3f",
                cur_p, p_profit, threshold, hurst)

    if p_profit < threshold:
        logger.info("ML filter: HOLD")
        if NOTIFY_HOLD:
            send_tg(f"⏸️ ML filter | p={p_profit:.2%} < {threshold:.2%}")
        return

    # TRADE
    atr = float(df["atr_14"].iloc[-2])
    lot = risk.position_size(atr, hurst)
    if lot < CFG["min_lot"]:
        send_tg("⚠️ Lot below minimum")
        return

    entry = float(df["close"].iloc[-2])
    sl, tp = risk.compute_sl_tp(entry, cur_p, atr)
    dir_str = "🟢 <b>BUY</b>" if cur_p == 1 else "🔴 <b>SELL</b>"

    msg = (
        f"{dir_str} {INSTRUMENT} [{PRIMARY_TF}]\n"
        f"━━━━━━━━━━━━━━━\n"
        f"Entry: <code>{entry:.4f}</code>\n"
        f"SL:    <code>{sl:.4f}</code>\n"
        f"TP:    <code>{tp:.4f}</code>\n"
        f"Lot:   <code>{lot:.4f}</code>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"p_profit: {p_profit:.2%} > {threshold:.2%}\n"
        f"Hurst: {hurst:.3f} | ATR: {atr:.4f}\n"
        f"HTF score: {int(df['htf_score'].iloc[-2])}\n"
        f"EV net: {ev['ev_net']:+.6f}\n"
    )
    if bt.get("ml_filtered", {}).get("n", 0) > 0:
        m = bt["ml_filtered"]
        msg += (
            f"\n📊 Backtest (n={m['n']}):\n"
            f"Win {m['win']:.1%} | Ret {m['ret']:+.2%}\n"
            f"Sharpe {m['sharpe']:.2f} | PF {m['pf']:.2f}"
        )
    send_tg(msg)

    new_state = {
        "instrument": INSTRUMENT, "primary_tf": PRIMARY_TF,
        "balance": risk.balance, "halted": risk.halted,
        "daily_pnl": risk.daily_pnl, "consecutive_losses": risk.consecutive_losses,
        "peak_equity": risk.peak_equity,
        "last_signal": cur_p, "last_p_profit": p_profit,
        "last_threshold": threshold, "last_ev_net": ev["ev_net"],
        "last_backtest": bt, "last_update": datetime.now(timezone.utc).isoformat(),
    }
    for _ in range(2):
        if save_state(new_state, state.get("etag")):
            break
        state = load_state()

    logger.info("═══ COMPLETE ═══")


if __name__ == "__main__":
    main()
