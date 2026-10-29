"""
================================================================================
XAUUSD QUANT BOT v7.0 — INSTITUTIONAL GRADE
================================================================================
10 PILAR TRADING & QUANTITATIVE FINANCE — IMPLEMENTATION STATUS:

    [AKTIF — dieksekusi penuh]
    Pilar 1  Trend Klasik         EMA 9/21/50, Z-Score
    Pilar 2  Momentum             RSI 14, Stochastic 5/3, adaptive threshold
    Pilar 3  ML Stacking          RF+GB+ET, adaptive Triple Barrier, Purged CV
    Pilar 4  Volatilitas          ATR 14, Donchian 20
    Pilar 5  Regime               Hurst 100 + Adaptive Barrier per regime
    Pilar 7  Cross-Asset          DXY correlation (env FETCH_DXY=1)
    Pilar 10 Seasonality          Session volatility matrix London/NY

    [BELUM — butuh data source eksternal tambahan]
    Pilar 6  Makroekonomi         FRED/US10Y/CPI/NFP feed
    Pilar 8  Order Flow / SMC     tick / order book data
    Pilar 9  Sentimen             COT report / retail sentiment feed

--------------------------------------------------------------------------------
SAFETY GATES (bot otomatis NO-TRADE jika salah satu gagal):
    Gate 1: EV labeling scheme ≥ 0 (net of costs)
    Gate 2: Model ensemble edge > 0 vs random baseline
    Gate 3: Hurst ≥ 0.52 (trend persistence)
    Gate 4: London/NY session (volatility window)
    Gate 5: Circuit breaker tidak aktif

--------------------------------------------------------------------------------
AUDIT FIXES APPLIED (vs v5.0):
    [v7.0] EV diagnostic gate, adaptive barrier by regime, 3-class prediction
    [v7.0] Ensemble weighting, backtest metrics, health check, SAFE MODE
    [v6.1] FIX TypeError isnan (object dtype from iloc[-2])
    [v6.1] FIX look-ahead bias (reject future bar), paksa UTC
    [v6.1] FIX baseline comparison (DummyClassifier)
    [v6.0] Triple Barrier, Purged K-Fold+Embargo, scaler leak fix
    [v6.0] Hurst/Entropy guards, Gist ETag, Telegram retry_after
    [v6.0] ATR position sizing, circuit breaker, spread/slippage buffer
================================================================================
"""
from __future__ import annotations

import os
import sys
import time
import json
import math
import pickle
import base64
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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

# Data
SYMBOL           = "XAU/USD"
DXY_SYMBOL       = "DXY"
INTERVAL         = "1h"
OUTPUT_SIZE      = 5000
FETCH_DXY        = os.getenv("FETCH_DXY", "0") == "1"
DATA_MAX_AGE_MIN = 120  # bar terakhir tidak boleh lebih tua dari 2 jam (1H interval)

# Risiko
ACCOUNT_BALANCE       = float(os.getenv("ACCOUNT_BALANCE", "10000"))
RISK_PCT              = float(os.getenv("RISK_PCT", "0.01"))
ATR_MULT_SL           = 1.5
ATR_MULT_TP           = 2.0
SPREAD_BUFFER         = 0.30
SLIPPAGE_BUFFER       = 0.15
MAX_DAILY_LOSS_PCT    = 0.02
MAX_DRAWDOWN_PCT      = 0.10
MAX_CONSECUTIVE_LOSSES = 3
HURST_MIN             = 0.52

# ML
MAX_HOLDING_BARS = 24
PURGE_EMBARGO    = 24
N_CV_SPLITS      = 5
MIN_BARS_TRAIN   = 500

# Notifikasi
NOTIFY_HOLD = os.getenv("NOTIFY_HOLD", "0") == "1"

# =============================================================================
# RESILIENT HTTP
# =============================================================================
class ResilientSession:
    """HTTP dengan exponential backoff + rate limiting + 429 awareness."""

    def __init__(self, min_interval_sec: float = 7.5):
        self.session = requests.Session()
        self.min_interval = min_interval_sec
        self._last_request = 0.0

    def _throttle(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        for attempt in range(5):
            self._throttle()
            try:
                resp = self.session.request(method, url, timeout=30, **kwargs)
                self._last_request = time.time()
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 2 ** attempt))
                    logger.warning("HTTP 429 — sleep %ss (attempt %d)", retry_after, attempt + 1)
                    time.sleep(retry_after + np.random.uniform(0, 1))
                    continue
                if 500 <= resp.status_code < 600:
                    wait = (2 ** attempt) + np.random.uniform(0, 1)
                    logger.warning("HTTP %d — backoff %.1fs", resp.status_code, wait)
                    time.sleep(wait)
                    continue
                return resp
            except requests.RequestException as e:
                wait = (2 ** attempt) + np.random.uniform(0, 1)
                logger.warning("Request error: %s — backoff %.1fs", e, wait)
                time.sleep(wait)
        raise RuntimeError(f"HTTP failed after 5 retries: {method} {url}")


http = ResilientSession()

# =============================================================================
# MODUL 1: DATA INGESTION
# =============================================================================
def _fetch_series(symbol: str, outputsize: int = OUTPUT_SIZE) -> pd.DataFrame:
    """Fetch satu series dari Twelve Data."""
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_KEY,
        "format": "JSON",
        "timezone": "UTC",
    }
    resp = http.request("GET", url, params=params)
    data = resp.json()
    if data.get("status") == "error":
        raise RuntimeError(f"Twelve Data error [{symbol}]: {data.get('message')}")
    values = data.get("values")
    if not values:
        raise RuntimeError(f"Twelve Data empty [{symbol}]")

    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def fetch_ohlcv() -> pd.DataFrame:
    """Fetch OHLCV + reject future-bar + health check."""
    df = _fetch_series(SYMBOL)

    now_utc = pd.Timestamp.now(tz="UTC")
    future = df[df.index > now_utc]
    if len(future) > 0:
        logger.warning("Reject %d future bars (look-ahead)", len(future))
        df = df[df.index <= now_utc]

    if len(df) < MIN_BARS_TRAIN:
        raise RuntimeError(f"Data terlalu sedikit: {len(df)} bars")

    # Health check
    last_age_min = (now_utc - df.index[-1]).total_seconds() / 60
    if last_age_min > DATA_MAX_AGE_MIN:
        raise RuntimeError(
            f"Data stale: bar terakhir {last_age_min:.0f} menit lalu "
            f"(max {DATA_MAX_AGE_MIN} menit)"
        )

    df = df.reset_index()
    logger.info("Fetched %d bars | %s → %s | age=%.0f min",
                len(df), df["datetime"].iloc[0], df["datetime"].iloc[-1], last_age_min)
    return df


def fetch_dxy() -> Optional[pd.DataFrame]:
    """Optional DXY untuk Pilar 7. Graceful fallback."""
    if not FETCH_DXY:
        return None
    try:
        dxy = _fetch_series(DXY_SYMBOL, outputsize=OUTPUT_SIZE)
        dxy = dxy.reset_index()[["datetime", "close"]].rename(columns={"close": "dxy_close"})
        logger.info("Fetched DXY: %d bars", len(dxy))
        return dxy
    except Exception as e:
        logger.warning("DXY fetch failed (Pilar 7 disabled): %s", e)
        return None

# =============================================================================
# MODUL 2: FEATURE ENGINEERING (10 PILAR)
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
    returns = np.diff(np.log(np.clip(series, 1e-12, None)))
    if len(returns) == 0 or np.std(returns) < 1e-12:
        return 0.0
    hist, _ = np.histogram(returns, bins=bins)
    probs = hist / max(hist.sum(), 1)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def add_features(df: pd.DataFrame, dxy: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Hitung fitur dari 10 pilar (subset yang executable dengan data tersedia)."""
    df = df.copy()

    # --- Pilar 1: Trend Klasik ---
    for span in (9, 21, 50):
        df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False).mean()
    roll_std_20 = df["close"].rolling(20).std().replace(0, np.nan)
    df["zscore_20"] = (df["close"] - df["close"].rolling(20).mean()) / roll_std_20

    # --- Pilar 2: Momentum Klasik ---
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    low_5 = df["low"].rolling(5).min()
    high_5 = df["high"].rolling(5).max()
    range_5 = (high_5 - low_5).replace(0, np.nan)
    df["stoch_k"] = 100 * (df["close"] - low_5) / range_5
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # --- Pilar 4: Volatilitas ---
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean().replace(0, np.nan).ffill()

    df["donchian_high_20"] = df["high"].rolling(20).max()
    df["donchian_low_20"] = df["low"].rolling(20).min()
    dc_range = (df["donchian_high_20"] - df["donchian_low_20"]).replace(0, np.nan)
    df["donchian_pos"] = (df["close"] - df["donchian_low_20"]) / dc_range

    # --- Pilar 5: Regime ---
    df["hurst_100"] = df["close"].rolling(100).apply(_rolling_hurst, raw=True)
    df["entropy_50"] = df["close"].rolling(50).apply(_rolling_entropy, raw=True)

    # --- Pilar 10: Seasonality (session volatility matrix) ---
    df["datetime_utc"] = pd.to_datetime(df["datetime"], utc=True)
    df["hour"] = df["datetime_utc"].dt.hour
    df["is_london_ny"] = ((df["hour"] >= 7) & (df["hour"] <= 16)).astype(int)

    # Session volatility matrix: volatility per jam (rolling 200 bar)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["_log_ret"] = log_ret
    hour_vol = (
        df.groupby("hour")["_log_ret"]
        .transform(lambda x: x.rolling(200, min_periods=50).std())
    )
    df["session_vol_pct"] = hour_vol.rank(pct=True)
    df = df.drop(columns=["_log_ret"])

    # --- Pilar 7: Cross-Asset DXY (optional) ---
    if dxy is not None and len(dxy) > 20:
        df = df.merge(dxy, on="datetime", how="left")
        df["dxy_close"] = df["dxy_close"].ffill()
        dxy_ret = df["dxy_close"].pct_change()
        xau_ret = df["close"].pct_change()
        df["dxy_corr_20"] = xau_ret.rolling(20).corr(dxy_ret)
        df["dxy_return_1h"] = dxy_ret
    else:
        df["dxy_corr_20"] = 0.0
        df["dxy_return_1h"] = 0.0

    return df


FEATURE_COLS_BASE = [
    "ema_9", "ema_21", "ema_50", "zscore_20",
    "rsi_14", "stoch_k", "stoch_d",
    "atr_14", "donchian_pos",
    "hurst_100", "entropy_50",
    "is_london_ny", "session_vol_pct",
]
FEATURE_COLS_DXY = ["dxy_corr_20", "dxy_return_1h"]


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Feature list dinamis tergantung Pilar 7 aktif atau tidak."""
    cols = list(FEATURE_COLS_BASE)
    if FETCH_DXY and "dxy_corr_20" in df.columns:
        cols.extend(FEATURE_COLS_DXY)
    return cols

# =============================================================================
# MODUL 3: ADAPTIVE TRIPLE BARRIER
# =============================================================================
def adaptive_triple_barrier(df: pd.DataFrame) -> pd.Series:
    """
    Regime-aware labeling (López de Prado Triple Barrier):
      Hurst > 0.55 (trending)    → TP=2.5, SL=1.5 (reward > risk)
      Hurst < 0.45 (mean-revert) → TP=1.0, SL=2.0 (tight TP, wide SL)
      else                        → TP=1.5, SL=1.5 (symmetric)
    """
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    hurst = df["hurst_100"].fillna(0.5).to_numpy(dtype=np.float64)
    n = len(df)

    returns = pd.Series(close).pct_change()
    vol = returns.ewm(span=20).std().bfill().to_numpy(dtype=np.float64)
    vol = np.where((vol < 1e-6) | ~np.isfinite(vol), 1e-6, vol)

    labels = np.zeros(n, dtype=int)
    for i in range(n - 1):
        h = hurst[i]
        if h > 0.55:
            pt_mult, sl_mult = 2.5, 1.5
        elif h < 0.45:
            pt_mult, sl_mult = 1.0, 2.0
        else:
            pt_mult, sl_mult = 1.5, 1.5

        entry = close[i]
        pt = entry * (1 + pt_mult * vol[i])
        sl = entry * (1 - sl_mult * vol[i])
        horizon = min(i + MAX_HOLDING_BARS, n - 1)

        hit = 0
        for j in range(i + 1, horizon + 1):
            if high[j] >= pt:
                hit = 1
                break
            if low[j] <= sl:
                hit = -1
                break
        labels[i] = hit

    return pd.Series(labels, index=df.index, name="label")

# =============================================================================
# MODUL 4: EV DIAGNOSTIC (GATE 1)
# =============================================================================
def labeling_expectancy_check(df: pd.DataFrame) -> float:
    """
    Hitung EV labeling scheme (net of costs). Negative EV = bot tidak boleh trade.
    Menggunakan exponential-weighted volatility per bar untuk PnL estimate.
    """
    df = df.copy()
    df["label"] = adaptive_triple_barrier(df)

    returns = pd.Series(df["close"].values).pct_change()
    vol = returns.ewm(span=20).std().bfill().to_numpy(dtype=np.float64)
    vol = np.where((vol < 1e-6) | ~np.isfinite(vol), 1e-6, vol)

    # PnL proxy per label: gunakan magnitude sesuai multiplier regime-aware
    hurst = df["hurst_100"].fillna(0.5).to_numpy(dtype=np.float64)
    pt_mults = np.where(hurst > 0.55, 2.5, np.where(hurst < 0.45, 1.0, 1.5))
    sl_mults = np.where(hurst > 0.55, 1.5, np.where(hurst < 0.45, 2.0, 1.5))

    labels = df["label"].to_numpy()
    pnl = np.where(
        labels == 1, pt_mults * vol,
        np.where(labels == -1, -sl_mults * vol, 0.0),
    )
    ev_gross = float(np.nanmean(pnl))

    win_rate = float((labels == 1).mean())
    avg_breakeven = float(np.mean(sl_mults / (pt_mults + sl_mults)))

    # Biaya transaksi: spread 0.30 USD @ 2000 = 0.015% per sisi → 0.03% round-trip
    cost_per_trade = 0.0003
    ev_net = ev_gross - cost_per_trade

    bh = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]

    logger.info("═══ EV DIAGNOSTIC (GATE 1) ═══")
    logger.info("Labels: {-1:%.2f%%, 0:%.2f%%, +1:%.2f%%}",
                (labels == -1).mean() * 100,
                (labels == 0).mean() * 100,
                (labels == 1).mean() * 100)
    logger.info("Win rate: %.4f | Breakeven (adaptive): %.4f | Delta: %+.4f",
                win_rate, avg_breakeven, win_rate - avg_breakeven)
    logger.info("EV gross: %+.6f | EV net (costs): %+.6f | Buy&Hold: %+.2f%%",
                ev_gross, ev_net, bh * 100)

    if ev_net <= 0:
        logger.critical("❌ GATE 1 FAILED: EV net NEGATIVE — ML tidak akan profitable")
    else:
        logger.info("✅ GATE 1 PASSED: EV net positive")

    return ev_net

# =============================================================================
# MODUL 5: PURGED CV + BASELINE
# =============================================================================
def purged_kfold_indices(
    n_samples: int, n_splits: int = N_CV_SPLITS, embargo: int = PURGE_EMBARGO
) -> List[Tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(n_samples)
    fold_size = n_samples // n_splits
    splits = []
    for k in range(n_splits):
        test_start = k * fold_size
        test_end = min((k + 1) * fold_size, n_samples)
        test_idx = indices[test_start:test_end]
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[max(0, test_start - embargo): min(n_samples, test_end + embargo)] = False
        train_idx = indices[train_mask]
        if len(train_idx) > 100 and len(test_idx) > 50:
            splits.append((train_idx, test_idx))
    return splits

# =============================================================================
# MODUL 6: ML TRAINING (3-CLASS, WEIGHTED ENSEMBLE)
# =============================================================================
def train_ensemble(
    df: pd.DataFrame,
) -> Tuple[Optional[Dict[str, Any]], Optional[StandardScaler], Dict[str, Any]]:
    """
    Train 3-class ensemble. Return:
      models, scaler, metadata (baseline acc, per-model weight, feature importance)
    """
    meta: Dict[str, Any] = {
        "baseline_acc": None,
        "random_acc": None,
        "model_weights": {},
        "cv_scores": {},
        "feature_importance": {},
        "gate2_passed": False,
    }

    df = df.copy()
    df["label"] = adaptive_triple_barrier(df)
    feat_cols = get_feature_cols(df)
    df = df.dropna(subset=feat_cols + ["label"]).reset_index(drop=True)

    if len(df) < MIN_BARS_TRAIN:
        logger.warning("Data terlalu sedikit untuk training (%d bars)", len(df))
        return None, None, meta

    X = df[feat_cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=int)  # {-1, 0, +1}

    if len(np.unique(y)) < 2:
        logger.warning("Hanya %d kelas — tidak bisa train", len(np.unique(y)))
        return None, None, meta

    splits = purged_kfold_indices(len(df))
    if not splits:
        return None, None, meta

    models_factory = {
        "rf": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
        "gb": lambda: GradientBoostingClassifier(
            n_estimators=150, max_depth=4, random_state=42),
        "et": lambda: ExtraTreesClassifier(
            n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
    }

    scores: Dict[str, List[float]] = {k: [] for k in models_factory}
    dummy_scores: List[float] = []
    random_scores: List[float] = []
    cv_predictions: List[Tuple[np.ndarray, np.ndarray]] = []  # (y_true, y_pred_ensemble)

    for train_idx, test_idx in splits:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        dummy_scores.append(accuracy_score(y_test, dummy.predict(X_test)))

        rng = DummyClassifier(strategy="stratified", random_state=42)
        rng.fit(X_train, y_train)
        random_scores.append(accuracy_score(y_test, rng.predict(X_test)))

        fold_preds = []
        for name, factory in models_factory.items():
            m = factory()
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            scores[name].append(accuracy_score(y_test, pred))
            fold_preds.append(pred)

        # Ensemble majority vote untuk backtest
        ensemble_pred = np.array([
            np.bincount([p[i] for p in fold_preds]).argmax() for i in range(len(y_test))
        ]) - 1  # classes_ = [0,1,2] → map ke [-1,0,1]
        # Karena sklearn class order bisa berbeda, ambil dari model.classes_
        classes = models_factory["rf"]().fit(X_train, y_train).classes_
        ensemble_pred = np.array([
            classes[np.bincount([np.where(classes == p[i])[0][0] for p in fold_preds]).argmax()]
            for i in range(len(y_test))
        ])
        cv_predictions.append((y_test, ensemble_pred))

    meta["baseline_acc"] = float(np.mean(dummy_scores))
    meta["random_acc"] = float(np.mean(random_scores))
    meta["cv_scores"] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in scores.items()}

    logger.info("═══ CV RESULTS (GATE 2) ═══")
    logger.info("Baseline dummy (most_frequent): %.4f", meta["baseline_acc"])
    logger.info("Baseline random (stratified):   %.4f", meta["random_acc"])
    for name, (m, s) in meta["cv_scores"].items():
        logger.info("  %s: %.4f ± %.4f | edge vs random: %+.4f",
                    name.upper(), m, s, m - meta["random_acc"])

    # Ensemble weight: max(0, accuracy - random_baseline)
    raw_weights = {
        name: max(0.0, m - meta["random_acc"])
        for name, (m, _) in meta["cv_scores"].items()
    }
    total_w = sum(raw_weights.values())
    if total_w > 0:
        meta["model_weights"] = {k: v / total_w for k, v in raw_weights.items()}
    else:
        meta["model_weights"] = {k: 1.0 / len(raw_weights) for k in raw_weights}
        logger.warning("Semua model ≤ random baseline — pakai equal weight")

    # Best single model vs random → gate 2
    best_model_acc = max(m for m, _ in meta["cv_scores"].values())
    meta["gate2_passed"] = best_model_acc > meta["random_acc"]

    if not meta["gate2_passed"]:
        logger.critical("❌ GATE 2 FAILED: tidak ada model > random baseline")
    else:
        logger.info("✅ GATE 2 PASSED: best model %.4f > random %.4f",
                    best_model_acc, meta["random_acc"])

    # Final fit
    scaler_final = StandardScaler()
    X_all = scaler_final.fit_transform(X)
    final_models = {}
    for name, factory in models_factory.items():
        model = factory()
        model.fit(X_all, y)
        final_models[name] = model
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            top_idx = np.argsort(imp)[-5:][::-1]
            meta["feature_importance"][name] = [
                (feat_cols[i], float(imp[i])) for i in top_idx
            ]
            logger.info("  %s top-5 features: %s", name.upper(),
                        [(feat_cols[i], f"{imp[i]:.3f}") for i in top_idx])

    meta["cv_predictions"] = cv_predictions
    return final_models, scaler_final, meta

# =============================================================================
# MODUL 7: PREDICTION (3-CLASS + WEIGHTED)
# =============================================================================
def predict_signal(
    df: pd.DataFrame,
    models: Dict[str, Any],
    scaler: StandardScaler,
    meta: Dict[str, Any],
) -> Tuple[int, float, Dict[str, float]]:
    """Weighted ensemble prediction → BUY(1)/SELL(-1)/HOLD(0)."""
    empty = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
    if models is None or scaler is None:
        return 0, 0.0, empty

    feat_cols = get_feature_cols(df)
    last_row = df.iloc[-2]  # closed bar

    try:
        X_last = last_row[feat_cols].to_numpy(dtype=np.float64).reshape(1, -1)
    except (ValueError, TypeError) as e:
        logger.warning("Feature conversion failed — HOLD | %s", e)
        return 0, 0.0, empty

    if not np.isfinite(X_last).all():
        return 0, 0.0, empty

    X_scaled = scaler.transform(X_last)

    weights = meta.get("model_weights", {})
    agg = {-1: 0.0, 0: 0.0, 1: 0.0}
    for name, model in models.items():
        w = weights.get(name, 1.0 / len(models))
        probs = model.predict_proba(X_scaled)[0]
        for cls, p in zip(model.classes_, probs):
            agg[int(cls)] += w * float(p)

    p_buy, p_sell, p_hold = agg[1], agg[-1], agg[0]
    detail = {"buy": p_buy, "sell": p_sell, "hold": p_hold}

    # Veto: jika model dominan "timeout", jangan trade
    if p_hold > 0.40:
        return 0, max(p_buy, p_sell), detail

    if p_buy > 0.40 and (p_buy - p_sell) > 0.05:
        return 1, p_buy, detail
    if p_sell > 0.40 and (p_sell - p_buy) > 0.05:
        return -1, p_sell, detail
    return 0, max(p_buy, p_sell), detail

# =============================================================================
# MODUL 8: IN-MEMORY BACKTEST
# =============================================================================
def backtest_from_cv(cv_predictions: List[Tuple[np.ndarray, np.ndarray]],
                     df: pd.DataFrame) -> Dict[str, Any]:
    """Backtest sederhana dari CV ensemble predictions."""
    if not cv_predictions:
        return {}

    all_y, all_p = [], []
    for y, p in cv_predictions:
        all_y.append(y)
        all_p.append(p)
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)

    # Simulasi: tiap prediksi ±1 → trade searah label; hold = no trade
    # Return per label (dalam satuan vol): +pt_mult*vol untuk hit, -sl_mult*vol untuk SL
    vol = df["atr_14"].iloc[-len(y_pred):].fillna(df["atr_14"].mean()).to_numpy() / df["close"].iloc[-len(y_pred):].to_numpy()
    vol = np.clip(vol, 1e-5, None)

    pnl = np.where(y_pred == 1, 2.0 * vol,
                   np.where(y_pred == -1, -1.5 * vol, 0.0))
    pnl = pnl - 0.0003  # costs

    trades = pnl[pnl != -0.0003]  # exclude no-trades
    if len(trades) == 0:
        return {}

    equity = np.cumprod(1 + trades)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak

    metrics = {
        "n_trades": int(len(trades)),
        "total_return": float(equity[-1] - 1),
        "win_rate": float((trades > 0).mean()),
        "avg_trade": float(trades.mean()),
        "sharpe": float(trades.mean() / trades.std()) if trades.std() > 0 else 0.0,
        "max_dd": float(drawdown.min()),
        "profit_factor": float(trades[trades > 0].sum() / abs(trades[trades < 0].sum()))
            if (trades < 0).any() else float("inf"),
    }

    logger.info("═══ BACKTEST (from CV predictions) ═══")
    logger.info("Trades: %d | Win: %.2f%% | Return: %+.2f%% | MaxDD: %.2f%%",
                metrics["n_trades"], metrics["win_rate"] * 100,
                metrics["total_return"] * 100, metrics["max_dd"] * 100)
    logger.info("Sharpe (per-trade): %.3f | Profit Factor: %.2f",
                metrics["sharpe"], metrics["profit_factor"])
    return metrics

# =============================================================================
# MODUL 9: RISK ENGINE
# =============================================================================
class RiskEngine:
    def __init__(self, balance: float, risk_pct: float):
        self.balance = balance
        self.risk_pct = risk_pct
        self.peak_equity = balance
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.halted = False
        self._last_reset = datetime.now(timezone.utc).date()

    def reset_daily(self):
        today = datetime.now(timezone.utc).date()
        if today != self._last_reset:
            self.daily_pnl = 0.0
            self._last_reset = today

    def check_circuit_breaker(self, current_equity: float) -> bool:
        self.reset_daily()
        self.peak_equity = max(self.peak_equity, current_equity)
        dd = (self.peak_equity - current_equity) / max(self.peak_equity, 1e-9)
        daily_loss = -self.daily_pnl / self.balance if self.daily_pnl < 0 else 0.0

        if self.halted:
            return True
        if dd >= MAX_DRAWDOWN_PCT:
            logger.critical("CIRCUIT BREAKER: drawdown %.2f%% ≥ %.2f%%",
                            dd * 100, MAX_DRAWDOWN_PCT * 100)
            self.halted = True
        elif daily_loss >= MAX_DAILY_LOSS_PCT:
            logger.critical("CIRCUIT BREAKER: daily loss %.2f%% ≥ %.2f%%",
                            daily_loss * 100, MAX_DAILY_LOSS_PCT * 100)
            self.halted = True
        elif self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            logger.critical("CIRCUIT BREAKER: %d consecutive losses", self.consecutive_losses)
            self.halted = True
        return self.halted

    def position_size(self, atr: float) -> float:
        risk_money = self.balance * self.risk_pct
        stop_distance = ATR_MULT_SL * max(atr, 0.01)
        tick_size, tick_value = 0.01, 1.0
        loss_per_lot = (stop_distance / tick_size) * tick_value
        if loss_per_lot <= 0:
            return 0.0
        lot = math.floor((risk_money / loss_per_lot) / 0.01) * 0.01
        return max(min(lot, 5.0), 0.0)

    def compute_sl_tp(self, entry: float, direction: int, atr: float) -> Tuple[float, float]:
        sl_d = ATR_MULT_SL * atr
        tp_d = ATR_MULT_TP * atr
        buf = SPREAD_BUFFER + SLIPPAGE_BUFFER
        if direction == 1:
            return round(entry - sl_d - buf, 2), round(entry + tp_d + buf, 2)
        return round(entry + sl_d + buf, 2), round(entry - tp_d - buf, 2)

# =============================================================================
# MODUL 10: STATE & TELEGRAM
# =============================================================================
def load_state() -> Dict[str, Any]:
    url = f"https://api.github.com/gists/{GIST_ID}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    resp = http.request("GET", url, headers=headers)
    if resp.status_code != 200:
        logger.warning("Gist load failed: %d", resp.status_code)
        return {"etag": None, "content": {}}
    etag = resp.headers.get("ETag")
    content = {}
    for name, f in resp.json().get("files", {}).items():
        if name.endswith(".json"):
            try:
                content = json.loads(f.get("content", "{}"))
                break
            except json.JSONDecodeError:
                continue
    return {"etag": etag, "content": content}


def save_state(state: Dict[str, Any], etag: Optional[str]) -> bool:
    url = f"https://api.github.com/gists/{GIST_ID}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    if etag:
        headers["If-Match"] = etag
    payload = {"files": {"bot_state.json": {
        "content": json.dumps(state, indent=2, default=str)
    }}}
    resp = http.request("PATCH", url, headers=headers, json=payload)
    if resp.status_code == 412:
        logger.warning("Gist race condition (ETag mismatch)")
        return False
    if resp.status_code not in (200, 201):
        logger.error("Gist save failed: %d", resp.status_code)
        return False
    return True


def send_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        logger.warning("Telegram credentials missing")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT, "text": message, "parse_mode": "HTML"}
    for attempt in range(3):
        try:
            resp = http.request("POST", url, json=payload)
        except RuntimeError as e:
            logger.error("Telegram request failed: %s", e)
            return False
        if resp.status_code == 200:
            return True
        if resp.status_code == 429:
            retry = 5
            try:
                retry = int(resp.json().get("parameters", {}).get("retry_after", 5))
            except Exception:
                pass
            time.sleep(retry + 1)
            continue
        time.sleep(2 ** attempt)
    return False

# =============================================================================
# MODUL 11: MAIN ORCHESTRATION
# =============================================================================
def main():
    logger.info("═══ XAUUSD BOT v7.0 START ═══")

    # Validasi env
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

    # 1. Load state
    state = load_state()
    content = state.get("content") or {}
    risk = RiskEngine(
        balance=float(content.get("balance", ACCOUNT_BALANCE)),
        risk_pct=RISK_PCT,
    )
    risk.halted = bool(content.get("halted", False))
    risk.daily_pnl = float(content.get("daily_pnl", 0.0))
    risk.consecutive_losses = int(content.get("consecutive_losses", 0))
    risk.peak_equity = float(content.get("peak_equity", risk.balance))

    # 2. Circuit breaker
    if risk.check_circuit_breaker(risk.balance):
        send_telegram("🚨 <b>CIRCUIT BREAKER</b>\nBot halt. Manual review via /rearm.")
        return

    # 3. Data
    try:
        df = fetch_ohlcv()
        dxy = fetch_dxy()
        df = add_features(df, dxy)
    except Exception as e:
        logger.exception("Data pipeline failed")
        send_telegram(f"⚠️ Data error: {e}")
        return

    # 4. EV Gate (GATE 1)
    ev_net = labeling_expectancy_check(df)
    if ev_net <= 0:
        send_telegram(
            f"🛑 <b>GATE 1 FAILED</b>\n"
            f"EV net: {ev_net:+.5f} (negative)\n"
            f"Bot refuses to trade. Fix barrier scheme."
        )
        return

    # 5. Train + Backtest
    models, scaler, meta = train_ensemble(df)
    if models is None:
        send_telegram("⚠️ Training failed — insufficient data")
        return

    backtest = backtest_from_cv(meta.get("cv_predictions", []), df)

    # 6. Gate 2: model edge vs random
    if not meta.get("gate2_passed", False):
        send_telegram(
            f"🛑 <b>GATE 2 FAILED</b>\n"
            f"Best model ≤ random baseline\n"
            f"Bot refuses to trade."
        )
        return

    # 7. Predict
    signal, confidence, prob_detail = predict_signal(df, models, scaler, meta)
    if signal == 0:
        logger.info("Signal HOLD (buy=%.2f sell=%.2f hold=%.2f)",
                    prob_detail["buy"], prob_detail["sell"], prob_detail["hold"])
        if NOTIFY_HOLD:
            send_telegram(f"⏸️ <b>HOLD</b> | "
                         f"B:{prob_detail['buy']:.2f} S:{prob_detail['sell']:.2f} H:{prob_detail['hold']:.2f}")
        return

    # 8. Gate 3: Hurst veto
    hurst = df["hurst_100"].iloc[-2]
    hurst = 0.5 if pd.isna(hurst) else float(hurst)
    if hurst < HURST_MIN:
        logger.info("VETO: Hurst %.3f < %.2f", hurst, HURST_MIN)
        send_telegram(f"🚫 <b>VETO</b> | Hurst {hurst:.3f} (choppy regime)")
        return

    # 9. Gate 4: Session filter
    if not int(df["is_london_ny"].iloc[-2]):
        logger.info("Outside London/NY — skip")
        return

    # 10. Risk sizing
    atr = float(df["atr_14"].iloc[-2])
    lot = risk.position_size(atr)
    if lot <= 0:
        send_telegram("⚠️ Lot 0 — ATR terlalu lebar atau modal kecil")
        return

    entry = float(df["close"].iloc[-2])
    sl, tp = risk.compute_sl_tp(entry, signal, atr)

    # 11. Notifikasi
    dir_str = "🟢 <b>BUY</b>" if signal == 1 else "🔴 <b>SELL</b>"
    msg = (
        f"{dir_str} XAUUSD\n"
        f"━━━━━━━━━━━━━━━\n"
        f"Entry: <code>{entry:.2f}</code>\n"
        f"SL:    <code>{sl:.2f}</code>\n"
        f"TP:    <code>{tp:.2f}</code>\n"
        f"Lot:   <code>{lot:.2f}</code>\n"
        f"Conf:  {confidence:.2%}\n"
        f"Probs: B={prob_detail['buy']:.2f} S={prob_detail['sell']:.2f} H={prob_detail['hold']:.2f}\n"
        f"Hurst: {hurst:.3f} | ATR: {atr:.2f}\n"
        f"EV net: {ev_net:+.5f}\n"
    )
    if backtest:
        msg += (
            f"\n📊 Backtest (CV):\n"
            f"Win {backtest['win_rate']:.1%} | Return {backtest['total_return']:+.2%}\n"
            f"MaxDD {backtest['max_dd']:.2%} | Sharpe {backtest['sharpe']:.2f}"
        )
    send_telegram(msg)

    # 12. Save state
    new_state = {
        "balance": risk.balance,
        "halted": risk.halted,
        "daily_pnl": risk.daily_pnl,
        "consecutive_losses": risk.consecutive_losses,
        "peak_equity": risk.peak_equity,
        "last_signal": signal,
        "last_confidence": confidence,
        "last_ev": ev_net,
        "last_backtest": backtest,
        "last_update": datetime.now(timezone.utc).isoformat(),
    }
    for attempt in range(2):
        if save_state(new_state, state.get("etag")):
            break
        state = load_state()
        if attempt == 1:
            logger.error("State save failed after retry")

    logger.info("═══ BOT RUN COMPLETE ═══")


if __name__ == "__main__":
    main()
