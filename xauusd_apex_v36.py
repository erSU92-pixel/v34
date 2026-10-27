"""
================================================================================
XAUUSD APEX V36.0 — INSTITUTIONAL QUANT RISK & EXECUTION ENGINE
Global Head of Quantitative Risk & Trading Systems Certified Architecture
================================================================================
Fixes vs V35.8 (audit ref: Forensic Report, 34 findings):
  [F1]  Purged Walk-Forward CV replaces anti-causal K-Fold
  [F2]  Barrier-consistent trade simulation replaces 1-bar return proxy
  [F3]  Triple Barrier uses high/low (wick-aware, pessimistic tie-break)
  [F4]  Per-trade cost application (not smeared across all bars)
  [F5]  Consistent annualization: BARS_PER_YEAR = 288 * 252
  [F6]  DSR unit-corrected: per-trade SR, T = n_trades, correct E[max SR]
  [F7]  Correct Shannon entropy (NaN-safe) & robust Hurst (clipped, -inf-safe)
  [F8]  Confidence-based position sizing via predict_proba
  [F9]  Kill switch: max DD, daily loss, consecutive losses
  [F10] Fail-closed data loading (no silent synthetic fallback in production)
  [F11] Fault-tolerant Gist / Telegram wrappers: timeout, retry, 429/5xx, ETag
  [F12] Idempotent order placement (SHA-256 idempotency key + state log)
  [F13] class_weight='balanced'; balanced_accuracy metric
  [F14] Explicit raises instead of assert (safe under python -O)
  [F15] OHLC integrity validation
  [F16] Monte Carlo block bootstrap (preserves autocorrelation)
  [F17] audit_status = PENDING_EXTERNAL_AUDIT (no self-certification)

Requirements: numpy, pandas, scikit-learn, scipy, requests
Usage:
    python xauusd_apex_v36.py --data xauusd_m5.csv
    python xauusd_apex_v36.py --data xauusd_m5.csv --allow-synthetic   # TESTING ONLY
================================================================================
"""

import os
import sys
import json
import time
import hashlib
import logging
import argparse
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, List, Optional, Iterator

import requests
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from scipy.stats import norm

# ==============================================================================
# 0. LOGGING & CONFIGURATION
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("V36_QUANT_ENGINE")


class GlobalConfig:
    SYMBOL = "XAUUSD"
    TIMEFRAME = "M5"
    BARS_PER_DAY = 288
    TRADING_DAYS_PER_YEAR = 252
    BARS_PER_YEAR = BARS_PER_DAY * TRADING_DAYS_PER_YEAR
    INITIAL_CAPITAL = 100_000.0
    RANDOM_STATE = 42
    N_STRATEGY_TRIALS = 10
    MIN_DATA_BARS = 1_500


class LabelConfig:
    PT_MULTIPLIER = 2.0
    SL_MULTIPLIER = 1.0
    MAX_VERTICAL_BARS = 12


class CostConfig:
    SPREAD_USD = 0.20
    SLIPPAGE_USD = 0.10
    COMMISSION_USD = 0.00

    @classmethod
    def round_turn_cost_frac(cls, price: float) -> float:
        total = cls.SPREAD_USD + cls.SLIPPAGE_USD + cls.COMMISSION_USD
        return total / max(float(price), 1e-9)


class RiskConfig:
    MAX_DRAWDOWN_KILL = 0.10
    MAX_DAILY_LOSS_KILL = 0.03
    MAX_CONSECUTIVE_LOSSES = 8
    MIN_CONFIDENCE = 0.40
    MAX_POSITION_FRAC = 1.0


class CVConfig:
    N_SPLITS = 5
    MIN_TRAIN_FRAC = 0.30
    EMBARGO_BARS = 12


# ==============================================================================
# 1. GOVERNANCE
# ==============================================================================
class ExperimentGovernance:
    def __init__(self):
        self.metadata = {
            "version": "36.0.0-PROD",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "author": "Global Head of Quantitative Risk & Trading Systems",
            "audit_status": "PENDING_EXTERNAL_AUDIT",
        }
        self.audit_trail: List[Dict[str, Any]] = []

    def log(self, event: str, details: Dict[str, Any]) -> None:
        entry = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "details": details,
        }
        self.audit_trail.append(entry)
        logger.info(f"[GOVERNANCE] {event}: {details}")


governance = ExperimentGovernance()


# ==============================================================================
# 2. DATA LAYER — FAIL CLOSED
# ==============================================================================
class DataLayer:
    @staticmethod
    def load_or_raise(
        filepath: str,
        allow_synthetic: bool = False,
        n_synthetic: int = 3_000,
    ) -> pd.DataFrame:
        governance.log("DATA_LOAD_ATTEMPT", {"source": filepath, "synthetic_ok": allow_synthetic})

        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df = df.sort_values('timestamp').reset_index(drop=True)
                df = DataLayer.validate(df)
                governance.log("DATA_LOAD_SUCCESS", {"rows": len(df)})
                return df
            except Exception as e:
                governance.log("DATA_LOAD_ERROR", {"error": str(e)})
                if not allow_synthetic:
                    raise
                logger.warning("[DATA] Load failed; falling back to synthetic (explicitly allowed).")

        if not allow_synthetic:
            raise FileNotFoundError(
                f"Data file '{filepath}' not found. Set allow_synthetic=True "
                f"only for research/testing. Production must fail closed."
            )

        logger.warning("[DATA] Generating synthetic XAUUSD feed (TESTING ONLY).")
        rng = np.random.default_rng(GlobalConfig.RANDOM_STATE)
        n = n_synthetic
        ts = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
        rets = rng.normal(0.00001, 0.0012, n)
        path = 2650.0 * np.exp(np.cumsum(rets))
        df = pd.DataFrame({
            "timestamp": ts,
            "open": path * (1 + rng.normal(0, 0.0001, n)),
            "high": path * (1 + np.abs(rng.normal(0, 0.0003, n))),
            "low": path * (1 - np.abs(rng.normal(0, 0.0003, n))),
            "close": path,
            "volume": rng.integers(100, 5000, n),
        })
        return DataLayer.validate(df)

    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        required = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if df['timestamp'].isna().any():
            raise ValueError("Null timestamps detected.")
        if df[list(required - {'timestamp'})].isna().any().any():
            raise ValueError("Null OHLCV values detected.")

        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

        bad_high = (df['high'] < df[['open', 'close']].max(axis=1))
        bad_low = (df['low'] > df[['open', 'close']].min(axis=1))
        if bad_high.any() or bad_low.any():
            raise ValueError(
                f"OHLC integrity violated: {int(bad_high.sum())} high, {int(bad_low.sum())} low."
            )

        governance.log("DATA_VALIDATION", {"status": "PASSED", "rows": len(df)})
        return df


# ==============================================================================
# 3. FEATURE ENGINEERING — CORRECTED ENTROPY & HURST
# ==============================================================================
class FeatureEngineering:
    @staticmethod
    def compute_shannon_entropy(x: np.ndarray, bins: int = 10) -> float:
        x = x[~np.isnan(x)]
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
            return float(np.clip(slope * 2.0, 0.0, 1.0))
        except Exception:
            return np.nan

    @classmethod
    def transform(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close, high, low = df['close'], df['high'], df['low']

        for lag in (1, 3, 5, 10):
            df[f'return_{lag}'] = close.pct_change(lag)

        for period in (9, 21, 50, 200):
            df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
        df['ema_spread_9_21'] = (df['ema_9'] - df['ema_21']) / close
        df['ema_slope_50'] = df['ema_50'].diff(3) / close

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-12)))

        tr = pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['volatility'] = close.pct_change().rolling(20).std()

        roll_mean = close.rolling(20).mean()
        roll_std = close.rolling(20).std() + 1e-12
        df['zscore'] = (close - roll_mean) / roll_std

        logger.info("[FEATURES] Computing rolling entropy & Hurst (bottleneck).")
        rets = close.pct_change()
        df['entropy'] = rets.rolling(100).apply(
            lambda w: cls.compute_shannon_entropy(w.values), raw=True
        )
        df['hurst'] = close.rolling(100).apply(
            lambda w: cls.compute_hurst(w.values), raw=True
        )

        governance.log("FEATURE_ENGINEERING", {"features": len(df.columns)})
        return df


# ==============================================================================
# 4. LABELING LAYER — WICK-AWARE, PESSIMISTIC TIE-BREAK
# ==============================================================================
class LabelingLayer:
    @staticmethod
    def apply_triple_barrier(
        df: pd.DataFrame,
        pt_mult: float = LabelConfig.PT_MULTIPLIER,
        sl_mult: float = LabelConfig.SL_MULTIPLIER,
        horizon: int = LabelConfig.MAX_VERTICAL_BARS,
    ) -> pd.DataFrame:
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        atr = df['atr'].values
        n = len(df)
        labels = np.full(n, np.nan, dtype=float)
        label_end = np.full(n, -1, dtype=int)

        for i in range(n - horizon):
            if not np.isfinite(atr[i]) or atr[i] <= 0:
                continue
            entry = close[i]
            upper = entry + pt_mult * atr[i]
            lower = entry - sl_mult * atr[i]
            end_j = min(i + horizon, n - 1)
            lab = 0
            for j in range(i + 1, end_j + 1):
                hit_up = high[j] >= upper
                hit_dn = low[j] <= lower
                if hit_up and hit_dn:
                    lab, end_j = -1, j
                    break
                if hit_up:
                    lab, end_j = 1, j
                    break
                if hit_dn:
                    lab, end_j = -1, j
                    break
            labels[i] = lab
            label_end[i] = end_j

        df = df.copy()
        df['target'] = labels
        df['label_end_idx'] = label_end
        dist = pd.Series(labels).value_counts(dropna=True).to_dict()
        governance.log("TRIPLE_BARRIER", {"distribution": dist})
        return df


# ==============================================================================
# 5. PURGED WALK-FORWARD CV (EXPANDING WINDOW)
# ==============================================================================
class PurgedWalkForwardCV:
    def __init__(self, n_splits=CVConfig.N_SPLITS,
                 min_train_frac=CVConfig.MIN_TRAIN_FRAC,
                 horizon=LabelConfig.MAX_VERTICAL_BARS,
                 embargo=CVConfig.EMBARGO_BARS):
        self.n_splits = n_splits
        self.min_train_frac = min_train_frac
        self.horizon = horizon
        self.embargo = max(embargo, horizon)

    def split(self, n: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if self.n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        fold_size = int((n * (1.0 - self.min_train_frac)) // self.n_splits)
        if fold_size <= 0:
            raise ValueError(f"n={n} too small for n_splits={self.n_splits}")

        initial_train = n - fold_size * self.n_splits
        for i in range(self.n_splits):
            test_start = initial_train + i * fold_size
            test_end = n if i == self.n_splits - 1 else test_start + fold_size
            train_end = test_start - self.horizon - self.embargo
            if train_end < 100:
                logger.warning(f"[CV] Skipping fold {i}: train_end={train_end} too small.")
                continue
            yield np.arange(0, train_end), np.arange(test_start, test_end)


# ==============================================================================
# 6. MODEL LAYER
# ==============================================================================
class ModelLayer:
    @staticmethod
    def build_ensemble() -> VotingClassifier:
        rs = GlobalConfig.RANDOM_STATE
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=6,
            class_weight='balanced', random_state=rs, n_jobs=-1,
        )
        et = ExtraTreesClassifier(
            n_estimators=200, max_depth=6,
            class_weight='balanced', random_state=rs, n_jobs=-1,
        )
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=rs,
        )
        return VotingClassifier(
            estimators=[('rf', rf), ('et', et), ('gb', gb)],
            voting='soft', n_jobs=-1,
        )


# ==============================================================================
# 7. TRADE SIMULATION — BARRIER-CONSISTENT PnL
# ==============================================================================
class TradeSimulator:
    @staticmethod
    def confidence_to_size(proba: np.ndarray) -> Tuple[int, float]:
        max_p = float(proba.max())
        if max_p < RiskConfig.MIN_CONFIDENCE:
            return 0, 0.0
        cls = int(proba.argmax()) - 1
        if cls == 0:
            return 0, 0.0
        size = (max_p - RiskConfig.MIN_CONFIDENCE) / (1.0 - RiskConfig.MIN_CONFIDENCE)
        return cls, float(np.clip(size, 0.0, RiskConfig.MAX_POSITION_FRAC))

    @staticmethod
    def simulate(
        df: pd.DataFrame, i: int, direction: int, size: float,
        pt_mult: float = LabelConfig.PT_MULTIPLIER,
        sl_mult: float = LabelConfig.SL_MULTIPLIER,
        horizon: int = LabelConfig.MAX_VERTICAL_BARS,
    ) -> Tuple[float, int]:
        if direction == 0 or size <= 0:
            return 0.0, i
        entry = df['close'].iat[i]
        atr_i = df['atr'].iat[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            return 0.0, i

        if direction > 0:
            tp, sl = entry + pt_mult * atr_i, entry - sl_mult * atr_i
        else:
            tp, sl = entry - pt_mult * atr_i, entry + sl_mult * atr_i

        end_max = min(i + horizon, len(df) - 1)
        for j in range(i + 1, end_max + 1):
            h, l = df['high'].iat[j], df['low'].iat[j]
            if direction > 0:
                hit_tp, hit_sl = h >= tp, l <= sl
            else:
                hit_tp, hit_sl = l <= tp, h >= sl

            if hit_tp and hit_sl:
                return size * (-sl_mult * atr_i / entry), j
            if hit_tp:
                return size * (pt_mult * atr_i / entry), j
            if hit_sl:
                return size * (-sl_mult * atr_i / entry), j

        end_close = df['close'].iat[end_max]
        ret = (end_close - entry) / entry if direction > 0 else (entry - end_close) / entry
        return size * ret, end_max


# ==============================================================================
# 8. RISK ENGINE — KILL SWITCH
# ==============================================================================
class RiskEngine:
    def __init__(self, cfg: RiskConfig = RiskConfig()):
        self.cfg = cfg
        self.tripped = False
        self.reason: Optional[str] = None
        self.consecutive_losses = 0

    def check(self, equity: float, peak_equity: float, day_start_equity: float) -> bool:
        if self.tripped:
            return False
        if peak_equity > 0 and (equity / peak_equity - 1) <= -self.cfg.MAX_DRAWDOWN_KILL:
            self.tripped, self.reason = True, "MAX_DRAWDOWN_KILL"
        elif day_start_equity > 0 and (equity / day_start_equity - 1) <= -self.cfg.MAX_DAILY_LOSS_KILL:
            self.tripped, self.reason = True, "MAX_DAILY_LOSS_KILL"
        elif self.consecutive_losses >= self.cfg.MAX_CONSECUTIVE_LOSSES:
            self.tripped, self.reason = True, "MAX_CONSECUTIVE_LOSSES"
        return not self.tripped

    def record(self, trade_return: float) -> None:
        self.consecutive_losses = self.consecutive_losses + 1 if trade_return < 0 else 0


# ==============================================================================
# 9. PERFORMANCE METRICS — TRADE-LEVEL, UNIT-CORRECT
# ==============================================================================
class PerformanceEngine:
    @staticmethod
    def trade_metrics(trade_returns: np.ndarray, n_test_bars: int) -> Dict[str, float]:
        if trade_returns.size == 0:
            return {
                "n_trades": 0, "CAGR": 0.0, "Max_Drawdown": 0.0,
                "Sharpe_Annualized": 0.0, "Sortino_Annualized": 0.0,
                "Terminal_Equity": GlobalConfig.INITIAL_CAPITAL,
                "SR_per_trade": 0.0,
            }

        equity = GlobalConfig.INITIAL_CAPITAL * np.cumprod(1 + trade_returns)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = float(dd.min())

        years = max(n_test_bars / GlobalConfig.BARS_PER_YEAR, 1e-9)
        cagr = float((equity[-1] / GlobalConfig.INITIAL_CAPITAL) ** (1 / years) - 1)

        mean_r = float(trade_returns.mean())
        std_r = float(trade_returns.std(ddof=1)) if trade_returns.size > 1 else 0.0
        sr_per_trade = mean_r / std_r if std_r > 0 else 0.0
        trades_per_year = trade_returns.size / years
        sr_annual = sr_per_trade * np.sqrt(trades_per_year)

        downside = trade_returns[trade_returns < 0]
        dstd = float(downside.std(ddof=1)) if downside.size > 1 else 1e-12
        sortino_per = mean_r / dstd if dstd > 0 else 0.0
        sortino_annual = sortino_per * np.sqrt(trades_per_year)

        return {
            "n_trades": int(trade_returns.size),
            "CAGR": cagr,
            "Max_Drawdown": max_dd,
            "Sharpe_Annualized": float(sr_annual),
            "Sortino_Annualized": float(sortino_annual),
            "Terminal_Equity": float(equity[-1]),
            "SR_per_trade": float(sr_per_trade),
        }

    @staticmethod
    def deflated_sharpe(
        sr_per_period: float, n_trials: int, T: int,
        skew: float = 0.0, kurt: float = 3.0,
    ) -> float:
        """Bailey & López de Prado (2014). sr_per_period is PER-TRADE, T = n_trades."""
        if T < 2 or n_trials < 2:
            return 0.0
        logN = np.log(n_trials)
        if logN <= 0:
            return 0.0
        expected_max = (
            np.sqrt(2 * logN)
            - (np.log(logN) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * logN))
        )
        var_sr = (1 - skew * sr_per_period + ((kurt - 1) / 4.0) * sr_per_period ** 2) / (T - 1)
        if var_sr <= 0:
            return 0.0
        z = (sr_per_period - expected_max) / np.sqrt(var_sr)
        return float(norm.cdf(z))

    @staticmethod
    def monte_carlo(
        trade_returns: np.ndarray,
        simulations: int = 1000,
        seed: int = 42,
    ) -> Dict[str, float]:
        n = trade_returns.size
        if n == 0:
            return {
                "MC_P5": GlobalConfig.INITIAL_CAPITAL,
                "MC_Median": GlobalConfig.INITIAL_CAPITAL,
                "MC_P95": GlobalConfig.INITIAL_CAPITAL,
                "P_Ruin_50": 0.0,
            }
        rng = np.random.default_rng(seed)
        block = min(5, n)
        terminals = np.empty(simulations)
        for s in range(simulations):
            sim = []
            while len(sim) < n:
                start = int(rng.integers(0, n))
                sim.extend(trade_returns[(start + k) % n] for k in range(block))
            path = np.array(sim[:n])
            terminals[s] = GlobalConfig.INITIAL_CAPITAL * np.prod(1 + path)
        return {
            "MC_P5": float(np.percentile(terminals, 5)),
            "MC_Median": float(np.median(terminals)),
            "MC_P95": float(np.percentile(terminals, 95)),
            "P_Ruin_50": float(np.mean(terminals < 0.5 * GlobalConfig.INITIAL_CAPITAL)),
        }


# ==============================================================================
# 10. API CLIENTS — FAULT-TOLERANT
# ==============================================================================
class GistStateManager:
    API = "https://api.github.com/gists/{gid}"

    def __init__(self, token: str, gist_id: str, timeout: int = 10, max_retries: int = 4):
        self.token, self.gist_id = token, gist_id
        self.timeout, self.max_retries = timeout, max_retries
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        }

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(min(2 ** attempt, 30) + float(np.random.uniform(0, 0.5)))

    def fetch(self) -> Tuple[Dict[str, Any], Optional[str]]:
        for attempt in range(self.max_retries):
            try:
                r = requests.get(
                    self.API.format(gid=self.gist_id),
                    headers=self.headers, timeout=self.timeout,
                )
                if r.status_code == 200:
                    return r.json(), r.headers.get("ETag")
                if r.status_code in (429, 500, 502, 503, 504):
                    self._backoff(attempt)
                    continue
                r.raise_for_status()
            except requests.RequestException as e:
                logger.warning(f"[GIST] fetch {attempt + 1} failed: {e}")
                self._backoff(attempt)
        raise RuntimeError("Gist fetch failed after retries")

    def atomic_write(
        self, content: str, etag: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        payload = {"files": {"state.json": {"content": content}}}
        headers = dict(self.headers)
        if etag:
            headers["If-Match"] = etag
        for attempt in range(self.max_retries):
            try:
                r = requests.patch(
                    self.API.format(gid=self.gist_id),
                    headers=headers, json=payload, timeout=self.timeout,
                )
                if r.status_code in (200, 201):
                    return r.json(), r.headers.get("ETag")
                if r.status_code in (409, 412):
                    raise RuntimeError("State conflict — refresh ETag and retry.")
                if r.status_code in (429, 500, 502, 503, 504):
                    self._backoff(attempt)
                    continue
                r.raise_for_status()
            except requests.RequestException as e:
                logger.warning(f"[GIST] write {attempt + 1} failed: {e}")
                self._backoff(attempt)
        raise RuntimeError("Gist write failed after retries")


class TelegramClient:
    def __init__(self, token: str, chat_id: str, timeout: int = 10, max_retries: int = 4):
        self.url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat_id, self.timeout, self.max_retries = chat_id, timeout, max_retries

    def send(self, text: str) -> Dict[str, Any]:
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}
        for attempt in range(self.max_retries):
            try:
                r = requests.post(self.url, json=payload, timeout=self.timeout)
                if r.status_code == 200 and r.json().get("ok"):
                    return r.json()
                if r.status_code == 429:
                    wait = int(r.json().get("parameters", {}).get("retry_after", 5))
                    time.sleep(wait + 1)
                    continue
                if 500 <= r.status_code < 600:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                r.raise_for_status()
            except requests.RequestException as e:
                logger.warning(f"[TELEGRAM] {attempt + 1} failed: {e}")
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("Telegram send failed after retries")


class OrderGuard:
    """Idempotent order dispatcher. Prevents duplicate submission on retry."""

    def __init__(self, gist: GistStateManager):
        self.gist = gist

    def submit(self, order: Dict[str, Any]) -> Dict[str, Any]:
        key = hashlib.sha256(json.dumps(order, sort_keys=True).encode()).hexdigest()
        try:
            state, etag = self.gist.fetch()
            content = json.loads(state["files"]["state.json"]["content"])
            sent = content.get("sent_orders", {})
            if key in sent:
                logger.info(f"[ORDER] Duplicate suppressed: {key[:12]}")
                return {"status": "DUPLICATE_SUPPRESSED", "key": key}
            sent[key] = {"ts": datetime.now(timezone.utc).isoformat(), "order": order}
            self.gist.atomic_write(
                json.dumps({"sent_orders": sent, "updated": sent[key]["ts"]}),
                etag=etag,
            )
            return {"status": "SUBMITTED", "key": key}
        except Exception as e:
            raise RuntimeError(f"Order submission failed: {e}")


# ==============================================================================
# 11. PIPELINE ORCHESTRATOR
# ==============================================================================
class Pipeline:
    @staticmethod
    def run(data_path: str, allow_synthetic: bool = False) -> Dict[str, Any]:
        governance.log("PIPELINE_START", {})

        raw = DataLayer.load_or_raise(data_path, allow_synthetic=allow_synthetic)
        if len(raw) < GlobalConfig.MIN_DATA_BARS:
            logger.warning(
                f"[PIPELINE] Only {len(raw)} bars — metrics below are NON-INFERENTIAL. "
                f"Require >= {GlobalConfig.MIN_DATA_BARS}."
            )

        df = FeatureEngineering.transform(raw)
        df = LabelingLayer.apply_triple_barrier(df)

        feature_cols = [
            c for c in df.columns
            if c not in {
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'target', 'label_end_idx',
            }
        ]

        df = df.dropna(subset=feature_cols + ['target']).reset_index(drop=True)
        X = df[feature_cols].values
        y_mapped = (df['target'].values + 1).astype(int)
        n = len(df)
        governance.log("PIPELINE_DATA_READY", {"rows": n, "features": len(feature_cols)})

        cv = PurgedWalkForwardCV()
        oof_idx: List[int] = []
        oof_proba: List[np.ndarray] = []
        fold_metrics: List[Dict[str, float]] = []

        for fold, (tr, te) in enumerate(cv.split(n)):
            logger.info(f"--- Fold {fold + 1}: train={len(tr)} test={len(te)} ---")
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y_mapped[tr], y_mapped[te]

            scaler = StandardScaler().fit(X_tr)
            X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

            model = ModelLayer.build_ensemble()
            model.fit(X_tr_s, y_tr)
            proba = model.predict_proba(X_te_s)
            preds = proba.argmax(axis=1)

            fold_metrics.append({
                "fold": fold + 1,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "accuracy": float(accuracy_score(y_te, preds)),
                "balanced_accuracy": float(balanced_accuracy_score(y_te, preds)),
                "f1_weighted": float(
                    f1_score(y_te, preds, average='weighted', zero_division=0)
                ),
            })
            oof_idx.extend(te.tolist())
            oof_proba.extend(proba)

        if not oof_idx:
            raise RuntimeError("CV produced no out-of-fold predictions.")

        order = np.argsort(np.array(oof_idx))
        oof_idx = [oof_idx[k] for k in order]
        oof_proba = [oof_proba[k] for k in order]

        risk = RiskEngine()
        equity = GlobalConfig.INITIAL_CAPITAL
        peak = GlobalConfig.INITIAL_CAPITAL
        day_start_equity = GlobalConfig.INITIAL_CAPITAL
        current_day = None
        trades: List[Dict[str, Any]] = []
        in_position_until = -1

        for i, proba in zip(oof_idx, oof_proba):
            if i <= in_position_until:
                continue

            bar_day = df['timestamp'].iat[i].date()
            if current_day is None or bar_day != current_day:
                current_day = bar_day
                day_start_equity = equity

            direction, size = TradeSimulator.confidence_to_size(proba)
            if direction == 0 or size <= 0:
                continue

            if not risk.check(equity, peak, day_start_equity):
                logger.error(f"[KILL SWITCH] {risk.reason} — halting trade generation.")
                break

            gross_ret, exit_j = TradeSimulator.simulate(df, i, direction, size)
            cost = size * CostConfig.round_turn_cost_frac(df['close'].iat[i])
            net_ret = gross_ret - cost

            equity *= (1 + net_ret)
            peak = max(peak, equity)
            risk.record(net_ret)

            trades.append({
                "entry_idx": int(i),
                "exit_idx": int(exit_j),
                "direction": int(direction),
                "size": float(size),
                "gross_return": float(gross_ret),
                "net_return": float(net_ret),
                "equity_after": float(equity),
            })
            in_position_until = exit_j

        trade_returns = (
            np.array([t["net_return"] for t in trades]) if trades else np.array([])
        )
        n_test_bars = len(oof_idx)

        metrics = PerformanceEngine.trade_metrics(trade_returns, n_test_bars)
        dsr = PerformanceEngine.deflated_sharpe(
            metrics["SR_per_trade"],
            GlobalConfig.N_STRATEGY_TRIALS,
            max(len(trade_returns), 2),
        )
        mc = PerformanceEngine.monte_carlo(trade_returns)

        report = {
            "metadata": governance.metadata,
            "fold_metrics": fold_metrics,
            "performance": metrics,
            "deflated_sharpe_probability": dsr,
            "monte_carlo": mc,
            "kill_switch_tripped": risk.tripped,
            "kill_switch_reason": risk.reason,
            "n_trades": len(trades),
            "audit_trail": governance.audit_trail,
        }

        out_path = "V36_QUANT_EXECUTION_REPORT.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("=" * 64)
        logger.info(f"PIPELINE COMPLETE -> {out_path}")
        logger.info(
            f"Trades: {len(trades)} | Net CAGR: {metrics['CAGR'] * 100:.2f}% | "
            f"Max DD: {metrics['Max_Drawdown'] * 100:.2f}%"
        )
        logger.info(
            f"Sharpe (ann): {metrics['Sharpe_Annualized']:.2f} | "
            f"DSR p: {dsr:.4f} | MC P5: {mc['MC_P5']:.0f}"
        )
        logger.info("=" * 64)

        tg_token = os.getenv("TELEGRAM_TOKEN")
        tg_chat = os.getenv("TELEGRAM_CHAT_ID")
        if tg_token and tg_chat:
            try:
                TelegramClient(tg_token, tg_chat).send(
                    f"<b>V36 Pipeline Complete</b>\n"
                    f"Trades: {len(trades)}\n"
                    f"CAGR: {metrics['CAGR'] * 100:.2f}% | "
                    f"Max DD: {metrics['Max_Drawdown'] * 100:.2f}%\n"
                    f"Sharpe: {metrics['Sharpe_Annualized']:.2f} | DSR p: {dsr:.4f}"
                )
            except Exception as e:
                logger.error(f"[TELEGRAM] Notification failed: {e}")

        return report


# ==============================================================================
# 12. ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAUUSD APEX V36.0")
    parser.add_argument("--data", default="xauusd_m5.csv")
    parser.add_argument(
        "--allow-synthetic", action="store_true",
        help="ONLY for testing. Production must fail closed.",
    )
    args = parser.parse_args()
    Pipeline.run(args.data, allow_synthetic=args.allow_synthetic)
