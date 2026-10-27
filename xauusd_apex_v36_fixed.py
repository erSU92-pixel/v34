"""
================================================================================
XAUUSD APEX V36.6 — FINAL
Institutional Quant Risk & Execution Engine (LIVE SIGNAL + ABLATION)

Changelog vs V36.5:
  M1.  Triple barrier: ambiguous (TP+SL same bar) → NaN, dropped from training.
  M2.  GB → HistGradientBoostingClassifier(class_weight='balanced').
  M4.  bars_per_year inferred from actual span (not 24*252 heuristic).
  M5.  Warn if trade count < 100 (DSR/MC unreliable).
  M6.  Diagnostic: position of MIN_CONFIDENCE within max_p distribution.
  M7.  Ablation framework (--ablation):
         none              → model + EMA filter (baseline)
         random            → model trained, predictions shuffled in time
                             (same confidence distribution, timing destroyed)
         no_filter         → model, no EMA filter
         random_no_filter  → shuffled predictions, no filter
================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import time
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
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from scipy.stats import norm


# ==============================================================================
# 0. LOGGING & CONFIG
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("V36_QUANT_ENGINE")


class GlobalConfig:
    SYMBOL = "XAUUSD"
    TIMEFRAME = "AUTO"
    BARS_PER_DAY = 24
    TRADING_DAYS_PER_YEAR = 252
    BARS_PER_YEAR = BARS_PER_DAY * TRADING_DAYS_PER_YEAR
    INITIAL_CAPITAL = 100_000.0
    RANDOM_STATE = 42
    N_STRATEGY_TRIALS = 10
    MIN_DATA_BARS = 2000
    MAX_LIVE_STALENESS = pd.Timedelta(hours=4)


class LabelConfig:
    PT_MULTIPLIER = 2.0
    SL_MULTIPLIER = 1.0
    MAX_VERTICAL_BARS = 12


class CostConfig:
    SPREAD_USD = 0.20
    SLIPPAGE_USD = 0.10
    COMMISSION_USD = 0.00
    STRESS_MULTIPLIER = 2.0

    @classmethod
    def round_turn_cost_frac(cls, price: float) -> float:
        return (cls.SPREAD_USD + cls.SLIPPAGE_USD + cls.COMMISSION_USD) / max(float(price), 1e-9)


class RiskConfig:
    MAX_DRAWDOWN_KILL = 0.10
    MAX_DAILY_LOSS_KILL = 0.03
    MAX_CONSECUTIVE_LOSSES = 20
    ENABLE_CONSECUTIVE_KILL = False   # off untuk backtest, nyalakan untuk live
    MIN_CONFIDENCE = 0.60
    MAX_POSITION_FRAC = 1.0


class CVConfig:
    N_SPLITS = 5
    MIN_TRAIN_FRAC = 0.30
    EMBARGO_BARS = 12


class AblationConfig:
    MODE = "none"
    RANDOM_SEED_OFFSET = 999
    VALID = ("none", "random", "no_filter", "random_no_filter")

    @classmethod
    def use_random(cls) -> bool:
        return cls.MODE in ("random", "random_no_filter")

    @classmethod
    def use_ema_filter(cls) -> bool:
        return cls.MODE not in ("no_filter", "random_no_filter")

    @classmethod
    def validate(cls) -> None:
        if cls.MODE not in cls.VALID:
            raise ValueError(f"Invalid ablation mode: {cls.MODE}. Valid: {cls.VALID}")


def validate_config() -> None:
    assert 0.5 <= RiskConfig.MIN_CONFIDENCE < 1.0
    assert 0.0 < RiskConfig.MAX_POSITION_FRAC <= 1.0
    assert LabelConfig.PT_MULTIPLIER > 0 and LabelConfig.SL_MULTIPLIER > 0
    assert LabelConfig.MAX_VERTICAL_BARS > 0
    assert CVConfig.N_SPLITS >= 2 and 0.0 < CVConfig.MIN_TRAIN_FRAC < 1.0
    assert GlobalConfig.MIN_DATA_BARS > 0
    assert GlobalConfig.N_STRATEGY_TRIALS >= 1
    AblationConfig.validate()


def seed_everything(seed: int) -> None:
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ==============================================================================
# 1. GOVERNANCE
# ==============================================================================
class ExperimentGovernance:
    def __init__(self):
        self.metadata = {
            "version": "36.6.0-FINAL",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "author": "Global Head of Quantitative Risk & Trading Systems",
            "audit_status": "PENDING_EXTERNAL_AUDIT",
        }
        self.audit_trail: List[Dict[str, Any]] = []

    def log(self, event: str, details: Dict[str, Any]) -> None:
        entry = {"time": datetime.now(timezone.utc).isoformat(),
                 "event": event, "details": details}
        self.audit_trail.append(entry)
        logger.info(f"[GOVERNANCE] {event}: {details}")


governance = ExperimentGovernance()


# ==============================================================================
# 2. DATA LAYER
# ==============================================================================
class DataLayer:
    @staticmethod
    def load_or_raise(filepath: str, allow_synthetic: bool = False,
                      n_synthetic: int = 3_000) -> pd.DataFrame:
        governance.log("DATA_LOAD_ATTEMPT",
                       {"source": filepath, "synthetic_ok": allow_synthetic})
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.sort_values("timestamp").reset_index(drop=True)
                df = DataLayer.validate(df)
                governance.log("DATA_LOAD_SUCCESS", {"rows": len(df)})
                return df
            except Exception as e:
                governance.log("DATA_LOAD_ERROR", {"error": str(e)})
                if not allow_synthetic:
                    raise

        if not allow_synthetic:
            raise FileNotFoundError(f"Data file '{filepath}' not found.")

        logger.warning("[DATA] Generating synthetic feed (TESTING).")
        rng = np.random.default_rng(GlobalConfig.RANDOM_STATE)
        n = n_synthetic
        ts = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
        rets = rng.normal(0.00001, 0.0012, n)
        path = 2650.0 * np.exp(np.cumsum(rets))
        open_p = path * (1 + rng.normal(0, 0.0001, n))
        close_p = path
        high_p = np.maximum(open_p, close_p) * (1 + np.abs(rng.normal(0, 0.0003, n)))
        low_p = np.minimum(open_p, close_p) * (1 - np.abs(rng.normal(0, 0.0003, n)))
        df = pd.DataFrame({
            "timestamp": ts, "open": open_p, "high": high_p,
            "low": low_p, "close": close_p,
            "volume": rng.integers(100, 5000, n),
        })
        return DataLayer.validate(df)

    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns: {required - set(df.columns)}")
        if df["timestamp"].isna().any():
            raise ValueError("Null timestamps.")
        df = (df.drop_duplicates(subset=["timestamp"])
                .sort_values("timestamp")
                .reset_index(drop=True))
        bad_high = df["high"] < df[["open", "close"]].max(axis=1)
        bad_low = df["low"] > df[["open", "close"]].min(axis=1)
        if bad_high.any() or bad_low.any():
            raise ValueError("OHLC integrity violated.")
        return df

    @staticmethod
    def infer_bars_per_day(df: pd.DataFrame) -> Tuple[int, pd.Timedelta]:
        diffs = df["timestamp"].diff().dropna()
        if diffs.empty:
            return GlobalConfig.BARS_PER_DAY, pd.NaT
        median_dt = diffs.median()
        if pd.isna(median_dt) or median_dt <= pd.Timedelta(0):
            return GlobalConfig.BARS_PER_DAY, median_dt
        bpd = max(1, int(round(pd.Timedelta(days=1) / median_dt)))
        return bpd, median_dt

    @staticmethod
    def infer_bars_per_year(df: pd.DataFrame) -> int:
        if len(df) < 2:
            return GlobalConfig.BARS_PER_YEAR
        span_sec = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()
        if span_sec <= 0:
            return GlobalConfig.BARS_PER_YEAR
        span_years = span_sec / (365.25 * 86400)
        return max(1, int(round(len(df) / span_years)))

    @staticmethod
    def read_source_tag(path: str = "data/source.txt") -> str:
        try:
            with open(path) as f:
                return f.read().strip() or "unknown"
        except Exception:
            return "unknown"


# ==============================================================================
# 3. FEATURE ENGINEERING
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
            return float(np.clip(slope, 0.0, 1.0))
        except Exception:
            return np.nan

    @classmethod
    def transform(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close, high, low = df["close"], df["high"], df["low"]

        for lag in (1, 3, 5, 10):
            df[f"return_{lag}"] = close.pct_change(lag)

        for period in (9, 21, 50, 200):
            df[f"ema_{period}"] = close.ewm(
                span=period, adjust=False, min_periods=period
            ).mean()

        df["ema_spread_9_21"] = (df["ema_9"] - df["ema_21"]) / close
        df["ema_slope_50"] = df["ema_50"].diff(3) / close

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-12)))

        tr = pd.concat(
            [high - low,
             (high - close.shift()).abs(),
             (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        df["volatility"] = close.pct_change().rolling(20).std()

        roll_mean = close.rolling(20).mean()
        roll_std = close.rolling(20).std() + 1e-12
        df["zscore"] = (close - roll_mean) / roll_std

        rets = close.pct_change()
        df["entropy"] = rets.rolling(100).apply(
            lambda w: cls.compute_shannon_entropy(w), raw=True)
        df["hurst"] = close.rolling(100).apply(
            lambda w: cls.compute_hurst(w), raw=True)

        return df


# ==============================================================================
# 4. LABELING
# ==============================================================================
class LabelingLayer:
    @staticmethod
    def apply_triple_barrier(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        atr = df["atr"].values
        n = len(df)

        labels = np.full(n, np.nan, dtype=float)
        label_end = np.full(n, -1, dtype=int)
        horizon = LabelConfig.MAX_VERTICAL_BARS
        pt_mult = LabelConfig.PT_MULTIPLIER
        sl_mult = LabelConfig.SL_MULTIPLIER

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
                    lab, end_j = np.nan, j
                    break
                if hit_up:
                    lab, end_j = 1, j
                    break
                if hit_dn:
                    lab, end_j = -1, j
                    break
            labels[i], label_end[i] = lab, end_j

        df = df.copy()
        df["target"] = labels
        df["label_end_idx"] = label_end
        return df


# ==============================================================================
# 5. PURGED WALK-FORWARD CV
# ==============================================================================
class PurgedWalkForwardCV:
    def __init__(self, n_splits: int = CVConfig.N_SPLITS,
                 min_train_frac: float = CVConfig.MIN_TRAIN_FRAC,
                 horizon: int = LabelConfig.MAX_VERTICAL_BARS,
                 embargo: int = CVConfig.EMBARGO_BARS):
        self.n_splits = n_splits
        self.min_train_frac = min_train_frac
        self.horizon = horizon
        self.embargo = max(embargo, horizon)

    def split(self, n: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        fold_size = int((n * (1.0 - self.min_train_frac)) // self.n_splits)
        if fold_size <= 0:
            raise ValueError("n_splits too large for data size")
        initial_train = n - fold_size * self.n_splits
        for i in range(self.n_splits):
            test_start = initial_train + i * fold_size
            test_end = n if i == self.n_splits - 1 else initial_train + (i + 1) * fold_size
            train_end = test_start - self.horizon - self.embargo
            if train_end < 100:
                continue
            yield np.arange(0, train_end), np.arange(test_start, test_end)


# ==============================================================================
# 6. MODEL
# ==============================================================================
class ModelLayer:
    @staticmethod
    def build_ensemble() -> VotingClassifier:
        rs = GlobalConfig.RANDOM_STATE
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced",
            random_state=rs, n_jobs=-1,
        )
        et = ExtraTreesClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced",
            random_state=rs, n_jobs=-1,
        )
        gb = HistGradientBoostingClassifier(
            max_iter=100, max_depth=4, class_weight="balanced",
            random_state=rs,
        )
        return VotingClassifier(
            estimators=[("rf", rf), ("et", et), ("gb", gb)],
            voting="soft", n_jobs=-1,
        )


# ==============================================================================
# 7. TRADE SIM & RISK
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
    def simulate(df: pd.DataFrame, i: int, direction: int, size: float
                 ) -> Tuple[float, int, float, float, float]:
        if direction == 0 or size <= 0:
            return 0.0, i, np.nan, np.nan, np.nan
        entry = float(df["close"].iat[i])
        atr_i = float(df["atr"].iat[i])
        if not np.isfinite(atr_i) or atr_i <= 0:
            return 0.0, i, np.nan, np.nan, np.nan

        if direction > 0:
            tp = entry + LabelConfig.PT_MULTIPLIER * atr_i
            sl = entry - LabelConfig.SL_MULTIPLIER * atr_i
        else:
            tp = entry - LabelConfig.PT_MULTIPLIER * atr_i
            sl = entry + LabelConfig.SL_MULTIPLIER * atr_i

        end_max = min(i + LabelConfig.MAX_VERTICAL_BARS, len(df) - 1)
        for j in range(i + 1, end_max + 1):
            h, l = float(df["high"].iat[j]), float(df["low"].iat[j])
            hit_tp = h >= tp if direction > 0 else l <= tp
            hit_sl = l <= sl if direction > 0 else h >= sl
            if hit_tp and hit_sl:
                return (size * (-LabelConfig.SL_MULTIPLIER * atr_i / entry),
                        j, tp, sl, sl)
            if hit_tp:
                return (size * (LabelConfig.PT_MULTIPLIER * atr_i / entry),
                        j, tp, sl, tp)
            if hit_sl:
                return (size * (-LabelConfig.SL_MULTIPLIER * atr_i / entry),
                        j, tp, sl, sl)

        end_close = float(df["close"].iat[end_max])
        ret = (end_close - entry) / entry if direction > 0 else (entry - end_close) / entry
        return size * ret, end_max, tp, sl, end_close


class RiskEngine:
    def __init__(self):
        self.tripped = False
        self.reason: Optional[str] = None
        self.consecutive_losses = 0

     def check(self, equity: float, peak: float, day_start: float) -> bool:
        if self.tripped:
            return False
        if peak > 0 and (equity / peak - 1) <= -RiskConfig.MAX_DRAWDOWN_KILL:
            self.tripped, self.reason = True, "MAX_DRAWDOWN_KILL"
        elif day_start > 0 and (equity / day_start - 1) <= -RiskConfig.MAX_DAILY_LOSS_KILL:
            self.tripped, self.reason = True, "MAX_DAILY_LOSS_KILL"
        elif (RiskConfig.ENABLE_CONSECUTIVE_KILL
              and self.consecutive_losses >= RiskConfig.MAX_CONSECUTIVE_LOSSES):
            self.tripped, self.reason = True, "MAX_CONSECUTIVE_LOSSES"
        return not self.tripped

    def record(self, net_ret: float) -> None:
        self.consecutive_losses = self.consecutive_losses + 1 if net_ret < 0 else 0


# ==============================================================================
# 8. PERFORMANCE METRICS
# ==============================================================================
class PerformanceEngine:
    @staticmethod
    def trade_metrics(trade_returns: np.ndarray, n_test_bars: int) -> Dict[str, float]:
        if trade_returns.size == 0:
            return {
                "n_trades": 0, "CAGR": 0.0, "Max_Drawdown": 0.0,
                "Sharpe_Annualized": 0.0, "Sortino_Annualized": 0.0,
                "Terminal_Equity": GlobalConfig.INITIAL_CAPITAL,
                "SR_per_trade": 0.0, "WinRate": 0.0,
                "ProfitFactor": 0.0, "Avg_Return": 0.0,
            }

        equity = GlobalConfig.INITIAL_CAPITAL * np.cumprod(1 + trade_returns)
        peak = np.maximum.accumulate(equity)
        max_dd = float(((equity - peak) / peak).min())
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

        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        win_rate = float(wins.size / trade_returns.size)
        profit_factor = (
            float(wins.sum() / abs(losses.sum()))
            if losses.size > 0 and losses.sum() != 0 else float("inf")
        )

        return {
            "n_trades": int(trade_returns.size),
            "CAGR": cagr,
            "Max_Drawdown": max_dd,
            "Sharpe_Annualized": float(sr_annual),
            "Sortino_Annualized": float(sortino_per * np.sqrt(trades_per_year)),
            "Terminal_Equity": float(equity[-1]),
            "SR_per_trade": float(sr_per_trade),
            "WinRate": win_rate,
            "ProfitFactor": profit_factor,
            "Avg_Return": mean_r,
        }

    @staticmethod
    def deflated_sharpe(sr: float, n_trials: int, T: int,
                        skew: float = 0.0, kurt: float = 3.0) -> float:
        if T < 2 or n_trials < 2:
            return 0.0
        logN = np.log(n_trials)
        if logN <= 0:
            return 0.0
        expected_max = (np.sqrt(2 * logN)
                        - (np.log(logN) + np.log(4 * np.pi))
                        / (2 * np.sqrt(2 * logN)))
        var_sr = (1 - skew * sr + ((kurt - 1) / 4.0) * sr ** 2) / (T - 1)
        if var_sr <= 0:
            return 0.0
        return float(norm.cdf((sr - expected_max) / np.sqrt(var_sr)))

    @staticmethod
    def monte_carlo(trade_returns: np.ndarray, simulations: int = 1000,
                    seed: int = 42) -> Dict[str, float]:
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
            sim: List[float] = []
            while len(sim) < n:
                start = int(rng.integers(0, n))
                sim.extend(trade_returns[(start + k) % n] for k in range(block))
            terminals[s] = GlobalConfig.INITIAL_CAPITAL * np.prod(1 + np.array(sim[:n]))
        return {
            "MC_P5": float(np.percentile(terminals, 5)),
            "MC_Median": float(np.median(terminals)),
            "MC_P95": float(np.percentile(terminals, 95)),
            "P_Ruin_50": float(np.mean(terminals < 0.5 * GlobalConfig.INITIAL_CAPITAL)),
        }


# ==============================================================================
# 9. TELEGRAM
# ==============================================================================
class TelegramClient:
    def __init__(self, token: str, chat_id: str, timeout: int = 10,
                 max_retries: int = 4):
        self.url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat_id = chat_id
        self.timeout = timeout
        self.max_retries = max_retries

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
                logger.warning(f"[TELEGRAM] attempt {attempt + 1} failed: {e}")
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("Telegram send failed after retries")


# ==============================================================================
# 10. PIPELINE
# ==============================================================================
class Pipeline:
    @staticmethod
    def run(data_path: str, allow_synthetic: bool = False,
            ablation: str = "none") -> Dict[str, Any]:
        AblationConfig.MODE = ablation
        validate_config()
        seed_everything(GlobalConfig.RANDOM_STATE)
        governance.log("PIPELINE_START", {"ablation": AblationConfig.MODE})
        logger.info(f"[ABLATION] mode={AblationConfig.MODE} "
                    f"(use_random={AblationConfig.use_random()}, "
                    f"use_ema_filter={AblationConfig.use_ema_filter()})")

        # ---- Load & validate ------------------------------------------------
        raw = DataLayer.load_or_raise(data_path, allow_synthetic=allow_synthetic)

        if len(raw) < GlobalConfig.MIN_DATA_BARS:
            raise ValueError(
                f"Data too short: {len(raw)} < {GlobalConfig.MIN_DATA_BARS} bars."
            )

        bars_per_day, median_dt = DataLayer.infer_bars_per_day(raw)
        bars_per_year = DataLayer.infer_bars_per_year(raw)
        GlobalConfig.BARS_PER_DAY = bars_per_day
        GlobalConfig.BARS_PER_YEAR = bars_per_year
        span = raw["timestamp"].iloc[-1] - raw["timestamp"].iloc[0]
        source_tag = DataLayer.read_source_tag()
        logger.info(
            f"[DATA] rows={len(raw)} | source={source_tag} | "
            f"median_dt={median_dt} | bars/day={bars_per_day} | "
            f"bars/year={bars_per_year} (span={span})"
        )

        # ---- Features (all rows) -------------------------------------------
        df_full = FeatureEngineering.transform(raw)

        # ---- Labels --------------------------------------------------------
        df = LabelingLayer.apply_triple_barrier(df_full)
        feature_cols = [
            c for c in df.columns
            if c not in {"timestamp", "open", "high", "low", "close",
                         "volume", "target", "label_end_idx"}
        ]
        n_with_features = df_full.dropna(subset=feature_cols).shape[0]
        df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
        logger.info(
            f"[DIAG] rows_with_features={n_with_features}, "
            f"labeled_trainable={len(df)}, "
            f"dropped={n_with_features - len(df)} "
            f"(ambiguous+unlabeled)"
        )

        X = df[feature_cols].values
        y_mapped = (df["target"].values + 1).astype(int)
        n = len(df)

        label_dist = (pd.Series(y_mapped)
                      .value_counts(normalize=True)
                      .sort_index().round(3).to_dict())
        logger.info(f"[DIAG] label distribution: {label_dist}")

        # ---- CV ------------------------------------------------------------
        cv = PurgedWalkForwardCV()
        splits = list(cv.split(n))
        fold_of = np.full(n, -1, dtype=int)
        for fold, (_tr, te) in enumerate(splits):
            fold_of[te] = fold

        oof_idx: List[int] = []
        oof_proba: List[np.ndarray] = []
        fold_metrics: List[Dict[str, Any]] = []

        for fold, (tr, te) in enumerate(splits):
            logger.info(f"--- Fold {fold + 1}: train={len(tr)} test={len(te)} ---")
            scaler = StandardScaler().fit(X[tr])
            X_tr_s = scaler.transform(X[tr])
            X_te_s = scaler.transform(X[te])

            model = ModelLayer.build_ensemble()
            model.fit(X_tr_s, y_mapped[tr])
            proba = model.predict_proba(X_te_s)
            preds = proba.argmax(axis=1)

            fm = {
                "fold": fold + 1,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "accuracy": float(accuracy_score(y_mapped[te], preds)),
                "balanced_accuracy": float(balanced_accuracy_score(y_mapped[te], preds)),
            }
            fold_metrics.append(fm)
            logger.info(
                f"[DIAG] Fold {fm['fold']}: acc={fm['accuracy']:.3f} "
                f"bal_acc={fm['balanced_accuracy']:.3f}"
            )

            # M7: shuffle predictions in time (same confidence, no timing skill)
            if AblationConfig.use_random():
                shuffle_rng = np.random.default_rng(
                    GlobalConfig.RANDOM_STATE + AblationConfig.RANDOM_SEED_OFFSET + fold
                )
                proba = proba[shuffle_rng.permutation(len(proba))]
                logger.info(f"[ABLATION] Fold {fold + 1}: predictions shuffled.")

            oof_idx.extend(te.tolist())
            oof_proba.extend(proba)

        if not oof_idx:
            raise RuntimeError("CV produced no predictions.")

        order = np.argsort(np.array(oof_idx))
        oof_idx = [oof_idx[k] for k in order]
        oof_proba = [oof_proba[k] for k in order]

        # ---- Prob diagnostics ----------------------------------------------
        proba_arr = np.asarray(oof_proba)
        argmax_counts = np.bincount(proba_arr.argmax(axis=1), minlength=3).tolist()
        max_p = proba_arr.max(axis=1)
        max_p_q = np.quantile(max_p, [0.5, 0.75, 0.90, 0.95]).round(4).tolist()
        logger.info(f"[DIAG] argmax counts [down,neutral,up]={argmax_counts}")
        logger.info(f"[DIAG] max_p quantiles [p50,p75,p90,p95]={max_p_q}")

        threshold_pct = float(np.mean(max_p < RiskConfig.MIN_CONFIDENCE)) * 100
        logger.info(
            f"[DIAG] MIN_CONFIDENCE={RiskConfig.MIN_CONFIDENCE:.2f} "
            f"rejects p{threshold_pct:.0f} of max_p distribution"
        )
        if threshold_pct > 85:
            logger.warning(
                f"[WARN] Threshold rejects {threshold_pct:.0f}% of bars. "
                f"Probabilities poorly calibrated — consider calibration before tuning."
            )

        # ---- Backtest ------------------------------------------------------
        risk = RiskEngine()
        equity = GlobalConfig.INITIAL_CAPITAL
        peak = GlobalConfig.INITIAL_CAPITAL
        day_start = GlobalConfig.INITIAL_CAPITAL
        current_day = None
        trades: List[Dict[str, Any]] = []
        in_pos_until = -1
        n_below_conf = n_ema_filtered = n_risk_blocked = 0

        for i, proba in zip(oof_idx, oof_proba):
            bar_day = df["timestamp"].iat[i].date()
            if current_day != bar_day:
                current_day = bar_day
                day_start = equity

            if i <= in_pos_until:
                continue

            direction, size = TradeSimulator.confidence_to_size(proba)
            if direction == 0 or size <= 0:
                n_below_conf += 1
                continue

            price = float(df["close"].iat[i])
            ema_200 = float(df["ema_200"].iat[i])

            if AblationConfig.use_ema_filter():
                if direction == 1 and price < ema_200:
                    n_ema_filtered += 1
                    continue
                if direction == -1 and price > ema_200:
                    n_ema_filtered += 1
                    continue

            if not risk.check(equity, peak, day_start):
                logger.error(f"[KILL SWITCH] {risk.reason} — halting.")
                n_risk_blocked += 1
                break

            gross_ret, exit_j, tp, sl, exit_price = TradeSimulator.simulate(
                df, i, direction, size
            )
            cost_ret = size * CostConfig.round_turn_cost_frac(price)
            net_ret = gross_ret - cost_ret

            equity *= (1 + net_ret)
            peak = max(peak, equity)
            risk.record(net_ret)

            trades.append({
                "entry_idx": int(i),
                "exit_idx": int(exit_j),
                "fold": int(fold_of[i]) if fold_of[i] >= 0 else -1,
                "direction": int(direction),
                "size": float(size),
                "confidence": float(proba.max()),
                "entry_price": price,
                "tp": float(tp),
                "sl": float(sl),
                "exit_price": float(exit_price),
                "atr": float(df["atr"].iat[i]),
                "gross_return": float(gross_ret),
                "cost_return": float(cost_ret),
                "net_return": float(net_ret),
                "equity_after": float(equity),
            })
            in_pos_until = exit_j

        logger.info(
            f"[DIAG] funnel: oof_bars={len(oof_idx)}, below_conf={n_below_conf}, "
            f"ema_filtered={n_ema_filtered}, risk_blocked={n_risk_blocked}, "
            f"traded={len(trades)}"
        )

        if 0 < len(trades) < 100:
            logger.warning(
                f"[WARN] Only {len(trades)} trades. DSR / Monte Carlo "
                f"unreliable below ~100 trades."
            )

        trade_returns = (np.array([t["net_return"] for t in trades])
                         if trades else np.array([]))
        n_test_bars = len(oof_idx)

        metrics = PerformanceEngine.trade_metrics(trade_returns, n_test_bars)
        dsr_prob = PerformanceEngine.deflated_sharpe(
            metrics["SR_per_trade"], GlobalConfig.N_STRATEGY_TRIALS,
            max(len(trade_returns), 2),
        )
        mc = PerformanceEngine.monte_carlo(trade_returns)

        stress_metrics: Dict[str, Any] = {}
        if trades:
            stress_rets = np.array([
                t["gross_return"] - CostConfig.STRESS_MULTIPLIER * t["cost_return"]
                for t in trades
            ])
            stress_metrics = PerformanceEngine.trade_metrics(stress_rets, n_test_bars)

        report: Dict[str, Any] = {
            "metadata": governance.metadata,
            "ablation": {
                "mode": AblationConfig.MODE,
                "use_random": AblationConfig.use_random(),
                "use_ema_filter": AblationConfig.use_ema_filter(),
            },
            "data": {
                "path": data_path,
                "source": source_tag,
                "rows": int(len(raw)),
                "median_dt": str(median_dt),
                "bars_per_day": int(bars_per_day),
                "bars_per_year": int(bars_per_year),
                "span": str(span),
            },
            "label_distribution": label_dist,
            "fold_metrics": fold_metrics,
            "performance": metrics,
            "deflated_sharpe_probability": dsr_prob,
            "deflated_sharpe_p_value": 1.0 - dsr_prob,
            "monte_carlo": mc,
            "cost_stress_2x": stress_metrics,
            "kill_switch_tripped": risk.tripped,
            "kill_switch_reason": risk.reason,
            "n_trades": len(trades),
            "signal_funnel": {
                "oof_bars": len(oof_idx),
                "below_conf": n_below_conf,
                "ema_filtered": n_ema_filtered,
                "risk_blocked": n_risk_blocked,
                "traded": len(trades),
            },
            "threshold_diagnostic": {
                "min_confidence": RiskConfig.MIN_CONFIDENCE,
                "pct_rejected": threshold_pct,
                "max_p_quantiles": dict(zip(["p50", "p75", "p90", "p95"], max_p_q)),
            },
            "audit_trail": governance.audit_trail,
        }

        # ---- Live signal ---------------------------------------------------
        if AblationConfig.use_random():
            logger.info("[LIVE] Skipped in ablation mode (shuffled predictions).")
            report["live_signal"] = {"status": "skipped",
                                     "reason": "ablation_random"}
        else:
            try:
                live_signal_info = Pipeline._live_signal(
                    df=df, df_full=df_full, feature_cols=feature_cols,
                    X=X, y_mapped=y_mapped,
                )
                report["live_signal"] = live_signal_info
            except Exception as e:
                logger.exception(f"[LIVE] failed: {e}")
                report["live_signal"] = {"error": str(e)}

        # ---- Persist -------------------------------------------------------
        out_name = f"V36_QUANT_EXECUTION_REPORT_{AblationConfig.MODE}.json"
        with open(out_name, "w") as f:
            json.dump(report, f, indent=2, default=str)
        with open("V36_QUANT_EXECUTION_REPORT.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"[REPORT] written: {out_name}")

        # ---- Summary -------------------------------------------------------
        logger.info("=" * 64)
        logger.info(
            f"[{AblationConfig.MODE.upper()}] Trades: {len(trades)} | "
            f"CAGR: {metrics['CAGR'] * 100:.2f}% | "
            f"Max DD: {metrics['Max_Drawdown'] * 100:.2f}%"
        )
        logger.info(
            f"Sharpe(ann): {metrics['Sharpe_Annualized']:.2f} | "
            f"WinRate: {metrics['WinRate'] * 100:.1f}% | "
            f"PF: {metrics['ProfitFactor']:.2f}"
        )
        logger.info(
            f"DSR prob: {dsr_prob:.4f} (p_value={1 - dsr_prob:.4f}) | "
            f"MC P5: {mc['MC_P5']:.0f}"
        )
        if stress_metrics:
            logger.info(
                f"[STRESS 2x] Trades: {stress_metrics['n_trades']} | "
                f"CAGR: {stress_metrics['CAGR'] * 100:.2f}% | "
                f"Sharpe: {stress_metrics['Sharpe_Annualized']:.2f}"
            )
        logger.info("=" * 64)

        return report

    # ------------------------------------------------------------------
    @staticmethod
    def _live_signal(df: pd.DataFrame, df_full: pd.DataFrame,
                     feature_cols: List[str], X: np.ndarray,
                     y_mapped: np.ndarray) -> Dict[str, Any]:
        logger.info("[LIVE] Predicting on true last raw bar...")

        last_row = df_full.iloc[[-1]]
        if last_row[feature_cols].isna().any().any():
            logger.warning("[LIVE] last raw row has NaN features; abort.")
            return {"status": "skipped", "reason": "nan_features"}

        last_ts = last_row["timestamp"].iat[0]
        age = pd.Timestamp.now(tz="UTC") - last_ts
        if age > GlobalConfig.MAX_LIVE_STALENESS:
            logger.warning(f"[LIVE] stale: last={last_ts} age={age}; abort.")
            return {
                "status": "skipped", "reason": "stale",
                "last_ts": str(last_ts), "age": str(age),
            }

        scaler_full = StandardScaler().fit(X)
        model_full = ModelLayer.build_ensemble()
        model_full.fit(scaler_full.transform(X), y_mapped)

        latest_features = last_row[feature_cols].values
        latest_proba = model_full.predict_proba(
            scaler_full.transform(latest_features)
        )[0]

        raw_direction, _ = TradeSimulator.confidence_to_size(latest_proba)
        price = float(last_row["close"].iat[0])
        ema_200 = float(last_row["ema_200"].iat[0])
        atr_last = float(last_row["atr"].iat[0])
        conf_pct = float(latest_proba.max()) * 100.0

        direction = raw_direction
        if direction == 1 and price <= ema_200:
            direction = 0
        if direction == -1 and price >= ema_200:
            direction = 0

        sl = tp = None
        if direction == 1:
            sl = price - LabelConfig.SL_MULTIPLIER * atr_last
            tp = price + LabelConfig.PT_MULTIPLIER * atr_last
            signal_text = "🟢 BUY (LONG)"
        elif direction == -1:
            sl = price + LabelConfig.SL_MULTIPLIER * atr_last
            tp = price - LabelConfig.PT_MULTIPLIER * atr_last
            signal_text = "🔴 SELL (SHORT)"
        else:
            signal_text = "⏸️ NEUTRAL"

        logger.info(
            f"[LIVE] ts={last_ts} close={price:.2f} ema200={ema_200:.2f} "
            f"raw_dir={raw_direction} final_dir={direction} "
            f"conf={conf_pct:.1f}% atr={atr_last:.2f} age={age}"
        )

        result: Dict[str, Any] = {
            "status": "ok",
            "signal": signal_text,
            "direction": int(direction),
            "confidence_pct": conf_pct,
            "price": price,
            "ema_200": ema_200,
            "atr": atr_last,
            "sl": sl, "tp": tp,
            "last_ts": str(last_ts),
            "age_seconds": float(age.total_seconds()),
        }

        tg_token = os.getenv("TELEGRAM_TOKEN")
        tg_chat = os.getenv("TELEGRAM_CHAT_ID")
        if tg_token and tg_chat and direction != 0:
            try:
                msg = (
                    f"<b>🚨 XAUUSD LIVE SIGNAL</b>\n\n"
                    f"Sinyal: <b>{signal_text}</b>\n"
                    f"Entry: {price:.2f}\n"
                    f"SL: {sl:.2f}\n"
                    f"TP: {tp:.2f}\n"
                    f"Confidence: {conf_pct:.1f}%\n"
                    f"ATR: {atr_last:.2f}\n"
                    f"EMA200: {ema_200:.2f}\n"
                    f"Bar: {last_ts.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                    f"<i>Max 1-2% risk per trade.</i>"
                )
                TelegramClient(tg_token, tg_chat).send(msg)
                result["telegram_sent"] = True
                logger.info("[TELEGRAM] sent.")
            except Exception as e:
                result["telegram_sent"] = False
                result["telegram_error"] = str(e)
                logger.error(f"[TELEGRAM] send failed: {e}")
        else:
            result["telegram_sent"] = False
            if direction == 0:
                logger.info("[TELEGRAM] neutral/filtered — not sent.")
            else:
                logger.info("[TELEGRAM] missing credentials — not sent.")

        return result


# ==============================================================================
# 11. ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAUUSD APEX V36.6 (final)")
    parser.add_argument("--data", default="data/xauusd_h1.csv")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument(
        "--ablation",
        choices=list(AblationConfig.VALID),
        default="none",
        help="Ablation mode for controlled experiments.",
    )
    args = parser.parse_args()
    Pipeline.run(
        args.data,
        allow_synthetic=args.allow_synthetic,
        ablation=args.ablation,
    )
