"""
================================================================================
XAUUSD APEX V36.4.1 — FIXED
Institutional Quant Risk & Execution Engine (LIVE SIGNAL)

Fixes vs V36.4.0 (see inline "# FIX Fn" comments):
  F1.  Runtime inference of bars/day & bars/year from actual bar spacing
       (was hard-coded H1=24/day, wrong if data is M5).
  F2.  compute_hurst: removed erroneous "* 2.0" scaling (slope IS H).
  F3.  EWM warmup: added min_periods so EMA values are not under-warmed.
  F4.  Live signal: predict on the TRUE last raw bar, not the 12-bar-stale
       row that survived dropna(subset=['target']).
  F5.  Telegram gate: respects EMA200 filter before sending; includes SL/TP.
  F6.  Daily-loss rollover: reset day_start BEFORE skipping in-position bars.
  F7.  DSR: log field renamed to probability, and 1-prob reported as p-value.
  F8.  Cost stress test (2x) added to report & log.
  F9.  Trade audit: per-trade fold, confidence, entry/exit, TP/SL stored.
  F10. Diagnostics: label dist, argmax counts, max_p quantiles, fold metrics,
       signal funnel, median bar spacing, data span.
  F11. MIN_DATA_BARS validation enforced.
================================================================================
"""

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
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
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
    TIMEFRAME = "AUTO"          # FIX F1: was "H1"; now inferred from data
    BARS_PER_DAY = 24            # FIX F1: overridden at runtime
    TRADING_DAYS_PER_YEAR = 252
    BARS_PER_YEAR = BARS_PER_DAY * TRADING_DAYS_PER_YEAR   # FIX F1: overridden
    INITIAL_CAPITAL = 100_000.0
    RANDOM_STATE = 42
    N_STRATEGY_TRIALS = 10       # bump this if you tune threshold/params
    MIN_DATA_BARS = 2000


class LabelConfig:
    PT_MULTIPLIER = 2.0
    SL_MULTIPLIER = 1.0
    MAX_VERTICAL_BARS = 12


class CostConfig:
    SPREAD_USD = 0.20
    SLIPPAGE_USD = 0.10
    COMMISSION_USD = 0.00
    STRESS_MULTIPLIER = 2.0      # FIX F8: 2x cost stress test

    @classmethod
    def round_turn_cost_frac(cls, price: float) -> float:
        return (cls.SPREAD_USD + cls.SLIPPAGE_USD + cls.COMMISSION_USD) / max(float(price), 1e-9)


class RiskConfig:
    MAX_DRAWDOWN_KILL = 0.10
    MAX_DAILY_LOSS_KILL = 0.03
    MAX_CONSECUTIVE_LOSSES = 10
    MIN_CONFIDENCE = 0.60
    MAX_POSITION_FRAC = 1.0      # consider lowering to 0.1-0.2 in production


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
            "version": "36.4.1-FIXED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "author": "Global Head of Quantitative Risk & Trading Systems",
            "audit_status": "PENDING_EXTERNAL_AUDIT",
        }
        self.audit_trail: List[Dict[str, Any]] = []

    def log(self, event: str, details: Dict[str, Any]) -> None:
        entry = {"time": datetime.now(timezone.utc).isoformat(), "event": event, "details": details}
        self.audit_trail.append(entry)
        logger.info(f"[GOVERNANCE] {event}: {details}")


governance = ExperimentGovernance()


# ==============================================================================
# 2. DATA LAYER
# ==============================================================================
class DataLayer:
    @staticmethod
    def load_or_raise(filepath: str, allow_synthetic: bool = False, n_synthetic: int = 3_000) -> pd.DataFrame:
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
        df = pd.DataFrame({"timestamp": ts, "open": open_p, "high": high_p,
                           "low": low_p, "close": close_p,
                           "volume": rng.integers(100, 5000, n)})
        return DataLayer.validate(df)

    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        required = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns: {required - set(df.columns)}")
        if df['timestamp'].isna().any():
            raise ValueError("Null timestamps.")
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        bad_high = (df['high'] < df[['open', 'close']].max(axis=1))
        bad_low = (df['low'] > df[['open', 'close']].min(axis=1))
        if bad_high.any() or bad_low.any():
            raise ValueError("OHLC integrity violated.")
        return df

    # FIX F1: infer bar spacing & annualization from the data itself.
    @staticmethod
    def infer_bars_per_day(df: pd.DataFrame) -> Tuple[int, pd.Timedelta]:
        diffs = df['timestamp'].diff().dropna()
        if diffs.empty:
            return GlobalConfig.BARS_PER_DAY, pd.NaT
        median_dt = diffs.median()
        if pd.isna(median_dt) or median_dt <= pd.Timedelta(0):
            return GlobalConfig.BARS_PER_DAY, median_dt
        bars_per_day = max(1, int(round(pd.Timedelta(days=1) / median_dt)))
        return bars_per_day, median_dt


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
            # FIX F2: slope already equals Hurst exponent; no "* 2.0".
            return float(np.clip(slope, 0.0, 1.0))
        except Exception:
            return np.nan

    @classmethod
    def transform(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close, high, low = df['close'], df['high'], df['low']
        for lag in (1, 3, 5, 10):
            df[f'return_{lag}'] = close.pct_change(lag)

        # FIX F3: min_periods ensures EWM is fully warmed before use.
        for period in (9, 21, 50, 200):
            df[f'ema_{period}'] = close.ewm(span=period, adjust=False, min_periods=period).mean()

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

        rets = close.pct_change()
        df['entropy'] = rets.rolling(100).apply(lambda w: cls.compute_shannon_entropy(w), raw=True)
        df['hurst'] = close.rolling(100).apply(lambda w: cls.compute_hurst(w), raw=True)
        return df


# ==============================================================================
# 4. LABELING LAYER
# ==============================================================================
class LabelingLayer:
    @staticmethod
    def apply_triple_barrier(df: pd.DataFrame) -> pd.DataFrame:
        close, high, low, atr = df['close'].values, df['high'].values, df['low'].values, df['atr'].values
        n = len(df)
        labels = np.full(n, np.nan, dtype=float)
        label_end = np.full(n, -1, dtype=int)
        horizon = LabelConfig.MAX_VERTICAL_BARS
        pt_mult, sl_mult = LabelConfig.PT_MULTIPLIER, LabelConfig.SL_MULTIPLIER

        for i in range(n - horizon):
            if not np.isfinite(atr[i]) or atr[i] <= 0:
                continue
            entry = close[i]
            upper = close[i] + pt_mult * atr[i]
            lower = close[i] - sl_mult * atr[i]
            end_j, lab = min(i + horizon, n - 1), 0
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
            labels[i], label_end[i] = lab, end_j

        df = df.copy()
        df['target'] = labels
        df['label_end_idx'] = label_end
        return df


# ==============================================================================
# 5. PURGED WALK-FORWARD CV
# ==============================================================================
class PurgedWalkForwardCV:
    def __init__(self, n_splits=CVConfig.N_SPLITS, min_train_frac=CVConfig.MIN_TRAIN_FRAC,
                 horizon=LabelConfig.MAX_VERTICAL_BARS, embargo=CVConfig.EMBARGO_BARS):
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
# 6. MODEL LAYER
# ==============================================================================
class ModelLayer:
    @staticmethod
    def build_ensemble() -> VotingClassifier:
        rs = GlobalConfig.RANDOM_STATE
        rf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                    class_weight='balanced', random_state=rs, n_jobs=-1)
        et = ExtraTreesClassifier(n_estimators=200, max_depth=6,
                                  class_weight='balanced', random_state=rs, n_jobs=-1)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=rs)
        return VotingClassifier(estimators=[('rf', rf), ('et', et), ('gb', gb)],
                                voting='soft', n_jobs=-1)


# ==============================================================================
# 7. TRADE SIMULATION & RISK
# ==============================================================================
class TradeSimulator:
    @staticmethod
    def confidence_to_size(proba: np.ndarray) -> Tuple[int, float]:
        max_p = float(proba.max())
        if max_p < RiskConfig.MIN_CONFIDENCE:
            return 0, 0.0
        cls = int(proba.argmax()) - 1   # 0->-1, 1->0, 2->+1
        if cls == 0:
            return 0, 0.0
        size = (max_p - RiskConfig.MIN_CONFIDENCE) / (1.0 - RiskConfig.MIN_CONFIDENCE)
        return cls, float(np.clip(size, 0.0, RiskConfig.MAX_POSITION_FRAC))

    # FIX F9: return (ret, exit_idx, tp, sl, exit_price) for audit trail.
    @staticmethod
    def simulate(df: pd.DataFrame, i: int, direction: int, size: float
                 ) -> Tuple[float, int, float, float, float]:
        if direction == 0 or size <= 0:
            return 0.0, i, np.nan, np.nan, np.nan
        entry = df['close'].iat[i]
        atr_i = df['atr'].iat[i]
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
            h, l = df['high'].iat[j], df['low'].iat[j]
            hit_tp = h >= tp if direction > 0 else l <= tp
            hit_sl = l <= sl if direction > 0 else h >= sl

            if hit_tp and hit_sl:
                return (size * (-LabelConfig.SL_MULTIPLIER * atr_i / entry), j, tp, sl, sl)
            if hit_tp:
                return (size * (LabelConfig.PT_MULTIPLIER * atr_i / entry), j, tp, sl, tp)
            if hit_sl:
                return (size * (-LabelConfig.SL_MULTIPLIER * atr_i / entry), j, tp, sl, sl)

        end_close = df['close'].iat[end_max]
        ret = (end_close - entry) / entry if direction > 0 else (entry - end_close) / entry
        return (size * ret, end_max, tp, sl, end_close)


class RiskEngine:
    def __init__(self):
        self.tripped = False
        self.reason = None
        self.consecutive_losses = 0

    def check(self, equity: float, peak: float, day_start: float) -> bool:
        if self.tripped:
            return False
        if peak > 0 and (equity / peak - 1) <= -RiskConfig.MAX_DRAWDOWN_KILL:
            self.tripped, self.reason = True, "MAX_DRAWDOWN_KILL"
        elif day_start > 0 and (equity / day_start - 1) <= -RiskConfig.MAX_DAILY_LOSS_KILL:
            self.tripped, self.reason = True, "MAX_DAILY_LOSS_KILL"
        elif self.consecutive_losses >= RiskConfig.MAX_CONSECUTIVE_LOSSES:
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
                "Terminal_Equity": GlobalConfig.INITIAL_CAPITAL, "SR_per_trade": 0.0,
            }
        equity = GlobalConfig.INITIAL_CAPITAL * np.cumprod(1 + trade_returns)
        peak = np.maximum.accumulate(equity)
        max_dd = float(((equity - peak) / peak).min())
        years = max(n_test_bars / GlobalConfig.BARS_PER_YEAR, 1e-9)   # FIX F1: uses inferred value
        cagr = float((equity[-1] / GlobalConfig.INITIAL_CAPITAL) ** (1 / years) - 1)
        mean_r = float(trade_returns.mean())
        std_r = float(trade_returns.std(ddof=1)) if trade_returns.size > 1 else 0.0
        sr_per_trade = mean_r / std_r if std_r > 0 else 0.0
        trades_per_year = trade_returns.size / years
        sr_annual = sr_per_trade * np.sqrt(trades_per_year)
        downside = trade_returns[trade_returns < 0]
        dstd = float(downside.std(ddof=1)) if downside.size > 1 else 1e-12
        sortino_per = mean_r / dstd if dstd > 0 else 0.0
        return {
            "n_trades": int(trade_returns.size),
            "CAGR": cagr,
            "Max_Drawdown": max_dd,
            "Sharpe_Annualized": float(sr_annual),
            "Sortino_Annualized": float(sortino_per * np.sqrt(trades_per_year)),
            "Terminal_Equity": float(equity[-1]),
            "SR_per_trade": float(sr_per_trade),
        }

    @staticmethod
    def deflated_sharpe(sr: float, n_trials: int, T: int,
                        skew: float = 0.0, kurt: float = 3.0) -> float:
        """Returns P(SR > expected_max under null).  Low value = no skill."""
        if T < 2 or n_trials < 2:
            return 0.0
        logN = np.log(n_trials)
        if logN <= 0:
            return 0.0
        expected_max = np.sqrt(2 * logN) - (np.log(logN) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * logN))
        var_sr = (1 - skew * sr + ((kurt - 1) / 4.0) * sr ** 2) / (T - 1)
        if var_sr <= 0:
            return 0.0
        return float(norm.cdf((sr - expected_max) / np.sqrt(var_sr)))

    @staticmethod
    def monte_carlo(trade_returns: np.ndarray, simulations: int = 1000, seed: int = 42) -> Dict[str, float]:
        n = trade_returns.size
        if n == 0:
            return {"MC_P5": GlobalConfig.INITIAL_CAPITAL,
                    "MC_Median": GlobalConfig.INITIAL_CAPITAL,
                    "MC_P95": GlobalConfig.INITIAL_CAPITAL,
                    "P_Ruin_50": 0.0}
        rng = np.random.default_rng(seed)
        block = min(5, n)
        terminals = np.empty(simulations)
        for s in range(simulations):
            sim = []
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
# 9. API CLIENTS
# ==============================================================================
class TelegramClient:
    def __init__(self, token: str, chat_id: str, timeout: int = 10, max_retries: int = 4):
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
                logger.warning(f"[TELEGRAM] {attempt + 1} failed: {e}")
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("Telegram send failed after retries")


# ==============================================================================
# 10. PIPELINE ORCHESTRATOR
# ==============================================================================
class Pipeline:
    @staticmethod
    def run(data_path: str, allow_synthetic: bool = False) -> Dict[str, Any]:
        governance.log("PIPELINE_START", {})

        # ---------------------------------------------------------------------
        # 10.1 Load + validate
        # ---------------------------------------------------------------------
        raw = DataLayer.load_or_raise(data_path, allow_synthetic=allow_synthetic)

        # FIX F11: enforce MIN_DATA_BARS before doing anything else.
        if len(raw) < GlobalConfig.MIN_DATA_BARS:
            raise ValueError(
                f"Data too short: {len(raw)} < {GlobalConfig.MIN_DATA_BARS} bars."
            )

        # FIX F1: infer bar spacing from data and log it.
        bars_per_day, median_dt = DataLayer.infer_bars_per_day(raw)
        GlobalConfig.BARS_PER_DAY = bars_per_day
        GlobalConfig.BARS_PER_YEAR = bars_per_day * GlobalConfig.TRADING_DAYS_PER_YEAR
        span = raw['timestamp'].iloc[-1] - raw['timestamp'].iloc[0]
        logger.info(
            f"[DATA] rows={len(raw)} | median_dt={median_dt} | "
            f"bars/day={bars_per_day} | bars/year={GlobalConfig.BARS_PER_YEAR} | span={span}"
        )

        # ---------------------------------------------------------------------
        # 10.2 Features (df_full keeps ALL rows incl. last H bars)
        # ---------------------------------------------------------------------
        df_full = FeatureEngineering.transform(raw)

        # ---------------------------------------------------------------------
        # 10.3 Labels + drop NaN for TRAINING only
        # ---------------------------------------------------------------------
        df = LabelingLayer.apply_triple_barrier(df_full)
        feature_cols = [
            c for c in df.columns
            if c not in {'timestamp', 'open', 'high', 'low', 'close',
                         'volume', 'target', 'label_end_idx'}
        ]
        df = df.dropna(subset=feature_cols + ['target']).reset_index(drop=True)

        X = df[feature_cols].values
        y_mapped = (df['target'].values + 1).astype(int)
        n = len(df)

        # FIX F10: label distribution diagnostic.
        label_dist = pd.Series(y_mapped).value_counts(normalize=True).sort_index().round(3).to_dict()
        logger.info(f"[DIAG] label distribution (0=down,1=neutral,2=up): {label_dist}")

        # ---------------------------------------------------------------------
        # 10.4 Purged walk-forward CV
        # ---------------------------------------------------------------------
        cv = PurgedWalkForwardCV()
        splits = list(cv.split(n))               # FIX F9: materialize for fold_of
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
            # FIX F10: per-fold metrics logged.
            logger.info(f"[DIAG] Fold {fm['fold']}: acc={fm['accuracy']:.3f} "
                        f"bal_acc={fm['balanced_accuracy']:.3f}")

            oof_idx.extend(te.tolist())
            oof_proba.extend(proba)

        if not oof_idx:
            raise RuntimeError("CV produced no predictions.")

        order = np.argsort(np.array(oof_idx))
        oof_idx = [oof_idx[k] for k in order]
        oof_proba = [oof_proba[k] for k in order]

        # FIX F10: signal-funnel diagnostics.
        oof_proba_arr = np.asarray(oof_proba)
        argmax_counts = np.bincount(oof_proba_arr.argmax(axis=1), minlength=3).tolist()
        max_p = oof_proba_arr.max(axis=1)
        max_p_q = np.quantile(max_p, [0.5, 0.75, 0.90, 0.95]).round(4).tolist()
        logger.info(f"[DIAG] argmax counts [down,neutral,up]={argmax_counts}")
        logger.info(f"[DIAG] max_p quantiles [p50,p75,p90,p95]={max_p_q}")

        # ---------------------------------------------------------------------
        # 10.5 Backtest on OOF predictions
        # ---------------------------------------------------------------------
        risk = RiskEngine()
        equity = GlobalConfig.INITIAL_CAPITAL
        peak = GlobalConfig.INITIAL_CAPITAL
        day_start = GlobalConfig.INITIAL_CAPITAL
        current_day = None
        trades: List[Dict[str, Any]] = []
        in_pos_until = -1

        n_below_conf = 0
        n_ema_filtered = 0
        n_risk_blocked = 0

        for i, proba in zip(oof_idx, oof_proba):
            # FIX F6: day rollover MUST happen before the in-position skip.
            bar_day = df['timestamp'].iat[i].date()
            if current_day != bar_day:
                current_day = bar_day
                day_start = equity

            if i <= in_pos_until:
                continue

            direction, size = TradeSimulator.confidence_to_size(proba)
            if direction == 0 or size <= 0:
                n_below_conf += 1
                continue

            # Macro trend filter
            current_price = df['close'].iat[i]
            ema_200 = df['ema_200'].iat[i]
            if direction == 1 and current_price < ema_200:
                n_ema_filtered += 1
                continue
            if direction == -1 and current_price > ema_200:
                n_ema_filtered += 1
                continue

            if not risk.check(equity, peak, day_start):
                logger.error(f"[KILL SWITCH] {risk.reason} — halting.")
                n_risk_blocked += 1
                break

            gross_ret, exit_j, tp, sl, exit_price = TradeSimulator.simulate(df, i, direction, size)
            cost_ret = size * CostConfig.round_turn_cost_frac(df['close'].iat[i])
            net_ret = gross_ret - cost_ret

            equity *= (1 + net_ret)
            peak = max(peak, equity)
            risk.record(net_ret)

            # FIX F9: richer audit trail.
            trades.append({
                "entry_idx": int(i),
                "exit_idx": int(exit_j),
                "fold": int(fold_of[i]) if fold_of[i] >= 0 else -1,
                "direction": int(direction),
                "size": float(size),
                "confidence": float(proba.max()),
                "entry_price": float(df['close'].iat[i]),
                "tp": float(tp),
                "sl": float(sl),
                "exit_price": float(exit_price),
                "atr": float(df['atr'].iat[i]),
                "gross_return": float(gross_ret),
                "cost_return": float(cost_ret),
                "net_return": float(net_ret),
                "equity_after": float(equity),
            })
            in_pos_until = exit_j

        # FIX F10: signal funnel.
        logger.info(
            f"[DIAG] signal funnel: oof_bars={len(oof_idx)}, "
            f"below_conf={n_below_conf}, ema_filtered={n_ema_filtered}, "
            f"risk_blocked={n_risk_blocked}, traded={len(trades)}"
        )

        trade_returns = np.array([t["net_return"] for t in trades]) if trades else np.array([])
        n_test_bars = len(oof_idx)

        metrics = PerformanceEngine.trade_metrics(trade_returns, n_test_bars)
        dsr_prob = PerformanceEngine.deflated_sharpe(
            metrics["SR_per_trade"], GlobalConfig.N_STRATEGY_TRIALS,
            max(len(trade_returns), 2)
        )
        mc = PerformanceEngine.monte_carlo(trade_returns)

        # FIX F8: cost stress test at 2x.
        stress_metrics: Dict[str, Any] = {}
        if trades:
            stress_rets = np.array(
                [t["gross_return"] - CostConfig.STRESS_MULTIPLIER * t["cost_return"]
                 for t in trades]
            )
            stress_metrics = PerformanceEngine.trade_metrics(stress_rets, n_test_bars)

        report = {
            "metadata": governance.metadata,
            "data": {
                "rows": int(len(raw)),
                "median_dt": str(median_dt),
                "bars_per_day": int(bars_per_day),
                "bars_per_year": int(GlobalConfig.BARS_PER_YEAR),
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
            "audit_trail": governance.audit_trail,
        }

        with open("V36_QUANT_EXECUTION_REPORT.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        # FIX F7: DSR is a probability, not a p-value. Log both.
        logger.info("=" * 64)
        logger.info(
            f"Trades: {len(trades)} | Net CAGR: {metrics['CAGR'] * 100:.2f}% | "
            f"Max DD: {metrics['Max_Drawdown'] * 100:.2f}%"
        )
        logger.info(
            f"Sharpe (ann): {metrics['Sharpe_Annualized']:.2f} | "
            f"DSR prob: {dsr_prob:.4f} (p_value={1.0 - dsr_prob:.4f}) | "
            f"MC P5: {mc['MC_P5']:.0f}"
        )
        if stress_metrics:
            logger.info(
                f"[STRESS 2x COST] Trades: {stress_metrics['n_trades']} | "
                f"Net CAGR: {stress_metrics['CAGR'] * 100:.2f}% | "
                f"Sharpe (ann): {stress_metrics['Sharpe_Annualized']:.2f}"
            )
        logger.info("=" * 64)

        # ---------------------------------------------------------------------
        # 10.6 LIVE SIGNAL — predict on the TRUE last bar
        # ---------------------------------------------------------------------
        logger.info("[LIVE SIGNAL] Memprediksi candle terakhir untuk sinyal live...")

        # FIX F4: use df_full (all rows) not df (last H rows dropped by label dropna).
        last_row = df_full.iloc[[-1]]
        if last_row[feature_cols].isna().any().any():
            logger.warning("[LIVE] Last raw row has NaN features; skipping live prediction.")
            return report

        scaler_full = StandardScaler().fit(X)
        model_full = ModelLayer.build_ensemble()
        model_full.fit(scaler_full.transform(X), y_mapped)

        latest_features = last_row[feature_cols].values
        latest_proba = model_full.predict_proba(scaler_full.transform(latest_features))[0]

        raw_direction, _ = TradeSimulator.confidence_to_size(latest_proba)
        current_price = float(last_row['close'].iat[0])
        ema_200 = float(last_row['ema_200'].iat[0])
        atr_last = float(last_row['atr'].iat[0])
        latest_ts = last_row['timestamp'].iat[0]
        confidence_pct = float(latest_proba.max()) * 100.0

        # FIX F5: apply EMA200 filter BEFORE deciding whether to notify.
        direction = raw_direction
        if direction == 1 and current_price <= ema_200:
            direction = 0
        if direction == -1 and current_price >= ema_200:
            direction = 0

        if direction == 1:
            signal_text = "🟢 BUY (LONG)"
            sl = current_price - LabelConfig.SL_MULTIPLIER * atr_last
            tp = current_price + LabelConfig.PT_MULTIPLIER * atr_last
        elif direction == -1:
            signal_text = "🔴 SELL (SHORT)"
            sl = current_price + LabelConfig.SL_MULTIPLIER * atr_last
            tp = current_price - LabelConfig.PT_MULTIPLIER * atr_last
        else:
            signal_text = "⏸️ NEUTRAL (Tidak Ada Sinyal)"
            sl = tp = None

        logger.info(
            f"[LIVE] ts={latest_ts} close={current_price:.2f} ema200={ema_200:.2f} "
            f"raw_dir={raw_direction} final_dir={direction} "
            f"conf={confidence_pct:.1f}% atr={atr_last:.2f}"
        )

        tg_token = os.getenv("TELEGRAM_TOKEN")
        tg_chat = os.getenv("TELEGRAM_CHAT_ID")
        if tg_token and tg_chat and direction != 0:
            try:
                msg = (
                    f"<b>🚨 XAUUSD LIVE SIGNAL</b>\n\n"
                    f"Sinyal: <b>{signal_text}</b>\n"
                    f"Entry: {current_price:.2f}\n"
                    f"SL: {sl:.2f}\n"
                    f"TP: {tp:.2f}\n"
                    f"Keyakinan AI: {confidence_pct:.1f}%\n"
                    f"ATR: {atr_last:.2f}\n"
                    f"EMA200: {ema_200:.2f}\n"
                    f"Waktu: {latest_ts.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                    f"<i>Gunakan Money Management (Max 1-2% Risk).</i>"
                )
                TelegramClient(tg_token, tg_chat).send(msg)
                logger.info(f"[TELEGRAM] Sinyal {signal_text} berhasil dikirim!")
            except Exception as e:
                logger.error(f"[TELEGRAM] Gagal mengirim sinyal: {e}")
        else:
            logger.info("[TELEGRAM] No actionable signal (neutral/filtered) — not sent.")

        return report


# ==============================================================================
# 11. ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAUUSD APEX V36.4.1 (fixed)")
    parser.add_argument("--data", default="data/xauusd_m5.csv")
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    Pipeline.run(args.data, allow_synthetic=args.allow_synthetic)
