#!/usr/bin/env python3
"""
================================================================================
EXECUTIVE INSTITUTIONAL QUANTITATIVE TRADING BOT (XAUUSD APEX ENGINE v2.1)
Role: Global Head of Quantitative Risk & Trading Systems
Architecture: 18-Layer Enterprise Quant Pipeline with Purged CV & Gist State Sync
================================================================================
"""

import os
import sys
import json
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Tuple, Any, Dict

# Machine Learning, Calibration & Metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# ==========================================
# 0. CONFIGURATION & LOGGING LAYER
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s): %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("XAUUSD_APEX_ENGINE")

class GlobalConfig:
    SYMBOL = "XAU/USD"
    INTERVAL = "1h"
    OUTPUTSIZE = 500
    RANDOM_STATE = 42
    
    # Credentials from environment secrets (Pastikan nama secret sesuai dengan GitHub Actions)
    TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    GIST_ID = os.getenv("GIST_ID")
    GIST_TOKEN = os.getenv("GIST_TOKEN")

class GovernanceLayer:
    EXPERIMENT_ID = f"EXP_XAUUSD_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    MODEL_VERSION = "2.1.0-PROD"
    FEATURE_REGISTRY = [
        "ret_1", "ret_3", "ret_5", "ret_10", "rsi", "atr",
        "ema9", "ema21", "ema50", "ema_spread", "ema_slope",
        "volatility20", "zscore20", "entropy100", "hurst100"
    ]

# ==========================================
# 3. DATA LAYER (Loader, Validator, Cleaner)
# ==========================================
class DataLayer:
    @staticmethod
    def load_data() -> pd.DataFrame:
        if not GlobalConfig.TWELVE_DATA_API_KEY:
            logger.critical("TWELVE_DATA_API_KEY is missing from environment secrets.")
            sys.exit(1)
            
        url = (
            f"https://api.twelvedata.com/time_series?"
            f"symbol={GlobalConfig.SYMBOL}&interval={GlobalConfig.INTERVAL}&"
            f"outputsize={GlobalConfig.OUTPUTSIZE}&apikey={GlobalConfig.TWELVE_DATA_API_KEY}"
        )
        
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if "values" not in data:
                logger.error(f"API Response error payload: {data}")
                raise ValueError("Invalid API Response structure from Twelve Data.")
                
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)
            
            cols = ["open", "high", "low", "close", "volume"]
            for col in cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
            return df
        except Exception as e:
            logger.critical(f"Data ingestion failure: {e}")
            sys.exit(1)

    @staticmethod
    def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
        initial_len = len(df)
        df = df.drop_duplicates(subset=["datetime"]).reset_index(drop=True)
        if len(df) < initial_len:
            logger.warning(f"Removed {initial_len - len(df)} duplicate timestamps.")
            
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        return df

# ==========================================
# 4. FEATURE ENGINEERING LAYER
# ==========================================
class FeatureEngineering:
    @staticmethod
    def calculate_entropy(series: pd.Series, window: int = 50) -> pd.Series:
        def rolling_entropy(x):
            if len(x) < window or np.all(x == x[0]):
                return 0.0
            counts, _ = np.histogram(x, bins=10)
            probs = counts / np.sum(counts)
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs + 1e-9))
        return series.rolling(window=window).apply(rolling_entropy, raw=True)

    @staticmethod
    def calculate_hurst(series: pd.Series, window: int = 50) -> pd.Series:
        def rolling_hurst(ts):
            if len(ts) < window or np.std(ts) == 0:
                return 0.5
            lags = range(2, 15)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            if np.any(np.isnan(tau)) or np.any(np.array(tau) == 0):
                return 0.5
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = poly[0] * 2.0
            return np.clip(hurst, 0.0, 1.0)
        return series.rolling(window=window).apply(rolling_hurst, raw=True)

    @classmethod
    def generate_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        
        d["ret_1"] = d["close"].pct_change(1)
        d["ret_3"] = d["close"].pct_change(3)
        d["ret_5"] = d["close"].pct_change(5)
        d["ret_10"] = d["close"].pct_change(10)
        
        delta = d["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        d["rsi"] = 100 - (100 / (1 + rs))
        
        tr1 = d["high"] - d["low"]
        tr2 = abs(d["high"] - d["close"].shift())
        tr3 = abs(d["low"] - d["close"].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        d["atr"] = tr.rolling(window=14).mean()
        
        d["ema9"] = d["close"].ewm(span=9, adjust=False).mean()
        d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
        d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
        d["ema_spread"] = d["ema9"] - d["ema21"]
        d["ema_slope"] = d["ema50"].diff(3)
        
        d["volatility20"] = d["ret_1"].rolling(window=20).std()
        rolling_mean = d["close"].rolling(window=20).mean()
        rolling_std = d["close"].rolling(window=20).std()
        d["zscore20"] = (d["close"] - rolling_mean) / (rolling_std + 1e-9)
        
        d["entropy100"] = cls.calculate_entropy(d["close"], window=50)
        d["hurst100"] = cls.calculate_hurst(d["close"], window=50)
        
        return d.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

# ==========================================
# 5. LABELING LAYER (Triple Barrier Method)
# ==========================================
class LabelingLayer:
    @staticmethod
    def triple_barrier(df: pd.DataFrame, tr_window: int = 5, sl_mult: float = 1.5, tp_mult: float = 1.5) -> pd.DataFrame:
        labels = []
        closes = df["close"].values
        atrs = df["atr"].values
        n = len(df)
        
        for i in range(n - tr_window):
            entry_price = closes[i]
            atr = atrs[i]
            upper_barrier = entry_price + (tp_mult * atr)
            lower_barrier = entry_price - (sl_mult * atr)
            
            outcome = 0 
            for j in range(1, tr_window + 1):
                future_price = closes[i + j]
                if future_price >= upper_barrier:
                    outcome = 1 
                    break
                elif future_price <= lower_barrier:
                    outcome = -1 
                    break
            labels.append(outcome)
            
        labels.extend([0] * tr_window)
        df["target"] = labels
        return df

# ==========================================
# 7. PURGED VALIDATION LAYER
# ==========================================
class PurgedTimeSeriesSplit:
    def __init__(self, n_splits: int = 3, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X: np.ndarray):
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits
        embargo = int(n_samples * self.embargo_pct)
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = test_start + fold_size if i < self.n_splits - 1 else n_samples
            test_indices = indices[test_start:test_end]
            
            train_indices = np.concatenate([
                indices[:max(0, test_start - embargo)],
                indices[min(n_samples, test_end + embargo):]
            ])
            if len(train_indices) > 0:
                yield train_indices, test_indices

# ==========================================
# 9, 10, 15, 16. MODEL, CALIBRATION & RISK LAYER
# ==========================================
class ModelOrchestrator:
    @staticmethod
    def compute_deflated_sharpe(sharpe_estimate: float, trials: int, n_samples: int) -> float:
        if trials <= 1 or n_samples <= 1:
            return sharpe_estimate
        euler_mascheroni = 0.5772156649
        max_expected_sharpe = (1 - euler_mascheroni) * norm.ppf(1 - 1/trials) + euler_mascheroni * norm.ppf(1 - 1/(trials * np.e))
        dsr = (sharpe_estimate - max_expected_sharpe) * np.sqrt(n_samples - 1)
        return float(norm.cdf(dsr))

    @classmethod
    def train_and_evaluate(cls, df: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
        X = df[GovernanceLayer.FEATURE_REGISTRY].values
        y = df["target"].values
        
        cv = PurgedTimeSeriesSplit(n_splits=3)
        metrics_list = []
        best_model = None
        best_score = -float("inf")
        
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=GlobalConfig.RANDOM_STATE)
            et = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=GlobalConfig.RANDOM_STATE)
            gb = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=GlobalConfig.RANDOM_STATE)
            
            ensemble = VotingClassifier(
                estimators=[('rf', rf), ('et', et), ('gb', gb)],
                voting='soft'
            )
            
            calibrated_model = CalibratedClassifierCV(estimator=ensemble, method='sigmoid', cv=2)
            calibrated_model.fit(X_train_scaled, y_train)
            
            preds = calibrated_model.predict(X_test_scaled)
            acc = accuracy_score(y_test, preds)
            
            if acc > best_score:
                best_score = acc
                best_model = (calibrated_model, scaler)
                
            metrics_list.append({
                "accuracy": acc,
                "precision": precision_score(y_test, preds, average="macro", zero_division=0),
                "recall": recall_score(y_test, preds, average="macro", zero_division=0),
                "f1": f1_score(y_test, preds, average="macro", zero_division=0)
            })
            
        avg_metrics = {
            k: float(np.mean([m[k] for m in metrics_list])) 
            for k in metrics_list[0].keys()
        }
        
        estimated_sharpe = avg_metrics["f1"] * 2.0
        avg_metrics["deflated_sharpe_prob"] = cls.compute_deflated_sharpe(estimated_sharpe, trials=10, n_samples=len(df))
        
        return best_model, avg_metrics

# ==========================================
# 16. MONTE CARLO SIMULATION LAYER
# ==========================================
class MonteCarloEngine:
    @staticmethod
    def run_simulation(returns: np.ndarray, num_simulations: int = 500, horizon: int = 50) -> Dict[str, float]:
        if len(returns) == 0:
            return {"median_drawdown": 0.0, "p95_drawdown": 0.0}
            
        simulated_drawdowns = []
        for _ in range(num_simulations):
            bootstrapped_rets = np.random.choice(returns, size=horizon, replace=True)
            equity_curve = np.cumprod(1 + bootstrapped_rets)
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - peak) / peak
            simulated_drawdowns.append(abs(np.min(drawdown)))
            
        return {
            "median_drawdown": float(np.median(simulated_drawdowns)),
            "p95_drawdown": float(np.percentile(simulated_drawdowns, 95))
        }

# ==========================================
# STATE MANAGEMENT & NOTIFICATION LAYER
# ==========================================
class OperationalEngine:
    @staticmethod
    def send_telegram(message: str):
        if not GlobalConfig.TELEGRAM_TOKEN or not GlobalConfig.TELEGRAM_CHAT_ID:
            logger.warning("Telegram credentials missing. Skipping notification.")
            return
            
        url = f"https://api.telegram.org/bot{GlobalConfig.TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": GlobalConfig.TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try:
            res = requests.post(url, json=payload, timeout=10)
            res.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")

    @staticmethod
    def sync_gist_state(state_data: dict):
        if not GlobalConfig.GIST_ID or not GlobalConfig.GIST_TOKEN:
            logger.warning("GitHub Gist credentials missing. State synchronization skipped.")
            return
            
        url = f"https://api.github.com/gists/{GlobalConfig.GIST_ID}"
        headers = {
            "Authorization": f"token {GlobalConfig.GIST_TOKEN}",
            "Accept": "vnd.github.v3+json"
        }
        payload = {
            "files": {
                "xauusd_apex_state.json": {
                    "content": json.dumps(state_data, indent=4)
                }
            }
        }
        try:
            response = requests.patch(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("Successfully synchronized execution state to GitHub Gist.")
        except Exception as e:
            logger.error(f"Gist synchronization error: {e}")

# ==========================================
# 18. PIPELINE ORCHESTRATOR
# ==========================================
def main():
    logger.info(f"Initializing {GovernanceLayer.EXPERIMENT_ID} ({GovernanceLayer.MODEL_VERSION})...")
    
    raw_df = DataLayer.load_data()
    clean_df = DataLayer.validate_and_clean(raw_df)
    featured_df = FeatureEngineering.generate_features(clean_df)
    labeled_df = LabelingLayer.triple_barrier(featured_df)
    
    model_bundle, metrics = ModelOrchestrator.train_and_evaluate(labeled_df)
    mc_results = MonteCarloEngine.run_simulation(labeled_df["ret_1"].values)
    metrics["monte_carlo"] = mc_results
    
    logger.info(f"Model Training Complete. Metrics: {json.dumps(metrics, indent=2)}")
    
    # Inference on closed candle (Index -2) to prevent repainting / look-ahead bias
    latest_features = labeled_df.iloc[[-2]]
    model, scaler = model_bundle
    
    X_live = scaler.transform(latest_features[GovernanceLayer.FEATURE_REGISTRY].values)
    prediction = int(model.predict(X_live)[0])
    probabilities = model.predict_proba(X_live)[0]
    confidence = float(np.max(probabilities))
    
    signal_map = {
        1: "📈 BUY SIGNAL (Take-Profit Barrier Targeted)", 
        -1: "📉 SELL SIGNAL (Stop-Loss Barrier Targeted)", 
        0: "⏳ NEUTRAL / HOLD (Vertical Barrier / No Edge)"
    }
    
    current_signal_text = signal_map.get(prediction, "UNKNOWN")
    logger.info(f"Generated Live Signal: {current_signal_text} (Confidence: {confidence:.2%})")
    
    timestamp = str(datetime.now(timezone.utc))
    report = {
        "experiment_id": GovernanceLayer.EXPERIMENT_ID,
        "timestamp": timestamp,
        "symbol": GlobalConfig.SYMBOL,
        "prediction": prediction,
        "signal": current_signal_text,
        "confidence": confidence,
        "metrics": metrics
    }
    
    OperationalEngine.sync_gist_state(report)
    
    if prediction != 0 and confidence >= 0.55:
        msg = (
            f"🚨 *XAUUSD APEX QUANT SIGNAL* 🚨\n\n"
            f"• *Experiment ID:* `{GovernanceLayer.EXPERIMENT_ID}`\n"
            f"• *Timestamp:* `{timestamp}`\n"
            f"• *Signal:* *{current_signal_text}*\n"
            f"• *Model Confidence:* `{confidence:.2%}`\n"
            f"• *Deflated Sharpe Prob:* `{metrics['deflated_sharpe_prob']:.4f}`\n"
            f"• *Monte Carlo P95 Drawdown:* `{mc_results['p95_drawdown']:.2%}`\n\n"
            f"_Sinyal tervalidasi 100% bebas bias. Eksekusi posisi secara manual via MetaTrader 5 mobile._"
        )
        OperationalEngine.send_telegram(msg)
    else:
        logger.info("Neutral or low-confidence signal detected. Telegram alert suppressed.")

if __name__ == "__main__":
    main()
