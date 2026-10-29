#!/usr/bin/env python3
"""
================================================================================
EXECUTIVE INSTITUTIONAL QUANTITATIVE TRADING BOT (XAUUSD APEX ENGINE v4.3)
Role: Global Head of Quantitative Risk & Trading Systems
Architecture: Ironclad Multi-Strategy Consensus Voting with Robust API Retry & Jitter
================================================================================
"""

import os
import sys
import json
import time
import random
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Tuple, Any, Dict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Machine Learning & Metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. CONFIGURATION & LOGGING LAYER
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s): %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("XAUUSD_APEX_V4_3")

class GlobalConfig:
    SYMBOL = "XAU/USD"
    INTERVAL = "1h"
    OUTPUTSIZE = 500
    RANDOM_STATE = 42
    
    TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    GIST_ID = os.getenv("GIST_ID")
    GIST_TOKEN = os.getenv("GIST_TOKEN")

class GovernanceLayer:
    EXPERIMENT_ID = f"EXP_XAUUSD_V43_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    MODEL_VERSION = "4.3.0-ENTERPRISE"
    FEATURE_REGISTRY = [
        "ret_1", "ret_3", "ret_5", "ret_10", "rsi", "atr",
        "ema9", "ema21", "ema50", "ema_spread", "ema_slope",
        "volatility20", "zscore20", "entropy50", "hurst50"
    ]

# ==========================================
# 3. DATA LAYER (Robust Retry & Session Handler)
# ==========================================
class DataLayer:
    @staticmethod
    def _create_robust_session() -> requests.Session:
        """Membuat sesi HTTP dengan automatic exponential backoff & status code handling"""
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False,
            respect_retry_after_header=True
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

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
        
        session = DataLayer._create_robust_session()
        
        try:
            logger.info("Mengirim permintaan data aman ke Twelve Data (dengan Enterprise Retry Session)...")
            # Jitter acak mikro untuk menghindari bentrok rate limit massal di jam kritikal cron
            time.sleep(random.uniform(0.1, 0.9))
            
            response = session.get(url, timeout=25)
            
            if response.status_code == 429:
                logger.error("Rate Limit (429) terdeteksi. Melakukan pendingin cooldown 10 detik...")
                time.sleep(10)
                response = session.get(url, timeout=25)
                
            response.raise_for_status()
            data = response.json()
            
            if "values" not in data:
                logger.error(f"API Response error payload: {data}")
                raise ValueError("Invalid API Response structure from Twelve Data.")
                
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)
            
            cols = ["open", "high", "low", "close"]
            for col in cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    raise ValueError(f"Required price column '{col}' missing from API response.")
            
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0) if "volume" in df.columns else 0.0
            return df
        except Exception as e:
            logger.critical(f"Kritikal: Gagal total mengambil data setelah mekanisme retry: {e}")
            sys.exit(1)

    @staticmethod
    def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
        initial_len = len(df)
        df = df.drop_duplicates(subset=["datetime"]).reset_index(drop=True)
        if len(df) < initial_len:
            logger.warning(f"Removed {initial_len - len(df)} duplicate timestamps.")
        cleaned = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        if len(cleaned) < 100:
            logger.critical("Insufficient data points after cleaning for institutional model training.")
            sys.exit(1)
        return cleaned

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
            return np.clip(poly[0] * 2.0, 0.0, 1.0)
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
        d["rsi"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        
        tr1 = d["high"] - d["low"]
        tr2 = abs(d["high"] - d["close"].shift())
        tr3 = abs(d["low"] - d["close"].shift())
        d["atr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(window=14).mean().clip(lower=0.0001)
        
        d["ema9"] = d["close"].ewm(span=9, adjust=False).mean()
        d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
        d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
        d["ema_spread"] = d["ema9"] - d["ema21"]
        d["ema_slope"] = d["ema50"].diff(3)
        
        d["volatility20"] = d["ret_1"].rolling(window=20).std()
        d["zscore20"] = (d["close"] - d["close"].rolling(window=20).mean()) / (d["close"].rolling(window=20).std() + 1e-9)
        
        d["entropy50"] = cls.calculate_entropy(d["close"], window=50)
        d["hurst50"] = cls.calculate_hurst(d["close"], window=50)
        
        return d.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

# ==========================================
# 5. LABELING LAYER (Triple Barrier)
# ==========================================
class LabelingLayer:
    @staticmethod
    def triple_barrier(df: pd.DataFrame, tr_window: int = 5, sl_mult: float = 1.2, tp_mult: float = 1.8) -> pd.DataFrame:
        labels = []
        closes = df["close"].values
        atrs = df["atr"].values
        n = len(df)
        
        for i in range(n - tr_window):
            entry_price = closes[i]
            upper = entry_price + (tp_mult * atrs[i])
            lower = entry_price - (sl_mult * atrs[i])
            outcome = 0 
            for j in range(1, tr_window + 1):
                if closes[i + j] >= upper:
                    outcome = 1
                    break
                elif closes[i + j] <= lower:
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
        n = len(X)
        indices = np.arange(n)
        fold_size = n // self.n_splits
        embargo = int(n * self.embargo_pct)
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = test_start + fold_size if i < self.n_splits - 1 else n
            test_indices = indices[test_start:test_end]
            train_indices = np.concatenate([indices[:max(0, test_start - embargo)], indices[min(n, test_end + embargo):]])
            if len(train_indices) > 0:
                yield train_indices, test_indices

# ==========================================
# 9. CONSENSUS MULTI-STRATEGY ORCHESTRATOR
# ==========================================
class ConsensusOrchestrator:
    @classmethod
    def train_strategies(cls, df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, float]]:
        X = df[GovernanceLayer.FEATURE_REGISTRY].values
        y = df["target"].values
        
        cv = PurgedTimeSeriesSplit(n_splits=3)
        
        model_a = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=GlobalConfig.RANDOM_STATE, class_weight='balanced')
        model_b = GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=GlobalConfig.RANDOM_STATE)
        model_c = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=GlobalConfig.RANDOM_STATE, class_weight='balanced')
        
        accuracies = []
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)
            
            model_a.fit(X_train_sc, y_train)
            model_b.fit(X_train_sc, y_train)
            model_c.fit(X_train_sc, y_train)
            
            preds_a = model_a.predict(X_test_sc).ravel()
            preds_b = model_b.predict(X_test_sc).ravel()
            preds_c = model_c.predict(X_test_sc).ravel()
            
            consensus_preds = []
            for pa, pb, pc in zip(preds_a, preds_b, preds_c):
                votes = [int(pa), int(pb), int(pc)]
                counts = {v: votes.count(v) for v in set(votes)}
                majority = max(counts, key=counts.get)
                if counts[majority] < 2:
                    consensus_preds.append(0)
                else:
                    consensus_preds.append(majority)
                    
            accuracies.append(accuracy_score(y_test, consensus_preds))
            
        scaler_final = StandardScaler()
        X_scaled = scaler_final.fit_transform(X)
        
        model_a.fit(X_scaled, y)
        model_b.fit(X_scaled, y)
        model_c.fit(X_scaled, y)
        
        models_bundle = {
            "model_a": model_a,
            "model_b": model_b,
            "model_c": model_c,
            "scaler": scaler_final
        }
        
        metrics = {
            "consensus_cv_accuracy": float(np.mean(accuracies))
        }
        return models_bundle, metrics

# ==========================================
# STATE & NOTIFICATION LAYER
# ==========================================
class OperationalEngine:
    @staticmethod
    def send_telegram(message: str):
        if not GlobalConfig.TELEGRAM_TOKEN or not GlobalConfig.TELEGRAM_CHAT_ID:
            logger.warning("Telegram credentials missing.")
            return
        url = f"https://api.telegram.org/bot{GlobalConfig.TELEGRAM_TOKEN}/sendMessage"
        try:
            requests.post(url, json={"chat_id": GlobalConfig.TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=10)
        except Exception as e:
            logger.error(f"Telegram error: {e}")

    @staticmethod
    def sync_gist(data: dict):
        if not GlobalConfig.GIST_ID or not GlobalConfig.GIST_TOKEN:
            return
        url = f"https://api.github.com/gists/{GlobalConfig.GIST_ID}"
        headers = {"Authorization": f"token {GlobalConfig.GIST_TOKEN}", "Accept": "vnd.github.v3+json"}
        try:
            requests.patch(url, headers=headers, json={"files": {"xauusd_consensus_state.json": {"content": json.dumps(data, indent=4)}}}, timeout=10)
        except Exception as e:
            logger.error(f"Gist error (non-fatal): {e}")

# ==========================================
# PIPELINE ORCHESTRATOR
# ==========================================
def main():
    logger.info(f"Initializing {GovernanceLayer.EXPERIMENT_ID} ({GovernanceLayer.MODEL_VERSION})...")
    
    df = DataLayer.load_data()
    clean_df = DataLayer.validate_and_clean(df)
    featured_df = FeatureEngineering.generate_features(clean_df)
    labeled_df = LabelingLayer.triple_barrier(featured_df)
    
    models_bundle, metrics = ConsensusOrchestrator.train_strategies(labeled_df)
    logger.info(f"Consensus Training Complete. Metrics: {metrics}")
    
    # Inference pada closed candle (Index -2) untuk menghindari repainting / look-ahead bias
    latest_row = labeled_df.iloc[[-2]]
    scaler = models_bundle["scaler"]
    X_live = scaler.transform(latest_row[GovernanceLayer.FEATURE_REGISTRY].values)
    
    pred_a = int(models_bundle["model_a"].predict(X_live)[0])
    pred_b = int(models_bundle["model_b"].predict(X_live)[0])
    pred_c = int(models_bundle["model_c"].predict(X_live)[0])
    
    votes = [pred_a, pred_b, pred_c]
    logger.info(f"Individual Strategy Votes -> Model A: {pred_a}, Model B: {pred_b}, Model C: {pred_c}")
    
    counts = {v: votes.count(v) for v in set(votes)}
    majority_vote = max(counts, key=counts.get)
    
    # Syarat Mutlak Konsensus: Minimal 2 dari 3 model harus sepakat
    final_decision = majority_vote if counts[majority_vote] >= 2 else 0
    
    current_close = float(latest_row["close"].values[0])
    current_atr = float(latest_row["atr"].values[0])
    
    sl_dist = current_atr * 1.2
    tp_dist = current_atr * 1.8
    
    if final_decision == 1:
        signal_text = "🟢 SINYAL XAUUSD (BUY)"
        entry, sl, tp = round(current_close, 2), round(current_close - sl_dist, 2), round(current_close + tp_dist, 2)
        trend = "UP"
    elif final_decision == -1:
        signal_text = "🔴 SINYAL XAUUSD (SELL)"
        entry, sl, tp = round(current_close, 2), round(current_close + sl_dist, 2), round(current_close - tp_dist, 2)
        trend = "DOWN"
    else:
        signal_text = "⏳ NEUTRAL / HOLD (Consensus Not Reached)"
        entry, sl, tp, trend = 0, 0, 0, "SIDEWAYS"

    logger.info(f"Consensus Result: {signal_text}")
    
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    report = {
        "experiment_id": GovernanceLayer.EXPERIMENT_ID,
        "timestamp": str(datetime.now(timezone.utc)),
        "final_decision": int(final_decision),
        "votes": {"model_a": pred_a, "model_b": pred_b, "model_c": pred_c},
        "metrics": metrics
    }
    OperationalEngine.sync_gist(report)
    
    if final_decision != 0:
        msg = (
            f"{signal_text}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Entry: {entry}\n"
            f"🛡️ Stop Loss: {sl}\n"
            f"🎯 Take Profit: {tp}\n"
            f"📏 Lot Size: 0.01 (Risk 1.0%)\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Consensus: {counts[majority_vote]}/3 Models Agree\n"
            f"📈 H1 Trend: {trend} | R:R: 1:1.5\n"
            f"🕒 {timestamp}\n"
            f"_Ironclad Multi-Strategy Consensus Engine v4.3_"
        )
        OperationalEngine.send_telegram(msg)
    else:
        logger.info("No consensus reached (models disagreed or split vote). Telegram notification suppressed.")

if __name__ == "__main__":
    main()
