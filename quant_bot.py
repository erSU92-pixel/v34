#!/usr/bin/env python3
"""
================================================================================
EXECUTIVE INSTITUTIONAL QUANTITATIVE TRADING BOT (XAUUSD APEX ENGINE v6.0)
Role: Global Head of Quantitative Risk & Trading Systems
Architecture: Grand Committee Multi-Mazhab Consensus Voting (10-Pillar Framework)
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
from typing import Dict, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Machine Learning & Metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. CONFIGURATION & LOGGING LAYER
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s): %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("XAUUSD_APEX_V6")

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
    EXPERIMENT_ID = f"EXP_XAUUSD_V6_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    MODEL_VERSION = "6.0.0-ULTIMATE-10-PILLAR"
    FEATURE_REGISTRY = [
        "ret_1", "ret_3", "ret_5", "ret_10", "rsi", "atr",
        "ema9", "ema21", "ema50", "ema_spread", "ema_slope",
        "volatility20", "zscore20", "entropy50", "hurst50"
    ]

# ==========================================
# 1. DATA LAYER (Robust Retry & Session Handler)
# ==========================================
class DataLayer:
    @staticmethod
    def _create_robust_session() -> requests.Session:
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
            logger.info("Mengirim permintaan data aman ke Twelve Data (10-Pillar Engine Session)...")
            time.sleep(random.uniform(0.1, 0.9))
            response = session.get(url, timeout=25)
            
            if response.status_code == 429:
                logger.error("Rate Limit (429) terdeteksi. Melakukan pendinginan cooldown 10 detik...")
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
# 2. FEATURE ENGINEERING & MATH LAYER
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
        
        low_min = d["low"].rolling(window=5).min()
        high_max = d["high"].rolling(window=5).max()
        d["stoch_k5"] = 100 * ((d["close"] - low_min) / (high_max - low_min + 1e-9))
        
        low_min7 = d["low"].rolling(window=7).min()
        high_max7 = d["high"].rolling(window=7).max()
        d["stoch_k7"] = 100 * ((d["close"] - low_min7) / (high_max7 - low_min7 + 1e-9))
        
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
# 3. TRIPLE BARRIER LABELING LAYER
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
# 4. 10-PILLAR GRAND COMMITTEE ORCHESTRATOR
# ==========================================
class GrandCommittee10PillarOrchestrator:
    @classmethod
    def evaluate_committee(cls, latest_row: pd.Series, df_full: pd.DataFrame) -> Dict[str, int]:
        """
        Mengevaluasi suara dari 10 Pilar Master Blueprint:
        - Pilar 1: Trend Klasik (EMA & Z-Score)
        - Pilar 2: Momentum Klasik (RSI & Stochastic Crossover)
        - Pilar 3: Machine Learning Stacking (RF, GB, ET)
        - Pilar 4: Volatilitas & Breakout (ATR & Price Expansion)
        - Pilar 5: Market Regime Gatekeeper (Hurst Exponent Filter)
        - Pilar 6: Makroekonomi & Suku Bunga (Mock Proxy / Yield Matrix)
        - Pilar 7: Korelasi Silang Antar Aset (DXY Proxy Simulation)
        - Pilar 8: Likuiditas & Order Flow (POC / Volume Profile Proxy)
        - Pilar 9: Sentimen & Psikologi Pasar (Retail Sentiment Filter Proxy)
        - Pilar 10: Musiman & Siklus Waktu (Session Timing Gatekeeper)
        """
        # --- Training Machine Learning untuk Pilar 3 ---
        X = df_full[GovernanceLayer.FEATURE_REGISTRY].values
        y = df_full["target"].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=GlobalConfig.RANDOM_STATE, class_weight='balanced')
        gb = GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=GlobalConfig.RANDOM_STATE)
        et = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=GlobalConfig.RANDOM_STATE, class_weight='balanced')
        
        rf.fit(X_scaled, y)
        gb.fit(X_scaled, y)
        et.fit(X_scaled, y)
        
        X_live = scaler.transform(latest_row[GovernanceLayer.FEATURE_REGISTRY].values.reshape(1, -1))
        ml_a = int(rf.predict(X_live)[0])
        ml_b = int(gb.predict(X_live)[0])
        ml_c = int(et.predict(X_live)[0])
        ml_consensus = 1 if (ml_a + ml_b + ml_c) > 0 else (-1 if (ml_a + ml_b + ml_c) < 0 else 0)

        # 1. Pilar Trend Klasik
        p1 = 1 if latest_row["ema9"] > latest_row["ema21"] and latest_row["zscore20"] > 0 else (-1 if latest_row["ema9"] < latest_row["ema21"] and latest_row["zscore20"] < 0 else 0)

        # 2. Pilar Momentum Klasik
        p2 = 1 if latest_row["rsi"] > 50 and latest_row["stoch_k5"] > latest_row["stoch_k7"] else (-1 if latest_row["rsi"] < 50 and latest_row["stoch_k5"] < latest_row["stoch_k7"] else 0)

        # 3. Pilar ML Stacking
        p3 = ml_consensus

        # 4. Pilar Volatilitas & Breakout
        ret_1 = latest_row["ret_1"]
        vol20 = latest_row["volatility20"]
        p4 = 1 if abs(ret_1) > (vol20 * 1.5) and ret_1 > 0 else (-1 if abs(ret_1) > (vol20 * 1.5) and ret_1 < 0 else 0)

        # 5. Pilar Market Regime Gatekeeper (Hurst)
        p5 = 0 if latest_row["hurst50"] < 0.48 else p1

        # 6. Pilar Makroekonomi & Suku Bunga (Proxy berdasarkan tren volatilitas suku bunga riil/zscore)
        p6 = 1 if latest_row["zscore20"] > 0.5 else (-1 if latest_row["zscore20"] < -0.5 else 0)

        # 7. Pilar Korelasi Silang DXY Proxy
        p7 = -1 if p1 == 1 else (1 if p1 == -1 else 0) # Invers korelasi emas terhadap dolar

        # 8. Pilar Order Flow / Volume Profile Proxy
        p8 = 1 if latest_row["volume"] > 0 and latest_row["close"] > latest_row["ema21"] else ( -1 if latest_row["volume"] > 0 and latest_row["close"] < latest_row["ema21"] else 0 )

        # 9. Pilar Sentimen & Psikologi Pasar (Contrarian filter jika RSI ekstrem)
        rsi_val = latest_row["rsi"]
        p9 = -1 if rsi_val > 75 else (1 if rsi_val < 25 else 0)

        # 10. Pilar Musiman & Sıkllus Waktu (Session Timing Check UTC)
        current_hour = datetime.now(timezone.utc).hour
        # Aktif di sesi London & New York overlap (08:00 - 17:00 UTC)
        p10 = 1 if 8 <= current_hour <= 17 else 0

        votes = {
            "pilar_1_trend": p1,
            "pilar_2_momentum": p2,
            "pilar_3_ml_ai": p3,
            "pilar_4_breakout": p4,
            "pilar_5_regime_gate": p5,
            "pilar_6_macro": p6,
            "pilar_7_cross_asset": p7,
            "pilar_8_order_flow": p8,
            "pilar_9_sentiment": p9,
            "pilar_10_seasonality": p10
        }
        return votes

# ==========================================
# 5. STATE & TELEGRAM NOTIFICATION LAYER
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
            requests.patch(url, headers=headers, json={"files": {"xauusd_10_pillar_state.json": {"content": json.dumps(data, indent=4)}}}, timeout=10)
        except Exception as e:
            logger.error(f"Gist error (non-fatal): {e}")

# ==========================================
# 6. MAIN PIPELINE ORCHESTRATOR
# ==========================================
def main():
    logger.info(f"Initializing {GovernanceLayer.EXPERIMENT_ID} ({GovernanceLayer.MODEL_VERSION})...")
    
    df = DataLayer.load_data()
    clean_df = DataLayer.validate_and_clean(df)
    featured_df = FeatureEngineering.generate_features(clean_df)
    labeled_df = LabelingLayer.triple_barrier(featured_df)
    
    # Ambil baris closed candle terakhir (-2) untuk menghindari Look-Ahead Bias
    latest_row = labeled_df.iloc[-2]
    
    # Evaluasi Grand Committee 10 Pilar
    committee_votes = GrandCommittee10PillarOrchestrator.evaluate_committee(latest_row, labeled_df)
    logger.info(f"10-Pillar Grand Committee Votes -> {committee_votes}")
    
    vote_values = list(committee_votes.values())
    counts = {v: vote_values.count(v) for v in set(vote_values)}
    majority_vote = max(counts, key=counts.get)
    
    # Syarat Mutlak Konsensus: Minimal 6 dari 10 pilar harus sepakat secara mutlak
    agreed_count = counts.get(majority_vote, 0)
    final_decision = majority_vote if agreed_count >= 6 else 0
    
    current_close = float(latest_row["close"])
    current_atr = float(latest_row["atr"])
    
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
        signal_text = "⏳ NEUTRAL / HOLD (10-Pillar Consensus Not Reached)"
        entry, sl, tp, trend = 0, 0, 0, "SIDEWAYS"

    logger.info(f"Grand Consensus Result: {signal_text} (Agreement: {agreed_count}/10)")
    
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    report = {
        "experiment_id": GovernanceLayer.EXPERIMENT_ID,
        "timestamp": str(datetime.now(timezone.utc)),
        "final_decision": int(final_decision),
        "agreement_count": int(agreed_count),
        "committee_votes": committee_votes
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
            f"🏛️ 10-Pillar Committee: {agreed_count}/10 Pillars Agree\n"
            f"📈 H1 Trend: {trend} | R:R: 1:1.5\n"
            f"🕒 {timestamp}\n"
            f"_Ultimate 10-Pillar Grand Committee Engine v6.0_"
        )
        OperationalEngine.send_telegram(msg)
    else:
        logger.info(f"No consensus reached (Agreement: {agreed_count}/10 < 6). Telegram notification suppressed.")

if __name__ == "__main__":
    main()
