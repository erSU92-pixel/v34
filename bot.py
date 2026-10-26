#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAUUSD Quant Bot - Apex Institutional Spot Edition v34.0
Author: Global Head of Quantitative Risk & Trading Systems

Core Architecture:
  1. Native XAUUSD Spot Data Feed (Direct Spot Pricing, Zero Yahoo Finance)
  2. Time-Based Session Filter (London & New York High-Liquidity Windows)
  3. Forensic Closed-Bar Analysis (Eliminates Repainting & Forming-Bar Hazard)
  4. Multi-Timeframe Trend Confirmation (M15 Execution + H1 EMA 50 Macro Filter)
  5. GitHub Gist State Persistence & Signal Deduplication
  6. Confidence-Scaled Risk Sizing & Dynamic ATR Stop-Loss/Take-Profit (1:2+ RR)
"""

import os
import sys
import json
import logging
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

try:
    import numpy as np
    import pandas as pd
    import requests
except ImportError as e:
    print(f"FATAL: missing dependency: {e}", file=sys.stderr)
    sys.exit(1)

# ===========================================================================
# ENVIRONMENT HELPERS
# ===========================================================================
def _env_float(k: str, d: float) -> float:
    v = (os.getenv(k) or "").strip()
    try:
        return float(v) if v else d
    except ValueError:
        return d

def _env_str(k: str, d: str = "") -> str:
    return (os.getenv(k) or "").strip() or d

def _env_bool(k: str, d: bool = True) -> bool:
    v = (os.getenv(k) or "").strip().lower()
    return v in ("true", "1", "yes", "on") if v else d

# ===========================================================================
# CONFIGURATION
# ===========================================================================
@dataclass(frozen=True)
class Config:
    telegram_token: str = _env_str("TELEGRAM_TOKEN")
    telegram_chat_id: str = _env_str("TELEGRAM_CHAT_ID")
    github_token: str = _env_str("GITHUB_TOKEN")
    gist_id: str = _env_str("GIST_ID")
    
    # Twelve Data Free API Key untuk XAU/USD Spot (atau gunakan key kustom Anda)
    twelve_data_api_key: str = _env_str("TWELVE_DATA_API_KEY", "demo")
    ticker: str = "XAU/USD"
    
    # Sesi Waktu Likuiditas Utama (UTC Hours)
    # London Session: 07:00 - 16:00 UTC | New York Session: 12:00 - 21:00 UTC
    enforce_session_filter: bool = _env_bool("ENFORCE_SESSION_FILTER", True)
    session_start_utc: int = 7   # Mulai Sesi London
    session_end_utc: int = 21    # Akhir Sesi New York
    
    timeframe_minutes: int = 15
    min_candles: int = 300
    
    account_equity: float = _env_float("ACCOUNT_EQUITY", 1000.0)
    risk_percent: float = _env_float("RISK_PERCENT", 1.0)
    
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 3.0  # Menghasilkan Minimum R:R 1:2
    min_rr: float = 2.0
    min_sl_distance: float = 1.50

CONFIG = Config()

# ===========================================================================
# LOGGING
# ===========================================================================
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        return json.dumps(payload, default=str)

def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("xauusd_v34")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(JsonFormatter())
    logger.addHandler(sh)
    return logger

log = _setup_logging()

# ===========================================================================
# TELEGRAM NOTIFIER
# ===========================================================================
def send_telegram_alert(message: str) -> None:
    if not CONFIG.telegram_token or not CONFIG.telegram_chat_id:
        log.info("Telegram credentials not set. Skipping notification.")
        return
    try:
        url = f"https://api.telegram.org/bot{CONFIG.telegram_token}/sendMessage"
        payload = {"chat_id": CONFIG.telegram_chat_id, "text": message, "parse_mode": "HTML"}
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            log.warning(f"Failed to send Telegram alert: {response.text}")
    except Exception as e:
        log.warning(f"Telegram alert error: {e}")

# ===========================================================================
# STATE MANAGEMENT (GitHub Gist Deduplication)
# ===========================================================================
def get_state() -> Dict[str, Any]:
    if not CONFIG.github_token or not CONFIG.gist_id:
        return {}
    url = f"https://api.github.com/gists/{CONFIG.gist_id}"
    headers = {"Authorization": f"Bearer {CONFIG.github_token}"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            content = r.json()["files"]["state.json"]["content"]
            return json.loads(content)
    except Exception as e:
        log.warning(f"Failed to fetch Gist state: {e}")
    return {}

def save_state(state: Dict[str, Any]) -> None:
    if not CONFIG.github_token or not CONFIG.gist_id:
        return
    url = f"https://api.github.com/gists/{CONFIG.gist_id}"
    headers = {"Authorization": f"Bearer {CONFIG.github_token}"}
    payload = {"files": {"state.json": {"content": json.dumps(state)}}}
    try:
        requests.patch(url, headers=headers, json=payload, timeout=10)
    except Exception as e:
        log.warning(f"Failed to save Gist state: {e}")

# ===========================================================================
# NATIVE SPOT DATA FETCHER (Twelve Data / Public Spot Feed)
# ===========================================================================
def fetch_spot_data(interval: str, outputsize: int = 300) -> pd.DataFrame:
    log.info(f"Fetching native XAUUSD spot data ({interval}) via Twelve Data Spot API...")
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": CONFIG.ticker,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": CONFIG.twelve_data_api_key,
        "format": "JSON"
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        
        if "values" not in data:
            log.warning(f"API Spot error or rate limit hit: {data.get('message', 'Unknown error')}")
            return pd.DataFrame()
            
        df = pd.DataFrame(data["values"])
        df = df.rename(columns={"datetime": "datetime", "open": "open", "high": "high", "low": "low", "close": "close"})
        
        # Konversi tipe data string ke float
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
            
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        
        log.info(f"Successfully loaded {len(df)} native spot candles for {interval}.")
        return df
    except Exception as e:
        log.error(f"Error fetching spot data {interval}: {e}")
        return pd.DataFrame()

# ===========================================================================
# INDICATORS & FEATURE ENGINEERING
# ===========================================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 50:
        return df
    df = df.copy()
    c = df["close"]
    
    # RSI (EWM Method)
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    rs = g / l.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)
    
    # ATR
    prev_c = c.shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # EMA Suite
    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["ema21"] = c.ewm(span=21, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    
    return df.dropna().reset_index(drop=True)

# ===========================================================================
# MAIN EXECUTION ROUTINE (Institutional Apex Edition)
# ===========================================================================
def main() -> None:
    log.info("XAUUSD Quant Bot v34.0 (Apex Institutional Spot) — Execution Started")
    
    current_utc_hour = datetime.now(timezone.utc).hour
    
    # 1. PILAR 1: TIME-BASED SESSION FILTER
    if CONFIG.enforce_session_filter:
        if not (CONFIG.session_start_utc <= current_utc_hour <= CONFIG.session_end_utc):
        log.info(f"Current UTC hour ({current_utc_hour}) is outside active liquidity hours ({CONFIG.session_start_utc}:00 - {CONFIG.session_end_utc}:00 UTC). Exiting cleanly.")
        sys.exit(0)

    # 2. Tarik Data Spot M15 dan H1
    df = fetch_spot_data("15min", outputsize=300)
    df_h1 = fetch_spot_data("1h", outputsize=200)
    
    if df.empty or len(df) < CONFIG.min_candles // 2:
        log.warning("Insufficient M15 spot candles retrieved.")
        sys.exit(0)
        
    df = add_features(df)
    df_h1 = add_features(df_h1)
    
    if len(df) < 3:
        log.warning("Insufficient data after indicator calculation.")
        sys.exit(0)
        
    # 3. PILAR 2: FORENSIC CLOSED-BAR ANALYSIS (iloc[-2])
    latest = df.iloc[-2]
    prev = df.iloc[-3]
    
    # 4. PILAR 3: MACRO TREND FILTER (H1 EMA 50)
    h1_trend = "NEUTRAL"
    if not df_h1.empty and len(df_h1) >= 2:
        h1_last = df_h1.iloc[-2]
        if h1_last["close"] > h1_last["ema50"]:
            h1_trend = "UP"
        elif h1_last["close"] < h1_last["ema50"]:
            h1_trend = "DOWN"
            
    # 5. Logika Sinyal & Penskalaan Keyakinan (Confidence Scoring)
    side = "NEUTRAL"
    conf = 0.0
    
    up = (latest["ema9"] > latest["ema21"]) and (latest["close"] > latest["ema50"])
    dn = (latest["ema9"] < latest["ema21"]) and (latest["close"] < latest["ema50"])
    gc = (prev["ema9"] <= prev["ema21"]) and (latest["ema9"] > latest["ema21"])
    dc = (prev["ema9"] >= prev["ema21"]) and (latest["ema9"] < latest["ema21"])
    
    if up and latest["rsi"] < 65 and h1_trend != "DOWN":
        side = "BUY"
        conf = 1.0 if gc else 0.6
    elif dn and latest["rsi"] > 35 and h1_trend != "UP":
        side = "SELL"
        conf = 1.0 if dc else 0.6
        
    log.info(f"Signal Evaluated -> Side: {side} | H1 Trend: {h1_trend} | RSI: {latest['rsi']:.2f} | Bar Time: {latest['datetime']}")
    
    if side == "NEUTRAL":
        log.info("No valid signal on closed bar during active session. Exiting cleanly.")
        sys.exit(0)
        
    # 6. Deduplikasi via Gist State
    bar_time = str(latest["datetime"])
    state = get_state()
    if state.get("last_bar") == bar_time and state.get("last_side") == side:
        log.info("Signal for this closed bar has already been dispatched. Skipping.")
        sys.exit(0)
        
    # 7. PILAR 4: RISK-FIRST MATHEMATICS & REWARD OPTIMIZATION
    close_price = float(latest["close"])
    atr_val = float(latest["atr"])
    
    sl_dist = max(atr_val * CONFIG.atr_sl_mult, CONFIG.min_sl_distance)
    tp_dist = atr_val * CONFIG.atr_tp_mult
    rr = tp_dist / sl_dist
    
    if rr < CONFIG.min_rr:
        log.info(f"Risk-Reward ratio too small ({rr:.2f}). Skipping signal.")
        sys.exit(0)
        
    if side == "BUY":
        sl = close_price - sl_dist
        tp = close_price + tp_dist
    else:
        sl = close_price + sl_dist
        tp = close_price - tp_dist
        
    risk_amount = CONFIG.account_equity * (CONFIG.risk_percent / 100.0) * conf
    lot_size = max(0.01, round(risk_amount / (sl_dist * 100), 2))
    
    # 8. Format Susunan Pesan Telegram (HTML Structured)
    emoji = "🟢" if side == "BUY" else "🔴"
    msg = (
        f"{emoji} <b>XAUUSD APEX SPOT SIGNAL ({side})</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 <b>Spot Entry (Closed Bar):</b> <code>{close_price:.2f}</code>\n"
        f"🛡️ <b>Stop Loss:</b> <code>{sl:.2f}</code>\n"
        f"🎯 <b>Take Profit:</b> <code>{tp:.2f}</code> (R:R 1:{rr:.1f})\n"
        f"📏 <b>Lot Size:</b> <code>{lot_size}</code> (Risk {CONFIG.risk_percent}% | Conf: {conf})\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>RSI:</b> {latest['rsi']:.1f} | <b>ATR:</b> {atr_val:.2f}\n"
        f"📈 <b>H1 Macro Trend:</b> {h1_trend}\n"
        f"🕒 <i>Bar Time: {bar_time} UTC</i>\n"
        f"<i>Apex Institutional Spot Execution Engine</i>"
    )
    
    print("\n=======================================================")
    print(msg.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "").replace("<i>", "").replace("</i>", ""))
    print("=======================================================\n")
    
    # Kirim Notifikasi ke Telegram
    send_telegram_alert(msg)
    
    # 9. Simpan State ke Gist
    state["last_bar"] = bar_time
    state["last_side"] = side
    save_state(state)
    log.info("Signal successfully validated, notified, and state synchronized.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.critical(f"Fatal error: {e}\n{traceback.format_exc()}")
        sys.exit(1)
