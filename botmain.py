import os
import re
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import yfinance as yf

# ============================================================
# CONFIG
# ============================================================
STATE_FILE = "signal_state.json"
REPORT_JSON = "xauusd_apex_report.json"
COOLDOWN_MINUTES = 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [ApexV6] - %(message)s")

def _redact(text): return re.sub(r"apikey=[^&\s'\"]+", "apikey=***", str(text))

# ============================================================
# DATA PROVIDER (FAILOVER CASCADE)
# ============================================================
TWELVE_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")

def _normalize_df(df):
    if df is None or len(df) == 0: return None
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(columns={'date': 'datetime', 't': 'datetime'})
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns: return None
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'volume' not in df.columns: df['volume'] = 100.0
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df = df.sort_values('datetime').reset_index(drop=True)
    return df

def fetch_xauusd_twelvedata(interval, outputsize):
    try:
        url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval={interval}&outputsize={outputsize}&apikey={TWELVE_API_KEY}"
        r = requests.get(url, timeout=10); r.raise_for_status()
        data = r.json()
        if "values" not in data: raise ValueError("No values")
        return _normalize_df(pd.DataFrame(data["values"]).iloc[::-1].reset_index(drop=True))
    except Exception as e:
        logging.warning(f"TwelveData XAUUSD gagal: {str(e)[:50]}"); return None

def fetch_xauusd_yfinance(interval, outputsize):
    try:
        yf_interval = '5m' if interval == '5min' else '1h'
        period = "60d" if yf_interval == '5m' else "1y"
        df = yf.download("GC=F", period=period, interval=yf_interval, progress=False, threads=False)
        if df.empty: raise ValueError("Empty")
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return _normalize_df(df.tail(outputsize))
    except Exception as e:
        logging.warning(f"yfinance XAUUSD gagal: {str(e)[:50]}"); return None

def fetch_xauusd(interval="5min", outputsize=250):
    for name, func in [("TwelveData", fetch_xauusd_twelvedata), ("yfinance", fetch_xauusd_yfinance)]:
        logging.info(f"Mencoba XAUUSD {interval} via {name}...")
        df = func(interval, outputsize)
        if df is not None and len(df) >= 50:
            logging.info(f"✅ XAUUSD via {name} ({len(df)} bar).")
            return df
        time.sleep(1)
    logging.error("❌ SEMUA sumber XAUUSD gagal."); return None

def fetch_dxy_twelvedata(interval, outputsize):
    try:
        url = f"https://api.twelvedata.com/time_series?symbol=UUP&interval={interval}&outputsize={outputsize}&apikey={TWELVE_API_KEY}"
        r = requests.get(url, timeout=10); r.raise_for_status()
        data = r.json()
        if "values" not in data: raise ValueError("No values")
        return _normalize_df(pd.DataFrame(data["values"]).iloc[::-1].reset_index(drop=True))
    except Exception as e:
        logging.warning(f"TwelveData DXY gagal: {str(e)[:50]}"); return None

def fetch_dxy_yfinance(interval, outputsize):
    try:
        yf_interval = '5m' if interval == '5min' else '1h'
        period = "60d" if yf_interval == '5m' else "1y"
        df = yf.download("UUP", period=period, interval=yf_interval, progress=False, threads=False)
        if df.empty: raise ValueError("Empty")
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return _normalize_df(df.tail(outputsize))
    except Exception as e:
        logging.warning(f"yfinance DXY gagal: {str(e)[:50]}"); return None

def fetch_dxy(interval="5min", outputsize=250):
    for name, func in [("TwelveData", fetch_dxy_twelvedata), ("yfinance", fetch_dxy_yfinance)]:
        logging.info(f"Mencoba DXY via {name}...")
        df = func(interval, outputsize)
        if df is not None and len(df) >= 50:
            logging.info(f"✅ DXY via {name} ({len(df)} bar)."); return df
        time.sleep(1)
    logging.error("❌ SEMUA sumber DXY gagal."); return None

# ============================================================
# HELPERS & INDICATORS
# ============================================================
def load_state():
    default = {"run_count": 0, "last_signal_type": "NEUTRAL", "last_signal_timestamp": ""}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f: 
                c = f.read().strip()
                if c: return {**default, **json.loads(c)}
        except: pass
    return default

def save_state(state):
    temp = STATE_FILE + ".tmp"
    with open(temp, "w") as f: json.dump(state, f, indent=2)
    os.replace(temp, STATE_FILE)

def append_report(report):
    history = []
    if os.path.exists(REPORT_JSON):
        try:
            with open(REPORT_JSON, "r") as f:
                c = f.read().strip()
                if c: history = json.loads(c) if isinstance(json.loads(c), list) else [json.loads(c)]
        except: pass
    history.append(report)
    temp = REPORT_JSON + ".tmp"
    with open(temp, "w") as f: json.dump(history[-500:], f, indent=2)
    os.replace(temp, REPORT_JSON)

def is_market_open(now_utc):
    wd, h = now_utc.weekday(), now_utc.hour
    if wd == 5: return False
    if wd == 6: return h >= 22
    if wd == 4: return h < 21
    return True

def check_news_shield():
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=5).json()
        now = datetime.now(timezone.utc)
        for e in r:
            if e['country'] == 'USD' and e['impact'] == 'High':
                diff = abs((datetime.strptime(e['date'], "%Y-%m-%dT%H:%M:%S%z") - now).total_seconds() / 60)
                if diff <= 30: return True, e['title']
    except Exception as e: logging.warning(f"News API gagal: {str(e)[:50]}")
    return False, ""

def detect_regime(df):
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    v = tr.rolling(14).mean().iloc[-1] / tr.rolling(50).mean().iloc[-1]
    if v > 2.5: return {"mode": "ANOMALY", "threshold": 99}
    if v > 1.25: return {"mode": "TREND", "threshold": 11}
    if v < 0.85: return {"mode": "CHOPPY", "threshold": 14}
    return {"mode": "NORMAL", "threshold": 12}

def run_pillar_analysis(df):
    close = df["close"].astype(float)
    ema9, ema21, ema50 = close.ewm(span=9).mean(), close.ewm(span=21).mean(), close.ewm(span=50).mean()
    scores = [1 if close.iloc[-1] > ema50.iloc[-1] else -1, 1 if ema9.iloc[-1] > ema21.iloc[-1] else -1]
    delta = close.diff()
    rsi = 100 - (100 / (1 + delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean()))
    r = rsi.iloc[-1]
    scores.append(1 if r < 40 else (-1 if r > 60 else 0))
    for i in range(1, 13): scores.append(1 if (close.iloc[-1] - close.iloc[-1-i]) > 0 else -1)
    return sum(scores)

def calculate_adx(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    plus_dm, minus_dm = high.diff(), low.diff()
    plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm > 0] = 0
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/period).mean() / atr))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(period).mean().iloc[-1]

def get_heikin_ashi_trend(df):
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = (df['open'].shift(1) + df['close'].shift(1)) / 2
    return "BULLISH" if ha_close.iloc[-1] > ha_open.iloc[-1] else "BEARISH"

def calculate_dynamic_tpsl_and_lot(df, signal_type, regime_mode):
    atr_val = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
    price = float(df["close"].iloc[-1])
    tp_mult = 3.0 if regime_mode == "TREND" else (1.5 if regime_mode == "CHOPPY" else 2.0)
    if signal_type == "BUY":
        sl, tp, be = round(price - atr_val*1.5, 2), round(price + atr_val*tp_mult, 2), round(price + atr_val, 2)
    elif signal_type == "SELL":
        sl, tp, be = round(price + atr_val*1.5, 2), round(price - atr_val*tp_mult, 2), round(price - atr_val, 2)
    else: return 0.0, 0.0, 0.0, 0.0, 0.0
    lot = round(max(0.01, min(5.0, (float(os.getenv("ACCOUNT_BALANCE", "10000")) * (float(os.getenv("RISK_PERCENT", "1"))/100)) / (abs(price-sl)*100))), 2)
    return sl, tp, be, round(atr_val, 2), lot

# ============================================================
# LIVE ENGINE
# ============================================================
def run_live_engine():
    now_utc = datetime.now(timezone.utc)
    state = load_state()
    state["run_count"] += 1
    logging.info(f"=== Apex Hybrid v6.0 Live Run #{state['run_count']} ===")

    if not is_market_open(now_utc): logging.info("Market closed."); save_state(state); return
    if check_news_shield()[0]: logging.warning("🛑 NEWS SHIELD AKTIF."); save_state(state); return

    df_m5, df_h1, df_dxy = fetch_xauusd("5min", 250), fetch_xauusd("1h", 100), fetch_dxy("5min", 250)
    if any(x is None for x in [df_m5, df_h1, df_dxy]): logging.error("Data gagal. Blocked."); save_state(state); return
        
    df_m5, df_h1 = df_m5.iloc[:-1].reset_index(drop=True), df_h1.iloc[:-1].reset_index(drop=True)
    regime = detect_regime(df_m5)
    if regime["mode"] == "ANOMALY": logging.warning("Anomali. Blocked."); save_state(state); return

    h1_trend = "BULLISH" if df_h1['close'].iloc[-1] > df_h1['close'].ewm(span=50).mean().iloc[-1] else "BEARISH"
    dxy_trend = "BEARISH" if df_dxy['close'].iloc[-1] < df_dxy['close'].ewm(span=50).mean().iloc[-1] else "BULLISH"
    tech_score, adx_val, ha_trend, threshold = run_pillar_analysis(df_m5), calculate_adx(df_m5), get_heikin_ashi_trend(df_m5), regime["threshold"]

    signal = "NEUTRAL"
    if tech_score >= threshold and h1_trend == "BULLISH" and dxy_trend == "BEARISH" and adx_val > 25 and ha_trend == "BULLISH": signal = "BUY"
    elif tech_score <= -threshold and h1_trend == "BEARISH" and dxy_trend == "BULLISH" and adx_val > 25 and ha_trend == "BEARISH": signal = "SELL"

    notify = False
    if signal != "NEUTRAL" and (signal != state["last_signal_type"] or (now_utc - datetime.strptime(state["last_signal_timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)).total_seconds() > COOLDOWN_MINUTES*60):
        notify = True
        state["last_signal_type"], state["last_signal_timestamp"] = signal, now_utc.strftime("%Y-%m-%d %H:%M:%S")
    save_state(state)

    price = float(df_m5["close"].iloc[-1])
    sl, tp, be, atr_val, lot = calculate_dynamic_tpsl_and_lot(df_m5, signal, regime["mode"])
    append_report({"timestamp": now_utc.strftime("%Y-%m-%d %H:%M:%S"), "run_count": state["run_count"], "regime": regime["mode"], "h1": h1_trend, "dxy": dxy_trend, "adx": round(adx_val, 2), "tech": tech_score, "signal": signal, "entry": price, "sl": sl, "tp": tp, "be": be, "lot": lot})

    if notify:
        msg = (
            f"⚛️ *APEX HYBRID V6.0* ⚛️\n\n"
            f"🎯 *Signal:* *{signal}*\n"
            f"📌 *Entry:* `{price}` | ⚖️ *Lot:* `{lot}`\n"
            f"🛑 *SL:* `{sl}` | ✅ *TP:* `{tp}`\n"
            f"⚖️ *BE Alert:* Geser SL ke `{be}` jika tersentuh.\n\n"
            f"📈 *H1:* `{h1_trend}` | 🌍 *DXY:* `{dxy_trend}`\n"
            f"📊 *ADX:* `{adx_val:.2f}` | *HA:* `{ha_trend}`\n"
            f"⏰ *Time:* `{now_utc.strftime('%Y-%m-%d %H:%M')} UTC`"
        )
        token, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
        if token and chat: requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat, "text": msg, "parse_mode": "Markdown"})
        logging.info(f"Sinyal {signal} dikirim!")
    else: logging.info(f"Signal: {signal}. No dispatch.")

# ============================================================
# BACKTEST ENGINE (FOR WEEKEND)
# ============================================================
def run_backtest_engine():
    logging.info("=== Running Weekly Backtest ===")
    df = yf.download("GC=F", period="1y", interval="1h", progress=False, threads=False)
    if df.empty: logging.error("Gagal load data historis."); return
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]

    close = df['close']
    df['ema50'] = close.ewm(span=50).mean()
    high, low = df['high'], df['low']
    plus_dm, minus_dm = high.diff(), low.diff()
    plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm > 0] = 0
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/14).mean() / atr))
    df['adx'] = ((abs(plus_di - minus_di) / (plus_di + minus_di)) * 100).rolling(14).mean()

    wins, losses = 0, 0
    for i in range(50, len(df)-1):
        row = df.iloc[i]
        if pd.isna(row['adx']): continue
        signal = "BUY" if (row['close'] > row['ema50'] and row['adx'] > 25) else ("SELL" if (row['close'] < row['ema50'] and row['adx'] > 25) else "NEUTRAL")
        if signal != "NEUTRAL":
            tp = row['close'] + (atr.iloc[i]*2.0) if signal == "BUY" else row['close'] - (atr.iloc[i]*2.0)
            sl = row['close'] - (atr.iloc[i]*1.5) if signal == "BUY" else row['close'] + (atr.iloc[i]*1.5)
            if signal == "BUY":
                if df['high'].iloc[i+1] >= tp: wins += 1
                elif df['low'].iloc[i+1] <= sl: losses += 1
            else:
                if df['low'].iloc[i+1] <= tp: wins += 1
                elif df['high'].iloc[i+1] >= sl: losses += 1

    total = wins + losses
    wr = (wins / total * 100) if total > 0 else 0
    msg = f"📊 *LAPORAN AUTO-BACKTEST*\n\nTrade: `{total}`\nWin Rate: `{wr:.1f}%`\n"
    token, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat: requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat, "text": msg, "parse_mode": "Markdown"})

# ============================================================
# EXECUTION ENTRY POINT
# ============================================================
if __name__ == "__main__":
    # Jika dijalankan oleh workflow backtest (via env var), jalankan backtest
    if os.getenv("RUN_BACKTEST") == "true":
        run_backtest_engine()
    else:
        run_live_engine()
