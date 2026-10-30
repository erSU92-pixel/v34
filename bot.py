import os
import re
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import xgboost as xgb
from data_provider import fetch_xauusd, fetch_dxy

# --- KONFIGURASI ---
STATE_FILE = "signal_state.json"
REPORT_JSON = "xauusd_apex_report.json"
COOLDOWN_MINUTES = 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [ApexV3] - %(message)s")

def _redact(text): return re.sub(r"apikey=[^&\s'\"]+", "apikey=***", str(text))

def load_state():
    default = {"last_signal_type": "NEUTRAL", "last_signal_timestamp": ""}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f: return {**default, **json.load(f)}
        except: pass
    return default

def save_state(state):
    with open(STATE_FILE, "w") as f: json.dump(state, f, indent=2)

def append_report(report):
    history = []
    if os.path.exists(REPORT_JSON):
        try:
            with open(REPORT_JSON, "r") as f:
                content = f.read().strip()
                if content: 
                    data = json.loads(content)
                    history = data if isinstance(data, list) else [data]
        except: pass
    history.append(report)
    with open(REPORT_JSON, "w") as f: json.dump(history[-500:], f, indent=2)

def is_market_open(now_utc):
    wd, h = now_utc.weekday(), now_utc.hour
    if wd == 5: return False
    if wd == 6: return h >= 22
    if wd == 4: return h < 21
    return True

def check_news_shield():
    """Memblokir sinyal 30 menit sebelum/sesudah berita High Impact USD"""
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        r = requests.get(url, timeout=5).json()
        now = datetime.now(timezone.utc)
        for event in r:
            if event['country'] == 'USD' and event['impact'] == 'High':
                event_time = datetime.strptime(event['date'], "%Y-%m-%dT%H:%M:%S%z")
                diff = abs((event_time - now).total_seconds() / 60)
                if diff <= 30:
                    return True, event['title']
    except Exception as e:
        logging.warning(f"News API gagal, bypass shield: {str(e)[:50]}")
    return False, ""

def run_ml_gate(df, dxy_df):
    """XGBoost + DXY Correlation + Predict Magnitude (> 0.5 ATR)"""
    d = df.copy().reset_index(drop=True)
    d['ret1'] = d['close'].pct_change()
    d['dxy_ret'] = dxy_df['close'].pct_change()
    
    tr = pd.concat([d['high'] - d['low'], (d['high'] - d['close'].shift()).abs(), (d['low'] - d['close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    d['target'] = (d['close'].shift(-1) > d['close'] + (atr * 0.5)).astype(int)
    
    feats = ['ret1', 'dxy_ret', 'close']
    train = d.dropna(subset=feats + ['target'])
    if len(train) < 50: return False, 0.0
    
    X, y = train[feats], train['target']
    if y.nunique() < 2: return False, 0.0
    
    clf = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=1)
    clf.fit(X, y)
    prob = clf.predict_proba(d[feats].tail(1))[0][1]
    return (prob > 0.60), prob

def calculate_lot(balance, risk_pct, sl_distance):
    """Dynamic Lot Calculator (Standar XAUUSD: $1 move = $100/lot)"""
    risk_amount = balance * (risk_pct / 100)
    lot = risk_amount / (sl_distance * 100)
    return round(max(0.01, min(10.0, lot)), 2)

def main():
    now_utc = datetime.now(timezone.utc)
    if not is_market_open(now_utc):
        logging.info("Market closed. Skip.")
        return

    # 1. Fetch Data via Failover
    df_m5, src_m5 = fetch_xauusd("5min", 250)
    df_h1, src_h1 = fetch_xauusd("1h", 100)
    df_dxy, src_dxy = fetch_dxy("5min", 250)

    if df_m5 is None or df_h1 is None or df_dxy is None:
        logging.error("Gagal total mendapatkan data. Sinyal diblokir.")
        return
        
    df_m5 = df_m5.iloc[:-1].reset_index(drop=True) # Drop candle running

    # 2. News Shield
    is_news, news_title = check_news_shield()
    if is_news:
        logging.warning(f"🛑 NEWS SHIELD AKTIF: {news_title}.")
        return

    # 3. Indikator 18 Pilar & H1 Confluence
    close = df_m5['close']
    ema50 = close.ewm(span=50, adjust=False).mean()
    h1_close = df_h1['close']
    h1_ema50 = h1_close.ewm(span=50, adjust=False).mean()
    
    h1_trend = "BULLISH" if h1_close.iloc[-1] > h1_ema50.iloc[-1] else "BEARISH"
    tech_score = 1 if close.iloc[-1] > ema50.iloc[-1] else -1
    if h1_trend == "BULLISH" and tech_score > 0: tech_score += 2
    if h1_trend == "BEARISH" and tech_score < 0: tech_score -= 2

    # 4. DXY Filter
    dxy_close = df_dxy['close']
    dxy_ema50 = dxy_close.ewm(span=50, adjust=False).mean()
    dxy_trend = "BEARISH" if dxy_close.iloc[-1] < dxy_ema50.iloc[-1] else "BULLISH"

    # 5. ML Gate
    ml_valid, ml_prob = run_ml_gate(df_m5, df_dxy)

    # 6. Evaluasi Sinyal
    signal = "NEUTRAL"
    if tech_score >= 3 and dxy_trend == "BEARISH" and ml_valid:
        signal = "BUY"
    elif tech_score <= -3 and dxy_trend == "BULLISH" and ml_valid:
        signal = "SELL"

    # 7. Cooldown & Dispatch
    state = load_state()
    if signal != "NEUTRAL" and (signal != state["last_signal_type"] or (now_utc - datetime.strptime(state["last_signal_timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)).total_seconds() > COOLDOWN_MINUTES*60):
        
        atr = (df_m5['high'] - df_m5['low']).rolling(14).mean().iloc[-1]
        price = close.iloc[-1]
        sl_dist = atr * 1.5
        tp_dist = atr * 3.0
        
        sl = round(price - sl_dist, 2) if signal == "BUY" else round(price + sl_dist, 2)
        tp = round(price + tp_dist, 2) if signal == "BUY" else round(price - tp_dist, 2)
        be_level = round(price + (sl_dist if signal == "BUY" else -sl_dist), 2)
        
        balance = float(os.getenv("ACCOUNT_BALANCE", "10000"))
        risk = float(os.getenv("RISK_PERCENT", "1"))
        lot = calculate_lot(balance, risk, sl_dist)

        msg = (
            f"🚀 *APEX V3.0 - INSTITUTIONAL* 🚀\n\n"
            f"🎯 *Signal:* *{signal}* ({src_m5})\n"
            f"📌 *Entry:* `{price}`\n"
            f"🛑 *SL:* `{sl}` (Jarak: {sl_dist:.2f})\n"
            f"✅ *TP:* `{tp}`\n"
            f"💰 *Saran Lot:* `{lot}` (Risiko {risk}% dari ${balance})\n\n"
            f"🌍 *DXY Trend:* `{dxy_trend}`\n"
            f"📈 *H1 Trend:* `{h1_trend}`\n"
            f"🧠 *XGBoost Prob:* `{ml_prob*100:.1f}%`\n\n"
            f"⚠️ *TRAILING STOP:* Jika harga menyentuh `{be_level}`, geser SL ke Entry."
        )
        
        token, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
        if token and chat:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat, "text": msg, "parse_mode": "Markdown"})
        
        state["last_signal_type"] = signal
        state["last_signal_timestamp"] = now_utc.strftime("%Y-%m-%d %H:%M:%S")
        save_state(state)
        
        append_report({
            "timestamp": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "signal": signal, "entry": price, "sl": sl, "tp": tp, 
            "lot": lot, "dxy": dxy_trend, "h1": h1_trend, "ml_prob": round(ml_prob, 2)
        })
        logging.info(f"Sinyal {signal} dikirim via {src_m5}!")
    else:
        logging.info(f"Signal: {signal}. No dispatch (Cooldown/Steril).")

if __name__ == "__main__":
    main()
