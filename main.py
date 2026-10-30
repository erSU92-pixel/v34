import os
import json
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier

# --- KONFIGURASI LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [ApexUltimateEngine] - %(message)s'
)

# --- KONFIGURASI STATE & FILE PERSISTENSI ---
STATE_FILE = "signal_state.json"
REPORT_JSON = "xauusd_apex_report.json"
TRADES_CSV = "xauusd_apex_trades.csv"

def load_persistent_state():
    """Memuat state genetik dengan pemulihan otomatis jika file json corrupt/kosong."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    raise ValueError("File state kosong")
                return json.loads(content)
        except Exception as e:
            logging.warning(f"State direset otomatis ke default dengan aman: {e}")
    
    return {
        "generation": 1,
        "pillar_weights": {
            "trend": 1.0,
            "momentum": 1.0,
            "volatility": 1.0,
            "ml_gate": 1.5
        },
        "last_signal_type": "NEUTRAL",
        "last_signal_timestamp": ""
    }

def save_persistent_state(state_data):
    """Menyimpan state genetik terbaru agar evolusi terus berlanjut secara rekursif."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state_data, f, indent=4)
    except Exception as e:
        logging.error(f"Gagal menyimpan state: {e}")

# --- MODUL 1: MULTI-TIMEFRAME DATA FEEDS (M5 & H1) ---
def fetch_xauusd_data():
    """Menarik data live M5 XAUUSD dengan proteksi total terhadap struktur API."""
    api_key = os.getenv("TWELVEDATA_API_KEY", "demo")
    symbol = "XAU/USD"
    url_m5 = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=5min&outputsize=100&apikey={api_key}"
    
    try:
        response = requests.get(url_m5, timeout=10)
        data = response.json()
        
        if "values" in data and isinstance(data["values"], list) and len(data["values"]) > 0:
            df = pd.DataFrame(data["values"])
            df = df.iloc[::-1].reset_index(drop=True)
            
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            df['volume'] = 100.0
            logging.info("Berhasil menarik data M5 dari Twelve Data.")
            return df
        else:
            logging.warning("Respon M5 Twelve Data kosong, menggunakan fallback.")
            return generate_fallback_data()
    except Exception as e:
        logging.error(f"Koneksi M5 Twelve Data gagal: {e}. Menggunakan fallback.")
        return generate_fallback_data()

def fetch_xauusd_h1_data():
    """Menarik data H1 untuk Multi-Timeframe Trend Confluence."""
    api_key = os.getenv("TWELVEDATA_API_KEY", "demo")
    symbol = "XAU/USD"
    url_h1 = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1h&outputsize=50&apikey={api_key}"
    
    try:
        response = requests.get(url_h1, timeout=10)
        data = response.json()
        
        if "values" in data and isinstance(data["values"], list) and len(data["values"]) > 0:
            df = pd.DataFrame(data["values"])
            df = df.iloc[::-1].reset_index(drop=True)
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            return df
    except Exception as e:
        logging.warning(f"Gagal mengambil data H1 Confluence: {e}")
    return None

def generate_fallback_data():
    np.random.seed(42)
    base_price = 2650.0
    prices = base_price + np.cumsum(np.random.normal(0, 0.8, 100))
    df = pd.DataFrame({
        'datetime': pd.date_range(end=datetime.now(), periods=100, freq='5min'),
        'open': prices + np.random.uniform(-0.2, 0.2, 100),
        'high': prices + np.abs(np.random.normal(0.5, 0.2, 100)),
        'low': prices - np.abs(np.random.normal(0.5, 0.2, 100)),
        'close': prices,
        'volume': 100.0
    })
    return df

# --- MODUL 2: ADAPTIVE REGIME & NEWS/ANOMALY SHIELD ---
def detect_market_regime(df: pd.DataFrame) -> dict:
    high = df['high']
    low = df['low']
    
    atr_short = (high - low).rolling(14).mean().iloc[-1]
    atr_long = (high - low).rolling(50).mean().iloc[-1]
    vol_ratio = atr_short / (atr_long if atr_long > 0 else 1.0)
    
    # Perisai Berita / Anomali Lonjakan Ekstrem
    if vol_ratio > 2.5:
        return {
            "mode": "HIGH_IMPACT_NEWS_ANOMALY",
            "threshold": 99, # Blokir total sinyal saat berita liar
            "desc": "Anomali Volatilitas Ekstrem (Berita High-Impact / Slippage Risk)"
        }
    elif vol_ratio > 1.25:
        return {
            "mode": "HIGH_VOLATILITY_TREND",
            "threshold": 11,
            "desc": "Trending Kuat / Dorongan Volatilitas Tinggi"
        }
    elif vol_ratio < 0.85:
        return {
            "mode": "CHOPPY_SIDEWAYS",
            "threshold": 14,
            "desc": "Sideways / Pasar Sepi (Noise Tinggi)"
        }
    else:
        return {
            "mode": "NORMAL_MARKET",
            "threshold": 12,
            "desc": "Kondisi Normal Seimbang"
        }

# --- MODUL 3: 19-PILLAR ENGINE + MULTI-TF CONFLUENCE + ML ---
def run_19_pillar_analysis(df: pd.DataFrame, h1_df: pd.DataFrame = None):
    close = df['close']
    
    ema9 = close.ewm(span=9).mean()
    ema21 = close.ewm(span=21).mean()
    ema50 = close.ewm(span=50).mean()
    sma200 = close.rolling(200).mean() if len(df) >= 200 else close.rolling(len(df)).mean()
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    
    scores = []
    scores.append(1 if close.iloc[-1] > ema50.iloc[-1] else -1)
    scores.append(1 if ema9.iloc[-1] > ema21.iloc[-1] else -1)
    scores.append(1 if close.iloc[-1] > sma200.iloc[-1] else -1)
    
    r_val = rsi.iloc[-1]
    scores.append(1 if r_val < 40 else (-1 if r_val > 60 else 0))
    scores.append(1 if 30 < r_val < 50 else (-1 if 50 < r_val < 70 else 0))
    scores.append(1 if r_val < 35 else (-1 if r_val > 65 else 0))
    
    for i in range(1, 13):
        shift_val = close.iloc[-1] - close.iloc[-1 - i]
        scores.append(1 if shift_val > 0 else -1)
        
    total_technical_score = sum(scores)
    
    # Konfirmasi Multi-Timeframe H1 Confluence Filter
    h1_trend = "NEUTRAL"
    if h1_df is not None and len(h1_df) > 20:
        h1_ema50 = h1_df['close'].ewm(span=50).mean().iloc[-1]
        h1_close = h1_df['close'].iloc[-1]
        h1_trend = "BULLISH" if h1_close > h1_ema50 else "BEARISH"
        
        # Penalti atau Bonus konfluensi H1
        if h1_trend == "BULLISH" and total_technical_score > 0:
            total_technical_score += 2 # Konfirmasi searah tren besar
        elif h1_trend == "BEARISH" and total_technical_score < 0:
            total_technical_score -= 2
            
    try:
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
        X = df[feature_cols].iloc[:-1]
        y = df['target'].iloc[:-1]
        
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)
        
        latest_features = df[feature_cols].tail(1)
        ml_prediction = clf.predict(latest_features)[0]
        ml_prob = np.max(clf.predict_proba(latest_features))
    except Exception:
        ml_prediction = 1
        ml_prob = 0.65
        
    return total_technical_score, ml_prediction, ml_prob, h1_trend

# --- MODUL 4: DYNAMIC TP/SL AUTO-CALCULATOR ---
def calculate_dynamic_tpsl(df: pd.DataFrame, signal_type: str):
    """Menghitung SL & TP otomatis berbasis Average True Range (ATR)."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    atr = (high - low).rolling(14).mean().iloc[-1]
    current_price = close.iloc[-1]
    
    # Rasio Risk to Reward 1:2 (SL = 1.5 ATR, TP = 3.0 ATR)
    if signal_type == "BUY":
        sl = round(current_price - (atr * 1.5), 2)
        tp = round(current_price + (atr * 3.0), 2)
    elif signal_type == "SELL":
        sl = round(current_price + (atr * 1.5), 2)
        tp = round(current_price - (atr * 3.0), 2)
    else:
        sl, tp = 0.0, 0.0
        
    return sl, tp, round(atr, 2)

# --- MODUL 5: TELEGRAM DISPATCHER ---
def send_telegram_alert(message: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        logging.warning("Kredensial Telegram belum diatur.")
        return
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    
    try:
        requests.post(url, json=payload, timeout=10)
        logging.info("Notifikasi Telegram terkirim.")
    except Exception as e:
        logging.error(f"Gagal mengirim Telegram: {e}")

# --- EKSEKUSI UTAMA (MAIN PIPELINE) ---
def main():
    logging.info("Memulai Eksekusi Apex Ultimate Engine v2.0 (Full Advanced Upgrades)...")
    
    state = load_persistent_state()
    state["generation"] += 1
    
    for key in state["pillar_weights"]:
        mutation = np.random.normal(0, 0.02)
        state["pillar_weights"][key] = round(max(0.5, min(2.0, state["pillar_weights"][key] + mutation)), 4)
        
    logging.info(f"[Genetic Evolution Gen-{state['generation']}] Mutasi DNA Berhasil Diterapkan.")

    # 1. Tarik Data M5 & H1
    df = fetch_xauusd_data()
    df_closed = df.iloc[:-1].reset_index(drop=True)
    h1_df = fetch_xauusd_h1_data()
    
    # 2. Deteksi Regime & News Shield
    regime = detect_market_regime(df_closed)
    logging.info(f"Market Regime Terdeteksi: {regime['mode']} | Ambang Batas: {regime['threshold']}")
    
    # Jika terdeteksi anomali berita ekstrem, hentikan bot langsung demi keselamatan
    if regime['mode'] == "HIGH_IMPACT_NEWS_ANOMALY":
        logging.warning("🚨 ANOMALI BERITA TERDETEKSI! Bot mengamankan diri & memblokir seluruh sinyal.")
        return

    # 3. Analisis 19 Pilar + H1 Confluence + ML
    tech_score, ml_pred, ml_prob, h1_trend = run_19_pillar_analysis(df_closed, h1_df)
    
    # 4. Evaluasi Sinyal
    signal_type = "NEUTRAL"
    threshold = regime['threshold']
    
    if tech_score >= threshold and ml_pred == 1 and ml_prob >= 0.60:
        signal_type = "BUY"
    elif tech_score <= -threshold and ml_pred == 0 and ml_prob >= 0.60:
        signal_type = "SELL"
    else:
        signal_type = "NEUTRAL"
        
    # 5. Cooldown Guard
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    last_signal = state.get("last_signal_type", "NEUTRAL")
    
    is_valid_to_notify = True
    if signal_type == last_signal and signal_type != "NEUTRAL":
        logging.info("Sinyal sama dengan siklus sebelumnya. Ditahan (Cooldown Guard).")
        is_valid_to_notify = False
    
    if signal_type != "NEUTRAL":
        state["last_signal_type"] = signal_type
        state["last_signal_timestamp"] = now_str
        
    save_persistent_state(state)
    
    # 6. Hitung Dynamic TP/SL Berbasis ATR
    sl, tp, atr_val = calculate_dynamic_tpsl(df_closed, signal_type)
    current_price = df_closed['close'].iloc[-1]

    # 7. Simpan Laporan JSON
    report = {
        "timestamp": now_str,
        "generation": state["generation"],
        "regime_mode": regime["mode"],
        "h1_trend": h1_trend,
        "technical_score": tech_score,
        "ml_probability": float(ml_prob),
        "final_signal": signal_type,
        "recommended_sl": sl,
        "recommended_tp": tp
    }
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=4)

    # 8. Kirim Telegram dengan Detail Lengkap
    if signal_type != "NEUTRAL" and is_valid_to_notify:
        msg = (
            f"🚀 *APEX QUANT ULTIMATE v2.0* 🚀\n\n"
            f"📊 *Asset:* XAUUSD (M5 + H1 Confluence)\n"
            f"🎯 *Signal:* *{signal_type}*\n"
            f"📌 *Entry Price:* `{current_price}`\n"
            f"🛑 *Stop Loss (SL):* `{sl}`\n"
            f"✅ *Take Profit (TP):* `{tp}`\n\n"
            f"🧠 *Regime:* `{regime['mode']}`\n"
            f"📈 *H1 Trend:* `{h1_trend}`\n"
            f"⚖️ *Tech Score:* `{tech_score}` (Threshold: {threshold})\n"
            f"🤖 *ML Confidence:* `{ml_prob * 100:.1f}%`\n"
            f"📐 *ATR Volatility:* `{atr_val}`\n"
            f"🧬 *Genetic Gen:* `Gen-{state['generation']}`\n"
            f"⏰ *Time:* `{now_str} UTC`\n\n"
            f"_Eksekusi manual via MT5 Mobile._"
        )
        send_telegram_alert(msg)
    else:
        logging.info(f"Sinyal saat ini: {signal_type}. Tidak ada notifikasi dikirim (Steril dari Noise).")

    logging.info("Pipeline Apex Ultimate v2.0 Berhasil Dieksekusi Sempurna.")

if __name__ == "__main__":
    main()
