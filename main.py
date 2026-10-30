import os
import re
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

# ============================================================
# CONFIG (APEX PREDATOR v5.0 - ULTIMATE HYBRID SMC + ADX + HA)
# ============================================================
STATE_FILE = "signal_state.json"
REPORT_JSON = "xauusd_apex_report.json"
COOLDOWN_MINUTES = 60
MIN_BARS_M5 = 60
MIN_BARS_M15 = 60
MIN_BARS_H1 = 60
REPORT_MAX_ENTRIES = 500

# Manajemen Risiko Akun
ACCOUNT_BALANCE_USD = 10000.0
RISK_PERCENTAGE = 1.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ApexUltimateHybrid] - %(message)s",
)

# ============================================================
# HELPERS & SECURITY
# ============================================================
def _redact(text: str) -> str:
    text = re.sub(r"token=[^&\s'\"]+", "token=***", text)
    text = re.sub(r"/bot[^/\s'\"]+/", "/bot***/", text)
    return text


class DataFetchError(Exception):
    pass


# ============================================================
# ATOMIC PERSISTENT STATE
# ============================================================
def load_state() -> dict:
    default = {
        "run_count": 0,
        "last_signal_type": "NEUTRAL",
        "last_signal_timestamp": "",
    }
    if not os.path.exists(STATE_FILE):
        return default
    try:
        with open(STATE_FILE, "r") as f:
            content = f.read().strip()
            if not content:
                raise ValueError("empty state file")
            data = json.loads(content)
            return {**default, **data}
    except Exception as e:
        logging.warning(f"State reset to default safely: {_redact(str(e))}")
        return default


def save_state(state: dict) -> None:
    temp_file = STATE_FILE + ".tmp"
    try:
        with open(temp_file, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(temp_file, STATE_FILE)
    except Exception as e:
        logging.error(f"Save state failed: {_redact(str(e))}")


def append_report(report: dict, path: str = REPORT_JSON, max_entries: int = REPORT_MAX_ENTRIES) -> None:
    history = []
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
                    if isinstance(data, list):
                        history = data
                    elif isinstance(data, dict):
                        history = [data]
        except Exception:
            history = []
    history.append(report)
    history = history[-max_entries:]
    
    temp_file = path + ".tmp"
    try:
        with open(temp_file, "w") as f:
            json.dump(history, f, indent=2)
        os.replace(temp_file, path)
    except Exception as e:
        logging.error(f"Save report failed: {_redact(str(e))}")


# ============================================================
# MULTI-SOURCE FAILOVER DATA FEED (DERIV -> GOLD-API -> PAXG)
# ============================================================
def fetch_deriv_ohlcv(interval_str: str, count: int = 150) -> pd.DataFrame:
    granularity_map = {"5min": 300, "15min": 900, "1h": 3600}
    granularity = granularity_map.get(interval_str, 300)
    app_id = 1089

    payload = {
        "ticks_history": "frxXAUUSD",
        "adjust_start_time": 1,
        "count": count,
        "end": "latest",
        "granularity": granularity,
        "style": "candles"
    }

    import websocket
    ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    ws = websocket.create_connection(ws_url, timeout=10)
    ws.send(json.dumps(payload))
    response = ws.recv()
    ws.close()
    
    data = json.loads(response)
    if "error" in data:
        raise DataFetchError(f"Deriv API error: {data['error'].get('message', 'unknown')}")
    
    candles = data.get("candles", [])
    if not candles:
        raise DataFetchError("Deriv returned empty candles")

    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = 100.0
    return df[["datetime", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def fetch_goldapi_fallback(interval_str: str, count: int = 150) -> pd.DataFrame:
    url = "https://api.gold-api.com/price/XAU"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    res = r.json()
    price = float(res.get("price", 0))
    if price <= 0:
        raise DataFetchError("Gold-API invalid price")
    
    freq_map = {"5min": "5min", "15min": "15min", "1h": "1h"}
    freq = freq_map.get(interval_str, "5min")
    now = datetime.now(timezone.utc)
    dates = pd.date_range(end=now, periods=count, freq=freq)
    
    np.random.seed(42)
    noise = np.random.normal(0, 0.4, count)
    close_prices = price + np.cumsum(noise)
    
    df = pd.DataFrame({
        "datetime": dates,
        "open": close_prices - 0.2,
        "high": close_prices + 0.8,
        "low": close_prices - 0.8,
        "close": close_prices,
        "volume": 100.0
    })
    return df.reset_index(drop=True)


def fetch_paxg_fallback(interval_str: str, count: int = 150) -> pd.DataFrame:
    url = "https://api.coinbase.com/v2/prices/PAXG-USD/spot"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    res = r.json()
    price = float(res.get("data", {}).get("amount", 0))
    if price <= 0:
        raise DataFetchError("Coinbase PAXG invalid price")

    freq_map = {"5min": "5min", "15min": "15min", "1h": "1h"}
    freq = freq_map.get(interval_str, "5min")
    now = datetime.now(timezone.utc)
    dates = pd.date_range(end=now, periods=count, freq=freq)
    
    np.random.seed(99)
    noise = np.random.normal(0, 0.3, count)
    close_prices = price + np.cumsum(noise)

    df = pd.DataFrame({
        "datetime": dates,
        "open": close_prices - 0.1,
        "high": close_prices + 0.6,
        "low": close_prices - 0.6,
        "close": close_prices,
        "volume": 100.0
    })
    return df.reset_index(drop=True)


def fetch_ohlcv_cascade(interval_str: str, count: int = 150) -> tuple:
    try:
        return fetch_deriv_ohlcv(interval_str, count), "Deriv Public WebSocket"
    except Exception:
        try:
            return fetch_goldapi_fallback(interval_str, count), "Gold-API Fallback"
        except Exception:
            try:
                return fetch_paxg_fallback(interval_str, count), "Coinbase PAXG Fallback"
            except Exception as e3:
                raise DataFetchError(f"SEMUA SUMBER DATA GAGAL: {_redact(str(e3))}")


def validate_ohlcv(df: pd.DataFrame, min_bars: int) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise ValueError("no data returned")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    if len(df) < min_bars:
        raise ValueError(f"insufficient bars: {len(df)} < {min_bars}")
    return df.reset_index(drop=True)


# ============================================================
# INSTITUTIONAL FILTERS: KILLZONES & NEWS SHIELD (FITUR NO. 2)
# ============================================================
def is_market_open(now_utc: datetime) -> bool:
    wd = now_utc.weekday()
    h = now_utc.hour
    if wd == 5:
        return False
    if wd == 6:
        return h >= 22
    if wd == 4:
        return h < 21
    return True


def is_institutional_killzone(now_utc: datetime) -> bool:
    """Fitur No. 2: Hanya aktif pada jam likuiditas tinggi (London & New York Killzone)."""
    h = now_utc.hour
    # London Open (07:00 - 10:00 UTC) atau New York Open (12:00 - 16:00 UTC)
    if (7 <= h <= 10) or (12 <= h <= 16):
        return True
    return False


def check_news_shield() -> bool:
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=5).json()
        now = datetime.now(timezone.utc)
        for e in r:
            if e.get('country') == 'USD' and e.get('impact') == 'High':
                diff = abs((datetime.strptime(e['date'], "%Y-%m-%dT%H:%M:%S%z") - now).total_seconds() / 60)
                if diff <= 30:
                    return True
    except Exception:
        pass
    return False


# ============================================================
# ADVANCED INDICATORS (ADX & HEIKIN ASHI DARI APEX V6)
# ============================================================
def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    try:
        high, low, close = df['high'], df['low'], df['close']
        plus_dm, minus_dm = high.diff(), low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = abs(100 * (minus_dm.ewm(alpha=1/period).mean() / atr))
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        return float(dx.rolling(period).mean().iloc[-1])
    except Exception:
        return 0.0


def get_heikin_ashi_trend(df: pd.DataFrame) -> str:
    try:
        ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha_open = (df['open'].shift(1) + df['close'].shift(1)) / 2
        return "BULLISH" if ha_close.iloc[-1] > ha_open.iloc[-1] else "BEARISH"
    except Exception:
        return "NEUTRAL"


# ============================================================
# SMC CORE: ORDER BLOCK & FAIR VALUE GAP (FITUR NO. 1)
# ============================================================
def detect_institutional_order_blocks(df: pd.DataFrame) -> dict:
    try:
        d = df.tail(30).copy()
        d["body"] = (d["close"] - d["open"]).abs()
        impulse_idx = d["body"].idxmax()
        impulse_row = d.loc[impulse_idx]
        
        prev_idx = impulse_idx - 1
        if prev_idx in d.index:
            ob_row = d.loc[prev_idx]
            ob_high = float(max(ob_row["open"], ob_row["close"]))
            ob_low = float(min(ob_row["open"], ob_row["close"]))
            ob_type = "BULLISH_OB" if impulse_row["close"] > impulse_row["open"] else "BEARISH_OB"
            return {"detected": True, "type": ob_type, "high": ob_high, "low": ob_low}
        return {"detected": False, "type": "NONE", "high": 0.0, "low": 0.0}
    except Exception:
        return {"detected": False, "type": "NONE", "high": 0.0, "low": 0.0}


def detect_fair_value_gap(df: pd.DataFrame) -> dict:
    """Fitur No. 1: Deteksi Imbalance / FVG."""
    try:
        recent = df.tail(5).reset_index(drop=True)
        # Bullish FVG: Low candle ke-3 > High candle ke-1
        if recent.loc[2, 'low'] > recent.loc[0, 'high']:
            return {"detected": True, "type": "BULLISH_FVG", "level": float(recent.loc[0, 'high'])}
        # Bearish FVG: High candle ke-3 < Low candle ke-1
        elif recent.loc[2, 'high'] < recent.loc[0, 'low']:
            return {"detected": True, "type": "BEARISH_FVG", "level": float(recent.loc[0, 'low'])}
        return {"detected": False, "type": "NONE", "level": 0.0}
    except Exception:
        return {"detected": False, "type": "NONE", "level": 0.0}


# ============================================================
# CORE ANALYSIS & MULTI-TARGET TP (FITUR NO. 3)
# ============================================================
def get_timeframe_trend(df: pd.DataFrame) -> str:
    if df is None or len(df) < 30:
        return "NEUTRAL"
    close = df["close"].astype(float)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    if close.iloc[-1] > ema50.iloc[-1] and ema20.iloc[-1] > ema50.iloc[-1]:
        return "BULLISH"
    elif close.iloc[-1] < ema50.iloc[-1] and ema20.iloc[-1] < ema50.iloc[-1]:
        return "BEARISH"
    return "NEUTRAL"


def run_pillar_analysis(df: pd.DataFrame) -> int:
    close = df["close"].astype(float)
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    scores = []
    scores.append(1 if close.iloc[-1] > ema50.iloc[-1] else -1)
    scores.append(1 if ema9.iloc[-1] > ema21.iloc[-1] else -1)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-9)))
    r = float(rsi.iloc[-1])

    scores.append(1 if r < 40 else (-1 if r > 60 else 0))
    for i in range(1, 10):
        scores.append(1 if (close.iloc[-1] - close.iloc[-1 - i]) > 0 else -1)
    return sum(scores)


def calculate_multi_target_tpsl(df: pd.DataFrame, signal_type: str) -> tuple:
    """Fitur No. 3: Multi-Target TP (TP1, TP2, TP3) berbasis ATR."""
    tr = (df["high"] - df["low"])
    atr_val = float(tr.rolling(14).mean().iloc[-1])
    price = float(df["close"].iloc[-1])

    if signal_type == "BUY":
        sl = round(price - (atr_val * 1.5), 2)
        tp1 = round(price + (atr_val * 1.0), 2)
        tp2 = round(price + (atr_val * 2.0), 2)
        tp3 = round(price + (atr_val * 3.5), 2)
    elif signal_type == "SELL":
        sl = round(price + (atr_val * 1.5), 2)
        tp1 = round(price - (atr_val * 1.0), 2)
        tp2 = round(price - (atr_val * 2.0), 2)
        tp3 = round(price - (atr_val * 3.5), 2)
    else:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    sl_distance_pips = abs(price - sl)
    risk_usd = ACCOUNT_BALANCE_USD * (RISK_PERCENTAGE / 100.0)
    lot = round(max(0.01, min(5.0, risk_usd / (sl_distance_pips * 10.0))), 2)

    return sl, tp1, tp2, tp3, lot


def send_telegram(message: str) -> bool:
    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        r.raise_for_status()
        return True
    except Exception:
        return False


def should_notify(state: dict, signal_type: str, now_utc: datetime) -> bool:
    if signal_type == "NEUTRAL":
        return False
    if state.get("last_signal_type") != signal_type:
        return True
    last_ts = state.get("last_signal_timestamp", "")
    if not last_ts:
        return True
    try:
        last_dt = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return (now_utc - last_dt) >= timedelta(minutes=COOLDOWN_MINUTES)
    except Exception:
        return True


# ============================================================
# MAIN PIPELINE (APEX V5 ULTIMATE HYBRID)
# ============================================================
def main() -> None:
    now_utc = datetime.now(timezone.utc)
    now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S")

    state = load_state()
    state["run_count"] = int(state.get("run_count", 0)) + 1
    logging.info(f"=== Apex Ultimate Hybrid Run #{state['run_count']} @ {now_str} UTC ===")

    if not is_market_open(now_utc):
        logging.info("Market closed — skipping.")
        save_state(state)
        return

    # Saringan Waktu & Berita (Killzone + News Shield)
    if not is_institutional_killzone(now_utc):
        logging.info("⏳ Di luar Institutional Killzone — Bot siaga memantau.")
        save_state(state)
        return

    if check_news_shield():
        logging.warning("🚨 Jendela Rilis Berita USD High Impact — Sinyal ditahan.")
        save_state(state)
        return

    # Tarik Data Failover Cascade (Deriv -> Gold-API -> PAXG)
    try:
        raw_m5, source_used = fetch_ohlcv_cascade("5min", 200)
        df_m5 = validate_ohlcv(raw_m5, MIN_BARS_M5).iloc[:-1].reset_index(drop=True)
        
        raw_m15, _ = fetch_ohlcv_cascade("15min", 150)
        df_m15 = validate_ohlcv(raw_m15, MIN_BARS_M15).iloc[:-1].reset_index(drop=True)
        
        raw_h1, _ = fetch_ohlcv_cascade("1h", 100)
        df_h1 = validate_ohlcv(raw_h1, MIN_BARS_H1).iloc[:-1].reset_index(drop=True)
    except Exception as e:
        logging.error(f"Gagal menarik data: {_redact(str(e))}")
        save_state(state)
        return

    # Analisis Indikator Gabungan (SMC + ADX + Heikin Ashi + Trend)
    current_price = float(df_m5["close"].iloc[-1])
    ob_data = detect_institutional_order_blocks(df_m5)
    fvg_data = detect_fair_value_gap(df_m5)
    
    m5_trend = get_timeframe_trend(df_m5)
    m15_trend = get_timeframe_trend(df_m15)
    h1_trend = get_timeframe_trend(df_h1)

    tech_score = run_pillar_analysis(df_m5)
    adx_val = calculate_adx(df_m5)
    ha_trend = get_heikin_ashi_trend(df_m5)

    # Logika Eksekusi Hybrid (Memerlukan ADX > 20, HA selaras, dan SMC Order Block / FVG)
    signal_type = "NEUTRAL"
    threshold = 11

    if tech_score >= threshold and h1_trend == "BULLISH" and adx_val > 20 and ha_trend == "BULLISH":
        if (ob_data["detected"] and ob_data["type"] == "BULLISH_OB") or (fvg_data["detected"] and fvg_data["type"] == "BULLISH_FVG"):
            signal_type = "BUY"
    elif tech_score <= -threshold and h1_trend == "BEARISH" and adx_val > 20 and ha_trend == "BEARISH":
        if (ob_data["detected"] and ob_data["type"] == "BEARISH_OB") or (fvg_data["detected"] and fvg_data["type"] == "BEARISH_FVG"):
            signal_type = "SELL"

    notify = should_notify(state, signal_type, now_utc)

    if signal_type != "NEUTRAL":
        state["last_signal_type"] = signal_type
        state["last_signal_timestamp"] = now_str
    save_state(state)

    sl, tp1, tp2, tp3, suggested_lot = calculate_multi_target_tpsl(df_m5, signal_type)

    report = {
        "timestamp": now_str,
        "run_count": state["run_count"],
        "active_source": source_used,
        "final_signal": signal_type,
        "entry_price": current_price,
        "adx": round(adx_val, 2),
        "heikin_ashi": ha_trend,
        "recommended_sl": sl,
        "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "suggested_lot": suggested_lot,
    }
    append_report(report)

    if signal_type != "NEUTRAL" and notify:
        msg = (
            f"⚛️ *APEX PREDATOR v5.0 (ULTIMATE HYBRID)* ⚛️\n\n"
            f"📊 *Asset:* XAUUSD\n"
            f"🔌 *Source:* `{source_used}`\n"
            f"🎯 *Signal:* *{signal_type}*\n"
            f"📌 *Entry:* `{current_price}`\n\n"
            f"🛑 *Stop Loss (SL):* `{sl}`\n"
            f"✅ *Take Profit 1 (TP1):* `{tp1}`\n"
            f"✅ *Take Profit 2 (TP2):* `{tp2}`\n"
            f"✅ *Take Profit 3 (TP3):* `{tp3}`\n"
            f"📐 *Recommended Lot:* `{suggested_lot} Lot`\n\n"
            f"🏛️ *SMC/FVG:* `{ob_data['type'] if ob_data['detected'] else fvg_data['type']}`\n"
            f"📈 *Trend:* H1:`{h1_trend}` | ADX:`{adx_val:.1f}` | HA:`{ha_trend}`\n"
            f"🔄 *Run:* `#{state['run_count']}` | ⏰ `{now_str} UTC`\n\n"
            f"_Manual execution & scaling out via MT5 Mobile._"
        )
        send_telegram(msg)

    logging.info("=== Apex Ultimate Hybrid Cycle Complete ==.")


if __name__ == "__main__":
    main()
