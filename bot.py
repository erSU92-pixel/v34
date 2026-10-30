"""
Apex Quant Engine — Production Hardened Edition
================================================
- No silent fallback (blocks signals when data invalid)
- Walk-forward validated ML gate (auto-disabled if below baseline)
- Time-based cooldown (resets on signal flip)
- OHLC integrity validation + spike detection
- Append-only report history (max 500 entries)
- Market-hours aware (skips when XAUUSD closed)
- Redacted credentials in all error logs
"""

import os
import re
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# CONFIG
# ============================================================
STATE_FILE = "signal_state.json"
REPORT_JSON = "xauusd_apex_report.json"
COOLDOWN_MINUTES = 60
MIN_BARS_M5 = 60
MIN_BARS_H1 = 60
M5_OUTPUTSIZE = 250
H1_OUTPUTSIZE = 100
BLOCK_ON_H1_FAILURE = False  # True = strict; False = warn + continue
REPORT_MAX_ENTRIES = 500

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ApexEngine] - %(message)s",
)

# ============================================================
# HELPERS
# ============================================================
def _redact(text: str) -> str:
    """Mask API keys and bot tokens from any string."""
    text = re.sub(r"apikey=[^&\s'\"]+", "apikey=***", text)
    text = re.sub(r"/bot[^/\s'\"]+/", "/bot***/", text)
    return text


class DataFetchError(Exception):
    pass


# ============================================================
# PERSISTENT STATE
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
        logging.warning(f"State reset to default: {_redact(str(e))}")
        return default


def save_state(state: dict) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logging.error(f"Save state failed: {_redact(str(e))}")


def append_report(report: dict, path: str = REPORT_JSON, max_entries: int = REPORT_MAX_ENTRIES) -> None:
    """Append-only history (list). Trims to last max_entries."""
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
    try:
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logging.error(f"Save report failed: {_redact(str(e))}")


# ============================================================
# DATA FETCH (NO FALLBACK)
# ============================================================
def fetch_ohlcv(interval: str, outputsize: int, retries: int = 2) -> pd.DataFrame:
    api_key = os.getenv("TWELVEDATA_API_KEY", "").strip()
    if not api_key or api_key == "demo":
        raise DataFetchError("TWELVEDATA_API_KEY missing or demo")

    url = (
        "https://api.twelvedata.com/time_series"
        f"?symbol=XAU/USD&interval={interval}&outputsize={outputsize}&apikey={api_key}"
    )

    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            if data.get("status") == "error":
                raise DataFetchError(f"API error: {data.get('message', 'unknown')}")
            values = data.get("values")
            if not isinstance(values, list) or len(values) == 0:
                raise DataFetchError("empty values array")

            df = pd.DataFrame(values).iloc[::-1].reset_index(drop=True)
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
            else:
                df["volume"] = np.nan
            return df

        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 ** attempt)

    raise DataFetchError(f"{type(last_err).__name__}: {_redact(str(last_err))}")


def validate_ohlcv(df: pd.DataFrame, min_bars: int) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise ValueError("no data returned")

    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    if len(df) < min_bars:
        raise ValueError(f"insufficient bars: {len(df)} < {min_bars}")

    prices = df[["open", "high", "low", "close"]]
    if (prices <= 0).any().any():
        raise ValueError("non-positive price detected")
    if (df["high"] < df["low"]).any():
        raise ValueError("high < low")
    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        raise ValueError("high < max(open, close)")
    if (df["low"] > df[["open", "close"]].min(axis=1)).any():
        raise ValueError("low > min(open, close)")

    pct = df["close"].pct_change().abs()
    if (pct > 0.15).any():
        idx = pct[pct > 0.15].index[0]
        raise ValueError(f"extreme spike {pct[idx] * 100:.1f}% at bar {idx}")

    return df.reset_index(drop=True)


# ============================================================
# MARKET HOURS (XAUUSD: Sun 22:00 UTC → Fri 21:00 UTC)
# ============================================================
def is_market_open(now_utc: datetime) -> bool:
    wd = now_utc.weekday()  # Mon=0 .. Sun=6
    h = now_utc.hour
    if wd == 5:            # Saturday
        return False
    if wd == 6:            # Sunday
        return h >= 22
    if wd == 4:            # Friday
        return h < 21
    return True            # Mon–Thu


# ============================================================
# REGIME DETECTION (True-Range ATR based)
# ============================================================
def _true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    return pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)


def detect_regime(df: pd.DataFrame) -> dict:
    tr = _true_range(df)
    atr_short = tr.rolling(14).mean().iloc[-1]
    atr_long = tr.rolling(50).mean().iloc[-1]
    vol_ratio = float(atr_short / atr_long) if atr_long > 0 else 1.0

    if vol_ratio > 2.5:
        return {"mode": "HIGH_IMPACT_NEWS_ANOMALY", "threshold": 99,
                "vol_ratio": round(vol_ratio, 2),
                "desc": "Extreme volatility / news anomaly"}
    if vol_ratio > 1.25:
        return {"mode": "HIGH_VOLATILITY_TREND", "threshold": 11,
                "vol_ratio": round(vol_ratio, 2),
                "desc": "Strong trending volatility"}
    if vol_ratio < 0.85:
        return {"mode": "CHOPPY_SIDEWAYS", "threshold": 14,
                "vol_ratio": round(vol_ratio, 2),
                "desc": "Sideways / noise"}
    return {"mode": "NORMAL_MARKET", "threshold": 12,
            "vol_ratio": round(vol_ratio, 2),
            "desc": "Normal balanced market"}


# ============================================================
# 18-PILLAR ANALYSIS + H1 CONFLUENCE
# ============================================================
def run_pillar_analysis(df: pd.DataFrame, h1_df: pd.DataFrame = None) -> tuple:
    close = df["close"].astype(float)

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan

    scores = []

    # --- Trend pillars (3) ---
    scores.append(1 if close.iloc[-1] > ema50.iloc[-1] else -1)
    scores.append(1 if ema9.iloc[-1] > ema21.iloc[-1] else -1)
    scores.append(0 if np.isnan(sma200) else (1 if close.iloc[-1] > sma200 else -1))

    # --- RSI pillars (3) ---
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-9)))
    r = float(rsi.iloc[-1])

    scores.append(1 if r < 40 else (-1 if r > 60 else 0))
    scores.append(1 if 30 < r < 50 else (-1 if 50 < r < 70 else 0))
    scores.append(1 if r < 35 else (-1 if r > 65 else 0))

    # --- Momentum lag pillars (12) ---
    for i in range(1, 13):
        scores.append(1 if (close.iloc[-1] - close.iloc[-1 - i]) > 0 else -1)

    total = sum(scores)

    # --- H1 confluence (±2) ---
    h1_trend = "NEUTRAL"
    if h1_df is not None and len(h1_df) >= 50:
        h1_close = h1_df["close"].astype(float)
        h1_ema50 = h1_close.ewm(span=50, adjust=False).mean()
        h1_trend = "BULLISH" if h1_close.iloc[-1] > h1_ema50.iloc[-1] else "BEARISH"
        if h1_trend == "BULLISH" and total > 0:
            total += 2
        elif h1_trend == "BEARISH" and total < 0:
            total -= 2

    return total, h1_trend


# ============================================================
# ML GATE (walk-forward validated — auto-disable if < baseline)
# ============================================================
def build_ml_gate(df: pd.DataFrame) -> dict:
    try:
        d = df.copy().reset_index(drop=True)

        d["ret1"] = d["close"].pct_change()
        d["ret3"] = d["close"].pct_change(3)
        d["ema9"] = d["close"].ewm(span=9, adjust=False).mean()
        d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
        d["dist_ema9"] = (d["close"] - d["ema9"]) / (d["ema9"] + 1e-9)
        d["dist_ema21"] = (d["close"] - d["ema21"]) / (d["ema21"] + 1e-9)

        delta = d["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        d["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

        tr = _true_range(d)
        d["atr_pct"] = tr.rolling(14).mean() / (d["close"] + 1e-9)

        d["target"] = (d["close"].shift(-1) > d["close"]).astype(float)
        d.loc[d.index[-1], "target"] = np.nan  # last row has no known future

        feats = ["ret1", "ret3", "dist_ema9", "dist_ema21", "rsi", "atr_pct"]
        train = d.dropna(subset=feats + ["target"])

        if len(train) < 60:
            return {"valid": False, "reason": f"insufficient_samples({len(train)})"}

        X = train[feats].values
        y = train["target"].astype(int).values

        split = int(len(train) * 0.7)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        clf = RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=10, random_state=42
        )
        clf.fit(X_tr, y_tr)
        acc = float(clf.score(X_te, y_te))
        baseline = float(max(y_tr.mean(), 1 - y_tr.mean()))

        latest_feats = d[feats].iloc[-1:].values
        if np.isnan(latest_feats).any():
            return {"valid": False, "reason": "latest_features_nan"}

        pred = int(clf.predict(latest_feats)[0])
        prob = float(np.max(clf.predict_proba(latest_feats)))

        valid = acc > max(0.55, baseline + 0.03)

        return {
            "valid": valid,
            "test_acc": round(acc, 4),
            "baseline": round(baseline, 4),
            "pred": pred,
            "prob": round(prob, 4),
        }

    except Exception as e:
        return {"valid": False, "reason": _redact(str(e))}


# ============================================================
# TP / SL (ATR-based, R:R = 1:2)
# ============================================================
def calculate_dynamic_tpsl(df: pd.DataFrame, signal_type: str) -> tuple:
    tr = _true_range(df)
    atr_val = float(tr.rolling(14).mean().iloc[-1])
    price = float(df["close"].iloc[-1])

    if signal_type == "BUY":
        sl = round(price - atr_val * 1.5, 2)
        tp = round(price + atr_val * 3.0, 2)
    elif signal_type == "SELL":
        sl = round(price + atr_val * 1.5, 2)
        tp = round(price - atr_val * 3.0, 2)
    else:
        sl, tp = 0.0, 0.0

    return sl, tp, round(atr_val, 2)


# ============================================================
# TELEGRAM
# ============================================================
def send_telegram(message: str) -> bool:
    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        logging.warning("Telegram credentials missing — skip")
        return False

    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        r.raise_for_status()
        logging.info("Telegram sent")
        return True
    except Exception as e:
        logging.error(f"Telegram failed: {_redact(str(e))}")
        return False


# ============================================================
# COOLDOWN (time-based, resets on direction flip)
# ============================================================
def should_notify(state: dict, signal_type: str, now_utc: datetime) -> bool:
    if signal_type == "NEUTRAL":
        return False
    if state.get("last_signal_type") != signal_type:
        return True  # direction flipped → always allow
    last_ts = state.get("last_signal_timestamp", "")
    if not last_ts:
        return True
    try:
        last_dt = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return (now_utc - last_dt) >= timedelta(minutes=COOLDOWN_MINUTES)
    except Exception:
        return True


# ============================================================
# MAIN PIPELINE
# ============================================================
def main() -> None:
    now_utc = datetime.now(timezone.utc)
    now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S")

    state = load_state()
    state["run_count"] = int(state.get("run_count", 0)) + 1
    logging.info(f"=== Apex Engine run #{state['run_count']} @ {now_str} UTC ===")

    # --- Market hours guard ---
    if not is_market_open(now_utc):
        logging.info("Market closed — skipping cycle.")
        save_state(state)
        return

    # --- Fetch M5 (no fallback) ---
    try:
        df_m5_raw = fetch_ohlcv("5min", M5_OUTPUTSIZE)
        df_m5 = validate_ohlcv(df_m5_raw, MIN_BARS_M5)
    except Exception as e:
        reason = _redact(str(e))
        logging.error(f"M5 fetch/validate failed: {reason}")
        send_telegram(
            f"⚠️ *ENGINE FAILURE*\n"
            f"M5 data unavailable. Signal *BLOCKED*.\n"
            f"Reason: `{reason}`"
        )
        save_state(state)
        return

    # drop last (still-forming) bar
    df_closed = df_m5.iloc[:-1].reset_index(drop=True)

    # --- Fetch H1 (confluence filter) ---
    h1_df = None
    h1_status = "unavailable"
    try:
        h1_raw = fetch_ohlcv("1h", H1_OUTPUTSIZE)
        h1_df = validate_ohlcv(h1_raw, MIN_BARS_H1)
        h1_status = "ok"
    except Exception as e:
        reason = _redact(str(e))
        logging.warning(f"H1 unavailable: {reason}")
        if BLOCK_ON_H1_FAILURE:
            send_telegram(
                f"⚠️ *ENGINE STOPPED*\n"
                f"H1 confluence unavailable. Signal *BLOCKED*.\n"
                f"Reason: `{reason}`"
            )
            save_state(state)
            return
        send_telegram(
            f"⚠️ *H1 Confluence unavailable* — running without multi-TF filter.\n"
            f"Reason: `{reason}`"
        )

    # --- Regime detection ---
    regime = detect_regime(df_closed)
    logging.info(f"Regime: {regime['mode']} (vol_ratio={regime['vol_ratio']})")

    if regime["mode"] == "HIGH_IMPACT_NEWS_ANOMALY":
        logging.warning("🚨 News anomaly detected — all signals blocked.")
        save_state(state)
        return

    # --- Analysis ---
    tech_score, h1_trend = run_pillar_analysis(df_closed, h1_df)
    ml_result = build_ml_gate(df_closed)

    logging.info(
        f"Tech: {tech_score} | H1: {h1_trend} | "
        f"ML: valid={ml_result.get('valid')} "
        f"acc={ml_result.get('test_acc')} "
        f"base={ml_result.get('baseline')} "
        f"prob={ml_result.get('prob')}"
    )

    # --- Signal determination ---
    threshold = regime["threshold"]
    ml_ok = ml_result.get("valid", False)
    ml_pred = ml_result.get("pred")
    ml_prob = ml_result.get("prob", 0.0)

    signal_type = "NEUTRAL"
    if tech_score >= threshold:
        if not ml_ok or (ml_pred == 1 and ml_prob >= 0.60):
            signal_type = "BUY"
    elif tech_score <= -threshold:
        if not ml_ok or (ml_pred == 0 and ml_prob >= 0.60):
            signal_type = "SELL"

    # --- Cooldown ---
    notify = should_notify(state, signal_type, now_utc)

    if signal_type != "NEUTRAL":
        state["last_signal_type"] = signal_type
        state["last_signal_timestamp"] = now_str
    save_state(state)

    # --- TP/SL ---
    sl, tp, atr_val = calculate_dynamic_tpsl(df_closed, signal_type)
    current_price = float(df_closed["close"].iloc[-1])

    # --- Persist report (append) ---
    report = {
        "timestamp": now_str,
        "run_count": state["run_count"],
        "regime_mode": regime["mode"],
        "vol_ratio": regime["vol_ratio"],
        "h1_status": h1_status,
        "h1_trend": h1_trend,
        "technical_score": tech_score,
        "threshold": threshold,
        "ml_valid": ml_ok,
        "ml_test_acc": ml_result.get("test_acc"),
        "ml_baseline": ml_result.get("baseline"),
        "ml_probability": ml_prob,
        "final_signal": signal_type,
        "entry_price": current_price,
        "recommended_sl": sl,
        "recommended_tp": tp,
        "atr": atr_val,
        "notify_sent": bool(notify and signal_type != "NEUTRAL"),
    }
    append_report(report)

    # --- Dispatch ---
    if signal_type != "NEUTRAL" and notify:
        h1_display = f"{h1_trend} ({h1_status})"
        if ml_ok:
            ml_display = (
                f"Active — ACC `{ml_result['test_acc']}` "
                f"vs base `{ml_result['baseline']}`"
            )
        else:
            ml_display = f"Disabled ({ml_result.get('reason', 'low acc')})"

        msg = (
            f"🚀 *APEX QUANT ENGINE* 🚀\n\n"
            f"📊 *Asset:* XAUUSD (M5 + H1)\n"
            f"🎯 *Signal:* *{signal_type}*\n"
            f"📌 *Entry:* `{current_price}`\n"
            f"🛑 *SL:* `{sl}`\n"
            f"✅ *TP:* `{tp}`\n\n"
            f"🧠 *Regime:* `{regime['mode']}`\n"
            f"📈 *H1 Trend:* `{h1_display}`\n"
            f"⚖️ *Tech Score:* `{tech_score}` / th `{threshold}`\n"
            f"🤖 *ML Gate:* {ml_display}\n"
            f"📐 *ATR:* `{atr_val}`\n"
            f"🔄 *Run:* `#{state['run_count']}`\n"
            f"⏰ *Time:* `{now_str} UTC`\n\n"
            f"_Manual execution only. Not financial advice._"
        )
        send_telegram(msg)
    else:
        logging.info(f"Signal: {signal_type} | notify={notify} — no dispatch")

    logging.info("=== Cycle complete ===")


if __name__ == "__main__":
    main()
