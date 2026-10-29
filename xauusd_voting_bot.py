"""
XAUUSD 10-PILLAR VOTING BOT v3.3 — FINAL FIX
================================================================
Fix v3.3:
  - FIX KeyError 'trend_daily' di merge_asof (rename kolom sebelum merge)
  - Semua fix v3.2 (cooldown, daily filter, R:R 2:1) tetap aktif

9 PILAR AKTIF (Pilar 9 Sentimen tidak tersedia):
  Voter: P1, P2, P3, P4, P6, P7, P8, P10
  Modifier: P5 (Hurst) → adaptive weights
================================================================
"""
from __future__ import annotations

import os, sys, time, json, math, logging
from io import StringIO
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("VOTING_BOT")

# =============================================================================
# ENV
# =============================================================================
TWELVE_DATA_KEY = os.getenv("TWELVEDATA_API_KEY")
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT   = os.getenv("TELEGRAM_CHAT_ID")
GIST_ID         = os.getenv("GIST_ID")
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN")

SYMBOL      = os.getenv("SYMBOL", "XAU/USD")
TF          = os.getenv("TF", "1h")
OUTPUT_SIZE = int(os.getenv("OUTPUT_SIZE", "5000"))
THRESHOLD   = float(os.getenv("THRESHOLD", "4.0"))

CONTRACT_SIZE    = 100
MIN_LOT          = 0.01
MAX_LOT          = 5.0
SPREAD_BUFFER    = 0.30
SLIPPAGE_BUFFER  = 0.15
ATR_SL_MULT      = 2.0
ATR_TP_MULT      = 4.0
MAX_HOLDING_BARS = 16
COOLDOWN_BARS    = 24

ACCOUNT_BALANCE   = float(os.getenv("ACCOUNT_BALANCE", "10000"))
RISK_PCT          = float(os.getenv("RISK_PCT", "0.003"))
MAX_DRAWDOWN_PCT  = 0.20
MAX_CONSEC_LOSSES = 5

LIVE_MODE     = os.getenv("LIVE_MODE", "1") == "1"
BACKTEST_MODE = os.getenv("BACKTEST_MODE", "0") == "1"
BACKTEST_DAYS = int(os.getenv("BACKTEST_DAYS", "90"))
NOTIFY_HOLD   = os.getenv("NOTIFY_HOLD", "0") == "1"

FRED_START = os.getenv("FRED_START", "2023-01-01")

# =============================================================================
# HTTP
# =============================================================================
class HTTP:
    def __init__(self, interval=7.5):
        self.s = requests.Session()
        self.interval = interval
        self._last = 0.0
    def _wait(self):
        d = time.time() - self._last
        if d < self.interval:
            time.sleep(self.interval - d)
    def req(self, method, url, **kw):
        for a in range(5):
            self._wait()
            try:
                r = self.s.request(method, url, timeout=30, **kw)
                self._last = time.time()
                if r.status_code == 429:
                    time.sleep(int(r.headers.get("Retry-After", 2 ** a)) + np.random.uniform(0, 1))
                    continue
                if 500 <= r.status_code < 600:
                    time.sleep((2 ** a) + np.random.uniform(0, 1))
                    continue
                return r
            except requests.RequestException:
                time.sleep((2 ** a) + np.random.uniform(0, 1))
        raise RuntimeError(f"HTTP failed: {method} {url}")

http = HTTP()

# =============================================================================
# DATA
# =============================================================================
def _interval_td():
    return pd.Timedelta(hours=1) if TF == "1h" else pd.Timedelta(hours=4)


def fetch_td(symbol, interval, outputsize):
    tf_map = {"1h": "1h", "4h": "4h", "1d": "1day"}
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol, "interval": tf_map.get(interval, interval),
        "outputsize": outputsize, "apikey": TWELVE_DATA_KEY,
        "format": "JSON", "timezone": "UTC",
    }
    r = http.req("GET", url, params=params)
    d = r.json()
    if d.get("status") == "error":
        raise RuntimeError(f"TD [{symbol}]: {d.get('message')}")
    vals = d.get("values")
    if not vals:
        raise RuntimeError(f"TD empty [{symbol}]")

    df = pd.DataFrame(vals)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df.reset_index()


def fetch_ohlcv():
    df = fetch_td(SYMBOL, TF, OUTPUT_SIZE)
    now = pd.Timestamp.now(tz="UTC")
    itd = _interval_td()
    if len(df) > 0:
        last_ts = df["datetime"].iloc[-1]
        if last_ts + itd > now:
            logger.info("Dropping in-progress bar: %s", last_ts)
            df = df.iloc[:-1].reset_index(drop=True)
    df = df[df["datetime"] + itd <= now]
    if len(df) < 250:
        raise RuntimeError(f"Insufficient bars: {len(df)}")
    logger.info("Fetched %d closed bars | %s → %s",
                len(df), df["datetime"].iloc[0], df["datetime"].iloc[-1])
    return df


def fetch_dxy():
    try:
        dxy = fetch_td("UUP", TF, OUTPUT_SIZE)
        now = pd.Timestamp.now(tz="UTC")
        itd = _interval_td()
        if len(dxy) > 0 and dxy["datetime"].iloc[-1] + itd > now:
            dxy = dxy.iloc[:-1].reset_index(drop=True)
        out = dxy[["datetime", "close"]].rename(columns={"close": "dxy_close"})
        logger.info("DXY proxy [UUP]: %d closed bars", len(out))
        return out
    except Exception as e:
        logger.warning("DXY fetch failed: %s", e)
        return None


def fetch_daily_trend():
    try:
        daily = fetch_td(SYMBOL, "1d", 500)
        now = pd.Timestamp.now(tz="UTC")
        daily = daily[daily["datetime"] + pd.Timedelta(days=1) <= now]
        daily["sma200"] = daily["close"].rolling(200).mean()
        daily["trend_daily"] = np.sign((daily["close"] - daily["sma200"]).fillna(0)).astype(int)
        out = daily[["datetime", "trend_daily"]]
        logger.info("Daily trend: %d bars | last trend=%+d",
                    len(out), int(out["trend_daily"].iloc[-1]))
        return out
    except Exception as e:
        logger.warning("Daily trend fetch failed: %s", e)
        return None


def _fetch_fred_csv(series_id, start):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    try:
        r = http.req("GET", url)
        if r.status_code != 200:
            return None
        df = pd.read_csv(StringIO(r.text))
        if df.shape[1] < 2:
            return None
        df.columns = ["datetime", "value"]
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna(subset=["value"]).set_index("datetime")["value"]
    except Exception:
        return None


def fetch_fred():
    try:
        t10y = _fetch_fred_csv("DGS10", FRED_START)
        cpi = _fetch_fred_csv("CPIAUCSL", FRED_START)
        if t10y is None or cpi is None or len(t10y) < 30 or len(cpi) < 15:
            return None
        cpi_yoy = cpi.pct_change(12).dropna() * 100
        if len(cpi_yoy) < 5:
            return None
        t10y_df = t10y.reset_index(); t10y_df.columns = ["datetime", "t10y"]
        cpi_df = cpi_yoy.reset_index(); cpi_df.columns = ["datetime", "cpi_yoy"]
        t10y_df = t10y_df.sort_values("datetime")
        cpi_df = cpi_df.sort_values("datetime")
        m = pd.merge_asof(t10y_df, cpi_df, on="datetime", direction="backward")
        m["real_yield"] = m["t10y"] - m["cpi_yoy"]
        ry = m[["datetime", "real_yield"]].dropna()
        if len(ry) < 30:
            return None
        logger.info("FRED real yield: %d obs | last %.2f",
                    len(ry), float(ry["real_yield"].iloc[-1]))
        return ry
    except Exception as e:
        logger.warning("FRED failed: %s", e)
        return None

# =============================================================================
# INDICATORS
# =============================================================================
def _rolling_hurst(series):
    if len(series) < 20: return 0.5
    lags = list(range(2, min(20, len(series) // 2)))
    if len(lags) < 2: return 0.5
    try:
        tau = [np.std(series[lag:] - series[:-lag]) for lag in lags]
        tau = [t for t in tau if t > 1e-12]
        if len(tau) < 2: return 0.5
        poly = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
        return float(np.clip(poly[0], 0.0, 1.0))
    except Exception:
        return 0.5


def add_indicators(df, dxy, fred, daily_trend):
    df = df.copy()

    for span in (9, 21, 50, 200):
        df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False).mean()

    roll_std = df["close"].rolling(20).std().replace(0, np.nan)
    df["zscore_20"] = (df["close"] - df["close"].rolling(20).mean()) / roll_std

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    low_5 = df["low"].rolling(5).min()
    high_5 = df["high"].rolling(5).max()
    rng = (high_5 - low_5).replace(0, np.nan)
    df["stoch_k"] = 100 * (df["close"] - low_5) / rng

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean().replace(0, np.nan).ffill()

    df["don_high"] = df["high"].rolling(20).max().shift(1)
    df["don_low"] = df["low"].rolling(20).min().shift(1)

    df["hurst_100"] = df["close"].rolling(100).apply(_rolling_hurst, raw=True)

    df["datetime_utc"] = pd.to_datetime(df["datetime"], utc=True)
    hour = df["datetime_utc"].dt.hour
    df["in_session"] = ((hour >= 7) & (hour < 16)).astype(int)

    # --- DXY merge ---
    df["dxy_momentum_20"] = 0.0
    if dxy is not None and len(dxy) > 20:
        try:
            df = pd.merge_asof(df.sort_values("datetime").reset_index(drop=True),
                               dxy.sort_values("datetime").reset_index(drop=True),
                               on="datetime", direction="backward")
            df["dxy_close"] = df["dxy_close"].ffill()
            df["dxy_momentum_20"] = df["dxy_close"].pct_change(20)
        except Exception as e:
            logger.warning("DXY merge failed: %s", e)

    # --- FRED merge ---
    df["real_yield_change_5d"] = 0.0
    if fred is not None and len(fred) > 10:
        try:
            fred_renamed = fred.rename(columns={"real_yield": "ry_merge"})
            df = pd.merge_asof(df.sort_values("datetime").reset_index(drop=True),
                               fred_renamed.sort_values("datetime").reset_index(drop=True),
                               on="datetime", direction="backward")
            df["ry_merge"] = df["ry_merge"].ffill()
            df["real_yield_change_5d"] = df["ry_merge"].diff(5)
            df = df.drop(columns=["ry_merge"], errors="ignore")
        except Exception as e:
            logger.warning("FRED merge failed: %s", e)

    # --- Daily trend merge (FIX v3.3: rename dulu) ---
    df["trend_daily"] = 0
    if daily_trend is not None and len(daily_trend) > 0:
        try:
            daily_renamed = daily_trend.rename(columns={"trend_daily": "td_merge"})
            df = pd.merge_asof(df.sort_values("datetime").reset_index(drop=True),
                               daily_renamed.sort_values("datetime").reset_index(drop=True),
                               on="datetime", direction="backward")
            df["trend_daily"] = df["td_merge"].fillna(0).astype(int)
            df = df.drop(columns=["td_merge"], errors="ignore")
        except Exception as e:
            logger.warning("Daily trend merge failed: %s", e)
            df["trend_daily"] = 0

    # --- Clean alt columns ---
    for c in ["dxy_momentum_20", "real_yield_change_5d"]:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(0.0).replace([np.inf, -np.inf], 0.0)

    if "trend_daily" not in df.columns:
        df["trend_daily"] = 0
    df["trend_daily"] = df["trend_daily"].fillna(0).astype(int)

    return df

# =============================================================================
# PILLAR VOTES
# =============================================================================
def p1_trend(row):
    try:
        e9, e21, e50 = row["ema_9"], row["ema_21"], row["ema_50"]
        if pd.isna(e9) or pd.isna(e21) or pd.isna(e50): return 0
        if e9 > e21 > e50: return 1
        if e9 < e21 < e50: return -1
    except Exception:
        return 0
    return 0


def p2_momentum(row):
    try:
        rsi = row["rsi_14"]
        if pd.isna(rsi): return 0
        if rsi < 30: return 1
        if rsi > 70: return -1
    except Exception:
        return 0
    return 0


def p3_ml_proxy(row):
    try:
        z = row["zscore_20"]
        stoch = row["stoch_k"]
        if pd.isna(z) or pd.isna(stoch): return 0
        score = 0
        if z < -1.5: score += 1
        if z > 1.5: score -= 1
        if stoch < 20: score += 1
        if stoch > 80: score -= 1
        if score >= 2: return 1
        if score <= -2: return -1
    except Exception:
        return 0
    return 0


def p4_volatility(row):
    try:
        c, dh, dl = row["close"], row["don_high"], row["don_low"]
        if pd.isna(dh) or pd.isna(dl): return 0
        if c > dh: return 1
        if c < dl: return -1
    except Exception:
        return 0
    return 0


def p6_macro(row):
    try:
        chg = row["real_yield_change_5d"]
        if pd.isna(chg) or abs(chg) < 1e-6: return 0
        if chg > 0.02: return -1
        if chg < -0.02: return 1
    except Exception:
        return 0
    return 0


def p7_cross_asset(row):
    try:
        mom = row["dxy_momentum_20"]
        if pd.isna(mom): return 0
        if mom > 0.005: return -1
        if mom < -0.005: return 1
    except Exception:
        return 0
    return 0


def p8_smc(df, i):
    try:
        if i < 25: return 0
        window = df.iloc[max(0, i - 20):i]
        if len(window) < 20: return 0
        swing_high = float(window["high"].max())
        swing_low = float(window["low"].min())
        for j in range(max(0, i - 2), i + 1):
            bar = df.iloc[j]
            h = float(bar["high"]); l = float(bar["low"]); c = float(bar["close"])
            if h > swing_high and c < swing_high: return -1
            if l < swing_low and c > swing_low: return 1
    except Exception:
        return 0
    return 0


def p10_session(row):
    try:
        if not row.get("in_session", 0): return 0
        e9, e21 = row["ema_9"], row["ema_21"]
        if pd.isna(e9) or pd.isna(e21): return 0
        if e9 > e21: return 1
        if e9 < e21: return -1
    except Exception:
        return 0
    return 0


def get_regime_weights(row):
    try:
        hurst = row.get("hurst_100", 0.5)
        if pd.isna(hurst): hurst = 0.5
    except Exception:
        hurst = 0.5
    if hurst > 0.55:
        return {"p1": 2.0, "p2": 1.0, "p3": 1.0, "p4": 1.5,
                "p6": 1.0, "p7": 1.0, "p8": 1.5, "p10": 1.0}
    elif hurst < 0.45:
        return {"p1": 0.5, "p2": 2.0, "p3": 1.5, "p4": 0.5,
                "p6": 1.0, "p7": 1.0, "p8": 1.5, "p10": 0.5}
    else:
        return {"p1": 1.0, "p2": 1.0, "p3": 1.0, "p4": 1.0,
                "p6": 1.0, "p7": 1.0, "p8": 1.0, "p10": 1.0}


def compute_votes(df, i):
    row = df.iloc[i]
    votes = {
        "p1": p1_trend(row), "p2": p2_momentum(row),
        "p3": p3_ml_proxy(row), "p4": p4_volatility(row),
        "p6": p6_macro(row), "p7": p7_cross_asset(row),
        "p8": p8_smc(df, i), "p10": p10_session(row),
    }
    weights = get_regime_weights(row)
    weighted_sum = sum(votes[k] * weights[k] for k in votes.keys())
    return votes, weighted_sum, weights

# =============================================================================
# SL/TP & SIZING
# =============================================================================
def compute_sltp(entry, direction, atr):
    sl_dist = ATR_SL_MULT * atr
    tp_dist = ATR_TP_MULT * atr
    buf = SPREAD_BUFFER + SLIPPAGE_BUFFER
    if direction == 1:
        return (round(entry - sl_dist - buf, 2),
                round(entry + tp_dist + buf, 2))
    return (round(entry + sl_dist + buf, 2),
            round(entry - tp_dist - buf, 2))


def position_size(entry, sl, balance):
    risk_money = balance * RISK_PCT
    sl_dist = abs(entry - sl)
    if sl_dist <= 0: return 0.0
    loss_per_lot = sl_dist * CONTRACT_SIZE
    if loss_per_lot <= 0: return 0.0
    lot = math.floor((risk_money / loss_per_lot) / MIN_LOT) * MIN_LOT
    return max(min(lot, MAX_LOT), 0.0)

# =============================================================================
# TELEGRAM
# =============================================================================
def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT, "text": msg, "parse_mode": "HTML"}
    for a in range(3):
        try:
            r = http.req("POST", url, json=payload)
        except RuntimeError:
            return False
        if r.status_code == 200:
            logger.info("Telegram sent OK")
            return True
        if r.status_code == 429:
            try:
                ra = int(r.json().get("parameters", {}).get("retry_after", 5))
            except Exception:
                ra = 5
            time.sleep(ra + 1)
            continue
        time.sleep(2 ** a)
    return False

# =============================================================================
# STATE
# =============================================================================
def load_state():
    url = f"https://api.github.com/gists/{GIST_ID}"
    h = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    try:
        r = http.req("GET", url, headers=h)
    except RuntimeError:
        return {"etag": None, "content": {}}
    if r.status_code != 200:
        return {"etag": None, "content": {}}
    content = {}
    for name, f in r.json().get("files", {}).items():
        if name.endswith(".json"):
            try:
                content = json.loads(f.get("content", "{}"))
                break
            except json.JSONDecodeError:
                continue
    return {"etag": r.headers.get("ETag"), "content": content}


def save_state(state, etag):
    url = f"https://api.github.com/gists/{GIST_ID}"
    payload = {"files": {"bot_state.json": {"content": json.dumps(state, indent=2, default=str)}}}
    h = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    if etag: h["If-Match"] = etag
    try:
        r = http.req("PATCH", url, headers=h, json=payload)
    except RuntimeError:
        return False
    if r.status_code in (200, 201): return True
    if r.status_code == 412:
        h.pop("If-Match", None)
        try:
            r = http.req("PATCH", url, headers=h, json=payload)
        except RuntimeError:
            return False
        return r.status_code in (200, 201)
    return False

# =============================================================================
# CIRCUIT BREAKER
# =============================================================================
def check_cb(state):
    bal = state.get("balance", ACCOUNT_BALANCE)
    peak = state.get("peak_equity", bal)
    consec = state.get("consecutive_losses", 0)
    if state.get("halted", False): return True
    peak = max(peak, bal); state["peak_equity"] = peak
    dd = (peak - bal) / max(peak, 1e-9)
    if dd >= MAX_DRAWDOWN_PCT:
        state["halted"] = True; return True
    if consec >= MAX_CONSEC_LOSSES:
        state["halted"] = True; return True
    return False

# =============================================================================
# TRADE MONITORING
# =============================================================================
def check_trade_outcome(df, trade):
    direction = trade["direction"]
    entry, sl, tp = trade["entry"], trade["sl"], trade["tp"]
    opened = pd.Timestamp(trade["opened_at"])
    after = df[df["datetime"] > opened].sort_values("datetime")
    for _, row in after.iterrows():
        h, l = float(row["high"]), float(row["low"])
        if direction == 1:
            if l <= sl: return {"outcome": "SL", "exit": sl, "time": str(row["datetime"])}
            if h >= tp: return {"outcome": "TP", "exit": tp, "time": str(row["datetime"])}
        else:
            if h >= sl: return {"outcome": "SL", "exit": sl, "time": str(row["datetime"])}
            if l <= tp: return {"outcome": "TP", "exit": tp, "time": str(row["datetime"])}
    return None

# =============================================================================
# LIVE MODE
# =============================================================================
def run_live():
    logger.info("═══ LIVE MODE | %s %s | THR=%.1f | R:R=%.1f | CD=%d ═══",
                SYMBOL, TF, THRESHOLD, ATR_TP_MULT/ATR_SL_MULT, COOLDOWN_BARS)

    s = load_state()
    state = s["content"] or {}
    state.setdefault("balance", ACCOUNT_BALANCE)
    state.setdefault("peak_equity", ACCOUNT_BALANCE)
    state.setdefault("consecutive_losses", 0)
    state.setdefault("halted", False)
    state.setdefault("active_trade", None)
    state.setdefault("last_signal_bar", None)

    if check_cb(state):
        send_telegram("🚨 <b>CIRCUIT BREAKER</b>")
        return

    try:
        df = fetch_ohlcv()
        dxy = fetch_dxy()
        daily_trend = fetch_daily_trend()
        fred = fetch_fred()
    except Exception as e:
        logger.exception("Data fetch failed")
        send_telegram(f"⚠️ Data error: {e}")
        return

    df = add_indicators(df, dxy, fred, daily_trend)
    if len(df) < 250:
        return

    trade = state.get("active_trade")
    if trade:
        outcome = check_trade_outcome(df, trade)
        if outcome:
            pnl_pips = ((outcome["exit"] - trade["entry"]) if trade["direction"] == 1
                        else (trade["entry"] - outcome["exit"]))
            pnl = pnl_pips * trade["lot"] * CONTRACT_SIZE
            state["balance"] = state.get("balance", ACCOUNT_BALANCE) + pnl
            if outcome["outcome"] == "SL":
                state["consecutive_losses"] = state.get("consecutive_losses", 0) + 1
            else:
                state["consecutive_losses"] = 0
            emoji = "✅" if outcome["outcome"] == "TP" else "❌"
            send_telegram(
                f"{emoji} <b>{outcome['outcome']}</b> — {trade['direction_str']} {SYMBOL}\n"
                f"Entry: {trade['entry']:.2f} → Exit: {outcome['exit']:.2f}\n"
                f"PnL: <b>${pnl:+.2f}</b> | Balance: ${state['balance']:.2f}"
            )
            state["active_trade"] = None
        else:
            logger.info("Trade active: %s @ %.2f", trade["direction_str"], trade["entry"])
    else:
        last_sig_bar = state.get("last_signal_bar")
        cur_bar_ts = str(df["datetime"].iloc[-1])
        if last_sig_bar:
            try:
                last_ts = pd.Timestamp(last_sig_bar)
                cur_ts = pd.Timestamp(cur_bar_ts)
                bars_since = (cur_ts - last_ts) / _interval_td()
                if bars_since < COOLDOWN_BARS:
                    logger.info("Cooldown active: %.0f bars since last signal (<%d)",
                                bars_since, COOLDOWN_BARS)
                    state["last_update"] = datetime.now(timezone.utc).isoformat()
                    save_state(state, s.get("etag"))
                    return
            except Exception:
                pass

        votes_cur, weighted_cur, _ = compute_votes(df, len(df) - 1)
        votes_prev, weighted_prev, _ = compute_votes(df, len(df) - 2)
        dir_cur = 1 if weighted_cur >= THRESHOLD else (-1 if weighted_cur <= -THRESHOLD else 0)
        dir_prev = 1 if weighted_prev >= THRESHOLD else (-1 if weighted_prev <= -THRESHOLD else 0)

        logger.info("Bar | weighted=%.2f | votes=%s | trend_daily=%+d",
                    weighted_cur, votes_cur, int(df["trend_daily"].iloc[-1]))

        if dir_cur != 0 and dir_cur != dir_prev:
            row = df.iloc[-1]
            trend_daily = int(row.get("trend_daily", 0))
            if trend_daily != 0 and dir_cur != trend_daily:
                logger.info("Signal %+d rejected by daily trend (%+d)", dir_cur, trend_daily)
                if NOTIFY_HOLD:
                    send_telegram(f"⏸️ Signal filtered by daily trend (dir={dir_cur}, trend={trend_daily})")
                state["last_update"] = datetime.now(timezone.utc).isoformat()
                save_state(state, s.get("etag"))
                return

            atr = float(row["atr_14"]) if not pd.isna(row["atr_14"]) else 10.0
            entry = float(row["close"])
            direction_str = "BUY" if dir_cur == 1 else "SELL"
            sl, tp = compute_sltp(entry, dir_cur, atr)
            lot = position_size(entry, sl, state["balance"])

            if lot >= MIN_LOT:
                state["active_trade"] = {
                    "direction": dir_cur, "direction_str": direction_str,
                    "entry": entry, "sl": sl, "tp": tp, "lot": lot,
                    "opened_at": str(row["datetime"]),
                }
                state["last_signal_bar"] = cur_bar_ts
                emoji = "🟢" if dir_cur == 1 else "🔴"
                rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
                active_votes = [f"P{k[1:]}" for k, v in votes_cur.items() if v == dir_cur]
                send_telegram(
                    f"{emoji} <b>{direction_str} {SYMBOL}</b> [{TF}]\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"Entry: <code>{entry:.2f}</code>\n"
                    f"SL:    <code>{sl:.2f}</code>\n"
                    f"TP:    <code>{tp:.2f}</code>\n"
                    f"R:R    {rr:.1f}:1\n"
                    f"Lot:   <code>{lot}</code>\n"
                    f"Risk:  ${state['balance'] * RISK_PCT:.2f}\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"Weighted: {weighted_cur:+.1f}\n"
                    f"Voting: {','.join(active_votes)}\n"
                    f"Daily trend: {trend_daily:+d}\n"
                    f"Balance: ${state['balance']:.2f}"
                )
                logger.info("SIGNAL: %s @ %.2f lot=%.2f w=%.1f", direction_str, entry, lot, weighted_cur)

    state["last_update"] = datetime.now(timezone.utc).isoformat()
    save_state(state, s.get("etag"))
    logger.info("═══ DONE ═══")

# =============================================================================
# BACKTEST MODE
# =============================================================================
def run_backtest():
    logger.info("═══ BACKTEST v3.3 | %s %s | THR=%.1f | R:R=%.1f | CD=%d ═══",
                SYMBOL, TF, THRESHOLD, ATR_TP_MULT/ATR_SL_MULT, COOLDOWN_BARS)

    try:
        df = fetch_ohlcv()
        dxy = fetch_dxy()
        daily_trend = fetch_daily_trend()
        fred = fetch_fred()
    except Exception as e:
        logger.exception("Data fetch failed")
        return

    df = add_indicators(df, dxy, fred, daily_trend)

    cutoff = df["datetime"].max() - pd.Timedelta(days=BACKTEST_DAYS)
    df = df[df["datetime"] >= cutoff].reset_index(drop=True)
    logger.info("Testing on %d bars (%s → %s)", len(df),
                df["datetime"].iloc[0].date(), df["datetime"].iloc[-1].date())
    if len(df) < 250:
        logger.error("Insufficient bars: %d", len(df))
        return

    logger.info("Computing votes for all bars...")
    all_votes, all_weighted = [], []
    for i in range(len(df)):
        v, w, _ = compute_votes(df, i)
        all_votes.append(v); all_weighted.append(w)

    pillar_counts = {k: {"BUY": 0, "SELL": 0, "NEUTRAL": 0}
                     for k in ["p1", "p2", "p3", "p4", "p6", "p7", "p8", "p10"]}
    for v in all_votes:
        for k, vote in v.items():
            if vote == 1: pillar_counts[k]["BUY"] += 1
            elif vote == -1: pillar_counts[k]["SELL"] += 1
            else: pillar_counts[k]["NEUTRAL"] += 1

    logger.info("")
    logger.info("═══ PILLAR ACTIVITY ═══")
    for k, c in pillar_counts.items():
        total = sum(c.values())
        logger.info("  %s: BUY=%d SELL=%d NEUTRAL=%d (active %.1f%%)",
                    k, c["BUY"], c["SELL"], c["NEUTRAL"],
                    100 * (c["BUY"] + c["SELL"]) / max(total, 1))

    n_buy = sum(1 for w in all_weighted if w >= THRESHOLD)
    n_sell = sum(1 for w in all_weighted if w <= -THRESHOLD)
    logger.info("")
    logger.info("Signal bars: BUY=%d SELL=%d HOLD=%d", n_buy, n_sell,
                len(all_weighted) - n_buy - n_sell)

    logger.info("")
    logger.info("Simulating trades (with cooldown + daily filter)...")
    trades = []
    balance = ACCOUNT_BALANCE
    i = 250
    last_trade_bar = -9999

    while i < len(df) - 1:
        if i - last_trade_bar < COOLDOWN_BARS:
            i += 1
            continue

        weighted = all_weighted[i]
        direction = 1 if weighted >= THRESHOLD else (-1 if weighted <= -THRESHOLD else 0)
        if direction == 0:
            i += 1
            continue

        prev_weighted = all_weighted[i - 1]
        prev_dir = 1 if prev_weighted >= THRESHOLD else (-1 if prev_weighted <= -THRESHOLD else 0)
        if direction == prev_dir:
            i += 1
            continue

        row = df.iloc[i]
        trend_daily = int(row.get("trend_daily", 0))
        if trend_daily != 0 and direction != trend_daily:
            i += 1
            continue

        atr = float(row["atr_14"]) if not pd.isna(row["atr_14"]) else 10.0
        entry = float(row["close"])
        sl, tp = compute_sltp(entry, direction, atr)
        lot = position_size(entry, sl, balance)
        if lot < MIN_LOT:
            i += 1
            continue

        exit_price, exit_reason = None, None
        j = i + 1
        for j in range(i + 1, min(i + MAX_HOLDING_BARS + 1, len(df))):
            b = df.iloc[j]
            h, l = float(b["high"]), float(b["low"])
            if direction == 1:
                if l <= sl: exit_price, exit_reason = sl, "SL"; break
                if h >= tp: exit_price, exit_reason = tp, "TP"; break
            else:
                if h >= sl: exit_price, exit_reason = sl, "SL"; break
                if l <= tp: exit_price, exit_reason = tp, "TP"; break
        if exit_price is None:
            b = df.iloc[min(i + MAX_HOLDING_BARS, len(df) - 1)]
            exit_price = float(b["close"]); exit_reason = "TIME"

        pnl_pips = (exit_price - entry) if direction == 1 else (entry - exit_price)
        pnl = pnl_pips * lot * CONTRACT_SIZE
        balance += pnl
        trades.append({"date": row["datetime"].date(),
                       "dir": "BUY" if direction == 1 else "SELL",
                       "entry": entry, "exit": exit_price,
                       "reason": exit_reason, "pnl": pnl, "lot": lot,
                       "weighted": weighted})
        logger.info("  %s %s entry=%.2f exit=%.2f %s pnl=$%+.2f w=%.1f trend_d=%+d",
                    row["datetime"].date(), "BUY " if direction == 1 else "SELL",
                    entry, exit_price, exit_reason, pnl, weighted, trend_daily)
        last_trade_bar = i
        i = j + 1

    if not trades:
        logger.warning("No trades")
        send_telegram(f"📊 Backtest {BACKTEST_DAYS} hari: 0 trade")
        return

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = gw / gl if gl > 0 else float('inf') if gw > 0 else 0.0
    wr = 100 * len(wins) / len(trades)
    tp_ct = sum(1 for t in trades if t["reason"] == "TP")
    sl_ct = sum(1 for t in trades if t["reason"] == "SL")
    to_ct = sum(1 for t in trades if t["reason"] == "TIME")
    total_pnl = sum(t["pnl"] for t in trades)

    logger.info("")
    logger.info("═══ BACKTEST SUMMARY ═══")
    logger.info("Trades: %d | Win: %d | Loss: %d", len(trades), len(wins), len(losses))
    logger.info("TP=%d | SL=%d | TIME=%d", tp_ct, sl_ct, to_ct)
    logger.info("Win Rate: %.1f%%", wr)
    logger.info("Total PnL: $%+.2f | Balance: $%.2f → $%.2f",
                total_pnl, ACCOUNT_BALANCE, balance)
    logger.info("Profit Factor: %.2f", pf)

    top_trades = sorted(trades, key=lambda t: abs(t["pnl"]), reverse=True)[:5]
    trades_str = "\n".join(f"  {t['date']} {t['dir']} {t['pnl']:+.2f}" for t in top_trades)

    send_telegram(
        f"📊 <b>BACKTEST v3.3 — {BACKTEST_DAYS} hari</b>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"Trades: {len(trades)} (TP={tp_ct} SL={sl_ct} TIME={to_ct})\n"
        f"Win rate: {wr:.1f}%\n"
        f"PnL: <b>${total_pnl:+.2f}</b>\n"
        f"Balance: ${ACCOUNT_BALANCE:.2f} → ${balance:.2f}\n"
        f"<b>Profit Factor: {pf:.2f}</b>\n"
        f"\nTop trades:\n{trades_str}"
    )

# =============================================================================
# MAIN
# =============================================================================
def main():
    logger.info("═" * 60)
    logger.info("XAUUSD VOTING BOT v3.3 | LIVE=%s | BT=%s | TF=%s",
                LIVE_MODE, BACKTEST_MODE, TF)
    logger.info("═" * 60)

    if not TWELVE_DATA_KEY:
        logger.error("TWELVEDATA_API_KEY missing")
        sys.exit(1)

    if BACKTEST_MODE:
        run_backtest()
    elif LIVE_MODE:
        run_live()
    else:
        logger.error("Both modes off")
        sys.exit(1)

    logger.info("═" * 60)
    logger.info("COMPLETE")


if __name__ == "__main__":
    main()
