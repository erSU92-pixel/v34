"""
LONDON BO v1.4 — BACKTEST COMPARE: Breakout vs Fade
Jalankan backtest dua arah sekaligus. Pilih yang PF > 1.0.
"""
from __future__ import annotations
import os, sys, time, json, math, logging
from datetime import datetime, timezone, date as date_cls
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np
import requests

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("LONDON_BO")

# ENV
TWELVE_DATA_KEY = os.getenv("TWELVEDATA_API_KEY")
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT   = os.getenv("TELEGRAM_CHAT_ID")
GIST_ID         = os.getenv("GIST_ID")
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN")

SYMBOL         = os.getenv("SYMBOL", "XAU/USD")
INTERVAL       = "1h"
OUTPUT_SIZE    = int(os.getenv("OUTPUT_SIZE", "500"))
CONTRACT_SIZE  = 100
MIN_LOT        = 0.01
MAX_LOT        = 5.0

SERVER_TZ_OFFSET = int(os.getenv("SERVER_TZ_OFFSET", "0"))
ASIA_START_UTC   = (int(os.getenv("ASIA_START_HOUR", "3")) - SERVER_TZ_OFFSET) % 24
ASIA_END_UTC     = (int(os.getenv("ASIA_END_HOUR", "8")) - SERVER_TZ_OFFSET) % 24
ENTRY_HOUR_UTC   = (int(os.getenv("ENTRY_HOUR", "8")) - SERVER_TZ_OFFSET) % 24
CLOSE_HOUR_UTC   = (int(os.getenv("CLOSE_HOUR", "21")) - SERVER_TZ_OFFSET) % 24

RANGE_MIN_PCT = float(os.getenv("RANGE_MIN_PCT", "0.0030"))
RANGE_MAX_PCT = float(os.getenv("RANGE_MAX_PCT", "0.0160"))
SL_MAX_PCT    = float(os.getenv("SL_MAX_PCT", "0.0060"))
TP_R          = float(os.getenv("TP_R", "2.5"))
BE_R          = float(os.getenv("BE_R", "1.5"))

ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "10000"))
RISK_PCT        = float(os.getenv("RISK_PCT", "0.003"))
MAX_DAILY_LOSS_PCT     = 0.02
MAX_DRAWDOWN_PCT       = 0.15
MAX_CONSECUTIVE_LOSSES = 4

BACKTEST_MODE = os.getenv("BACKTEST_MODE", "0") == "1"
BACKTEST_DAYS = int(os.getenv("BACKTEST_DAYS", "14"))
FADE_MODE     = os.getenv("FADE_MODE", "0") == "1"   # untuk live: 1=fade, 0=breakout

# HTTP
class ResilientSession:
    def __init__(self, min_interval_sec=7.5):
        self.session = requests.Session()
        self.min_interval = min_interval_sec
        self._last = 0.0
    def _throttle(self):
        elapsed = time.time() - self._last
        if elapsed < self.min_interval: time.sleep(self.min_interval - elapsed)
    def request(self, method, url, **kwargs):
        for attempt in range(5):
            self._throttle()
            try:
                r = self.session.request(method, url, timeout=30, **kwargs)
                self._last = time.time()
                if r.status_code == 429:
                    ra = int(r.headers.get("Retry-After", 2 ** attempt))
                    time.sleep(ra + np.random.uniform(0, 1)); continue
                if 500 <= r.status_code < 600:
                    time.sleep((2 ** attempt) + np.random.uniform(0, 1)); continue
                return r
            except requests.RequestException:
                time.sleep((2 ** attempt) + np.random.uniform(0, 1))
        raise RuntimeError(f"HTTP failed: {method} {url}")

http = ResilientSession()

def fetch_ohlcv() -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": SYMBOL, "interval": INTERVAL, "outputsize": OUTPUT_SIZE,
              "apikey": TWELVE_DATA_KEY, "format": "JSON", "timezone": "UTC"}
    resp = http.request("GET", url, params=params)
    data = resp.json()
    if data.get("status") == "error": raise RuntimeError(f"TD: {data.get('message')}")
    values = data.get("values")
    if not values: raise RuntimeError("TD empty")
    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    now = pd.Timestamp.now(tz="UTC")
    df = df[df.index <= now].reset_index()
    logger.info("Fetched %d bars | %s → %s", len(df),
                df["datetime"].iloc[0], df["datetime"].iloc[-1])
    return df

def compute_asia_range(df, trade_date):
    today = df[df["datetime"].dt.date == trade_date]
    if len(today) == 0: return None
    m = (today["datetime"].dt.hour >= ASIA_START_UTC) & (today["datetime"].dt.hour < ASIA_END_UTC)
    asia = today[m]
    hours_needed = (ASIA_END_UTC - ASIA_START_UTC) % 24
    if len(asia) < max(3, hours_needed - 1): return None
    hi = float(asia["high"].max()); lo = float(asia["low"].min())
    cl = float(asia["close"].iloc[-1])
    rng = (hi - lo) / cl if cl > 0 else 0
    if rng < RANGE_MIN_PCT or rng > RANGE_MAX_PCT: return None
    return {"high": hi, "low": lo, "range_pct": rng}

def detect_signal(df, trade_date, asia_high, asia_low, fade=False):
    """Detect breakout, apply fade if requested."""
    today = df[df["datetime"].dt.date == trade_date].sort_values("datetime")
    if len(today) == 0: return None
    after = today[today["datetime"].dt.hour >= ENTRY_HOUR_UTC]
    if len(after) == 0: return None
    for _, row in after.iterrows():
        c = float(row["close"]); breakout_dir = None
        if c > asia_high: breakout_dir = "BUY"
        elif c < asia_low: breakout_dir = "SELL"
        if breakout_dir is None: continue
        actual_dir = breakout_dir
        if fade: actual_dir = "BUY" if breakout_dir == "SELL" else "SELL"
        entry = c
        sl_dist = entry * SL_MAX_PCT
        if actual_dir == "BUY":
            sl = entry - sl_dist; tp = entry + TP_R * sl_dist
            be_t = entry + BE_R * sl_dist
        else:
            sl = entry + sl_dist; tp = entry - TP_R * sl_dist
            be_t = entry - BE_R * sl_dist
        return {"direction": actual_dir, "entry": entry, "bar_time": row["datetime"],
                "sl": sl, "tp": tp, "be_trigger": be_t,
                "sl_pct": sl_dist / entry * 100}
    return None

def simulate_trade(df, trade_date, sig):
    direction = sig["direction"]; entry = sig["entry"]
    sl = sig["sl"]; tp = sig["tp"]; be_t = sig["be_trigger"]
    be_hit = False
    after = df[(df["datetime"] > sig["bar_time"]) &
               (df["datetime"].dt.date == trade_date)].sort_values("datetime")
    for _, row in after.iterrows():
        h = float(row["high"]); l = float(row["low"]); bt = row["datetime"]
        if not be_hit:
            if (direction == "BUY" and h >= be_t) or (direction == "SELL" and l <= be_t):
                be_hit = True; sl = entry
        sl_hit = (direction == "BUY" and l <= sl) or (direction == "SELL" and h >= sl)
        if sl_hit:
            return {"outcome": "BE" if be_hit and sl == entry else "SL",
                    "exit": sl, "be_hit": be_hit}
        tp_hit = (direction == "BUY" and h >= tp) or (direction == "SELL" and l <= tp)
        if tp_hit:
            return {"outcome": "TP", "exit": tp, "be_hit": be_hit}
        if bt.hour >= CLOSE_HOUR_UTC:
            return {"outcome": "TIMEOUT", "exit": float(row["close"]), "be_hit": be_hit}
    return None

def position_size(entry, sl, balance):
    risk = balance * RISK_PCT; dist = abs(entry - sl)
    if dist <= 0: return 0.0
    per_lot = dist * CONTRACT_SIZE
    lot = math.floor((risk / per_lot) / MIN_LOT) * MIN_LOT
    return max(min(lot, MAX_LOT), 0.0)

def backtest_mode(df, n_days, fade=False):
    label = "FADE (inverted)" if fade else "BREAKOUT (original)"
    logger.info("\n═══ MODE: %s — %d hari ═══", label, n_days)
    logger.info("%-12s | %-5s | %-8s | %-6s | %-8s | %-5s | %s",
                "Date", "Dir", "Entry", "SL%", "Range%", "BE", "Outcome")
    logger.info("-" * 85)
    all_dates = sorted(df["datetime"].dt.date.unique())
    recent = all_dates[-n_days:]
    trades = []; balance = ACCOUNT_BALANCE
    for d in recent:
        asia = compute_asia_range(df, d)
        if asia is None:
            logger.info("%-12s | SKIP (range)", d); continue
        sig = detect_signal(df, d, asia["high"], asia["low"], fade=fade)
        if sig is None:
            logger.info("%-12s | SKIP (no breakout)", d); continue
        outcome = simulate_trade(df, d, sig)
        if outcome is None:
            logger.info("%-12s | %-5s | %-8.2f | %-6.2f | %-8.3f | OPEN",
                        d, sig["direction"], sig["entry"], sig["sl_pct"],
                        asia["range_pct"] * 100); continue
        lot = position_size(sig["entry"], sig["sl"], balance)
        pnl = ((outcome["exit"] - sig["entry"]) if sig["direction"] == "BUY"
               else (sig["entry"] - outcome["exit"])) * lot * CONTRACT_SIZE
        balance += pnl
        logger.info("%-12s | %-5s | %-8.2f | %-6.2f | %-8.3f | %-5s | %s PnL=$%+.2f",
                    d, sig["direction"], sig["entry"], sig["sl_pct"],
                    asia["range_pct"] * 100, "yes" if outcome["be_hit"] else "no",
                    outcome["outcome"], pnl)
        trades.append({"pnl": pnl, "outcome": outcome["outcome"]})
    if not trades:
        logger.warning("No trades")
        return {"label": label, "n": 0, "wins": 0, "losses": 0, "pnl": 0.0, "pf": 0.0,
                "tp": 0, "sl": 0, "be": 0, "to": 0, "wr": 0.0}
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    gw = sum(t["pnl"] for t in wins); gl = abs(sum(t["pnl"] for t in losses))
    pf = gw / gl if gl > 0 else float('inf')
    tp = sum(1 for t in trades if t["outcome"] == "TP")
    sl = sum(1 for t in trades if t["outcome"] == "SL")
    be = sum(1 for t in trades if t["outcome"] == "BE")
    to = sum(1 for t in trades if t["outcome"] == "TIMEOUT")
    total = sum(t["pnl"] for t in trades)
    wr = 100 * len(wins) / len(trades)
    logger.info("-" * 85)
    logger.info("RESULT: n=%d | Win=%d | Loss=%d | TP=%d SL=%d BE=%d TO=%d",
                len(trades), len(wins), len(losses), tp, sl, be, to)
    logger.info("WR=%.1f%% | PnL=$%+.2f | PF=%.2f", wr, total, pf)
    return {"label": label, "n": len(trades), "wins": len(wins),
            "losses": len(losses), "pnl": total, "pf": pf,
            "tp": tp, "sl": sl, "be": be, "to": to, "wr": wr}

def send_tg(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT: return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT, "text": msg, "parse_mode": "HTML"}
    for a in range(3):
        try: r = http.request("POST", url, json=payload)
        except RuntimeError: return False
        if r.status_code == 200: return True
        if r.status_code == 429:
            try: ra = int(r.json().get("parameters", {}).get("retry_after", 5))
            except: ra = 5
            time.sleep(ra + 1); continue
        time.sleep(2 ** a)
    return False

def load_state():
    url = f"https://api.github.com/gists/{GIST_ID}"
    h = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    try: r = http.request("GET", url, headers=h)
    except RuntimeError: return {"etag": None, "content": {}}
    if r.status_code != 200: return {"etag": None, "content": {}}
    content = {}
    for name, f in r.json().get("files", {}).items():
        if name.endswith(".json"):
            try: content = json.loads(f.get("content", "{}")); break
            except: continue
    return {"etag": r.headers.get("ETag"), "content": content}

def save_state(state, etag):
    url = f"https://api.github.com/gists/{GIST_ID}"
    payload = {"files": {"london_state.json": {
        "content": json.dumps(state, indent=2, default=str)}}}
    h = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    if etag: h["If-Match"] = etag
    try: r = http.request("PATCH", url, headers=h, json=payload)
    except RuntimeError: return False
    if r.status_code in (200, 201): return True
    if r.status_code == 412:
        h.pop("If-Match", None)
        try: r = http.request("PATCH", url, headers=h, json=payload)
        except RuntimeError: return False
        if r.status_code in (200, 201): return True
    return False

def check_cb(state):
    bal = state.get("balance", ACCOUNT_BALANCE)
    peak = state.get("peak_equity", bal)
    daily = state.get("daily_pnl", 0.0)
    consec = state.get("consecutive_losses", 0)
    if state.get("halted", False): return True
    peak = max(peak, bal); state["peak_equity"] = peak
    dd = (peak - bal) / max(peak, 1e-9)
    dl = -daily / bal if daily < 0 else 0.0
    if dd >= MAX_DRAWDOWN_PCT: state["halted"] = True; return True
    if dl >= MAX_DAILY_LOSS_PCT: state["halted"] = True; return True
    if consec >= MAX_CONSECUTIVE_LOSSES: state["halted"] = True; return True
    return False

def check_trade_outcome(df, trade):
    direction = trade["direction"]; entry = trade["entry"]
    sl = trade["sl"]; tp = trade["tp"]; be_t = trade["be_trigger"]
    be_hit = trade.get("be_hit", False)
    opened = pd.Timestamp(trade["opened_at"])
    after = df[df["datetime"] > opened].sort_values("datetime")
    for _, row in after.iterrows():
        h = float(row["high"]); l = float(row["low"]); bt = row["datetime"]
        if not be_hit:
            if (direction == "BUY" and h >= be_t) or (direction == "SELL" and l <= be_t):
                be_hit = True; sl = entry
        sl_hit = (direction == "BUY" and l <= sl) or (direction == "SELL" and h >= sl)
        if sl_hit:
            return {"outcome": "BE" if be_hit and sl == entry else "SL",
                    "exit_price": sl, "be_hit": be_hit}
        tp_hit = (direction == "BUY" and h >= tp) or (direction == "SELL" and l <= tp)
        if tp_hit:
            return {"outcome": "TP", "exit_price": tp, "be_hit": be_hit}
        if bt.hour >= CLOSE_HOUR_UTC:
            return {"outcome": "TIMEOUT", "exit_price": float(row["close"]), "be_hit": be_hit}
    return None

def main():
    logger.info("═══ LONDON BO v1.4 | BACKTEST=%s | FADE=%s ═══", BACKTEST_MODE, FADE_MODE)

    if BACKTEST_MODE:
        try: df = fetch_ohlcv()
        except Exception as e: logger.error("Data: %s", e); return

        r_bo = backtest_mode(df, BACKTEST_DAYS, fade=False)
        r_fade = backtest_mode(df, BACKTEST_DAYS, fade=True)

        logger.info("\n" + "=" * 65)
        logger.info("═══ COMPARISON ═══")
        logger.info("=" * 65)
        logger.info("%-20s | %-4s | %-6s | %-10s | %-6s",
                    "Mode", "n", "WR%", "PnL", "PF")
        logger.info("-" * 65)
        for r in [r_bo, r_fade]:
            logger.info("%-20s | %-4d | %-6.1f | $%+.2f | %.2f",
                        r["label"][:20], r["n"], r["wr"], r["pnl"], r["pf"])

        logger.info("\n" + "=" * 65)
        if r_fade["pf"] > r_bo["pf"] and r_fade["pf"] > 1.0:
            logger.info("WINNER: FADE (PF=%.2f) → set FADE_MODE=1 untuk live", r_fade["pf"])
        elif r_bo["pf"] > 1.0:
            logger.info("WINNER: BREAKOUT (PF=%.2f) → set FADE_MODE=0 untuk live", r_bo["pf"])
        else:
            logger.info("BOTH FAILED (best PF=%.2f) → JANGAN LIVE TRADING",
                        max(r_bo["pf"], r_fade["pf"]))
        logger.info("=" * 65)

        send_tg(
            f"📊 <b>COMPARE: Breakout vs Fade</b> — {BACKTEST_DAYS} hari\n"
            f"━━━━━━━━━━━━━━━\n"
            f"<b>BREAKOUT</b>: n={r_bo['n']} | WR={r_bo['wr']:.1f}% | "
            f"PnL=${r_bo['pnl']:+.2f} | PF={r_bo['pf']:.2f}\n"
            f"<b>FADE</b>: n={r_fade['n']} | WR={r_fade['wr']:.1f}% | "
            f"PnL=${r_fade['pnl']:+.2f} | PF={r_fade['pf']:.2f}\n"
            f"━━━━━━━━━━━━━━━\n"
            + (f"✅ FADE lebih baik (PF={r_fade['pf']:.2f})"
               if r_fade["pf"] > r_bo["pf"] else
               f"✅ BREAKOUT lebih baik (PF={r_bo['pf']:.2f})")
        )
        return

    # LIVE MODE (sama seperti v1.3, tapi FADE_MODE toggle)
    missing = [n for n, v in [("TWELVEDATA_API_KEY", TWELVE_DATA_KEY),
        ("TELEGRAM_TOKEN", TELEGRAM_TOKEN), ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT),
        ("GIST_ID", GIST_ID), ("GITHUB_TOKEN", GITHUB_TOKEN)] if not v]
    if missing: logger.error("Env missing: %s", missing); return

    s = load_state()
    state = s["content"] or {}
    state.setdefault("balance", ACCOUNT_BALANCE)
    state.setdefault("peak_equity", ACCOUNT_BALANCE)
    state.setdefault("daily_pnl", 0.0)
    state.setdefault("consecutive_losses", 0)
    state.setdefault("halted", False)
    state.setdefault("active_trade", None)

    if check_cb(state):
        send_tg("🚨 CIRCUIT BREAKER"); return

    try: df = fetch_ohlcv()
    except Exception as e:
        logger.error("Data: %s", e); send_tg(f"⚠️ {e}"); return

    now = datetime.now(timezone.utc)
    today = now.date(); today_str = str(today)

    trade = state.get("active_trade")
    if trade and trade.get("trade_date") == today_str:
        outcome = check_trade_outcome(df, trade)
        if outcome:
            entry = trade["entry"]; exit_px = outcome["exit_price"]
            lot = trade.get("lot", 0.01)
            pnl = ((exit_px - entry) if trade["direction"] == "BUY"
                   else (entry - exit_px)) * lot * CONTRACT_SIZE
            state["balance"] = state.get("balance", ACCOUNT_BALANCE) + pnl
            state["daily_pnl"] = state.get("daily_pnl", 0.0) + pnl
            if outcome["outcome"] == "SL":
                state["consecutive_losses"] = state.get("consecutive_losses", 0) + 1
            elif outcome["outcome"] == "TP":
                state["consecutive_losses"] = 0
            emoji = {"TP": "✅", "SL": "❌", "BE": "⚖️", "TIMEOUT": "⏰"}[outcome["outcome"]]
            send_tg(f"{emoji} <b>{outcome['outcome']}</b> {trade['direction']}\n"
                    f"PnL ${pnl:+.2f} | Balance ${state['balance']:.2f}")
            state["active_trade"] = None
    elif trade:
        state["active_trade"] = None

    if state.get("active_trade") is None:
        asia = compute_asia_range(df, today)
        if asia:
            sig = detect_signal(df, today, asia["high"], asia["low"], fade=FADE_MODE)
            if sig:
                last_closed = df.iloc[-1]
                if sig["bar_time"] == last_closed["datetime"]:
                    age_min = (now - sig["bar_time"].to_pydatetime()).total_seconds() / 60
                    if age_min <= 90:
                        lot = position_size(sig["entry"], sig["sl"], state["balance"])
                        if lot >= MIN_LOT:
                            state["active_trade"] = {
                                "direction": sig["direction"], "entry": sig["entry"],
                                "sl": sig["sl"], "tp": sig["tp"],
                                "be_trigger": sig["be_trigger"], "be_hit": False,
                                "lot": lot, "opened_at": str(now),
                                "trade_date": today_str}
                            emoji = "🟢" if sig["direction"] == "BUY" else "🔴"
                            mode_str = "FADE" if FADE_MODE else "BREAKOUT"
                            send_tg(
                                f"{emoji} <b>{sig['direction']}</b> {SYMBOL} — {mode_str}\n"
                                f"Entry {sig['entry']:.2f} | SL {sig['sl']:.2f} | TP {sig['tp']:.2f}\n"
                                f"Lot {lot} | BE {sig['be_trigger']:.2f}")

    state["last_update"] = now.isoformat()
    for _ in range(2):
        if save_state(state, s.get("etag")): break
        s = load_state()

    logger.info("═══ COMPLETE ═══")

if __name__ == "__main__":
    main()
