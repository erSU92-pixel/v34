"""
XAUUSD LONDON RANGE BREAKOUT v1.3 — FINAL
================================================================
Fitur:
  - London Range Breakout (backtest 3.7 tahun: PF 1.38, OOS 2026 PF 1.61)
  - BACKTEST_MODE=1 → langsung test historis, tidak nunggu jam 08:00 UTC
  - BACKTEST_MODE=0 → live mode (cron tiap jam)
  - Idempotent signal detection (tidak fire duplikat meskipun state gagal)
  - Gist error handling dengan diagnostik lengkap (404/401/403/412)
  - Circuit breaker: DD 15%, daily loss 2%, 4 consecutive losses
  - Telegram notifikasi untuk signal + outcome + backtest summary
================================================================
"""
from __future__ import annotations

import os
import sys
import time
import json
import math
import logging
from datetime import datetime, timezone, date as date_cls
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("LONDON_BO")

# =============================================================================
# ENV
# =============================================================================
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
NOTIFY_HOLD = os.getenv("NOTIFY_HOLD", "0") == "1"

BACKTEST_MODE = os.getenv("BACKTEST_MODE", "0") == "1"
BACKTEST_DAYS = int(os.getenv("BACKTEST_DAYS", "14"))

# =============================================================================
# HTTP
# =============================================================================
class ResilientSession:
    def __init__(self, min_interval_sec: float = 7.5):
        self.session = requests.Session()
        self.min_interval = min_interval_sec
        self._last = 0.0

    def _throttle(self):
        elapsed = time.time() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        for attempt in range(5):
            self._throttle()
            try:
                r = self.session.request(method, url, timeout=30, **kwargs)
                self._last = time.time()
                if r.status_code == 429:
                    ra = int(r.headers.get("Retry-After", 2 ** attempt))
                    logger.warning("HTTP 429 — sleep %ss", ra)
                    time.sleep(ra + np.random.uniform(0, 1))
                    continue
                if 500 <= r.status_code < 600:
                    time.sleep((2 ** attempt) + np.random.uniform(0, 1))
                    continue
                return r
            except requests.RequestException as e:
                logger.warning("HTTP err: %s", e)
                time.sleep((2 ** attempt) + np.random.uniform(0, 1))
        raise RuntimeError(f"HTTP failed: {method} {url}")

http = ResilientSession()

# =============================================================================
# DATA
# =============================================================================
def fetch_ohlcv() -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL, "interval": INTERVAL, "outputsize": OUTPUT_SIZE,
        "apikey": TWELVE_DATA_KEY, "format": "JSON", "timezone": "UTC",
    }
    resp = http.request("GET", url, params=params)
    data = resp.json()
    if data.get("status") == "error":
        raise RuntimeError(f"TD error: {data.get('message')}")
    values = data.get("values")
    if not values:
        raise RuntimeError("TD empty")
    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    now = pd.Timestamp.now(tz="UTC")
    df = df[df.index <= now].reset_index()
    logger.info("Fetched %d bars | %s → %s", len(df),
                df["datetime"].iloc[0], df["datetime"].iloc[-1])
    return df

# =============================================================================
# CORE LOGIC
# =============================================================================
def compute_asia_range(df: pd.DataFrame, trade_date: date_cls) -> Optional[Dict[str, Any]]:
    today = df[df["datetime"].dt.date == trade_date]
    if len(today) == 0:
        return None
    m = (today["datetime"].dt.hour >= ASIA_START_UTC) & \
        (today["datetime"].dt.hour < ASIA_END_UTC)
    asia = today[m]
    hours_needed = (ASIA_END_UTC - ASIA_START_UTC) % 24
    if len(asia) < max(3, hours_needed - 1):
        return None
    hi = float(asia["high"].max())
    lo = float(asia["low"].min())
    cl = float(asia["close"].iloc[-1])
    rng = (hi - lo) / cl if cl > 0 else 0
    if rng < RANGE_MIN_PCT or rng > RANGE_MAX_PCT:
        return None
    return {"high": hi, "low": lo, "range_pct": rng, "bars": len(asia)}


def detect_breakout(df: pd.DataFrame, trade_date: date_cls,
                    hi: float, lo: float) -> Optional[Dict[str, Any]]:
    today = df[df["datetime"].dt.date == trade_date].sort_values("datetime")
    if len(today) == 0:
        return None
    after = today[today["datetime"].dt.hour >= ENTRY_HOUR_UTC]
    if len(after) == 0:
        return None
    for _, row in after.iterrows():
        c = float(row["close"])
        if c > hi:
            return {"direction": "BUY", "entry": c,
                    "bar_time": row["datetime"],
                    "asia_high": hi, "asia_low": lo}
        if c < lo:
            return {"direction": "SELL", "entry": c,
                    "bar_time": row["datetime"],
                    "asia_high": hi, "asia_low": lo}
    return None


def compute_sl_tp(direction: str, entry: float,
                  hi: float, lo: float) -> Dict[str, Any]:
    raw = (entry - lo) if direction == "BUY" else (hi - entry)
    cap = entry * SL_MAX_PCT
    sl_dist = min(raw, cap)
    if sl_dist <= 0:
        sl_dist = entry * 0.001
    if direction == "BUY":
        return {"sl": round(entry - sl_dist, 2),
                "tp": round(entry + TP_R * sl_dist, 2),
                "be_trigger": round(entry + BE_R * sl_dist, 2),
                "sl_dist": round(sl_dist, 2),
                "sl_pct": round(sl_dist / entry * 100, 3),
                "capped": raw > cap}
    return {"sl": round(entry + sl_dist, 2),
            "tp": round(entry - TP_R * sl_dist, 2),
            "be_trigger": round(entry - BE_R * sl_dist, 2),
            "sl_dist": round(sl_dist, 2),
            "sl_pct": round(sl_dist / entry * 100, 3),
            "capped": raw > cap}


def simulate_outcome(df: pd.DataFrame, trade_date: date_cls,
                     breakout: Dict[str, Any],
                     sltp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Simulasi outcome trade pada data historis."""
    direction = breakout["direction"]
    entry = breakout["entry"]
    sl = sltp["sl"]
    tp = sltp["tp"]
    be_t = sltp["be_trigger"]
    be_hit = False

    after = df[(df["datetime"] > breakout["bar_time"]) &
               (df["datetime"].dt.date == trade_date)].sort_values("datetime")
    for _, row in after.iterrows():
        h = float(row["high"])
        l = float(row["low"])
        bt = row["datetime"]

        if not be_hit:
            if (direction == "BUY" and h >= be_t) or \
               (direction == "SELL" and l <= be_t):
                be_hit = True
                sl = entry

        sl_hit = (direction == "BUY" and l <= sl) or \
                 (direction == "SELL" and h >= sl)
        if sl_hit:
            return {"outcome": "BE" if be_hit and sl == entry else "SL",
                    "exit": sl, "be_hit": be_hit}

        tp_hit = (direction == "BUY" and h >= tp) or \
                 (direction == "SELL" and l <= tp)
        if tp_hit:
            return {"outcome": "TP", "exit": tp, "be_hit": be_hit}

        if bt.hour >= CLOSE_HOUR_UTC:
            return {"outcome": "TIMEOUT",
                    "exit": float(row["close"]), "be_hit": be_hit}
    return None


def position_size(entry: float, sl: float, balance: float) -> float:
    risk = balance * RISK_PCT
    dist = abs(entry - sl)
    if dist <= 0:
        return 0.0
    per_lot = dist * CONTRACT_SIZE
    if per_lot <= 0:
        return 0.0
    lot = math.floor((risk / per_lot) / MIN_LOT) * MIN_LOT
    return max(min(lot, MAX_LOT), 0.0)

# =============================================================================
# BACKTEST
# =============================================================================
def backtest_recent(df: pd.DataFrame, n_days: int = 14) -> None:
    logger.info("═══ BACKTEST MODE — %d hari terakhir ═══", n_days)
    all_dates = sorted(df["datetime"].dt.date.unique())
    recent = all_dates[-n_days:]

    trades = []
    balance = ACCOUNT_BALANCE

    logger.info("%-12s | %-5s | %-8s | %-6s | %-8s | %-5s | %s",
                "Date", "Dir", "Entry", "SL%", "Range%", "BE", "Outcome")
    logger.info("-" * 85)

    for d in recent:
        asia = compute_asia_range(df, d)
        if asia is None:
            logger.info("%-12s | SKIP (range filter)", d)
            continue
        br = detect_breakout(df, d, asia["high"], asia["low"])
        if br is None:
            logger.info("%-12s | SKIP (no breakout)", d)
            continue
        sltp = compute_sl_tp(br["direction"], br["entry"],
                             asia["high"], asia["low"])
        outcome = simulate_outcome(df, d, br, sltp)
        if outcome is None:
            logger.info("%-12s | %-5s | %-8.2f | %-6.2f | %-8.3f | OPEN",
                        d, br["direction"], br["entry"],
                        sltp["sl_pct"], asia["range_pct"] * 100)
            continue

        lot = position_size(br["entry"], sltp["sl"], balance)
        pnl = ((outcome["exit"] - br["entry"]) if br["direction"] == "BUY"
               else (br["entry"] - outcome["exit"])) * lot * CONTRACT_SIZE
        balance += pnl
        be_str = "yes" if outcome["be_hit"] else "no"

        logger.info("%-12s | %-5s | %-8.2f | %-6.2f | %-8.3f | %-5s | %s PnL=$%+.2f",
                    d, br["direction"], br["entry"], sltp["sl_pct"],
                    asia["range_pct"] * 100, be_str, outcome["outcome"], pnl)

        trades.append({"date": str(d), "direction": br["direction"],
                       "outcome": outcome["outcome"], "pnl": pnl,
                       "range_pct": asia["range_pct"], "be_hit": outcome["be_hit"]})

    if not trades:
        logger.warning("No trades in %d days — range filter atau tidak ada breakout", n_days)
        send_tg(f"📊 <b>BACKTEST</b> ({n_days} hari)\n0 trade — filter terlalu ketat atau tidak ada breakout")
        return

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    total_pnl = sum(t["pnl"] for t in trades)
    gross_w = sum(t["pnl"] for t in wins)
    gross_l = abs(sum(t["pnl"] for t in losses))
    pf = gross_w / gross_l if gross_l > 0 else float('inf')

    tp = sum(1 for t in trades if t["outcome"] == "TP")
    sl = sum(1 for t in trades if t["outcome"] == "SL")
    be = sum(1 for t in trades if t["outcome"] == "BE")
    to = sum(1 for t in trades if t["outcome"] == "TIMEOUT")
    wr = 100 * len(wins) / len(trades)

    logger.info("-" * 85)
    logger.info("═══ SUMMARY ═══")
    logger.info("Trades: %d | Win: %d | Loss: %d", len(trades), len(wins), len(losses))
    logger.info("TP=%d SL=%d BE=%d TIMEOUT=%d", tp, sl, be, to)
    logger.info("Win rate: %.1f%%", wr)
    logger.info("Total PnL: $%+.2f | Balance: $%.2f → $%.2f",
                total_pnl, ACCOUNT_BALANCE, balance)
    logger.info("Profit Factor: %.2f", pf)

    send_tg(
        f"📊 <b>BACKTEST London BO — {n_days} hari</b>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"Trades: {len(trades)} (TP={tp} SL={sl} BE={be} TO={to})\n"
        f"Win rate: {wr:.1f}%\n"
        f"PnL: <b>${total_pnl:+.2f}</b>\n"
        f"Balance: ${ACCOUNT_BALANCE:.2f} → ${balance:.2f}\n"
        f"<b>Profit Factor: {pf:.2f}</b>"
    )

# =============================================================================
# STATE (Gist with full diagnostics)
# =============================================================================
def load_state() -> Dict[str, Any]:
    url = f"https://api.github.com/gists/{GIST_ID}"
    h = {"Authorization": f"token {GITHUB_TOKEN}",
         "Accept": "application/vnd.github+json"}
    try:
        r = http.request("GET", url, headers=h)
    except RuntimeError as e:
        logger.error("Gist GET exception: %s", e)
        return {"etag": None, "content": {}}

    if r.status_code != 200:
        logger.warning("Gist GET status=%d body=%s", r.status_code, r.text[:150])
        if r.status_code == 404:
            logger.error("→ GIST_ID tidak ditemukan. Verifikasi secret GIST_ID")
        elif r.status_code in (401, 403):
            logger.error("→ GIST_TOKEN invalid / tanpa scope 'gist'. Buat PAT baru")
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


def save_state(state: Dict[str, Any], etag: Optional[str]) -> bool:
    url = f"https://api.github.com/gists/{GIST_ID}"
    payload = {"files": {"london_state.json": {
        "content": json.dumps(state, indent=2, default=str)}}}

    h = {"Authorization": f"token {GITHUB_TOKEN}",
         "Accept": "application/vnd.github+json"}
    if etag:
        h["If-Match"] = etag

    try:
        r = http.request("PATCH", url, headers=h, json=payload)
    except RuntimeError as e:
        logger.error("Gist PATCH exception: %s", e)
        return False

    if r.status_code in (200, 201):
        logger.info("Gist saved OK")
        return True

    if r.status_code == 412:
        logger.warning("Gist 412 — retry tanpa If-Match")
        h.pop("If-Match", None)
        try:
            r = http.request("PATCH", url, headers=h, json=payload)
        except RuntimeError as e:
            logger.error("Gist PATCH retry exception: %s", e)
            return False
        if r.status_code in (200, 201):
            logger.info("Gist saved (tanpa If-Match)")
            return True

    logger.error("Gist PATCH failed: status=%d body=%s",
                 r.status_code, r.text[:200])
    if r.status_code == 404:
        logger.error("→ GIST_ID tidak ditemukan. Cek secret GIST_ID")
    elif r.status_code in (401, 403):
        logger.error("→ Token tanpa akses. Cek scope 'gist' pada GIST_TOKEN")
    return False

# =============================================================================
# TELEGRAM
# =============================================================================
def send_tg(msg: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        logger.warning("Telegram not configured")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT, "text": msg, "parse_mode": "HTML"}
    for a in range(3):
        try:
            r = http.request("POST", url, json=payload)
        except RuntimeError:
            return False
        if r.status_code == 200:
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
# RISK
# =============================================================================
def check_circuit_breaker(state: Dict[str, Any]) -> bool:
    bal = state.get("balance", ACCOUNT_BALANCE)
    peak = state.get("peak_equity", bal)
    daily = state.get("daily_pnl", 0.0)
    consec = state.get("consecutive_losses", 0)
    if state.get("halted", False):
        return True
    peak = max(peak, bal)
    state["peak_equity"] = peak
    dd = (peak - bal) / max(peak, 1e-9)
    dl = -daily / bal if daily < 0 else 0.0
    if dd >= MAX_DRAWDOWN_PCT:
        logger.critical("CB: drawdown %.2f%%", dd * 100)
        state["halted"] = True
        return True
    if dl >= MAX_DAILY_LOSS_PCT:
        logger.critical("CB: daily loss %.2f%%", dl * 100)
        state["halted"] = True
        return True
    if consec >= MAX_CONSECUTIVE_LOSSES:
        logger.critical("CB: %d consecutive losses", consec)
        state["halted"] = True
        return True
    return False


def check_trade_outcome(df: pd.DataFrame, trade: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    direction = trade["direction"]
    entry = trade["entry"]
    sl = trade["sl"]
    tp = trade["tp"]
    be_t = trade["be_trigger"]
    be_hit = trade.get("be_hit", False)
    opened = pd.Timestamp(trade["opened_at"])

    after = df[df["datetime"] > opened].sort_values("datetime")
    for _, row in after.iterrows():
        h = float(row["high"])
        l = float(row["low"])
        bt = row["datetime"]

        if not be_hit:
            if (direction == "BUY" and h >= be_t) or \
               (direction == "SELL" and l <= be_t):
                be_hit = True
                sl = entry

        sl_hit = (direction == "BUY" and l <= sl) or \
                 (direction == "SELL" and h >= sl)
        if sl_hit:
            return {"outcome": "BE" if be_hit and sl == entry else "SL",
                    "exit_price": sl, "be_hit": be_hit}

        tp_hit = (direction == "BUY" and h >= tp) or \
                 (direction == "SELL" and l <= tp)
        if tp_hit:
            return {"outcome": "TP", "exit_price": tp, "be_hit": be_hit}

        if bt.hour >= CLOSE_HOUR_UTC:
            return {"outcome": "TIMEOUT",
                    "exit_price": float(row["close"]), "be_hit": be_hit}
    return None

# =============================================================================
# MAIN
# =============================================================================
def main():
    logger.info("═══ LONDON BO v1.3 | %s | BACKTEST=%s ═══",
                SYMBOL, BACKTEST_MODE)

    if BACKTEST_MODE:
        try:
            df = fetch_ohlcv()
        except Exception as e:
            logger.error("Data failed: %s", e)
            return
        backtest_recent(df, BACKTEST_DAYS)
        return

    logger.info("Asia %02d-%02d UTC | Entry %02d UTC | Close %02d UTC",
                ASIA_START_UTC, ASIA_END_UTC, ENTRY_HOUR_UTC, CLOSE_HOUR_UTC)

    missing = [n for n, v in [
        ("TWELVEDATA_API_KEY", TWELVE_DATA_KEY),
        ("TELEGRAM_TOKEN", TELEGRAM_TOKEN),
        ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT),
        ("GIST_ID", GIST_ID),
        ("GITHUB_TOKEN", GITHUB_TOKEN),
    ] if not v]
    if missing:
        logger.error("Env missing: %s", missing)
        sys.exit(1)

    s = load_state()
    state = s["content"] or {}
    state.setdefault("balance", ACCOUNT_BALANCE)
    state.setdefault("peak_equity", ACCOUNT_BALANCE)
    state.setdefault("daily_pnl", 0.0)
    state.setdefault("consecutive_losses", 0)
    state.setdefault("halted", False)
    state.setdefault("active_trade", None)

    if check_circuit_breaker(state):
        send_tg("🚨 <b>CIRCUIT BREAKER</b> — bot halted")
        return

    try:
        df = fetch_ohlcv()
    except Exception as e:
        logger.error("Data failed: %s", e)
        send_tg(f"⚠️ Data: {e}")
        return

    now = datetime.now(timezone.utc)
    today = now.date()
    today_str = str(today)

    # ==== PHASE 1: Monitor existing trade ====
    trade = state.get("active_trade")
    if trade:
        if trade.get("trade_date") != today_str:
            logger.info("Closing stale trade from %s", trade.get("trade_date"))
            state["active_trade"] = None
        else:
            outcome = check_trade_outcome(df, trade)
            if outcome:
                entry = trade["entry"]
                exit_px = outcome["exit_price"]
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
                send_tg(
                    f"{emoji} <b>{outcome['outcome']}</b> — {trade['direction']} {SYMBOL}\n"
                    f"Entry {entry:.2f} → Exit {exit_px:.2f}\n"
                    f"PnL <b>${pnl:+.2f}</b> | Balance ${state['balance']:.2f}"
                )
                state["active_trade"] = None
            else:
                logger.info("Trade active: %s @ %.2f", trade["direction"], trade["entry"])

    # ==== PHASE 2: Signal detection (idempotent) ====
    if state.get("active_trade") is None:
        asia = compute_asia_range(df, today)
        if asia:
            br = detect_breakout(df, today, asia["high"], asia["low"])
            if br:
                last_closed = df.iloc[-1]
                if br["bar_time"] != last_closed["datetime"]:
                    logger.info("Breakout bar %s is not last closed — skip (idempotent)",
                                br["bar_time"])
                else:
                    age_min = (now - br["bar_time"].to_pydatetime()).total_seconds() / 60
                    if age_min > 90:
                        logger.info("Breakout bar is %.0f min old — stale, skip", age_min)
                    else:
                        sltp = compute_sl_tp(br["direction"], br["entry"],
                                             asia["high"], asia["low"])
                        lot = position_size(br["entry"], sltp["sl"], state["balance"])
                        if lot < MIN_LOT:
                            logger.warning("Lot %.4f < min — skip", lot)
                        else:
                            state["active_trade"] = {
                                "direction": br["direction"],
                                "entry": br["entry"],
                                "sl": sltp["sl"],
                                "tp": sltp["tp"],
                                "be_trigger": sltp["be_trigger"],
                                "be_hit": False,
                                "lot": lot,
                                "opened_at": str(now),
                                "trade_date": today_str,
                                "asia_high": asia["high"],
                                "asia_low": asia["low"],
                            }
                            emoji = "🟢" if br["direction"] == "BUY" else "🔴"
                            send_tg(
                                f"{emoji} <b>{br['direction']}</b> {SYMBOL} — LONDON BO\n"
                                f"━━━━━━━━━━━━━━━\n"
                                f"Asia range: {asia['low']:.2f}—{asia['high']:.2f} "
                                f"({asia['range_pct']*100:.2f}%)\n"
                                f"Entry: <code>{br['entry']:.2f}</code>\n"
                                f"SL: <code>{sltp['sl']:.2f}</code> ({sltp['sl_pct']:.2f}%)\n"
                                f"TP: <code>{sltp['tp']:.2f}</code>\n"
                                f"BE: <code>{sltp['be_trigger']:.2f}</code>\n"
                                f"Lot: <code>{lot}</code>\n"
                                f"Close by {CLOSE_HOUR_UTC:02d}:00 UTC\n"
                                f"Balance ${state['balance']:.2f}"
                            )
                            logger.info("SIGNAL: %s @ %.2f lot=%.2f",
                                        br["direction"], br["entry"], lot)

    # ==== PHASE 3: Save ====
    state["last_update"] = now.isoformat()
    saved = False
    for _ in range(2):
        if save_state(state, s.get("etag")):
            saved = True
            break
        s = load_state()
    if not saved:
        logger.warning("⚠️ State NOT saved — signal tetap jalan via idempotent detection")
        logger.warning("⚠️ Live monitoring mungkin miss — verifikasi GIST_ID + GIST_TOKEN")

    logger.info("═══ COMPLETE ═══")


if __name__ == "__main__":
    main()
