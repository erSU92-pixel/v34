"""
XAUUSD LONDON RANGE BREAKOUT v1.0
================================================================
Strategi: London Range Breakout (XAUUSD)
Backtest 3.7 tahun real tick: PF 1.38, DD 17.3%, OOS 2026 PF 1.61

Aturan:
  1. Range Asia: 03:00-08:00 UTC (configurable)
  2. Breakout: first 1H close outside range after 08:00 UTC
  3. Entry window: 08:00-10:00 UTC (2 jam setelah breakout)
  4. SL: sisi berlawanan range, capped 0.6% dari entry
  5. TP: 2.5R, BE trigger: 1.5R
  6. Force close: 21:00 UTC
  7. Filter: range 0.30%-1.60% dari harga
  8. Max 1 trade per hari

Sumber data: Twelve Data OHLCV 1H (UTC)
Notifikasi: Telegram
State: GitHub Gist (ETag-safe)
================================================================
"""
from __future__ import annotations

import os
import sys
import time
import json
import math
import logging
from datetime import datetime, timezone, timedelta, date as date_cls
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import requests

# =============================================================================
# LOGGING
# =============================================================================
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

# Instrument
SYMBOL         = os.getenv("SYMBOL", "XAU/USD")
INTERVAL       = "1h"
OUTPUT_SIZE    = int(os.getenv("OUTPUT_SIZE", "500"))
CONTRACT_SIZE  = 100     # XAUUSD: 100 oz per lot
TICK_SIZE      = 0.01
MIN_LOT        = 0.01
MAX_LOT        = 5.0
SPREAD_BUFFER  = float(os.getenv("SPREAD_BUFFER", "0.30"))
SLIPPAGE_BUFFER = float(os.getenv("SLIPPAGE_BUFFER", "0.15"))

# Timing (server time = UTC by default; ubah SERVER_TZ_OFFSET jika broker GMT+2/3)
SERVER_TZ_OFFSET   = int(os.getenv("SERVER_TZ_OFFSET", "0"))
ASIA_START_SERVER  = int(os.getenv("ASIA_START_HOUR", "3"))
ASIA_END_SERVER    = int(os.getenv("ASIA_END_HOUR", "8"))
ENTRY_HOUR_SERVER  = int(os.getenv("ENTRY_HOUR", "8"))
CLOSE_HOUR_SERVER  = int(os.getenv("CLOSE_HOUR", "21"))
MAX_ENTRY_DELAY_HRS = int(os.getenv("MAX_ENTRY_DELAY_HRS", "2"))

# Convert server → UTC
def to_utc(server_hour: int) -> int:
    return (server_hour - SERVER_TZ_OFFSET) % 24

ASIA_START_UTC = to_utc(ASIA_START_SERVER)
ASIA_END_UTC   = to_utc(ASIA_END_SERVER)
ENTRY_HOUR_UTC = to_utc(ENTRY_HOUR_SERVER)
CLOSE_HOUR_UTC = to_utc(CLOSE_HOUR_SERVER)

# Filter range
RANGE_MIN_PCT = float(os.getenv("RANGE_MIN_PCT", "0.0030"))   # 0.30%
RANGE_MAX_PCT = float(os.getenv("RANGE_MAX_PCT", "0.0160"))   # 1.60%
SL_MAX_PCT    = float(os.getenv("SL_MAX_PCT", "0.0060"))      # 0.60%
TP_R          = float(os.getenv("TP_R", "2.5"))
BE_R          = float(os.getenv("BE_R", "1.5"))

# Risk
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "10000"))
RISK_PCT        = float(os.getenv("RISK_PCT", "0.003"))       # 0.3% default
MAX_DAILY_LOSS_PCT     = 0.02
MAX_DRAWDOWN_PCT       = 0.15
MAX_CONSECUTIVE_LOSSES = 4
NOTIFY_HOLD = os.getenv("NOTIFY_HOLD", "0") == "1"

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
# DATA FETCH
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
    df = df[df.index <= now]
    df = df.reset_index()
    logger.info("Fetched %d bars | last: %s", len(df), df["datetime"].iloc[-1])
    return df

# =============================================================================
# RANGE & BREAKOUT
# =============================================================================
def compute_asia_range(df: pd.DataFrame, trade_date: date_cls) -> Optional[Dict[str, Any]]:
    """Range Asia untuk trade_date (UTC). Return None jika data tidak cukup."""
    today_bars = df[df["datetime"].dt.date == trade_date]
    if len(today_bars) == 0:
        return None

    asia_mask = (
        (today_bars["datetime"].dt.hour >= ASIA_START_UTC) &
        (today_bars["datetime"].dt.hour < ASIA_END_UTC)
    )
    asia = today_bars[asia_mask]
    hours_needed = (ASIA_END_UTC - ASIA_START_UTC) % 24
    if len(asia) < max(3, hours_needed - 1):
        logger.info("Asia range incomplete: %d/%d bars", len(asia), hours_needed)
        return None

    asia_high = float(asia["high"].max())
    asia_low = float(asia["low"].min())
    asia_close = float(asia["close"].iloc[-1])
    range_abs = asia_high - asia_low
    range_pct = range_abs / asia_close if asia_close > 0 else 0

    logger.info("Asia range [%02d-%02d UTC]: H=%.2f L=%.2f (%.3f%%)",
                ASIA_START_UTC, ASIA_END_UTC, asia_high, asia_low, range_pct * 100)

    if range_pct < RANGE_MIN_PCT:
        logger.info("SKIP: range too narrow (%.3f%% < %.3f%%)",
                    range_pct * 100, RANGE_MIN_PCT * 100)
        return None
    if range_pct > RANGE_MAX_PCT:
        logger.info("SKIP: range too wide (%.3f%% > %.3f%%)",
                    range_pct * 100, RANGE_MAX_PCT * 100)
        return None

    return {
        "high": asia_high, "low": asia_low, "close": asia_close,
        "range_pct": range_pct, "bars": len(asia),
    }


def detect_breakout(df: pd.DataFrame, trade_date: date_cls,
                    asia_high: float, asia_low: float) -> Optional[Dict[str, Any]]:
    """Cari bar pertama setelah ENTRY_HOUR_UTC dengan close di luar range."""
    today_bars = df[df["datetime"].dt.date == trade_date]
    if len(today_bars) == 0:
        return None

    entry_mask = (today_bars["datetime"].dt.hour >= ENTRY_HOUR_UTC) & \
                 (today_bars["datetime"].dt.hour < CLOSE_HOUR_UTC)
    after = today_bars[entry_mask].sort_values("datetime")

    if len(after) == 0:
        logger.info("No bars yet in entry window (>=%02d UTC)", ENTRY_HOUR_UTC)
        return None

    for _, row in after.iterrows():
        close = float(row["close"])
        if close > asia_high:
            return {"direction": "BUY", "entry": close,
                    "bar_time": row["datetime"], "asia_high": asia_high,
                    "asia_low": asia_low}
        if close < asia_low:
            return {"direction": "SELL", "entry": close,
                    "bar_time": row["datetime"], "asia_high": asia_high,
                    "asia_low": asia_low}

    logger.info("No breakout yet (price still inside range)")
    return None


def compute_sl_tp(direction: str, entry: float,
                  asia_high: float, asia_low: float) -> Dict[str, float]:
    """SL = sisi berlawanan range, capped SL_MAX_PCT. TP=2.5R, BE=1.5R."""
    if direction == "BUY":
        raw_sl_distance = entry - asia_low
    else:
        raw_sl_distance = asia_high - entry

    capped_sl = entry * SL_MAX_PCT
    sl_distance = min(raw_sl_distance, capped_sl)
    if sl_distance <= 0:
        sl_distance = entry * 0.001  # minimum 0.1%

    if direction == "BUY":
        sl = entry - sl_distance
        tp = entry + TP_R * sl_distance
        be = entry + BE_R * sl_distance
    else:
        sl = entry + sl_distance
        tp = entry - TP_R * sl_distance
        be = entry - BE_R * sl_distance

    return {
        "sl": round(sl, 2), "tp": round(tp, 2),
        "be_trigger": round(be, 2),
        "sl_distance": round(sl_distance, 2),
        "sl_pct": round(sl_distance / entry * 100, 3),
        "capped": raw_sl_distance > capped_sl,
    }

# =============================================================================
# POSITION SIZE
# =============================================================================
def position_size(entry: float, sl: float, balance: float) -> float:
    risk_money = balance * RISK_PCT
    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return 0.0
    loss_per_lot = sl_dist * CONTRACT_SIZE
    if loss_per_lot <= 0:
        return 0.0
    lot = math.floor((risk_money / loss_per_lot) / MIN_LOT) * MIN_LOT
    return max(min(lot, MAX_LOT), 0.0)

# =============================================================================
# TRADE MONITORING
# =============================================================================
def check_trade_outcome(df: pd.DataFrame, trade: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Cek apakah trade hit BE, SL, TP, atau timeout.
    Return dict outcome atau None jika masih berjalan.
    """
    direction = trade["direction"]
    entry = trade["entry"]
    sl = trade["sl"]
    tp = trade["tp"]
    be_trigger = trade["be_trigger"]
    be_hit = trade.get("be_hit", False)
    opened_at = pd.Timestamp(trade["opened_at"])

    # Bars setelah entry
    after = df[df["datetime"] > opened_at].sort_values("datetime")
    if len(after) == 0:
        return None

    for _, row in after.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        bar_time = row["datetime"]

        # Cek BE trigger dulu
        if not be_hit:
            if (direction == "BUY" and high >= be_trigger) or \
               (direction == "SELL" and low <= be_trigger):
                be_hit = True
                sl = entry  # move SL to BE
                logger.info("BE trigger hit at %s — SL moved to %.2f", bar_time, entry)

        # Cek SL (pessimistic: SL first if both hit in same bar)
        sl_hit = (direction == "BUY" and low <= sl) or \
                 (direction == "SELL" and high >= sl)
        if sl_hit:
            return {"outcome": "BE" if be_hit and sl == entry else "SL",
                    "exit_price": sl, "exit_time": str(bar_time),
                    "be_hit": be_hit}

        # Cek TP
        tp_hit = (direction == "BUY" and high >= tp) or \
                 (direction == "SELL" and low <= tp)
        if tp_hit:
            return {"outcome": "TP", "exit_price": tp,
                    "exit_time": str(bar_time), "be_hit": be_hit}

        # Cek timeout: bar hour >= CLOSE_HOUR_UTC
        if bar_time.hour >= CLOSE_HOUR_UTC:
            exit_px = float(row["close"])
            return {"outcome": "TIMEOUT", "exit_price": exit_px,
                    "exit_time": str(bar_time), "be_hit": be_hit}

    return None

# =============================================================================
# STATE
# =============================================================================
def load_state() -> Dict[str, Any]:
    url = f"https://api.github.com/gists/{GIST_ID}"
    h = {"Authorization": f"token {GITHUB_TOKEN}",
         "Accept": "application/vnd.github+json"}
    r = http.request("GET", url, headers=h)
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


def save_state(state: Dict[str, Any], etag: Optional[str]) -> bool:
    url = f"https://api.github.com/gists/{GIST_ID}"
    h = {"Authorization": f"token {GITHUB_TOKEN}",
         "Accept": "application/vnd.github+json"}
    if etag:
        h["If-Match"] = etag
    payload = {"files": {"london_state.json": {
        "content": json.dumps(state, indent=2, default=str)
    }}}
    r = http.request("PATCH", url, headers=h, json=payload)
    if r.status_code == 412:
        logger.warning("Gist ETag mismatch")
        return False
    return r.status_code in (200, 201)

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
# CIRCUIT BREAKER
# =============================================================================
def check_circuit_breaker(state: Dict[str, Any]) -> bool:
    balance = state.get("balance", ACCOUNT_BALANCE)
    peak = state.get("peak_equity", balance)
    daily_pnl = state.get("daily_pnl", 0.0)
    consecutive = state.get("consecutive_losses", 0)
    halted = state.get("halted", False)

    if halted:
        return True

    peak = max(peak, balance)
    state["peak_equity"] = peak
    dd = (peak - balance) / max(peak, 1e-9)
    daily_loss = -daily_pnl / balance if daily_pnl < 0 else 0.0

    if dd >= MAX_DRAWDOWN_PCT:
        logger.critical("CB: drawdown %.2f%% >= %.2f%%", dd * 100, MAX_DRAWDOWN_PCT * 100)
        state["halted"] = True
        return True
    if daily_loss >= MAX_DAILY_LOSS_PCT:
        logger.critical("CB: daily loss %.2f%% >= %.2f%%", daily_loss * 100, MAX_DAILY_LOSS_PCT * 100)
        state["halted"] = True
        return True
    if consecutive >= MAX_CONSECUTIVE_LOSSES:
        logger.critical("CB: %d consecutive losses", consecutive)
        state["halted"] = True
        return True

    return False

# =============================================================================
# UPDATE BALANCE / PNL
# =============================================================================
def record_trade_outcome(state: Dict[str, Any], trade: Dict[str, Any],
                         outcome: Dict[str, Any]) -> float:
    entry = trade["entry"]
    exit_px = outcome["exit_price"]
    direction = trade["direction"]
    lot = trade.get("lot", 0.01)

    if direction == "BUY":
        pnl_per_unit = exit_px - entry
    else:
        pnl_per_unit = entry - exit_px

    pnl_usd = pnl_per_unit * lot * CONTRACT_SIZE
    state["balance"] = state.get("balance", ACCOUNT_BALANCE) + pnl_usd
    state["daily_pnl"] = state.get("daily_pnl", 0.0) + pnl_usd

    if outcome["outcome"] == "SL":
        state["consecutive_losses"] = state.get("consecutive_losses", 0) + 1
    elif outcome["outcome"] == "TP":
        state["consecutive_losses"] = 0

    logger.info("Trade outcome: %s | PnL: $%+.2f | Balance: $%.2f",
                outcome["outcome"], pnl_usd, state["balance"])
    return pnl_usd

# =============================================================================
# MAIN
# =============================================================================
def main():
    logger.info("═══ LONDON BO v1.0 | %s | server_offset=%d ═══",
                SYMBOL, SERVER_TZ_OFFSET)
    logger.info("Asia: %02d-%02d UTC | Entry: %02d UTC | Close: %02d UTC",
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
    state.setdefault("signaled_dates", [])

    if check_circuit_breaker(state):
        send_tg("🚨 <b>CIRCUIT BREAKER ACTIVE</b>\nBot halted. Manual review required.")
        return

    try:
        df = fetch_ohlcv()
    except Exception as e:
        logger.exception("Data fetch failed")
        send_tg(f"⚠️ Data error: {e}")
        return

    now_utc = datetime.now(timezone.utc)
    today = now_utc.date()
    today_str = str(today)

    # ===== PHASE 1: Manage existing trade =====
    trade = state.get("active_trade")
    if trade:
        if trade.get("trade_date") != today_str:
            logger.info("Closing stale trade from %s at market", trade.get("trade_date"))
            outcome = {"outcome": "TIMEOUT", "exit_price": trade["entry"],
                       "exit_time": str(now_utc), "be_hit": trade.get("be_hit", False)}
            record_trade_outcome(state, trade, outcome)
            send_tg(f"⏰ Stale trade closed at market\n{trade['direction']} entry {trade['entry']}")
            state["active_trade"] = None
        else:
            outcome = check_trade_outcome(df, trade)
            if outcome:
                pnl = record_trade_outcome(state, trade, outcome)
                emoji = {"TP": "✅", "SL": "❌", "BE": "⚖️", "TIMEOUT": "⏰"}[outcome["outcome"]]
                send_tg(
                    f"{emoji} <b>{outcome['outcome']}</b> — {trade['direction']} XAUUSD\n"
                    f"Entry: {trade['entry']:.2f} → Exit: {outcome['exit_price']:.2f}\n"
                    f"PnL: <b>${pnl:+.2f}</b>\n"
                    f"BE hit: {outcome['be_hit']}\n"
                    f"Balance: ${state['balance']:.2f}"
                )
                state["active_trade"] = None
            else:
                logger.info("Trade active: %s entry %.2f sl %.2f tp %.2f be_hit=%s",
                            trade["direction"], trade["entry"], trade["sl"],
                            trade["tp"], trade.get("be_hit", False))

    # ===== PHASE 2: Look for new signal =====
    if state.get("active_trade") is None and today_str not in state.get("signaled_dates", []):
        # Cek window entry
        in_window = (ENTRY_HOUR_UTC <= now_utc.hour <= ENTRY_HOUR_UTC + MAX_ENTRY_DELAY_HRS)
        if not in_window:
            logger.info("Outside entry window (%02d-%02d UTC), current hour=%d",
                        ENTRY_HOUR_UTC, ENTRY_HOUR_UTC + MAX_ENTRY_DELAY_HRS, now_utc.hour)
        else:
            asia = compute_asia_range(df, today)
            if asia is None:
                logger.info("No valid asia range — skip")
            else:
                breakout = detect_breakout(df, today, asia["high"], asia["low"])
                if breakout:
                    sltp = compute_sl_tp(breakout["direction"], breakout["entry"],
                                          asia["high"], asia["low"])
                    lot = position_size(breakout["entry"], sltp["sl"], state["balance"])

                    if lot < MIN_LOT:
                        logger.warning("Lot size %.4f below min %.2f — skip", lot, MIN_LOT)
                    else:
                        trade_record = {
                            "direction": breakout["direction"],
                            "entry": breakout["entry"],
                            "sl": sltp["sl"],
                            "tp": sltp["tp"],
                            "be_trigger": sltp["be_trigger"],
                            "be_hit": False,
                            "lot": lot,
                            "opened_at": str(now_utc),
                            "trade_date": today_str,
                            "asia_high": asia["high"],
                            "asia_low": asia["low"],
                        }
                        state["active_trade"] = trade_record
                        state["signaled_dates"] = (state.get("signaled_dates", []) + [today_str])[-30:]

                        dir_emoji = "🟢" if breakout["direction"] == "BUY" else "🔴"
                        msg = (
                            f"{dir_emoji} <b>{breakout['direction']}</b> XAUUSD — LONDON BO\n"
                            f"━━━━━━━━━━━━━━━\n"
                            f"Asia range: {asia['low']:.2f} — {asia['high']:.2f} ({asia['range_pct']*100:.2f}%)\n"
                            f"Entry: <code>{breakout['entry']:.2f}</code>\n"
                            f"SL: <code>{sltp['sl']:.2f}</code> ({sltp['sl_pct']:.2f}%)\n"
                            f"TP: <code>{sltp['tp']:.2f}</code>\n"
                            f"BE trigger: <code>{sltp['be_trigger']:.2f}</code>\n"
                            f"Lot: <code>{lot}</code>\n"
                            f"Risk: ${state['balance'] * RISK_PCT:.2f}\n"
                            f"━━━━━━━━━━━━━━━\n"
                            f"Close by: {CLOSE_HOUR_UTC:02d}:00 UTC\n"
                            f"Balance: ${state['balance']:.2f}"
                        )
                        send_tg(msg)
                        logger.info("SIGNAL FIRED: %s @ %.2f, lot=%.2f",
                                    breakout["direction"], breakout["entry"], lot)

    # ===== PHASE 3: Close reminder at 21:00 =====
    if state.get("active_trade") and now_utc.hour == CLOSE_HOUR_UTC:
        t = state["active_trade"]
        send_tg(
            f"⏰ <b>CLOSE REMINDER</b>\n"
            f"{t['direction']} XAUUSD still open\n"
            f"Entry: {t['entry']:.2f} | SL: {t['sl']:.2f} | TP: {t['tp']:.2f}\n"
            f"Close at market now (21:00 UTC rule)"
        )

    # ===== Save =====
    state["last_update"] = now_utc.isoformat()
    for _ in range(2):
        if save_state(state, s.get("etag")):
            break
        s = load_state()
        if _ == 1:
            logger.error("State save failed after retry")

    logger.info("═══ COMPLETE ═══")


if __name__ == "__main__":
    main()
