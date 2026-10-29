"""
XAUUSD TRADING BOT v2.0 — LIVE SIGNAL GENERATOR
================================================================
Fungsi: setiap jam, cek XAUUSD, kirim sinyal BUY/SELL ke Telegram.

Strategy default: Mean Reversion (Bollinger + IBS + Trend Filter)
  LONG  : close < lower_BB + IBS < 0.2 + close > SMA200
  SHORT : close > upper_BB + IBS > 0.8 + close < SMA200
  SL    : 2× ATR
  TP    : middle Bollinger Band (SMA20)

Mode:
  LIVE_MODE=1 (default)  → kirim sinyal BUY/SELL ke Telegram
  BACKTEST_MODE=1        → test historis, tidak kirim sinyal live

Strategi bisa diganti via env STRATEGY=mean_rev|trend|breakout
================================================================
"""
from __future__ import annotations
import os, sys, time, json, math, logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("XAUUSD_BOT")

# =============================================================================
# ENV
# =============================================================================
TWELVE_DATA_KEY = os.getenv("TWELVEDATA_API_KEY")
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT   = os.getenv("TELEGRAM_CHAT_ID")
GIST_ID         = os.getenv("GIST_ID")
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN")

SYMBOL      = os.getenv("SYMBOL", "XAU/USD")
TF          = os.getenv("TF", "4h")
OUTPUT_SIZE = int(os.getenv("OUTPUT_SIZE", "500"))
STRATEGY    = os.getenv("STRATEGY", "mean_rev")   # mean_rev | trend | breakout

CONTRACT_SIZE = 100
MIN_LOT = 0.01
MAX_LOT = 5.0
SPREAD_BUFFER = 0.30
SLIPPAGE_BUFFER = 0.15

# Mean-reversion params
BB_PERIOD = 20
BB_STD    = 2.0
SMA_TREND = 200
ATR_PERIOD = 14
ATR_SL_MULT = 2.0
IBS_LOW   = 0.20
IBS_HIGH  = 0.80
MAX_HOLDING_BARS = 20

# Risk
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "10000"))
RISK_PCT        = float(os.getenv("RISK_PCT", "0.003"))
MAX_DRAWDOWN_PCT = 0.20
MAX_CONSEC_LOSSES = 5

# Modes
LIVE_MODE     = os.getenv("LIVE_MODE", "1") == "1"
BACKTEST_MODE = os.getenv("BACKTEST_MODE", "0") == "1"
BACKTEST_DAYS = int(os.getenv("BACKTEST_DAYS", "90"))
NOTIFY_HOLD   = os.getenv("NOTIFY_HOLD", "0") == "1"

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
        if d < self.interval: time.sleep(self.interval - d)
    def req(self, method, url, **kw):
        for a in range(5):
            self._wait()
            try:
                r = self.s.request(method, url, timeout=30, **kw)
                self._last = time.time()
                if r.status_code == 429:
                    time.sleep(int(r.headers.get("Retry-After", 2**a)) + np.random.uniform(0,1))
                    continue
                if 500 <= r.status_code < 600:
                    time.sleep((2**a) + np.random.uniform(0,1)); continue
                return r
            except requests.RequestException:
                time.sleep((2**a) + np.random.uniform(0,1))
        raise RuntimeError(f"HTTP failed: {method} {url}")

http = HTTP()

# =============================================================================
# DATA
# =============================================================================
def fetch_ohlcv() -> pd.DataFrame:
    tf_map = {"1h": "1h", "4h": "4h", "1d": "1day"}
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": SYMBOL, "interval": tf_map.get(TF, TF),
              "outputsize": OUTPUT_SIZE, "apikey": TWELVE_DATA_KEY,
              "format": "JSON", "timezone": "UTC"}
    r = http.req("GET", url, params=params)
    d = r.json()
    if d.get("status") == "error":
        raise RuntimeError(f"TD: {d.get('message')}")
    vals = d.get("values")
    if not vals: raise RuntimeError("TD empty")
    df = pd.DataFrame(vals)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df.index <= pd.Timestamp.now(tz="UTC")]
    df = df.reset_index()
    logger.info("Fetched %d bars | last closed: %s",
                len(df), df["datetime"].iloc[-1])
    return df

# =============================================================================
# INDICATORS
# =============================================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Bollinger
    df["sma_bb"] = df["close"].rolling(BB_PERIOD).mean()
    df["bb_std"] = df["close"].rolling(BB_PERIOD).std()
    df["upper_bb"] = df["sma_bb"] + BB_STD * df["bb_std"]
    df["lower_bb"] = df["sma_bb"] - BB_STD * df["bb_std"]

    # IBS
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["ibs"] = (df["close"] - df["low"]) / rng

    # Trend
    df["sma_trend"] = df["close"].rolling(SMA_TREND).mean()

    # ATR
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()

    # EMA untuk strategi trend
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    return df

# =============================================================================
# SIGNAL GENERATION
# =============================================================================
def signal_mean_rev(row: pd.Series) -> int:
    """Mean-reversion: beli oversold di uptrend, jual overbought di downtrend."""
    if pd.isna(row["lower_bb"]) or pd.isna(row["sma_trend"]) or pd.isna(row["ibs"]):
        return 0
    if (row["close"] < row["lower_bb"] and row["ibs"] < IBS_LOW
            and row["close"] > row["sma_trend"]):
        return 1
    if (row["close"] > row["upper_bb"] and row["ibs"] > IBS_HIGH
            and row["close"] < row["sma_trend"]):
        return -1
    return 0


def signal_trend(row: pd.Series) -> int:
    """Trend-following: EMA alignment searah SMA200."""
    if pd.isna(row["sma_trend"]) or pd.isna(row["ema_50"]):
        return 0
    if (row["close"] > row["sma_trend"]
            and row["ema_9"] > row["ema_21"] > row["ema_50"]):
        return 1
    if (row["close"] < row["sma_trend"]
            and row["ema_9"] < row["ema_21"] < row["ema_50"]):
        return -1
    return 0


def signal_breakout(row: pd.Series) -> int:
    """Breakout dari Donchian 20."""
    if pd.isna(row.get("don_high")) or pd.isna(row.get("don_low")):
        return 0
    if row["close"] > row["don_high"]:
        return 1
    if row["close"] < row["don_low"]:
        return -1
    return 0


def get_signal(row: pd.Series) -> int:
    if STRATEGY == "mean_rev": return signal_mean_rev(row)
    if STRATEGY == "trend":    return signal_trend(row)
    if STRATEGY == "breakout":
        # tambah Donchian kalau perlu
        return signal_breakout(row)
    return 0

# =============================================================================
# SL/TP
# =============================================================================
def compute_sltp(row: pd.Series, direction: int) -> Dict[str, float]:
    entry = float(row["close"])
    atr = float(row["atr"])

    if STRATEGY == "mean_rev":
        # TP = middle Bollinger (reversion target)
        tp = float(row["sma_bb"])
        sl_dist = ATR_SL_MULT * atr
        sl = entry - sl_dist if direction == 1 else entry + sl_dist
    elif STRATEGY == "trend":
        sl_dist = 1.5 * atr
        tp_dist = 3.0 * atr
        sl = entry - sl_dist if direction == 1 else entry + sl_dist
        tp = entry + tp_dist if direction == 1 else entry - tp_dist
    else:  # breakout
        sl_dist = 1.5 * atr
        tp_dist = 2.5 * atr
        sl = entry - sl_dist if direction == 1 else entry + sl_dist
        tp = entry + tp_dist if direction == 1 else entry - tp_dist

    # Spread + slippage buffer
    buf = SPREAD_BUFFER + SLIPPAGE_BUFFER
    if direction == 1:
        sl = round(sl - buf, 2)
        tp = round(tp + buf, 2)
    else:
        sl = round(sl + buf, 2)
        tp = round(tp - buf, 2)

    return {"entry": round(entry, 2), "sl": sl, "tp": tp,
            "sl_distance": round(abs(entry - sl), 2)}

# =============================================================================
# POSITION SIZE
# =============================================================================
def position_size(entry: float, sl: float, balance: float) -> float:
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
def send_telegram(msg: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        logger.warning("Telegram not configured")
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
            try: ra = int(r.json().get("parameters", {}).get("retry_after", 5))
            except Exception: ra = 5
            time.sleep(ra + 1); continue
        time.sleep(2**a)
    return False

# =============================================================================
# STATE (Gist)
# =============================================================================
def load_state() -> Dict[str, Any]:
    url = f"https://api.github.com/gists/{GIST_ID}"
    h = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    try:
        r = http.req("GET", url, headers=h)
    except RuntimeError:
        return {"etag": None, "content": {}}
    if r.status_code != 200:
        logger.warning("Gist GET: %d", r.status_code)
        return {"etag": None, "content": {}}
    content = {}
    for name, f in r.json().get("files", {}).items():
        if name.endswith(".json"):
            try: content = json.loads(f.get("content", "{}")); break
            except: continue
    return {"etag": r.headers.get("ETag"), "content": content}


def save_state(state: Dict[str, Any], etag: Optional[str]) -> bool:
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
        except RuntimeError: return False
        return r.status_code in (200, 201)
    return False

# =============================================================================
# CIRCUIT BREAKER
# =============================================================================
def check_circuit_breaker(state: Dict[str, Any]) -> bool:
    bal = state.get("balance", ACCOUNT_BALANCE)
    peak = state.get("peak_equity", bal)
    consec = state.get("consecutive_losses", 0)
    if state.get("halted", False): return True
    peak = max(peak, bal); state["peak_equity"] = peak
    dd = (peak - bal) / max(peak, 1e-9)
    if dd >= MAX_DRAWDOWN_PCT:
        logger.critical("CB: DD %.1f%%", dd * 100); state["halted"] = True; return True
    if consec >= MAX_CONSEC_LOSSES:
        logger.critical("CB: %d consec losses", consec); state["halted"] = True; return True
    return False

# =============================================================================
# TRADE MONITORING
# =============================================================================
def check_open_trade(df: pd.DataFrame, trade: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    direction = trade["direction"]
    entry = trade["entry"]; sl = trade["sl"]; tp = trade["tp"]
    opened = pd.Timestamp(trade["opened_at"])
    after = df[df["datetime"] > opened].sort_values("datetime")
    for _, row in after.iterrows():
        h = float(row["high"]); l = float(row["low"])
        if direction == 1:
            if l <= sl: return {"outcome": "SL", "exit": sl, "time": str(row["datetime"])}
            if h >= tp: return {"outcome": "TP", "exit": tp, "time": str(row["datetime"])}
        else:
            if h >= sl: return {"outcome": "SL", "exit": sl, "time": str(row["datetime"])}
            if l <= tp: return {"outcome": "TP", "exit": tp, "time": str(row["datetime"])}
    return None

# =============================================================================
# LIVE MODE — kirim sinyal
# =============================================================================
def run_live():
    logger.info("═══ LIVE MODE | %s %s | Strategy=%s ═══", SYMBOL, TF, STRATEGY)

    s = load_state()
    state = s["content"] or {}
    state.setdefault("balance", ACCOUNT_BALANCE)
    state.setdefault("peak_equity", ACCOUNT_BALANCE)
    state.setdefault("consecutive_losses", 0)
    state.setdefault("halted", False)
    state.setdefault("active_trade", None)

    if check_circuit_breaker(state):
        send_telegram("🚨 <b>CIRCUIT BREAKER</b>\nBot halted. Manual review.")
        return

    df = fetch_ohlcv()
    df = add_indicators(df)

    # Cek trade aktif
    trade = state.get("active_trade")
    if trade:
        outcome = check_open_trade(df, trade)
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
                f"{emoji} <b>{outcome['outcome']}</b> — {trade['direction_str']} XAUUSD\n"
                f"━━━━━━━━━━━━━━━\n"
                f"Entry: {trade['entry']:.2f}\n"
                f"Exit:  {outcome['exit']:.2f}\n"
                f"PnL:   <b>${pnl:+.2f}</b>\n"
                f"Balance: ${state['balance']:.2f}"
            )
            state["active_trade"] = None
        else:
            logger.info("Trade active: %s @ %.2f", trade["direction_str"], trade["entry"])

    # Cek sinyal baru (hanya bar closed terakhir)
    last_row = df.iloc[-1]  # bar terakhir yang closed
    sig = get_signal(last_row)

    if sig != 0 and state.get("active_trade") is None:
        # Pastikan sinyal BARU: cek bar sebelumnya tidak punya sinyal yang sama
        prev_row = df.iloc[-2] if len(df) > 1 else last_row
        prev_sig = get_signal(prev_row)

        if sig != prev_sig:  # sinyal baru
            direction_str = "BUY" if sig == 1 else "SELL"
            sltp = compute_sltp(last_row, sig)
            lot = position_size(sltp["entry"], sltp["sl"], state["balance"])

            if lot >= MIN_LOT:
                state["active_trade"] = {
                    "direction": sig,
                    "direction_str": direction_str,
                    "entry": sltp["entry"],
                    "sl": sltp["sl"],
                    "tp": sltp["tp"],
                    "lot": lot,
                    "opened_at": str(last_row["datetime"]),
                }

                emoji = "🟢" if sig == 1 else "🔴"
                rr = abs(sltp["tp"] - sltp["entry"]) / abs(sltp["entry"] - sltp["sl"])
                send_telegram(
                    f"{emoji} <b>{direction_str} {SYMBOL}</b> [{TF}]\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"Entry: <code>{sltp['entry']:.2f}</code>\n"
                    f"SL:    <code>{sltp['sl']:.2f}</code>\n"
                    f"TP:    <code>{sltp['tp']:.2f}</code>\n"
                    f"R:R    {rr:.1f}:1\n"
                    f"Lot:   <code>{lot}</code>\n"
                    f"Risk:  ${state['balance'] * RISK_PCT:.2f}\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"Strategy: {STRATEGY}\n"
                    f"Balance: ${state['balance']:.2f}"
                )
                logger.info("✅ SIGNAL SENT: %s @ %.2f lot=%.2f",
                            direction_str, sltp["entry"], lot)
            else:
                logger.warning("Lot too small (%.4f) — skip signal", lot)
        else:
            logger.info("Signal %d same as previous bar — not new", sig)
    elif sig == 0 and NOTIFY_HOLD:
        send_telegram(f"⏸️ HOLD — no setup on {SYMBOL} {TF}")
    else:
        logger.info("No new signal (sig=%d)", sig)

    state["last_update"] = datetime.now(timezone.utc).isoformat()
    save_state(state, s.get("etag"))
    logger.info("═══ DONE ═══")

# =============================================================================
# BACKTEST MODE — verifikasi historis
# =============================================================================
def run_backtest():
    logger.info("═══ BACKTEST MODE | %s %s | Strategy=%s | %d days ═══",
                SYMBOL, TF, STRATEGY, BACKTEST_DAYS)

    df = fetch_ohlcv()
    df = add_indicators(df)

    # Batasi ke N hari terakhir
    cutoff = df["datetime"].max() - pd.Timedelta(days=BACKTEST_DAYS)
    df = df[df["datetime"] >= cutoff].reset_index(drop=True)
    logger.info("Testing on %d bars (%s → %s)", len(df),
                df["datetime"].iloc[0].date(), df["datetime"].iloc[-1].date())

    trades = []
    balance = ACCOUNT_BALANCE
    i = SMA_TREND + BB_PERIOD
    while i < len(df) - 1:
        row = df.iloc[i]
        sig = get_signal(row)
        if sig == 0 or pd.isna(row["atr"]):
            i += 1; continue
        # Idempotent: hanya entry jika bar sebelumnya tidak sama sinyal
        prev_sig = get_signal(df.iloc[i-1]) if i > 0 else 0
        if sig == prev_sig:
            i += 1; continue

        sltp = compute_sltp(row, sig)
        entry = sltp["entry"]; sl = sltp["sl"]; tp = sltp["tp"]
        lot = position_size(entry, sl, balance)
        if lot < MIN_LOT:
            i += 1; continue

        # Simulasi forward
        exit_price = exit_reason = None
        for j in range(i+1, min(i+MAX_HOLDING_BARS+1, len(df))):
            b = df.iloc[j]
            h = float(b["high"]); l = float(b["low"])
            if sig == 1:
                if l <= sl: exit_price, exit_reason = sl, "SL"; break
                if h >= tp: exit_price, exit_reason = tp, "TP"; break
            else:
                if h >= sl: exit_price, exit_reason = sl, "SL"; break
                if l <= tp: exit_price, exit_reason = tp, "TP"; break
        if exit_price is None:
            b = df.iloc[min(i+MAX_HOLDING_BARS, len(df)-1)]
            exit_price = float(b["close"]); exit_reason = "TIME"

        pnl_pips = (exit_price - entry) if sig == 1 else (entry - exit_price)
        pnl = pnl_pips * lot * CONTRACT_SIZE
        balance += pnl
        trades.append({"date": row["datetime"].date(), "dir": "BUY" if sig == 1 else "SELL",
                       "entry": entry, "exit": exit_price, "reason": exit_reason,
                       "pnl": pnl, "lot": lot})
        i = j + 1

    # Ringkasan
    if not trades:
        logger.warning("No trades in %d days", BACKTEST_DAYS)
        send_telegram(f"📊 Backtest {BACKTEST_DAYS} hari: 0 trade")
        return

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = gw / gl if gl > 0 else float('inf')
    wr = 100 * len(wins) / len(trades)
    tp_ct = sum(1 for t in trades if t["reason"] == "TP")
    sl_ct = sum(1 for t in trades if t["reason"] == "SL")
    to_ct = sum(1 for t in trades if t["reason"] == "TIME")

    logger.info("")
    logger.info("═══ BACKTEST SUMMARY ═══")
    logger.info("Trades: %d | Win: %d | Loss: %d", len(trades), len(wins), len(losses))
    logger.info("TP=%d | SL=%d | TIME=%d", tp_ct, sl_ct, to_ct)
    logger.info("Win Rate: %.1f%%", wr)
    logger.info("Total PnL: $%+.2f | Balance: $%.2f → $%.2f",
                sum(t["pnl"] for t in trades), ACCOUNT_BALANCE, balance)
    logger.info("Profit Factor: %.2f", pf)

    send_telegram(
        f"📊 <b>BACKTEST {STRATEGY} — {BACKTEST_DAYS} hari</b>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"Trades: {len(trades)} (TP={tp_ct} SL={sl_ct} TIME={to_ct})\n"
        f"Win rate: {wr:.1f}%\n"
        f"PnL: <b>${sum(t['pnl'] for t in trades):+.2f}</b>\n"
        f"Balance: ${ACCOUNT_BALANCE:.2f} → ${balance:.2f}\n"
        f"<b>Profit Factor: {pf:.2f}</b>"
    )

# =============================================================================
# MAIN
# =============================================================================
def main():
    logger.info("═" * 60)
    logger.info("XAUUSD BOT v2.0 | LIVE=%s | BACKTEST=%s | Strategy=%s",
                LIVE_MODE, BACKTEST_MODE, STRATEGY)
    logger.info("═" * 60)

    if not TWELVE_DATA_KEY:
        logger.error("TWELVEDATA_API_KEY missing"); sys.exit(1)

    if BACKTEST_MODE:
        run_backtest()
    elif LIVE_MODE:
        run_live()
    else:
        logger.error("Both LIVE_MODE and BACKTEST_MODE are off — nothing to do")
        sys.exit(1)

    logger.info("═" * 60)
    logger.info("COMPLETE")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
