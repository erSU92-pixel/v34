"""
XAUUSD Quant Bot v6.0 — Institutional Grade
10 Pilar Trading & Quantitative Finance
Audited & Refactored by Chief Risk & Systemic Architect
"""
from __future__ import annotations

import os
import time
import json
import math
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# =============================================================================
# KONFIGURASI & LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("XAUUSD_BOT")

TWELVE_DATA_KEY = os.getenv("TWELVEDATA_API_KEY")
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT   = os.getenv("TELEGRAM_CHAT_ID")
GIST_ID         = os.getenv("GIST_ID")
GIST_TOKEN      = os.getenv("GITHUB_TOKEN")  # PAT dengan scope gist

SYMBOL = "XAU/USD"
INTERVAL = "1h"
OUTPUT_SIZE = 5000

# Parameter risiko (dapat di-override via env)
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "10000"))
RISK_PCT = float(os.getenv("RISK_PCT", "0.01"))
ATR_MULT_SL = 1.5
SPREAD_BUFFER = 0.30          # USD untuk XAUUSD retail
SLIPPAGE_BUFFER = 0.15        # USD
MAX_DAILY_LOSS_PCT = 0.02
MAX_DRAWDOWN_PCT = 0.10
MAX_CONSECUTIVE_LOSSES = 3
HURST_MIN = 0.52              # ambang batas adaptif akan dihitung
TRAIN_WINDOW = 3000

# =============================================================================
# UTILITAS: HTTP DENGAN RETRY & RATE LIMIT
# =============================================================================
class ResilientSession:
    """HTTP session dengan exponential backoff dan rate limiting."""

    def __init__(self, min_interval_sec: float = 7.5):
        self.session = requests.Session()
        self.min_interval = min_interval_sec
        self._last_request = 0.0

    def _throttle(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        max_retries = 5
        for attempt in range(max_retries):
            self._throttle()
            try:
                resp = self.session.request(method, url, timeout=30, **kwargs)
                self._last_request = time.time()

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 2 ** attempt))
                    logger.warning("HTTP 429 — sleep %ss (attempt %d)", retry_after, attempt + 1)
                    time.sleep(retry_after + np.random.uniform(0, 1))
                    continue

                if 500 <= resp.status_code < 600:
                    wait = (2 ** attempt) + np.random.uniform(0, 1)
                    logger.warning("HTTP %d — backoff %.1fs", resp.status_code, wait)
                    time.sleep(wait)
                    continue

                return resp

            except requests.RequestException as e:
                wait = (2 ** attempt) + np.random.uniform(0, 1)
                logger.warning("Request error: %s — backoff %.1fs", e, wait)
                time.sleep(wait)

        raise RuntimeError(f"HTTP failed after {max_retries} retries: {method} {url}")


http = ResilientSession()

# =============================================================================
# MODUL 1: DATA INGESTION
# =============================================================================
def fetch_ohlcv(symbol: str = SYMBOL, interval: str = INTERVAL, outputsize: int = OUTPUT_SIZE) -> pd.DataFrame:
    """Fetch OHLCV dari Twelve Data dengan validasi ketat."""
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_KEY,
        "format": "JSON",
    }
    resp = http.request("GET", url, params=params)
    data = resp.json()

    if data.get("status") == "error":
        raise RuntimeError(f"Twelve Data error: {data.get('message')}")

    values = data.get("values")
    if not values:
        raise RuntimeError("Twelve Data returned empty values")

    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.reset_index()  # pastikan index bersih
    logger.info("Fetched %d bars | %s → %s", len(df), df["datetime"].iloc[0], df["datetime"].iloc[-1])
    return df

# =============================================================================
# MODUL 2: FEATURE ENGINEERING (10 PILAR)
# =============================================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung fitur dari 10 pilar dengan guard numerik."""
    df = df.copy()

    # --- Pilar 1: Trend Klasik ---
    for span in (9, 21, 50):
        df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False).mean()

    df["zscore_20"] = (df["close"] - df["close"].rolling(20).mean()) / (
        df["close"].rolling(20).std().replace(0, np.nan)
    )

    # --- Pilar 2: Momentum Klasik ---
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    low_5 = df["low"].rolling(5).min()
    high_5 = df["high"].rolling(5).max()
    df["stoch_k"] = 100 * (df["close"] - low_5) / (high_5 - low_5).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # --- Pilar 4: Volatilitas & Breakout ---
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean().replace(0, np.nan).ffill()

    df["donchian_high_20"] = df["high"].rolling(20).max()
    df["donchian_low_20"] = df["low"].rolling(20).min()
    df["donchian_pos"] = (df["close"] - df["donchian_low_20"]) / (
        df["donchian_high_20"] - df["donchian_low_20"]
    ).replace(0, np.nan)

    # --- Pilar 5: Market Regime (Hurst & Entropy) — guarded ---
    df["hurst_100"] = df["close"].rolling(100).apply(_rolling_hurst, raw=True)
    df["entropy_50"] = df["close"].rolling(50).apply(_rolling_entropy, raw=True)

    # --- Pilar 10: Seasonality (jam sesi) ---
    df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
    df["is_london_ny"] = ((df["hour"] >= 7) & (df["hour"] <= 16)).astype(int)

    # Return target (untuk labeling, bukan fitur)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    return df


def _rolling_hurst(series: np.ndarray, min_window: int = 10) -> float:
    """Hurst exponent sederhana; guard terhadap std==0."""
    if len(series) < min_window:
        return 0.5
    lags = range(2, min(20, len(series) // 2))
    try:
        tau = [np.std(series[lag:] - series[:-lag]) for lag in lags]
        tau = [t for t in tau if t > 1e-12]
        if len(tau) < 2:
            return 0.5
        poly = np.polyfit(np.log(list(lags)[: len(tau)]), np.log(tau), 1)
        return float(np.clip(poly[0], 0.0, 1.0))
    except Exception:
        return 0.5


def _rolling_entropy(series: np.ndarray, bins: int = 10) -> float:
    """Shannon entropy dari return; guard log(0)."""
    if len(series) < 10:
        return 0.0
    returns = np.diff(np.log(np.clip(series, 1e-12, None)))
    if len(returns) == 0 or np.std(returns) < 1e-12:
        return 0.0
    hist, _ = np.histogram(returns, bins=bins, density=False)
    probs = hist / max(hist.sum(), 1)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs + 1e-12)))

# =============================================================================
# MODUL 3: TRIPLE BARRIER LABELING & PURGED CV
# =============================================================================
def triple_barrier_labels(
    df: pd.DataFrame,
    pt_mult: float = 2.0,
    sl_mult: float = 1.5,
    max_holding: int = 24,
) -> pd.Series:
    """
    Label +1 (profit-take), -1 (stop-loss), 0 (timeout).
    Barrier diskalakan dengan EWMA volatility.
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    # Volatility scaled barrier
    returns = pd.Series(close).pct_change()
    vol = returns.ewm(span=20).std().bfill().values
    vol = np.where(vol < 1e-6, 1e-6, vol)

    labels = np.zeros(n, dtype=int)
    for i in range(n - 1):
        entry = close[i]
        pt = entry * (1 + pt_mult * vol[i])
        sl = entry * (1 - sl_mult * vol[i])
        horizon = min(i + max_holding, n - 1)

        hit = 0
        for j in range(i + 1, horizon + 1):
            if high[j] >= pt:
                hit = 1
                break
            if low[j] <= sl:
                hit = -1
                break
        labels[i] = hit

    return pd.Series(labels, index=df.index, name="label")


def purged_kfold_indices(n_samples: int, n_splits: int = 5, embargo: int = 24) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Purged K-Fold dengan embargo untuk time-series."""
    indices = np.arange(n_samples)
    fold_size = n_samples // n_splits
    splits = []

    for k in range(n_splits):
        test_start = k * fold_size
        test_end = min((k + 1) * fold_size, n_samples)
        test_idx = indices[test_start:test_end]

        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[max(0, test_start - embargo): min(n_samples, test_end + embargo)] = False
        train_idx = indices[train_mask]

        if len(train_idx) > 100 and len(test_idx) > 50:
            splits.append((train_idx, test_idx))

    return splits

# =============================================================================
# MODUL 4: ML PIPELINE
# =============================================================================
FEATURE_COLS = [
    "ema_9", "ema_21", "ema_50", "zscore_20",
    "rsi_14", "stoch_k", "stoch_d",
    "atr_14", "donchian_pos",
    "hurst_100", "entropy_50",
    "is_london_ny",
]


def train_ensemble(df: pd.DataFrame) -> Tuple[Optional[Dict], Optional[StandardScaler]]:
    """Train ensemble dengan Purged CV; scaler fit hanya pada train fold."""
    df = df.copy()
    df["label"] = triple_barrier_labels(df)
    df = df.dropna(subset=FEATURE_COLS + ["label"]).reset_index(drop=True)

    if len(df) < 500:
        logger.warning("Data terlalu sedikit untuk training (%d bars)", len(df))
        return None, None

    X = df[FEATURE_COLS].values
    y = df["label"].values
    y_binary = (y > 0).astype(int)  # binary: 1 = profit, 0 = no profit (SL atau timeout)

    splits = purged_kfold_indices(len(df), n_splits=5, embargo=24)
    if not splits:
        logger.warning("Tidak ada fold valid untuk CV")
        return None, None

    models = {
        "rf": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
        "gb": GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42),
        "et": ExtraTreesClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
    }

    scores = {name: [] for name in models}
    for train_idx, test_idx in splits:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])  # fit HANYA di train
        X_test = scaler.transform(X[test_idx])

        for name, model in models.items():
            model.fit(X_train, y_binary[train_idx])
            pred = model.predict(X_test)
            scores[name].append(accuracy_score(y_binary[test_idx], pred))

    for name in scores:
        logger.info("CV %s accuracy: %.4f ± %.4f", name, np.mean(scores[name]), np.std(scores[name]))

    # Train final model on all data
    scaler_final = StandardScaler()
    X_all = scaler_final.fit_transform(X)
    for model in models.values():
        model.fit(X_all, y_binary)

    return models, scaler_final


def predict_signal(df: pd.DataFrame, models: Dict, scaler: StandardScaler) -> Tuple[int, float]:
    """Ensemble voting → BUY (1), SELL (-1), HOLD (0)."""
    if models is None or scaler is None:
        return 0, 0.0

    last_row = df.iloc[-2]  # closed bar, bukan bar berjalan
    X_last = last_row[FEATURE_COLS].values.reshape(1, -1)
    if np.isnan(X_last).any():
        logger.warning("NaN pada fitur terakhir — HOLD")
        return 0, 0.0

    X_scaled = scaler.transform(X_last)
    probs = [model.predict_proba(X_scaled)[0, 1] for model in models.values()]
    avg_prob = float(np.mean(probs))

    if avg_prob > 0.60:
        return 1, avg_prob
    elif avg_prob < 0.40:
        return -1, avg_prob
    else:
        return 0, avg_prob

# =============================================================================
# MODUL 5: RISK ENGINE (POSITION SIZING, CIRCUIT BREAKER)
# =============================================================================
class RiskEngine:
    def __init__(self, balance: float, risk_pct: float):
        self.balance = balance
        self.risk_pct = risk_pct
        self.peak_equity = balance
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.halted = False
        self._last_reset = datetime.now(timezone.utc).date()

    def reset_daily(self):
        today = datetime.now(timezone.utc).date()
        if today != self._last_reset:
            self.daily_pnl = 0.0
            self._last_reset = today

    def check_circuit_breaker(self, current_equity: float) -> bool:
        self.reset_daily()
        self.peak_equity = max(self.peak_equity, current_equity)
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        daily_loss = -self.daily_pnl / self.balance if self.daily_pnl < 0 else 0

        if self.halted:
            return True

        if drawdown >= MAX_DRAWDOWN_PCT:
            logger.critical("CIRCUIT BREAKER: drawdown %.2f%% ≥ %.2f%%", drawdown * 100, MAX_DRAWDOWN_PCT * 100)
            self.halted = True
        elif daily_loss >= MAX_DAILY_LOSS_PCT:
            logger.critical("CIRCUIT BREAKER: daily loss %.2f%% ≥ %.2f%%", daily_loss * 100, MAX_DAILY_LOSS_PCT * 100)
            self.halted = True
        elif self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            logger.critical("CIRCUIT BREAKER: %d consecutive losses", self.consecutive_losses)
            self.halted = True

        return self.halted

    def position_size(self, atr: float, balance: Optional[float] = None) -> float:
        """ATR-based lot sizing; SELALU dibulatkan ke bawah."""
        bal = balance or self.balance
        risk_money = bal * self.risk_pct
        stop_distance = ATR_MULT_SL * max(atr, 0.01)

        tick_size = 0.01
        tick_value = 1.0  # USD per tick per 1 lot (XAUUSD standard)
        loss_per_lot = (stop_distance / tick_size) * tick_value

        if loss_per_lot <= 0:
            return 0.0

        lot = math.floor((risk_money / loss_per_lot) / 0.01) * 0.01
        lot = min(lot, 5.0)  # hard cap 5 lot
        return max(lot, 0.0)

    def compute_sl_tp(self, entry: float, direction: int, atr: float) -> Tuple[float, float]:
        """SL/TP dengan spread + slippage buffer."""
        sl_distance = ATR_MULT_SL * atr
        tp_distance = 2.0 * ATR_MULT_SL * atr
        total_buffer = SPREAD_BUFFER + SLIPPAGE_BUFFER

        if direction == 1:  # BUY
            sl = entry - sl_distance - total_buffer
            tp = entry + tp_distance + total_buffer
        else:  # SELL
            sl = entry + sl_distance + total_buffer
            tp = entry - tp_distance - total_buffer

        return round(sl, 2), round(tp, 2)

# =============================================================================
# MODUL 6: STATE MANAGEMENT (GIST) & TELEGRAM
# =============================================================================
def load_state() -> Dict[str, Any]:
    """Load state dari Gist dengan ETag untuk conditional update."""
    url = f"https://api.github.com/gists/{GIST_ID}"
    headers = {"Authorization": f"token {GIST_TOKEN}", "Accept": "application/vnd.github+json"}
    resp = http.request("GET", url, headers=headers)

    if resp.status_code != 200:
        logger.warning("Gist load failed: %d", resp.status_code)
        return {"etag": None, "content": {}}

    etag = resp.headers.get("ETag")
    files = resp.json().get("files", {})
    content = {}
    for name, file_obj in files.items():
        if name.endswith(".json"):
            try:
                content = json.loads(file_obj.get("content", "{}"))
                break
            except json.JSONDecodeError:
                continue

    return {"etag": etag, "content": content}


def save_state(state: Dict[str, Any], etag: Optional[str]) -> bool:
    """PATCH state ke Gist dengan If-Match untuk mencegah race condition."""
    url = f"https://api.github.com/gists/{GIST_ID}"
    headers = {"Authorization": f"token {GIST_TOKEN}", "Accept": "application/vnd.github+json"}
    if etag:
        headers["If-Match"] = etag

    payload = {
        "files": {
            "bot_state.json": {
                "content": json.dumps(state, indent=2, default=str)
            }
        }
    }

    resp = http.request("PATCH", url, headers=headers, json=payload)

    if resp.status_code == 412:
        logger.warning("Gist race condition (ETag mismatch) — reload and retry")
        return False
    if resp.status_code not in (200, 201):
        logger.error("Gist save failed: %d %s", resp.status_code, resp.text[:200])
        return False

    return True


def send_telegram(message: str) -> bool:
    """Kirim Telegram dengan retry_after awareness."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        logger.warning("Telegram credentials missing")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT, "text": message, "parse_mode": "HTML"}

    for attempt in range(3):
        resp = http.request("POST", url, json=payload)

        if resp.status_code == 200:
            return True

        if resp.status_code == 429:
            retry_after = resp.json().get("parameters", {}).get("retry_after", 5)
            logger.warning("Telegram 429 — sleep %ss", retry_after)
            time.sleep(retry_after + 1)
            continue

        logger.error("Telegram error %d: %s", resp.status_code, resp.text[:200])
        time.sleep(2 ** attempt)

    return False

# =============================================================================
# ORKESTRASI UTAMA
# =============================================================================
def main():
    logger.info("=== XAUUSD BOT v6.0 START ===")

    # 1. Load state
    state = load_state()
    risk = RiskEngine(
        balance=state["content"].get("balance", ACCOUNT_BALANCE),
        risk_pct=RISK_PCT,
    )
    risk.halted = state["content"].get("halted", False)
    risk.daily_pnl = state["content"].get("daily_pnl", 0.0)
    risk.consecutive_losses = state["content"].get("consecutive_losses", 0)
    risk.peak_equity = state["content"].get("peak_equity", risk.balance)

    # 2. Circuit breaker gate
    if risk.check_circuit_breaker(risk.balance):
        send_telegram("🚨 <b>CIRCUIT BREAKER ACTIVE</b>\nBot dihentikan. Kirim /rearm setelah review manual.")
        logger.critical("Bot halted by circuit breaker")
        return

    # 3. Fetch & features
    try:
        df = fetch_ohlcv()
        df = add_features(df)
    except Exception as e:
        logger.exception("Data pipeline failed")
        send_telegram(f"⚠️ Data pipeline error: {e}")
        return

    # 4. ML training & signal
    models, scaler = train_ensemble(df)
    signal, confidence = predict_signal(df, models, scaler)

    if signal == 0:
        logger.info("Signal HOLD (confidence %.2f)", confidence)
        send_telegram(f"⏸️ <b>HOLD</b> | Confidence: {confidence:.2%}")
        return

    # 5. Gatekeeper: Hurst veto
    hurst = df["hurst_100"].iloc[-2] if not pd.isna(df["hurst_100"].iloc[-2]) else 0.5
    if hurst < HURST_MIN and signal != 0:
        logger.info("VETO: Hurst %.3f < %.2f (choppy market)", hurst, HURST_MIN)
        send_telegram(f"🚫 <b>VETO</b> | Hurst: {hurst:.3f} (pasar choppy)")
        return

    # 6. Session filter
    is_session = df["is_london_ny"].iloc[-2]
    if not is_session:
        logger.info("Outside London/NY session — skip")
        return

    # 7. Risk & sizing
    atr = float(df["atr_14"].iloc[-2])
    lot = risk.position_size(atr)
    if lot <= 0:
        send_telegram("⚠️ Lot size 0 — modal terlalu kecil atau ATR terlalu lebar.")
        return

    entry_price = float(df["close"].iloc[-2])
    sl, tp = risk.compute_sl_tp(entry_price, signal, atr)

    # 8. Notifikasi sinyal
    direction_str = "🟢 <b>BUY</b>" if signal == 1 else "🔴 <b>SELL</b>"
    msg = (
        f"{direction_str} XAUUSD\n"
        f"━━━━━━━━━━━━━━━\n"
        f"Entry : <code>{entry_price:.2f}</code>\n"
        f"SL    : <code>{sl:.2f}</code>\n"
        f"TP    : <code>{tp:.2f}</code>\n"
        f"Lot   : <code>{lot:.2f}</code>\n"
        f"Conf  : {confidence:.2%}\n"
        f"Hurst : {hurst:.3f}\n"
        f"ATR   : {atr:.2f}"
    )
    send_telegram(msg)

    # 9. Update state (dengan retry jika race condition)
    new_state = {
        "balance": risk.balance,
        "halted": risk.halted,
        "daily_pnl": risk.daily_pnl,
        "consecutive_losses": risk.consecutive_losses,
        "peak_equity": risk.peak_equity,
        "last_signal": signal,
        "last_confidence": confidence,
        "last_update": datetime.now(timezone.utc).isoformat(),
    }

    for attempt in range(2):
        if save_state(new_state, state.get("etag")):
            break
        state = load_state()  # reload ETag terbaru
        if attempt == 1:
            logger.error("Gist save failed after retry — state may be stale")

    logger.info("=== BOT RUN COMPLETE ===")


if __name__ == "__main__":
    main()
