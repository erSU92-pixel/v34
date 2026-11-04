#!/usr/bin/env python3
"""
APEX PREDATOR v19.1 — Hotfix: TwelveData missing volume field
=============================================================
Patch: handle missing 'volume' key from TwelveData XAU/USD response
"""
from __future__ import annotations
import os
import sys
import json
import time
import math
import logging
import hashlib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo
from functools import wraps

# ════════════════════════════════════════════════════════════
# CONFIGURATION & ENVIRONMENT
# ════════════════════════════════════════════════════════════
@dataclass
class Config:
    symbol: str = "XAUUSD"
    contract_size: float = 100.0
    account_currency: str = "USD"
    price_decimals: int = 2
    lot_step: float = 0.01
    min_lot: float = 0.01
    max_lot: float = 5.0

    account_balance: float = 1000.0
    max_risk_pct_per_trade: float = 1.0
    max_leverage: float = 30.0
    max_daily_loss_pct: float = 3.0
    max_consecutive_losses: int = 3
    max_daily_signals: int = 5
    dd_pause_hours: int = 24

    atr_period: int = 14
    atr_sl_mult: float = 1.8
    atr_tp1_mult: float = 1.0
    atr_tp2_mult: float = 2.4
    atr_tp3_mult: float = 4.0
    spread_buffer: float = 0.35
    adx_min: float = 22.0
    parliament_min_active: int = 4
    parliament_min_ratio: float = 0.55
    parliament_half_ratio: float = 0.65
    parliament_full_ratio: float = 0.75

    min_bars_m5: int = 100
    min_bars_m15: int = 100
    min_bars_h1: int = 100
    min_bars_h4: int = 50
    max_stale_mult: float = 3.0
    http_timeout: Tuple[int, int] = (5, 15)
    max_retry: int = 3

    news_blackout_min: int = 30
    cooldown_minutes: int = 45
    opposite_cooldown_minutes: int = 120
    skip_killzone_filter: bool = False
    enable_ml_models: bool = False
    dry_run: bool = True
    kill_switch: bool = False

    gist_id: str = ""
    gist_token: str = ""
    state_file: str = "apex_state_v19.json"

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        for name in cls.__dataclass_fields__:
            env_key = f"APEX_{name.upper()}"
            raw = os.getenv(env_key)
            if raw is None or not str(raw).strip():
                continue
            cur = getattr(cfg, name)
            try:
                if isinstance(cur, bool):
                    setattr(cfg, name, str(raw).strip().lower() in ("1", "true", "yes", "on"))
                elif isinstance(cur, int):
                    setattr(cfg, name, int(raw))
                elif isinstance(cur, float):
                    setattr(cfg, name, float(raw))
                elif isinstance(cur, str):
                    setattr(cfg, name, str(raw).strip())
                elif isinstance(cur, tuple):
                    setattr(cfg, name, tuple(map(int, raw.split(','))))
            except (ValueError, TypeError):
                logging.warning(f"Config: ignore invalid env {env_key}")

        cfg.gist_id = os.getenv("GIST_ID", cfg.gist_id).strip()
        cfg.gist_token = os.getenv("GIST_TOKEN", cfg.gist_token).strip()
        if os.getenv("APEX_KILL_SWITCH", "").lower() in ("1", "true", "yes"):
            cfg.kill_switch = True
        return cfg

# ════════════════════════════════════════════════════════════
# UTILITIES & LOGGING
# ════════════════════════════════════════════════════════════
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def retry_on_failure(max_retries: int = 3, backoff_factor: float = 1.5):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    wait = backoff_factor * (2 ** attempt)
                    logging.warning(f"Retry {attempt+1}/{max_retries} for {func.__name__}: {e}. Waiting {wait:.1f}s")
                    time.sleep(wait)
            raise last_exc
        return wrapper
    return decorator

class DataFetchError(Exception):
    pass

class StateError(Exception):
    pass

# ════════════════════════════════════════════════════════════
# STATE STORE
# ════════════════════════════════════════════════════════════
class StateStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.use_gist = bool(cfg.gist_id and cfg.gist_token)
        self._cache: Optional[dict] = None

    def _load(self) -> dict:
        if self._cache is not None:
            return self._cache
        if self.use_gist:
            try:
                headers = {"Authorization": f"Bearer {self.cfg.gist_token}"}
                r = requests.get(
                    f"https://api.github.com/gists/{self.cfg.gist_id}",
                    headers=headers, timeout=self.cfg.http_timeout
                )
                r.raise_for_status()
                content = r.json().get("files", {}).get(self.cfg.state_file, {}).get("content", "{}")
                self._cache = json.loads(content)
            except Exception as e:
                logging.error(f"Gist load failed: {e}")
                self._cache = self._load_local()
        else:
            self._cache = self._load_local()
        return self._cache

    def _load_local(self) -> dict:
        p = Path(self.cfg.state_file)
        return json.loads(p.read_text()) if p.exists() else {}

    def _save(self, data: dict) -> bool:
        if self.use_gist:
            try:
                headers = {"Authorization": f"Bearer {self.cfg.gist_token}"}
                body = {"files": {self.cfg.state_file: {"content": json.dumps(data, default=str)}}}
                requests.patch(
                    f"https://api.github.com/gists/{self.cfg.gist_id}",
                    headers=headers, json=body, timeout=self.cfg.http_timeout
                )
                self._cache = data
                return True
            except Exception as e:
                logging.error(f"Gist save failed: {e}")
        return self._save_local(data)

    def _save_local(self, data: dict) -> bool:
        try:
            Path(self.cfg.state_file).write_text(json.dumps(data, default=str, indent=2))
            self._cache = data
            return True
        except Exception as e:
            logging.error(f"Local save failed: {e}")
            return False

    def get(self, key: str, default=None):
        return self._load().get(key, default)

    def set(self, key: str, value) -> bool:
        data = dict(self._load())
        data[key] = value
        return self._save(data)

# ════════════════════════════════════════════════════════════
# MARKET DATA HANDLER — FIXED FOR MISSING VOLUME
# ════════════════════════════════════════════════════════════
class MarketDataHandler:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @retry_on_failure(max_retries=3)
    def _fetch_twelvedata(self, interval: str, outputsize: int) -> pd.DataFrame:
        key = os.getenv("TWELVE_DATA_API_KEY")
        if not key:
            raise DataFetchError("No TwelveData API Key")
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": "XAU/USD", "interval": interval,
            "outputsize": outputsize, "apikey": key, "timezone": "UTC"
        }
        r = requests.get(url, params=params, timeout=self.cfg.http_timeout)
        r.raise_for_status()
        data = r.json()
        if "values" not in data or not data["values"]:
            raise DataFetchError(data.get("message", "Invalid TD response"))

        df = pd.DataFrame(data["values"])

        # === HOTFIX: safe parse — handle missing 'volume' ===
        for c in ["open", "high", "low", "close"]:
            if c not in df.columns:
                raise DataFetchError(f"Missing required column: {c}")
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(100.0)
        else:
            # Forex/CFD often has no volume — use synthetic default
            df["volume"] = 100.0

        df = df.dropna(subset=["open", "high", "low", "close"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df.sort_values("datetime").reset_index(drop=True)

    @staticmethod
    def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Safe resample that handles missing/zero volume."""
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
        if "volume" in df.columns:
            agg["volume"] = "sum"
        out = (
            df.set_index("datetime")
            .resample(rule, label="left", closed="left")
            .agg(agg)
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
        )
        if "volume" not in out.columns:
            out["volume"] = 100.0
        return out

    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        dfs = {}
        try:
            dfs["5min"] = self._fetch_twelvedata("5min", 800)
            dfs["1h"] = self._fetch_twelvedata("1h", 500)
            dfs["4h"] = self._fetch_twelvedata("4h", 300)
            dfs["15min"] = self._resample(dfs["5min"], "15min")
        except Exception as e:
            raise DataFetchError(f"Data fetch failed: {e}")

        now = datetime.now(timezone.utc)
        for tf, mins in [("5min", 5), ("15min", 15), ("1h", 60), ("4h", 240)]:
            df = dfs[tf]
            last_ts = df["datetime"].iloc[-1]
            if (now - last_ts).total_seconds() < (mins * 60 * 0.9):
                df = df.iloc[:-1]
            min_bars_attr = f"min_bars_{tf.replace('min','m').replace('h','h')}"
            if len(df) < getattr(self.cfg, min_bars_attr):
                raise DataFetchError(f"Insufficient bars for {tf}")
            dfs[tf] = df.reset_index(drop=True)
        return dfs

# ════════════════════════════════════════════════════════════
# SIGNAL GENERATOR
# ════════════════════════════════════════════════════════════
class Indicators:
    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"].shift(1)
        tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period * 2:
            return 0.0
        up, dn = df["high"].diff(), -df["low"].diff()
        pdm = up.where((up > dn) & (up > 0), 0.0)
        mdm = dn.where((dn > up) & (dn > 0), 0.0)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs()
        ], axis=1).max(axis=1)
        atr_s = tr.ewm(alpha=1/period, adjust=False).mean()
        pdi = 100 * pdm.ewm(alpha=1/period, adjust=False).mean() / atr_s
        mdi = 100 * mdm.ewm(alpha=1/period, adjust=False).mean() / atr_s
        dx = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100
        return float(dx.ewm(alpha=1/period, adjust=False).mean().iloc[-1])

class SignalGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _ema_cross(self, df: pd.DataFrame) -> Tuple[int, float]:
        e9, e21 = Indicators.ema(df["close"], 9), Indicators.ema(df["close"], 21)
        diff = e9 - e21
        if diff.iloc[-1] > 0 and diff.diff().iloc[-1] > 0:
            return 1, 0.8
        if diff.iloc[-1] < 0 and diff.diff().iloc[-1] < 0:
            return -1, 0.8
        return 0, 0.0

    def _macd(self, df: pd.DataFrame) -> Tuple[int, float]:
        macd = Indicators.ema(df["close"], 12) - Indicators.ema(df["close"], 26)
        sig = Indicators.ema(macd, 9)
        hist = macd - sig
        if macd.iloc[-1] > sig.iloc[-1] and hist.diff().iloc[-1] > 0:
            return 1, 0.9
        if macd.iloc[-1] < sig.iloc[-1] and hist.diff().iloc[-1] < 0:
            return -1, 0.9
        return 0, 0.0

    def _rsi_signal(self, df: pd.DataFrame) -> Tuple[int, float]:
        rsi = Indicators.rsi(df["close"], 14)
        if rsi.iloc[-1] < 30 and rsi.iloc[-1] > rsi.iloc[-2]:
            return 1, 0.85
        if rsi.iloc[-1] > 70 and rsi.iloc[-1] < rsi.iloc[-2]:
            return -1, 0.85
        return 0, 0.0

    def _supertrend(self, df: pd.DataFrame) -> Tuple[int, float]:
        atr = Indicators.atr(df, 10)
        hl2 = (df["high"] + df["low"]) / 2
        upper, lower = (hl2 + 3.0 * atr).values, (hl2 - 3.0 * atr).values
        close, trend = df["close"].values, np.ones(len(df))
        for i in range(1, len(df)):
            if close[i] > upper[i-1]:
                trend[i] = 1
            elif close[i] < lower[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]
        if trend[-1] == 1 and trend[-2] == 1:
            return 1, 0.9
        if trend[-1] == -1 and trend[-2] == -1:
            return -1, 0.9
        return 0, 0.0

    def generate(self, m5: pd.DataFrame, m15: pd.DataFrame,
                 h1: pd.DataFrame, h4: pd.DataFrame) -> Dict[str, Any]:
        adx_val = Indicators.adx(m5)
        if adx_val < self.cfg.adx_min:
            return {"direction": "NEUTRAL", "skip_reason": f"ADX {adx_val:.1f} < {self.cfg.adx_min}"}

        models = [
            ("EMA_Cross", self._ema_cross, 1.0),
            ("MACD", self._macd, 1.2),
            ("RSI", self._rsi_signal, 1.0),
            ("SuperTrend", self._supertrend, 1.3),
        ]

        buy_w, sell_w, active = 0.0, 0.0, 0
        for name, fn, weight in models:
            vote, conf = fn(m5)
            if vote == 0:
                continue
            active += 1
            score = weight * conf
            if vote > 0:
                buy_w += score
            else:
                sell_w += score

        total = buy_w + sell_w
        if total < 1e-6 or active < self.cfg.parliament_min_active:
            return {"direction": "NEUTRAL", "skip_reason": "Weak consensus"}

        buy_ratio, sell_ratio = buy_w / total, sell_w / total
        dom = max(buy_ratio, sell_ratio)

        if dom < self.cfg.parliament_min_ratio:
            return {"direction": "NEUTRAL", "skip_reason": f"Consensus {dom:.0%} too weak"}

        direction = "BUY" if buy_ratio > sell_ratio else "SELL"

        h4_ema50 = Indicators.ema(h4["close"], 50)
        h4_trend = "BULLISH" if h4["close"].iloc[-1] > h4_ema50.iloc[-1] else "BEARISH"

        if direction == "BUY" and h4_trend == "BEARISH":
            return {"direction": "NEUTRAL", "skip_reason": "BUY blocked by H4 Bearish"}
        if direction == "SELL" and h4_trend == "BULLISH":
            return {"direction": "NEUTRAL", "skip_reason": "SELL blocked by H4 Bullish"}

        if dom >= self.cfg.parliament_full_ratio:
            verdict = "EXECUTE_FULL"
        elif dom >= self.cfg.parliament_half_ratio:
            verdict = "EXECUTE_HALF"
        else:
            verdict = "EXECUTE_MICRO"

        return {
            "direction": direction, "verdict": verdict, "adx": adx_val,
            "buy_ratio": buy_ratio, "sell_ratio": sell_ratio, "active": active
        }

# ════════════════════════════════════════════════════════════
# RISK MANAGEMENT ENGINE
# ════════════════════════════════════════════════════════════
class RiskManagementEngine:
    def __init__(self, cfg: Config, store: StateStore):
        self.cfg = cfg
        self.store = store

    def calculate_position_size(self, balance: float, entry: float,
                                 sl: float, risk_pct: float) -> float:
        sl_dist = abs(entry - sl)
        if sl_dist <= 0:
            raise ValueError("Invalid SL distance")

        risk_amount = balance * (risk_pct / 100.0)
        lot_size = risk_amount / (sl_dist * self.cfg.contract_size)

        notional = lot_size * self.cfg.contract_size * entry
        max_notional = balance * self.cfg.max_leverage
        if notional > max_notional:
            lot_size = max_notional / (self.cfg.contract_size * entry)

        lot_size = math.floor(lot_size / self.cfg.lot_step) * self.cfg.lot_step
        return max(self.cfg.min_lot, min(self.cfg.max_lot, lot_size))

    def check_circuit_breakers(self, balance: float) -> Tuple[bool, str]:
        journal = self.store.get("journal", []) or []
        today = datetime.now(timezone.utc).date().isoformat()

        daily_loss = 0.0
        for j in journal:
            if j.get("timestamp", "").startswith(today) and j.get("outcome") == "LOSS":
                r = abs(float(j.get("r_achieved", -1.0)))
                rp = float(j.get("risk_pct", 1.0))
                daily_loss += r * (rp / 100.0) * balance

        if balance > 0 and (daily_loss / balance * 100) >= self.cfg.max_daily_loss_pct:
            return False, f"MAX_DAILY_LOSS ({daily_loss/balance*100:.1f}%)"

        closed = [j for j in journal if j.get("outcome") in ("WIN_TP3", "WIN_BE", "LOSS")]
        closed.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        consec = 0
        for j in closed:
            if j["outcome"] == "LOSS":
                consec += 1
            else:
                break
        if consec >= self.cfg.max_consecutive_losses:
            return False, f"MAX_CONSEC_LOSSES ({consec})"

        return True, ""

# ════════════════════════════════════════════════════════════
# NEWS SAFETY FILTER
# ════════════════════════════════════════════════════════════
class NewsSafetyFilter:
    def __init__(self, cfg: Config, store: StateStore):
        self.cfg = cfg
        self.store = store

    def is_blackout(self) -> Tuple[bool, str]:
        cache = self.store.get("news_cache", {}) or {}
        now_ts = time.time()
        if cache.get("ts", 0) > now_ts - 1800:
            return cache.get("blackout", False), cache.get("title", "")

        try:
            r = requests.get(
                "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
                timeout=5
            )
            r.raise_for_status()
            events = r.json()
            now = datetime.now(timezone.utc)
            for ev in events:
                if ev.get("country") != "USD" or ev.get("impact") != "High":
                    continue
                try:
                    evt_time = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
                    if abs((evt_time - now).total_seconds()) / 60 <= self.cfg.news_blackout_min:
                        res = (True, ev.get("title", "High Impact"))
                        self.store.set("news_cache", {
                            "ts": now_ts, "blackout": True, "title": res[1]
                        })
                        return res
                except Exception:
                    continue
            self.store.set("news_cache", {"ts": now_ts, "blackout": False, "title": ""})
            return False, ""
        except Exception as e:
            logging.warning(f"News fetch failed: {e}")
            return False, ""

# ════════════════════════════════════════════════════════════
# TELEGRAM NOTIFIER
# ════════════════════════════════════════════════════════════
class TelegramNotifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.token = os.getenv("TELEGRAM_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    @retry_on_failure(max_retries=2)
    def send(self, msg: str) -> bool:
        if not (self.token and self.chat_id):
            return False
        if self.cfg.dry_run:
            msg = "🧪 <b>DRY RUN</b>\n" + msg
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": msg, "parse_mode": "HTML"}
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return r.json().get("ok", False)

# ════════════════════════════════════════════════════════════
# TRADING BOT ORCHESTRATOR
# ════════════════════════════════════════════════════════════
class TradingBotOrchestrator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.store = StateStore(cfg)
        self.data = MarketDataHandler(cfg)
        self.signal = SignalGenerator(cfg)
        self.risk = RiskManagementEngine(cfg, self.store)
        self.news = NewsSafetyFilter(cfg, self.store)
        self.tg = TelegramNotifier(cfg)

    def run(self) -> None:
        if self.cfg.kill_switch:
            logging.info("KILL SWITCH ENGAGED")
            return

        is_blackout, news_title = self.news.is_blackout()
        if is_blackout:
            logging.info(f"News Blackout: {news_title}")
            return

        try:
            dfs = self.data.fetch_all()
        except Exception as e:
            logging.error(f"Data Fetch Error: {e}")
            self.tg.send(f"⚠️ <b>Data Error</b>\n<code>{e}</code>")
            return

        balance = self.cfg.account_balance
        ok, reason = self.risk.check_circuit_breakers(balance)
        if not ok:
            logging.info(f"Circuit Breaker: {reason}")
            return

        result = self.signal.generate(dfs["5min"], dfs["15min"], dfs["1h"], dfs["4h"])
        if result["direction"] == "NEUTRAL":
            logging.info(f"Skip: {result.get('skip_reason', 'No signal')}")
            return

        m5 = dfs["5min"]
        atr_val = Indicators.atr(m5).iloc[-1]
        price = m5["close"].iloc[-1]

        if result["direction"] == "BUY":
            entry = price
            sl = entry - (atr_val * self.cfg.atr_sl_mult)
            tp1 = entry + (atr_val * self.cfg.atr_tp1_mult)
            tp2 = entry + (atr_val * self.cfg.atr_tp2_mult)
            tp3 = entry + (atr_val * self.cfg.atr_tp3_mult)
        else:
            entry = price
            sl = entry + (atr_val * self.cfg.atr_sl_mult)
            tp1 = entry - (atr_val * self.cfg.atr_tp1_mult)
            tp2 = entry - (atr_val * self.cfg.atr_tp2_mult)
            tp3 = entry - (atr_val * self.cfg.atr_tp3_mult)

        conf_pct = max(result["buy_ratio"], result["sell_ratio"]) * 100
        risk_pct = self.cfg.max_risk_pct_per_trade * (conf_pct / 100.0)
        lot = self.risk.calculate_position_size(balance, entry, sl, risk_pct)

        msg = (
            f"<b>🏛️ APEX v19.1 — {result['direction']}</b>\n"
            f"<b>{self.cfg.symbol}</b> @ <code>{entry:.2f}</code>\n"
            f"SL: <code>{sl:.2f}</code> | TP1: <code>{tp1:.2f}</code>\n"
            f"TP2: <code>{tp2:.2f}</code> | TP3: <code>{tp3:.2f}</code>\n"
            f"Lot: <code>{lot}</code> | Risk: {risk_pct:.2f}%\n"
            f"Verdict: {result['verdict']} | ADX: {result['adx']:.1f}"
        )

        if self.tg.send(msg):
            logging.info(f"Signal Sent: {result['direction']} @ {entry:.2f}")
            journal = self.store.get("journal", []) or []
            journal.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal": result["direction"], "entry": entry, "sl": sl,
                "tp1": tp1, "tp2": tp2, "tp3": tp3, "lot": lot,
                "outcome": "PENDING", "risk_pct": risk_pct
            })
            self.store.set("journal", journal[-50:])

# ════════════════════════════════════════════════════════════
# CLI COMMANDS
# ════════════════════════════════════════════════════════════
def cmd_selftest(cfg: Config) -> None:
    errors = []
    if cfg.max_lot < cfg.min_lot:
        errors.append("max_lot < min_lot")
    if not (cfg.parliament_min_ratio < cfg.parliament_half_ratio < cfg.parliament_full_ratio):
        errors.append("ratio ordering invalid")
    if cfg.account_balance <= 0:
        errors.append("account_balance <= 0")
    if errors:
        print("❌ SELFTEST FAILED:", "; ".join(errors))
        sys.exit(1)
    print("✅ SELFTEST PASSED (v19.1 Hotfix)")

def cmd_smoke(cfg: Config) -> None:
    data = MarketDataHandler(cfg)
    signal = SignalGenerator(cfg)
    try:
        dfs = data.fetch_all()
    except Exception as e:
        print(f"❌ Data Fetch Failed: {e}")
        sys.exit(1)

    m5c = dfs["5min"]
    print(f"Bars: M5={len(m5c)} M15={len(dfs['15min'])} H1={len(dfs['1h'])} H4={len(dfs['4h'])}")
    print(f"Last M5: {m5c.iloc[-1]['datetime']} close={m5c.iloc[-1]['close']:.2f}")

    res = signal.generate(m5c, dfs["15min"], dfs["1h"], dfs["4h"])
    print(f"Direction: {res['direction']} | Verdict: {res.get('verdict', 'CANCEL')}")
    print(f"Skip Reason: {res.get('skip_reason', 'None')}")

def cmd_test_tg(cfg: Config) -> None:
    tg = TelegramNotifier(cfg)
    ok = tg.send("<b>🔔 TEST APEX v19.1</b>\nTelegram connection successful.")
    print("✅ SENT" if ok else "❌ FAILED")

# ════════════════════════════════════════════════════════════
# MAIN ENTRYPOINT
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    setup_logging()
    try:
        config = Config.from_env()
    except Exception as e:
        print(f"CONFIG ERROR: {e}")
        sys.exit(2)

    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else "run"

    if cmd == "selftest":
        cmd_selftest(config)
    elif cmd == "smoke":
        cmd_smoke(config)
    elif cmd == "test-tg":
        cmd_test_tg(config)
    elif cmd == "run":
        try:
            TradingBotOrchestrator(config).run()
        except Exception as e:
            logging.exception("FATAL ERROR")
            sys.exit(1)
    else:
        print("Usage: python apex_v19_production.py [selftest|smoke|test-tg|run]")
