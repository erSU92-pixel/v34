"""
================================================================================
XAUUSD QUANTITATIVE INSTITUTIONAL SIGNAL BOT — v8.1.1 TITANIUM FINAL
================================================================================
Data Cascade: TwelveData -> Yahoo Finance -> XAUS-API (Free/Keyless).
================================================================================
"""
from __future__ import annotations

import html
import json
import logging
import os
import re
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from supabase import create_client, Client
except ImportError:
    Client = None

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================
VERSION = "8.1.1"
RUN_ID = uuid.uuid4().hex[:8]

@dataclass
class BotConfig:
    min_bars: int = 250
    retry_total: int = 3
    http_timeout: int = 12
    account_size_usd: float = 10000.0
    risk_pct: float = 1.0
    contract_size: float = 100.0
    signal_cooldown_min: int = 45
    news_buffer_min: int = 45
    twelve_data_api_key: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    tickerlayer_api_key: Optional[str] = None
    
    @classmethod
    def load(cls) -> "BotConfig":
        return cls(
            account_size_usd=float(os.getenv("ACCOUNT_SIZE_USD", "10000")),
            risk_pct=float(os.getenv("RISK_PCT", "1.0")),
            twelve_data_api_key=os.getenv("TWELVE_DATA_API_KEY"),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
            supabase_url=os.getenv("SUPABASE_URL"),
            supabase_key=os.getenv("SUPABASE_KEY"),
            tickerlayer_api_key=os.getenv("TICKERLAYER_API_KEY")
        )

# ==============================================================================
# 2. STRUCTURED LOGGING
# ==============================================================================
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        p = {"ts": datetime.now(timezone.utc).isoformat(), "run_id": RUN_ID,
             "level": record.levelname, "msg": record.getMessage()}
        if hasattr(record, "extra_data"): p.update(record.extra_data)
        if record.exc_info: p["exc"] = self.formatException(record.exc_info)
        return json.dumps(p, default=str)

LOG = logging.getLogger("xauusd_quant")
if not LOG.handlers:
    LOG.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    LOG.addHandler(h)

def log_evt(level: int, event: str, **kw) -> None:
    rec = LOG.makeRecord(LOG.name, level, "", 0, event, (), None)
    rec.extra_data = kw
    LOG.handle(rec)

def is_market_open() -> bool:
    now = datetime.now(timezone.utc)
    wd, h = now.weekday(), now.hour
    if wd == 5: return False
    if wd == 6: return h >= 22
    if wd == 4: return h < 21
    return True

# ==============================================================================
# 3. MATHEMATICAL CORE & INDICATORS
# ==============================================================================
def ema(s: pd.Series, p: int) -> pd.Series: return s.ewm(span=p, adjust=False).mean()
def sma(s: pd.Series, p: int) -> pd.Series: return s.rolling(p, min_periods=p).mean()

def rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1/p, adjust=False).mean()
    loss = -d.clip(upper=0).ewm(alpha=1/p, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + gain / (loss + 1e-10))

def atr(df: pd.DataFrame, p: int = 14) -> pd.Series:
    tr = pd.concat([df["high"] - df["low"], 
                    (df["high"] - df["close"].shift(1)).abs(), 
                    (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()

def adx(df: pd.DataFrame, p: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up, dn = df["high"].diff(), -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = atr(df, p)
    p_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/p, adjust=False).mean() / (tr + 1e-10)
    m_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/p, adjust=False).mean() / (tr + 1e-10)
    dx = 100 * (p_di - m_di).abs() / (p_di + m_di + 1e-10)
    return dx.ewm(alpha=1/p, adjust=False).mean(), p_di, m_di

def volume_profile_poc(df: pd.DataFrame, bins: int = 25) -> float:
    if df["volume"].sum() <= 0: return float(df["close"].median())
    hist, edges = np.histogram(df["close"], bins=bins, weights=df["volume"])
    return float((edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2.0)

# ==============================================================================
# 4. DATA INTEGRITY & FETCHING
# ==============================================================================
def validate_ohlcv(df: pd.DataFrame, label: str) -> None:
    req = {"time", "open", "high", "low", "close", "volume"}
    if req - set(df.columns): raise ValueError(f"{label}: Missing columns")
    if df[["open","high","low","close"]].isna().any().any(): raise ValueError(f"{label}: NaN detected")
    if (df["high"] < df["low"]).any(): raise ValueError(f"{label}: high < low")
    if df["volume"].sum() <= 0: raise ValueError(f"{label}: zero volume")
    
    last_ts = pd.to_datetime(df["time"].iloc[-1])
    age_h = (datetime.now(timezone.utc).replace(tzinfo=None) - last_ts).total_seconds() / 3600
    if age_h > (6 if is_market_open() else 72):
        raise ValueError(f"{label}: Stale data ({age_h:.1f}h old)")

@dataclass
class MarketData:
    h1: pd.DataFrame
    m15: pd.DataFrame
    source: str
    m15_synthetic: bool = False

class DataFetcher:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.5)))
        self.session.headers.update({"User-Agent": f"XAUUSD-Quant/{VERSION}"})

    def _fetch_twelvedata(self, interval: str, size: int) -> Optional[pd.DataFrame]:
        if not self.cfg.twelve_data_api_key: return None
        url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval={interval}&outputsize={size}&apikey={self.cfg.twelve_data_api_key}"
        r = self.session.get(url, timeout=self.cfg.http_timeout)
        r.raise_for_status()
        j = r.json()
        if j.get("status") == "error": raise RuntimeError(j.get("message"))
        vals = j.get("values", [])
        if not vals: return None
        
        df = pd.DataFrame(vals).rename(columns={"datetime": "time"})
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
        for c in ["open","high","low","close"]: df[c] = pd.to_numeric(df[c], errors="coerce")
        
        # FIX: Total rewrite for Volume handling in TwelveData
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1000.0)
        else:
            df["volume"] = 1000.0
            
        return df.sort_values("time").reset_index(drop=True)

    def _fetch_yfinance(self) -> MarketData:
        if not yf: raise RuntimeError("yfinance missing")
        tk = yf.Ticker("GC=F")
        h1_raw, m15_raw = tk.history(period="60d", interval="1h"), tk.history(period="60d", interval="15m")
        if h1_raw.empty: raise RuntimeError("YF H1 empty")

        def _norm(d):
            d = d.reset_index().rename(columns={"Datetime": "time", "Date": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            d["time"] = pd.to_datetime(d["time"], utc=True).dt.tz_localize(None)
            for c in ["open","high","low","close"]: d[c] = pd.to_numeric(d[c], errors="coerce")
            
            if "volume" in d.columns:
                d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(1000.0)
            else:
                d["volume"] = 1000.0
                
            return d.dropna(subset=["open","high","low","close"]).sort_values("time").reset_index(drop=True)

        h1 = _norm(h1_raw)
        if not m15_raw.empty and len(_norm(m15_raw)) >= 50:
            return MarketData(h1, _norm(m15_raw), "Yahoo", False)
        
        m15 = h1.set_index("time").resample("15min").ffill().reset_index()
        m15["volume"] /= 4.0
        return MarketData(h1, m15, "Yahoo(synth)", True)

    def _fetch_xaus_api(self) -> MarketData:
        try:
            url_h1 = "https://xaus.com/api/v1/chart?symbol=gold&range=1mo&interval=1h"
            r_h1 = self.session.get(url_h1, timeout=10)
            r_h1.raise_for_status()
            j_h1 = r_h1.json()
            bars_h1 = j_h1.get("bars") or j_h1.get("data") or []
            if not bars_h1: raise RuntimeError("XAUS H1 empty")
            
            df_h1 = pd.DataFrame(bars_h1)
            col_map = {"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
            df_h1 = df_h1.rename(columns=col_map)
            if "time" not in df_h1.columns: df_h1["time"] = df_h1.iloc[:, 0]
            df_h1["time"] = pd.to_datetime(df_h1["time"], unit="ms", utc=True).dt.tz_localize(None)
            for c in ["open","high","low","close"]: df_h1[c] = pd.to_numeric(df_h1[c], errors="coerce")
            df_h1["volume"] = pd.to_numeric(df_h1.get("volume", 1000.0), errors="coerce").fillna(1000.0)
            df_h1 = df_h1.dropna(subset=["open","high","low","close"]).sort_values("time").reset_index(drop=True)
            
            url_m15 = "https://xaus.com/api/v1/chart?symbol=gold&range=1mo&interval=15m"
            r_m15 = self.session.get(url_m15, timeout=10)
            r_m15.raise_for_status()
            j_m15 = r_m15.json()
            bars_m15 = j_m15.get("bars") or j_m15.get("data") or []
            if not bars_m15: raise RuntimeError("XAUS M15 empty")
            
            df_m15 = pd.DataFrame(bars_m15).rename(columns=col_map)
            if "time" not in df_m15.columns: df_m15["time"] = df_m15.iloc[:, 0]
            df_m15["time"] = pd.to_datetime(df_m15["time"], unit="ms", utc=True).dt.tz_localize(None)
            for c in ["open","high","low","close"]: df_m15[c] = pd.to_numeric(df_m15[c], errors="coerce")
            df_m15["volume"] = pd.to_numeric(df_m15.get("volume", 1000.0), errors="coerce").fillna(1000.0)
            df_m15 = df_m15.dropna(subset=["open","high","low","close"]).sort_values("time").reset_index(drop=True)
            
            return MarketData(df_h1, df_m15, "XAUS-API", False)
        except Exception as e:
            raise RuntimeError(f"XAUS-API failed: {e}")

    def fetch(self) -> MarketData:
        try:
            h1 = self._fetch_twelvedata("1h", 500)
            m15 = self._fetch_twelvedata("15min", 200)
            if h1 is not None and len(h1) >= self.cfg.min_bars:
                validate_ohlcv(h1, "12data-H1")
                if m15 is None or len(m15) < 50:
                    m15 = h1.set_index("time").resample("15min").ffill().reset_index()
                    m15["volume"] /= 4.0
                    synth = True
                else:
                    validate_ohlcv(m15, "12data-M15")
                    synth = False
                return MarketData(h1, m15, "TwelveData", synth)
        except Exception as e:
            log_evt(logging.WARNING, "twelvedata_fail", err=str(e)[:100])

        try:
            md = self._fetch_yfinance()
            validate_ohlcv(md.h1, "YF-H1")
            return md
        except Exception as e:
            log_evt(logging.WARNING, "yfinance_fail", err=str(e)[:100])

        try:
            md = self._fetch_xaus_api()
            validate_ohlcv(md.h1, "XAUS-H1")
            return md
        except Exception as e:
            log_evt(logging.WARNING, "xaus_fail", err=str(e)[:100])
            
        raise RuntimeError("All data sources failed.")

    def fetch_spot(self, h1: pd.DataFrame) -> float:
        if self.cfg.twelve_data_api_key:
            try:
                r = self.session.get(f"https://api.twelvedata.com/price?symbol=XAU/USD&apikey={self.cfg.twelve_data_api_key}", timeout=5)
                return float(r.json().get("price", 0) or 0) or float(h1["close"].iloc[-1])
            except: pass
        try:
            r = self.session.get("https://xaus.com/api/v1/spot?compact=1", timeout=5)
            data = r.json()
            return float(data.get("xau", {}).get("price") or data.get("spot_usd_oz", 0.0)) or float(h1["close"].iloc[-1])
        except: pass
        return float(h1["close"].iloc[-1])

# ==============================================================================
# 5. MACRO, NEWS, & REGIME ENGINE
# ==============================================================================
class MacroEngine:
    @staticmethod
    def check_correlation() -> Tuple[bool, str]:
        if not yf: return True, "Bypass"
        try:
            dxy = yf.Ticker("DX-Y.NYB").history(period="5d")["Close"]
            tnx = yf.Ticker("^TNX").history(period="5d")["Close"]
            notes = []
            if len(dxy) >= 2:
                ch = (dxy.iloc[-1] - dxy.iloc[-2]) / dxy.iloc[-2] * 100
                if ch > 0.4: notes.append(f"DXY+{ch:.2f}%")
            if len(tnx) >= 2:
                ch = (tnx.iloc[-1] - tnx.iloc[-2]) / tnx.iloc[-2] * 100
                if ch > 0.5: notes.append(f"10Y+{ch:.2f}%")
            if notes: return False, ", ".join(notes)
            return True, "Normal"
        except: return True, "Bypass"

    @staticmethod
    def check_news(cfg: BotConfig) -> Tuple[bool, str]:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        try:
            r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
            for e in r.json():
                if e["country"] == "USD" and e["impact"] == "High":
                    ts = datetime.fromisoformat(e["date"].replace("Z", "+00:00")).replace(tzinfo=None)
                    if abs((ts - now).total_seconds()) / 60 <= cfg.news_buffer_min:
                        return False, f"News Block: {e['title']}"
            return True, "Clear"
        except:
            if now.weekday() in (2,4) and 12 <= now.hour <= 15: return False, "Static Macro Block"
            return True, "Static Clear"

def get_fuzzy_regime(adx_val: float) -> Dict[str, float]:
    def _tri(x, a, b, c):
        if x <= a or x >= c: return 0.0
        if x <= b: return (x - a) / (b - a)
        return (c - x) / (c - b)
    
    m_r = _tri(adx_val, 0, 15, 25)
    m_t = _tri(adx_val, 20, 40, 60)
    raw = {"RANGING": max(m_r, 0.0), "TRENDING": max(m_t, 0.0)}
    s = sum(raw.values())
    if s > 1.0: 
        for k in raw: raw[k] /= s
        s = 1.0
    raw["TRANSITION"] = max(1.0 - s, 0.0)
    total = sum(raw.values())
    return {k: v/total if total>0 else 0.33 for k,v in raw.items()}

# ==============================================================================
# 6. ENSEMBLE & SIGNAL GENERATION
# ==============================================================================
@dataclass
class SignalResult:
    action: str
    score: float
    regime: str
    notes: str
    atr: float

def evaluate_market(md: MarketData, cfg: BotConfig) -> SignalResult:
    df, m15 = md.h1, md.m15
    c = float(df["close"].iloc[-1])
    
    adx_val = float(adx(df, 14)[0].iloc[-1])
    weights = get_fuzzy_regime(adx_val)
    dom_regime = max(weights, key=weights.get)
    atr_val = float(atr(df, 14).iloc[-1])
    
    votes = []
    votes.extend([1 if c > sma(df["close"], p).iloc[-1] else -1 for p in (5,10,20,50,100,200)])
    votes.extend([1 if ema(df["close"], 9).iloc[-1] > ema(df["close"], 21).iloc[-1] else -1,
                  1 if ema(df["close"], 12).iloc[-1] > ema(df["close"], 26).iloc[-1] else -1,
                  1 if ema(df["close"], 50).iloc[-1] > ema(df["close"], 200).iloc[-1] else -1])
    r = float(rsi(df["close"]).iloc[-1])
    votes.extend([1 if r > 50 else -1, 1 if r < 70 else -1, 1 if r > 30 else -1])
    _, p_di, m_di = adx(df, 14)
    votes.extend([1 if p_di.iloc[-1] > m_di.iloc[-1] else -1, 1 if adx_val > 20 else -1,
                  1 if c > df["close"].iloc[-2] else -1, 1 if c > df["close"].iloc[-5] else -1, 1 if c > df["close"].iloc[-10] else -1])
    for n in (5,10):
        hl = df["high"].iloc[-n:].max() - df["low"].iloc[-n:].min()
        votes.append(1 if (c - df["low"].iloc[-n:].min()) / max(hl, 1e-10) > 0.5 else -1)
    votes.append(1 if df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] else -1)
    votes.append(1 if c > volume_profile_poc(df) else -1)

    bull_count = sum(1 for v in votes if v == 1)
    bear_count = 21 - bull_count
    voter_agree = max(bull_count, bear_count) / 21.0
    intended = "BUY" if bull_count > bear_count else "SELL"

    last, prev = df.iloc[-1], df.iloc[-21:-1]
    rng = max(last["high"] - last["low"], 0.01)
    wick = (max(last["high"] - max(last["open"], last["close"]), min(last["open"], last["close"]) - last["low"]) / rng) * 100
    sweep = 35 if (last["high"] > prev["high"].max() and last["close"] < prev["high"].max()) or \
                  (last["low"] < prev["low"].min() and last["close"] > prev["low"].min()) else 0
    f_score = min(30 + sweep + (20 if wick > 40 else 0), 100)

    combined = round((voter_agree * 0.6 + (f_score / 100.0) * 0.4) * 100, 1)

    base_th = 65.0 if dom_regime == "TRENDING" else (74.0 if dom_regime == "RANGING" else 70.0)
    macro_ok, macro_msg = MacroEngine.check_correlation()
    news_ok, news_msg = MacroEngine.check_news(cfg)
    
    action, notes = "WAIT", ""
    
    if not is_market_open(): notes = "Market Closed"
    elif not news_ok: notes = f"News Veto: {news_msg}"
    elif not macro_ok: notes = f"Macro Veto: {macro_msg}"
    elif combined < base_th: notes = f"Score {combined}% < {base_th}%"
    else:
        m15_bull = ema(m15["close"], 9).iloc[-1] > ema(m15["close"], 21).iloc[-1]
        if intended == "BUY" and m15_bull and c > ema(df["close"], 200).iloc[-1]:
            action = "BUY"
        elif intended == "SELL" and not m15_bull and c < ema(df["close"], 200).iloc[-1]:
            action = "SELL"
        else:
            notes = "M15/EMA200 Misalignment"

    return SignalResult(action, combined, dom_regime, notes, atr_val)

# ==============================================================================
# 7. RISK MANAGEMENT & DATABASE
# ==============================================================================
class DatabaseManager:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.supabase = None
        self.sqlite = sqlite3.connect("data/audit.db", check_same_thread=False)
        self.sqlite.execute("CREATE TABLE IF NOT EXISTS history (ts TEXT, action TEXT, score REAL, r_multiple REAL)")
        
        if cfg.supabase_url and cfg.supabase_key and Client:
            try:
                self.supabase = create_client(cfg.supabase_url, cfg.supabase_key)
            except Exception as e:
                log_evt(logging.ERROR, "supabase_init_fail", err=str(e))

    def get_kelly_risk(self) -> Tuple[float, str]:
        try:
            if self.supabase:
                res = self.supabase.table("signals").select("r_multiple").in_("action", ["BUY","SELL"]).not_.is_("r_multiple","null").order("ts", desc=True).limit(50).execute()
                R = [float(x["r_multiple"]) for x in res.data if x["r_multiple"]]
            else:
                rows = self.sqlite.execute("SELECT r_multiple FROM history WHERE r_multiple IS NOT NULL ORDER BY ts DESC LIMIT 50").fetchall()
                R = [float(r[0]) for r in rows]
                
            if len(R) < 20: return self.cfg.risk_pct, f"Fallback {self.cfg.risk_pct}% (Need 20 trades)"
            
            wins = [x for x in R if x > 0]
            losses = [abs(x) for x in R if x < 0]
            if not wins or not losses: return self.cfg.risk_pct, "Fallback (No losses/wins)"
            
            p = len(wins) / len(R)
            b = (sum(wins)/len(wins)) / (sum(losses)/len(losses))
            f_star = max((p * b - (1-p)) / b, 0.0) * 0.5
            risk_pct = min(round(f_star * 100, 2), 3.0)
            return risk_pct if risk_pct > 0 else 0.1, f"Kelly: {risk_pct}% (Win:{p:.2f}, B:{b:.2f})"
        except Exception as e:
            return self.cfg.risk_pct, "Fallback (DB Error)"

    def log_signal(self, sig: SignalResult, spot: float, levels: dict) -> None:
        row = {"ts": datetime.now(timezone.utc).isoformat(), "action": sig.action, "spot": spot, "score": sig.score, 
               "regime": sig.regime, "notes": sig.notes, "sl": levels.get("sl"), "tp1": levels.get("tp1")}
        if self.supabase:
            try: self.supabase.table("signals").insert(row).execute()
            except: pass

def calculate_levels(action: str, price: float, atr: float, risk_pct: float, cfg: BotConfig) -> Optional[Dict]:
    if action not in ("BUY", "SELL"): return None
    
    sl_dist = max(atr * 1.5, 2.0)
    risk_usd = cfg.account_size_usd * (risk_pct / 100.0)
    lot = round(risk_usd / (sl_dist * cfg.contract_size), 2)
    lot = max(lot, 0.01)
    
    if action == "BUY":
        return {"entry": price, "sl": price - sl_dist, "tp1": price + sl_dist*1.5, "tp2": price + sl_dist*2.5, "lot": lot}
    else:
        return {"entry": price, "sl": price + sl_dist, "tp1": price - sl_dist*1.5, "tp2": price - sl_dist*2.5, "lot": lot}

# ==============================================================================
# 8. TELEGRAM EXECUTION
# ==============================================================================
def send_telegram(cfg: BotConfig, sig: SignalResult, spot: float, levels: dict, risk_reason: str) -> None:
    if not cfg.telegram_bot_token or not cfg.telegram_chat_id: return
    if sig.action == "WAIT": return
        
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    emoji = "🟢" if sig.action == "BUY" else "🔴"
    
    text = (f"{emoji} <b>XAUUSD {sig.action} (v{VERSION})</b>\n"
            f"⏰ <code>{now}</code>\n"
            f"───────────────────\n"
            f"<b>Confluence:</b> {sig.score}% | <b>Regime:</b> {sig.regime}\n"
            f"<code>ENTRY : {levels['entry']:.2f}\n"
            f"SL    : {levels['sl']:.2f}\n"
            f"TP1   : {levels['tp1']:.2f}\n"
            f"TP2   : {levels['tp2']:.2f}</code>\n"
            f"<b>Lot Size:</b> {levels['lot']} | <b>Risk:</b> {risk_reason}")

    url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage"
    payload = {"chat_id": cfg.telegram_chat_id, "text": text, "parse_mode": "HTML"}
    
    try:
        r = requests.post(url, json=payload, timeout=8)
        if r.status_code == 400:
            payload["text"] = re.sub(r'<[^>]+>', '', text)
            del payload["parse_mode"]
            requests.post(url, json=payload, timeout=8)
    except Exception as e:
        log_evt(logging.ERROR, "telegram_fail", err=str(e))

# ==============================================================================
# 9. MAIN ORCHESTRATOR
# ==============================================================================
def main() -> int:
    log_evt(logging.INFO, "engine_start", version=VERSION)
    if not is_market_open():
        print("ACTION: SKIP (Market Closed)")
        return 0

    cfg = BotConfig.load()
    fetcher = DataFetcher(cfg)
    db = DatabaseManager(cfg)

    try:
        md = fetcher.fetch()
        spot = fetcher.fetch_spot(md.h1)
        
        sig = evaluate_market(md, cfg)
        log_evt(logging.INFO, "analysis_done", action=sig.action, score=sig.score, regime=sig.regime, note=sig.notes, source=md.source)
        
        if sig.action in ("BUY", "SELL"):
            risk_pct, risk_reason = db.get_kelly_risk()
            levels = calculate_levels(sig.action, spot, sig.atr, risk_pct, cfg)
            
            if levels:
                db.log_signal(sig, spot, levels)
                send_telegram(cfg, sig, spot, levels, risk_reason)
                print(f"EXECUTED: {sig.action} | Score: {sig.score}% | Lot: {levels['lot']}")
        else:
            print(f"WAIT: {sig.notes} (Score: {sig.score}%) | Source: {md.source}")
            
        return 0
    except Exception as e:
        log_evt(logging.CRITICAL, "fatal_error", err=str(e))
        return 1

if __name__ == "__main__":
    sys.exit(main())
