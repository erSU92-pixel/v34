import os
import time
import logging
import requests
import pandas as pd
import yfinance as yf

TWELVE_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
LSE_API_KEY = os.getenv("LSE_API_KEY", "")

def _normalize_df(df):
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    
    # Standarisasi nama kolom (lowercase)
    df.columns = [str(c).lower() for c in df.columns]
    
    rename_map = {'date': 'datetime', 't': 'datetime'}
    df = df.rename(columns=rename_map)

    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            return None
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    if 'volume' not in df.columns:
        df['volume'] = 0

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df = df.sort_values('datetime').reset_index(drop=True)
        
    return df

# --- SUMBER XAUUSD ---
def fetch_xauusd_twelvedata(interval, outputsize):
    try:
        url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval={interval}&outputsize={outputsize}&apikey={TWELVE_API_KEY}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "values" not in data: raise ValueError("No values")
        df = pd.DataFrame(data["values"]).iloc[::-1].reset_index(drop=True)
        return _normalize_df(df)
    except Exception as e:
        logging.warning(f"TwelveData XAUUSD gagal: {str(e)[:50]}")
        return None

def fetch_xauusd_yfinance(interval, outputsize):
    try:
        yf_interval = '5m' if interval == '5min' else '1h'
        period = "60d" if yf_interval == '5m' else "1y"
        df = yf.download("GC=F", period=period, interval=yf_interval, progress=False, threads=False)
        if df.empty: raise ValueError("Empty")
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.tail(outputsize)
        return _normalize_df(df)
    except Exception as e:
        logging.warning(f"yfinance XAUUSD gagal: {str(e)[:50]}")
        return None

def fetch_xauusd(interval="5min", outputsize=250):
    for name, func in [("TwelveData", fetch_xauusd_twelvedata), ("yfinance", fetch_xauusd_yfinance)]:
        logging.info(f"Mencoba sumber XAUUSD: {name}...")
        df = func(interval, outputsize)
        if df is not None and len(df) >= 50:
            logging.info(f"✅ XAUUSD via {name} ({len(df)} bar).")
            return df, name
        time.sleep(1)
    logging.error("❌ SEMUA sumber XAUUSD gagal.")
    return None, None

# --- SUMBER DXY (MENGGUNAKAN UUP SEBAGAI PROXY) ---
def fetch_dxy_twelvedata(interval, outputsize):
    try:
        # UUP adalah ETF pelacak DXY yang support di API gratis
        url = f"https://api.twelvedata.com/time_series?symbol=UUP&interval={interval}&outputsize={outputsize}&apikey={TWELVE_API_KEY}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "values" not in data: raise ValueError("No values")
        df = pd.DataFrame(data["values"]).iloc[::-1].reset_index(drop=True)
        return _normalize_df(df)
    except Exception as e:
        logging.warning(f"TwelveData DXY (UUP) gagal: {str(e)[:50]}")
        return None

def fetch_dxy_yfinance(interval, outputsize):
    try:
        yf_interval = '5m' if interval == '5min' else '1h'
        period = "60d" if yf_interval == '5m' else "1y"
        # MENGGUNAKAN UUP DAN threads=False UNTUK MENGHINDARI BLOCKIRAN YAHOO
        df = yf.download("UUP", period=period, interval=yf_interval, progress=False, threads=False)
        if df.empty: raise ValueError("Empty")
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.tail(outputsize)
        return _normalize_df(df)
    except Exception as e:
        logging.warning(f"yfinance DXY (UUP) gagal: {str(e)[:50]}")
        return None

def fetch_dxy(interval="5min", outputsize=250):
    for name, func in [("TwelveData", fetch_dxy_twelvedata), ("yfinance", fetch_dxy_yfinance)]:
        logging.info(f"Mencoba sumber DXY (UUP): {name}...")
        df = func(interval, outputsize)
        if df is not None and len(df) >= 50:
            logging.info(f"✅ DXY (UUP) via {name} ({len(df)} bar).")
            return df, name
        time.sleep(1)
    logging.error("❌ SEMUA sumber DXY gagal.")
    return None, None
