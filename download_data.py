"""XAUUSD data fetcher: HuggingFace (20 Years) -> TwelveData (Recent) -> Yahoo."""
import os, sys, json, asyncio, requests
import pandas as pd
import yfinance as yf
import websockets
from datetime import datetime, timezone

MIN_REQUIRED_BARS = 2000
MAX_STALENESS = pd.Timedelta(days=7) # Longgar karena data HF mungkin telat beberapa hari
OUT_PATH = 'data/xauusd_h1.csv'
SOURCE_PATH = 'data/source.txt'

# URL Dataset 20 Tahun (2004 - 2025) dari Hugging Face
HF_URL = "https://huggingface.co/datasets/ZombitX64/xauusd-gold-price-historical-data-2004-2025/resolve/main/XAU_1h_data.csv"

def save(df, source_name):
    if len(df) < MIN_REQUIRED_BARS:
        raise ValueError(f"{source_name}: too few bars ({len(df)}).")
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    os.makedirs('data', exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    with open(SOURCE_PATH, 'w') as f:
        f.write(source_name)
    print(f"[SUKSES] {source_name}: {len(df)} bars, last={df['timestamp'].iloc[-1]}")

def get_huggingface():
    print("[1/3] Hugging Face (20 Years Historical)...")
    try:
        df = pd.read_csv(HF_URL)
        # Normalisasi nama kolom agar sesuai dengan engine
        df.columns = [c.lower() for c in df.columns]
        
        # Deteksi nama kolom timestamp
        ts_col = 'timestamp' if 'timestamp' in df.columns else ('datetime' if 'datetime' in df.columns else 'date')
        df = df.rename(columns={ts_col: 'timestamp'})
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Deteksi volume
        if 'volume' not in df.columns:
            df['volume'] = 0
            
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        save(df, "HuggingFace XAUUSD 1h (2004-2025)")
    except Exception as e:
        raise RuntimeError(f"HuggingFace failed: {e}")

def get_twelve_data():
    key = os.environ.get('TWELVEDATA_API_KEY')
    if not key:
        raise ValueError("no API key")
    print("[2/3] Twelve Data (Recent 5000 bars)...")
    url = (f'https://api.twelvedata.com/time_series'
           f'?symbol=XAU/USD&interval=1h&outputsize=5000&apikey={key}')
    r = requests.get(url, timeout=15).json()
    if 'values' not in r:
        raise ValueError(str(r)[:200])
    df = pd.DataFrame(r['values'])
    df['timestamp'] = pd.to_datetime(df['datetime'], utc=True)
    for c in ['open','high','low','close']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['volume'] = 0
    df = df[['timestamp','open','high','low','close','volume']].iloc[::-1]
    save(df, "Twelve Data XAU/USD 1h")

def get_yahoo():
    print("[3/3] Yahoo GC=F 1h...")
    df = yf.download('GC=F', period='2y', interval='1h', progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError("empty")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    if 'datetime' in df.columns:
        df = df.rename(columns={'datetime':'timestamp'})
    elif 'date' in df.columns:
        df = df.rename(columns={'date':'timestamp'})
    df = df[['timestamp','open','high','low','close','volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    save(df, "Yahoo GC=F 1h")

if __name__ == "__main__":
    for name, fn in [("HuggingFace", get_huggingface),
                     ("TwelveData", get_twelve_data),
                     ("Yahoo", get_yahoo)]:
        try:
            fn()
            sys.exit(0)
        except Exception as e:
            print(f"[-] {name} failed: {e}")
    sys.exit("ALL DATA SOURCES FAILED")
