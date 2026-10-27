import os, sys, requests
import pandas as pd
from datetime import datetime, timezone

OUT_PATH = 'data/xauusd_h1.csv'
SOURCE_PATH = 'data/source.txt'
MIN_REQUIRED_BARS = 2000

def save(df, source_name):
    now_utc = pd.Timestamp.now(tz='UTC')
    df = df[df['timestamp'] <= now_utc]
    if len(df) < MIN_REQUIRED_BARS:
        raise ValueError(f"{source_name}: too few bars ({len(df)}).")
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    os.makedirs('data', exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    with open(SOURCE_PATH, 'w') as f:
        f.write(source_name)
    print(f"[SUKSES] {source_name}: {len(df)} bars, last={df['timestamp'].iloc[-1]}")

def get_twelve_data():
    key = os.environ.get('TWELVEDATA_API_KEY')
    if not key:
        raise ValueError("no API key")
    print("[1/1] Twelve Data (5000 bars)...")
    url = (f'https://api.twelvedata.com/time_series'
           f'?symbol=XAU/USD&interval=1h&outputsize=5000&apikey={key}')
    r = requests.get(url, timeout=15).json()
    if 'values' not in r:
        raise ValueError(f"Twelve Data Error: {r}")
    df = pd.DataFrame(r['values'])
    df['timestamp'] = pd.to_datetime(df['datetime'], utc=True)
    for c in ['open','high','low','close']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['volume'] = 0
    df = df[['timestamp','open','high','low','close','volume']].iloc[::-1]
    save(df, "Twelve Data XAU/USD 1h")

if __name__ == "__main__":
    try:
        get_twelve_data()
        sys.exit(0)
    except Exception as e:
        print(f"[-] GAGAL TOTAL: {e}")
        sys.exit(1)
