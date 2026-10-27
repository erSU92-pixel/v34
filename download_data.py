"""XAUUSD data fetcher: TwelveData (Multi-Year Loop) -> Yahoo."""
import os, sys, time, requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

MIN_REQUIRED_BARS = 2000
MAX_STALENESS = pd.Timedelta(days=7)
OUT_PATH = 'data/xauusd_h1.csv'
SOURCE_PATH = 'data/source.txt'

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
    print("[1/2] Twelve Data (Multi-Year Pagination)...")
    
    all_dfs = []
    start_year = 2015
    current_year = datetime.now(timezone.utc).year
    
    for year in range(start_year, current_year + 1):
        for half in [1, 2]:
            start_month = 1 if half == 1 else 7
            end_month = 6 if half == 1 else 12
            end_day = 30 if end_month == 6 else 31
            
            start_date = f"{year}-{start_month:02d}-01"
            end_date = f"{year}-{end_month:02d}-{end_day:02d}"
            
            if year == current_year and half == 2:
                end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            
            url = (f'https://api.twelvedata.com/time_series'
                   f'?symbol=XAU/USD&interval=1h&outputsize=5000'
                   f'&start_date={start_date}&end_date={end_date}&apikey={key}')
            
            try:
                r = requests.get(url, timeout=15).json()
                if 'values' not in r:
                    print(f"[-] Skip {start_date} to {end_date}: {r.get('message', 'No data')}")
                    continue
                df = pd.DataFrame(r['values'])
                all_dfs.append(df)
                print(f"[+] Fetched {len(df)} bars for {start_date} to {end_date}")
            except Exception as e:
                print(f"[-] Failed chunk {start_date}: {e}")
            
            # Wajib jeda 9 detik agar tidak diblokir API Twelve Data
            time.sleep(9) 
            
    if not all_dfs:
        raise ValueError("No data fetched from Twelve Data")
        
    df = pd.concat(all_dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['datetime'], utc=True)
    for c in ['open','high','low','close']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['volume'] = 0
    df = df[['timestamp','open','high','low','close','volume']].iloc[::-1]
    save(df, "Twelve Data XAU/USD 1h (10 Years)")

def get_yahoo():
    print("[2/2] Yahoo GC=F 1h (Fallback)...")
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
    save(df, "Yahoo GC=F 1h (Fallback)")

if __name__ == "__main__":
    for name, fn in [("TwelveData", get_twelve_data),
                     ("Yahoo", get_yahoo)]:
        try:
            fn()
            sys.exit(0)
        except Exception as e:
            print(f"[-] {name} failed: {e}")
    sys.exit("ALL DATA SOURCES FAILED")
