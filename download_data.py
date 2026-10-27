"""XAUUSD data fetcher: TwelveData -> Deriv -> Yahoo."""
import os, sys, json, asyncio, requests
import pandas as pd
import yfinance as yf
import websockets
from datetime import datetime, timezone

MIN_REQUIRED_BARS = 2000
MAX_STALENESS = pd.Timedelta(hours=12) # Diperlonggar untuk jaga-jaga weekend
OUT_PATH = 'data/xauusd_h1.csv'
SOURCE_PATH = 'data/source.txt'

def save(df, source_name):
    # FIX TIMEZONE: Potong data masa depan (Twelve Data kadang ngaco)
    now_utc = pd.Timestamp.now(tz='UTC')
    df = df[df['timestamp'] <= now_utc]
    
    if len(df) < MIN_REQUIRED_BARS:
        raise ValueError(f"{source_name}: too few bars ({len(df)}).")
        
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    last = df['timestamp'].iloc[-1]
    age = now_utc - last
    
    if age > MAX_STALENESS:
        raise ValueError(f"{source_name}: stale last={last} age={age}.")
        
    os.makedirs('data', exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    with open(SOURCE_PATH, 'w') as f:
        f.write(source_name)
    print(f"[SUKSES] {source_name}: {len(df)} bars, last={last}")

def get_twelve_data():
    key = os.environ.get('TWELVEDATA_API_KEY')
    if not key:
        raise ValueError("no API key")
    print("[1/3] Twelve Data...")
    url = (f'https://api.twelvedata.com/time_series'
           f'?symbol=XAU/USD&interval=1h&outputsize=5000&apikey={key}')
    r = requests.get(url, timeout=15).json()
    if 'values' not in r:
        raise ValueError(str(r)[:200])
    df = pd.DataFrame(r['values'])
    df['timestamp'] = pd.to_datetime(df['datetime'], utc=True)
    for c in ['open','high','low','close']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    else:
        df['volume'] = 0
    df = df[['timestamp','open','high','low','close','volume']].iloc[::-1]
    save(df, "Twelve Data XAU/USD 1h")

async def _fetch_deriv():
    url = 'wss://api.derivws.com/trading/v1/options/ws/public'
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({
            'ticks_history':'frxXAUUSD','style':'candles',
            'granularity':3600,'count':5000,'end':'latest','req_id':1,
        }))
        data = json.loads(await ws.recv())
        if 'candles' not in data:
            raise ValueError(str(data)[:200])
        df = pd.DataFrame(data['candles'])
        df['timestamp'] = pd.to_datetime(df['epoch'], unit='s', utc=True)
        df['volume'] = 0
        df = df[['timestamp','open','high','low','close','volume']]
        save(df, "Deriv frxXAUUSD 1h")

def get_deriv():
    print("[2/3] Deriv API...")
    asyncio.run(_fetch_deriv())

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
    for name, fn in [("TwelveData", get_twelve_data),
                     ("Deriv", get_deriv),
                     ("Yahoo", get_yahoo)]:
        try:
            fn()
            sys.exit(0)
        except Exception as e:
            print(f"[-] {name} failed: {e}")
    sys.exit("ALL DATA SOURCES FAILED")
