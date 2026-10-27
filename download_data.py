import os
import sys
import pandas as pd
from datetime import datetime, timezone

# --- Konfigurasi -------------------------------------------------------------
SYMBOL = "XAUUSD"
TIMEFRAME = "h1"
START_YEAR = 2010          # Dukascopy punya data XAUUSD dari ~2003; 2010 cukup aman
MIN_REQUIRED_BARS = 5000   # H1 15 tahun > 50.000 bar
OUT_PATH = "data/xauusd_h1.csv"
SOURCE_PATH = "data/source.txt"

def fetch_dukascopy():
    """Unduh data H1 XAUUSD dari Dukascopy menggunakan library Python."""
    print(f"[DUKASCOPY] Mengunduh {SYMBOL} {TIMEFRAME} dari {START_YEAR}...")
    try:
        from dukascopy_downloader import Downloader

        # Inisialisasi downloader; data disimpan sementara di folder 'duka_data'
        dl = Downloader("duka_data")
        dl.download(
            instruments=[SYMBOL],
            tf=TIMEFRAME,
            start=f"{START_YEAR}-01-01",
            sides="BID",          # BID sudah cukup untuk backtest OHLC
        )

        # Ekspor ke DataFrame
        # Library menyimpan dalam format Parquet; kita baca lalu konversi
        import glob
        import pandas as pd

        # Cari file Parquet yang dihasilkan
        pattern = f"duka_data/{TIMEFRAME}/**/{SYMBOL}*.parquet"
        files = glob.glob(pattern, recursive=True)
        if not files:
            # Fallback: cari file apa pun yang mengandung XAUUSD dan h1
            pattern = f"duka_data/**/*{SYMBOL}*.parquet"
            files = glob.glob(pattern, recursive=True)

        if not files:
            raise FileNotFoundError("File Parquet dari Dukascopy tidak ditemukan.")

        # Gabungkan semua file jika ada beberapa
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)

        # Normalisasi kolom
        # Dukascopy biasanya menghasilkan kolom: timestamp, open, high, low, close, volume
        # atau: time, open, high, low, close, volume
        df.columns = [c.lower() for c in df.columns]
        if "time" in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={"time": "timestamp"})
        if "tick_volume" in df.columns:
            df = df.rename(columns={"tick_volume": "volume"})
        if "volume" not in df.columns:
            df["volume"] = 0

        # Pastikan timestamp UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("timestamp").reset_index(drop=True)

        print(f"[DUKASCOPY] Berhasil: {len(df)} bar, {df['timestamp'].min()} s/d {df['timestamp'].max()}")
        return df

    except Exception as e:
        print(f"[-] Dukascopy gagal: {e}")
        return None

def fetch_yahoo_fallback():
    """Fallback: ambil 2 tahun terakhir dari Yahoo Finance (GC=F)."""
    print("[YAHOO] Fallback: mengunduh GC=F 1h (2 tahun)...")
    try:
        import yfinance as yf

        df = yf.download("GC=F", period="2y", interval="1h", progress=False)
        if df.empty:
            raise ValueError("Yahoo mengembalikan data kosong.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})
        elif "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})

        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        print(f"[YAHOO] Berhasil: {len(df)} bar")
        return df

    except Exception as e:
        print(f"[-] Yahoo gagal: {e}")
        return None

def save_data(df, source_name):
    """Validasi dan simpan ke CSV."""
    if df is None or len(df) < MIN_REQUIRED_BARS:
        raise ValueError(f"{source_name}: data terlalu sedikit ({len(df) if df is not None else 0} bar).")

    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    with open(SOURCE_PATH, "w") as f:
        f.write(source_name)

    print(f"[SIMPAN] {source_name} -> {OUT_PATH} ({len(df)} bar)")

def main():
    # 1. Coba Dukascopy (sumber utama, data panjang)
    df = fetch_dukascopy()
    source = "Dukascopy XAUUSD H1"

    # 2. Jika gagal, fallback ke Yahoo
    if df is None:
        df = fetch_yahoo_fallback()
        source = "Yahoo GC=F H1 (fallback)"

    # 3. Jika keduanya gagal, berhenti
    if df is None:
        sys.exit("SEMUA SUMBER DATA GAGAL.")

    save_data(df, source)

if __name__ == "__main__":
    main()
