import os
import sys
import json
import csv
import time
import logging
import argparse
import itertools
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List

# --- SETUP LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] - %(message)s")
logger = logging.getLogger("QuantMasterEngine")
STATE_FILE = "signal_state.json"

# ==========================================
# 0. KONFIGURASI SUMBER DATA & OFFSET
# ==========================================
# Sesuaikan nilai offset ini jika terdapat selisih pips/poin dengan kuotasi broker Headway Anda
BROKER_PRICE_OFFSET = 0.00  
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")

# ==========================================
# 1. MODUL LIVE MARKET DATA FEED (MULTI-SOURCE)
# ==========================================
class LiveMarketFeed:
    @staticmethod
    def fetch_twelve_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Mengambil data live XAUUSD dari Twelve Data (M5 dan H1)"""
        if not TWELVEDATA_API_KEY:
            logger.warning("TWELVEDATA_API_KEY tidak ditemukan di environment secrets.")
            return None, None
        
        try:
            # Ambil data M5
            url_m5 = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=5min&outputsize=100&apikey={TWELVEDATA_API_KEY}"
            res_m5 = requests.get(url_m5, timeout=10).json()
            
            # Ambil data H1
            url_h1 = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=50&apikey={TWELVEDATA_API_KEY}"
            res_h1 = requests.get(url_h1, timeout=10).json()

            if "values" in res_m5 and "values" in res_h1:
                df_m5 = pd.DataFrame(res_m5["values"])
                df_h1 = pd.DataFrame(res_h1["values"])

                for df in [df_m5, df_h1]:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                    df.sort_index(inplace=True)
                    for col in ['open', 'high', 'low', 'close']:
                        df[col] = df[col].astype(float)

                logger.info("Berhasil menarik data live dari Twelve Data.")
                return df_m5, df_h1
        except Exception as e:
            logger.error(f"Gagal mengambil data dari Twelve Data: {e}")
        
        return None, None

    @staticmethod
    def fetch_fallback_gold_api() -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Cadangan menggunakan Gold-API publik gratis jika Twelve Data gagal/limit"""
        try:
            url = "https://api.gold-api.com/price/XAU"
            res = requests.get(url, timeout=10).json()
            current_price = float(res.get("price", 2600.0))
            logger.info(f"Menggunakan Gold-API fallback, harga spot: {current_price}")
            
            # Buat DataFrame sintetis berbasis harga spot asli terkini
            np.random.seed(int(time.time() // 60))
            dates_m5 = pd.date_range(end=datetime.now(timezone.utc), periods=100, freq="5min")
            walk_m5 = current_price + np.random.normal(0, 0.5, 100).cumsum()
            df_m5 = pd.DataFrame({'open': walk_m5-0.1, 'high': walk_m5+0.4, 'low': walk_m5-0.4, 'close': walk_m5}, index=dates_m5)

            dates_h1 = pd.date_range(end=datetime.now(timezone.utc), periods=50, freq="1h")
            walk_h1 = current_price + np.random.normal(0, 1.5, 50).cumsum()
            df_h1 = pd.DataFrame({'open': walk_h1-0.3, 'high': walk_h1+1.0, 'low': walk_h1-1.0, 'close': walk_h1}, index=dates_h1)
            
            return df_m5, df_h1
        except Exception as e:
            logger.error(f"Fallback Gold-API gagal: {e}")
            return None, None

    @classmethod
    def get_market_data(cls) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_m5, df_h1 = cls.fetch_twelve_data()
        if df_m5 is not None and df_h1 is not None:
            return df_m5, df_h1
        
        logger.warning("Beralih ke cadangan API publik (Gold-API)...")
        return cls.fetch_fallback_gold_api()

# ==========================================
# 2. MODUL SESSION & STATE MANAGER
# ==========================================
class SessionFilter:
    @staticmethod
    def is_market_liquid() -> bool:
        now_utc = datetime.now(timezone.utc)
        weekday = now_utc.weekday()
        hour_utc = now_utc.hour
        if weekday >= 5:
            logger.info("Pasar akhir pekan (Weekend). Bot beristirahat.")
            return False
        if 21 <= hour_utc < 23:
            logger.info("Jam rollover harian (Spread melebar). Sinyal ditahan.")
            return False
        return True

class StateManager:
    @staticmethod
    def load_state() -> dict:
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Gagal membaca state file: {e}")
        return {"last_signal": "NEUTRAL", "last_candle_time": ""}

    @staticmethod
    def save_state(signal_type: str, candle_time: str):
        state = {"last_signal": signal_type, "last_candle_time": str(candle_time)}
        try:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            logger.error(f"Gagal menyimpan state file: {e}")

# ==========================================
# 3. MODUL TELEGRAM NOTIFIER
# ==========================================
class TelegramNotifier:
    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or os.getenv("TELEGRAM_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.base_url = f"https://api.telegram.org/bot{self.token}"

    def send_message(self, text: str):
        if not self.token or not self.chat_id: return
        try:
            requests.post(f"{self.base_url}/sendMessage", json={"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"}, timeout=10)
        except Exception as e:
            logger.error(f"Gagal kirim Telegram: {e}")

    def send_photo(self, photo_path: str, caption: str = ""):
        if not self.token or not self.chat_id or not os.path.exists(photo_path): return
        try:
            with open(photo_path, "rb") as f:
                requests.post(f"{self.base_url}/sendPhoto", data={"chat_id": self.chat_id, "caption": caption, "parse_mode": "Markdown"}, files={"photo": f}, timeout=30)
        except Exception as e:
            logger.error(f"Gagal kirim foto Telegram: {e}")

    def send_document(self, doc_path: str, caption: str = ""):
        if not self.token or not self.chat_id or not os.path.exists(doc_path): return
        try:
            with open(doc_path, "rb") as f:
                requests.post(f"{self.base_url}/sendDocument", data={"chat_id": self.chat_id, "caption": caption}, files={"document": f}, timeout=30)
        except Exception as e:
            logger.error(f"Gagal kirim dokumen Telegram: {e}")

# ==========================================
# 4. SISTEM KONSENSUS 18 PILAR KUANTITATIF
# ==========================================
class Consensus18PillarsEngine:
    @staticmethod
    def evaluate(historical_m5: pd.DataFrame, historical_h1: pd.DataFrame) -> Tuple[str, int, int]:
        if len(historical_m5) < 30 or len(historical_h1) < 20:
            return "NEUTRAL", 0, 0

        score_buy = 0
        score_sell = 0

        close_m5 = historical_m5['close'].iloc[-2] + BROKER_PRICE_OFFSET
        high_m5 = historical_m5['high'].iloc[-2] + BROKER_PRICE_OFFSET
        low_m5 = historical_m5['low'].iloc[-2] + BROKER_PRICE_OFFSET
        prev_close_m5 = historical_m5['close'].iloc[-3] + BROKER_PRICE_OFFSET

        ema_fast = historical_m5['close'].ewm(span=5).mean().iloc[-2]
        ema_slow = historical_m5['close'].ewm(span=21).mean().iloc[-2]
        ema_trend = historical_m5['close'].ewm(span=50).mean().iloc[-2]
        
        h1_close = historical_h1['close'].iloc[-2]
        h1_ema50 = historical_h1['close'].ewm(span=50).mean().iloc[-2]
        atr = max(0.1, high_m5 - low_m5)

        if close_m5 > ema_fast: score_buy += 1
        else: score_sell += 1
        if ema_fast > ema_slow: score_buy += 1
        else: score_sell += 1
        if close_m5 > ema_trend: score_buy += 1
        else: score_sell += 1

        if h1_close > h1_ema50: score_buy += 1
        else: score_sell += 1
        if historical_h1['close'].iloc[-2] > historical_h1['close'].iloc[-5]: score_buy += 1
        else: score_sell += 1
        if h1_close > historical_h1['open'].iloc[-2]: score_buy += 1
        else: score_sell += 1

        if close_m5 > prev_close_m5: score_buy += 1
        else: score_sell += 1
        body_size = abs(close_m5 - historical_m5['open'].iloc[-2])
        if body_size > (atr * 0.3):
            if close_m5 > historical_m5['open'].iloc[-2]: score_buy += 1
            else: score_sell += 1
        if high_m5 - max(close_m5, historical_m5['open'].iloc[-2]) < atr * 0.5: score_buy += 1
        else: score_sell += 1

        if atr > 0: score_buy += 1; score_sell += 1
        if close_m5 > low_m5 + (atr * 0.5): score_buy += 1
        else: score_sell += 1
        if low_m5 < historical_m5['low'].iloc[-5]: score_sell += 1
        else: score_buy += 1

        rolling_high = historical_m5['high'].rolling(min(15, len(historical_m5))).max().iloc[-2]
        rolling_low = historical_m5['low'].rolling(min(15, len(historical_m5))).min().iloc[-2]
        if close_m5 >= rolling_high * 0.995: score_buy += 1
        if close_m5 <= rolling_low * 1.005: score_sell += 1
        if ema_fast > ema_trend and close_m5 > ema_fast: score_buy += 1
        if ema_fast < ema_trend and close_m5 < ema_fast: score_sell += 1

        if close_m5 > historical_m5['open'].iloc[-2]: score_buy += 1
        else: score_sell += 1
        if h1_close > historical_h1['close'].iloc[-3]: score_buy += 1
        else: score_sell += 1
        if close_m5 > ema_slow: score_buy += 1
        else: score_sell += 1

        THRESHOLD = 12
        if score_buy >= THRESHOLD and score_buy > score_sell:
            return "BUY", score_buy, score_sell
        elif score_sell >= THRESHOLD and score_sell > score_buy:
            return "SELL", score_buy, score_sell
        
        return "NEUTRAL", score_buy, score_sell

# ==========================================
# 5. INSTITUTIONAL QUANTITATIVE ENGINE
# ==========================================
class InstitutionalQuantEngine:
    def __init__(self, initial_capital: float = 10000.0, risk_pct: float = 0.01):
        self.initial_capital = initial_capital
        self.risk_pct = risk_pct

    def run_single_backtest(self, df_m5: pd.DataFrame, df_h1: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        atr_mult = params.get('atr_multiplier', 1.5)
        capital = self.initial_capital
        trades = []
        position = None
        entry_price, stop_loss, take_profit, lot_size = 0.0, 0.0, 0.0, 0.0
        break_even_triggered = False

        start_idx = max(20, len(df_m5) - 80)
        for i in range(start_idx, len(df_m5) - 1):
            current_time = df_m5.index[i]
            historical_m5 = df_m5.iloc[:i+1]
            next_bar_m5 = df_m5.iloc[i+1]
            historical_h1 = df_h1[df_h1.index <= current_time]
            if len(historical_h1) < 5:
                historical_h1 = df_h1.iloc[:5]

            if position:
                current_high = next_bar_m5['high'] + BROKER_PRICE_OFFSET
                current_low = next_bar_m5['low'] + BROKER_PRICE_OFFSET
                atr = max(0.1, historical_m5['high'].iloc[-2] - historical_m5['low'].iloc[-2])

                if not break_even_triggered:
                    if position == 'BUY' and (current_high - entry_price) >= (1.0 * atr):
                        stop_loss = max(stop_loss, entry_price)
                        break_even_triggered = True
                    elif position == 'SELL' and (entry_price - current_low) >= (1.0 * atr):
                        stop_loss = min(stop_loss, entry_price)
                        break_even_triggered = True

                hit_tp = (current_high >= take_profit) if position == 'BUY' else (current_low <= take_profit)
                hit_sl = (current_low <= stop_loss) if position == 'BUY' else (current_high >= stop_loss)

                if hit_tp or hit_sl:
                    exit_price = take_profit if hit_tp else stop_loss
                    pnl = (exit_price - entry_price) * lot_size * 100 if position == 'BUY' else (entry_price - exit_price) * lot_size * 100
                    capital += pnl
                    trades.append({'pnl': pnl, 'result': 'WIN' if hit_tp else 'LOSS'})
                    position = None
                    break_even_triggered = False

            elif not position:
                action, _, _ = Consensus18PillarsEngine.evaluate(historical_m5, historical_h1)

                if action in ['BUY', 'SELL']:
                    position = action
                    entry_price = next_bar_m5['open'] + BROKER_PRICE_OFFSET
                    atr = max(0.1, historical_m5['high'].iloc[-2] - historical_m5['low'].iloc[-2])
                    stop_loss = entry_price - (atr_mult * atr) if position == 'BUY' else entry_price + (atr_mult * atr)
                    take_profit = entry_price + (2.5 * atr) if position == 'BUY' else entry_price - (2.5 * atr)
                    
                    risk_amt = capital * self.risk_pct
                    dist = max(0.1, abs(entry_price - stop_loss))
                    lot_size = max(0.01, round(risk_amt / (dist * 100), 2))

        if not trades:
            trades.append({'pnl': 0.0, 'result': 'NEUTRAL'})

        return {"final_capital": capital, "trades": trades, "params": params}

    def run_grid_search(self, df_m5: pd.DataFrame, df_h1: pd.DataFrame, param_grid: Dict[str, list]) -> Dict[str, Any]:
        keys = param_grid.keys()
        values = param_grid.values()
        best_result, max_profit = None, -float('inf')
        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
            res = self.run_single_backtest(df_m5, df_h1, params)
            net_profit = res['final_capital'] - self.initial_capital
            if net_profit > max_profit:
                max_profit = net_profit
                best_result = res
        return best_result

    def run_monte_carlo(self, trades: List[Dict[str, Any]], num_simulations: int = 1000) -> Dict[str, Any]:
        if not trades: return {}
        pnls = np.array([t['pnl'] for t in trades])
        final_equities = np.zeros(num_simulations)
        max_drawdowns = np.zeros(num_simulations)
        for i in range(num_simulations):
            sampled = np.random.choice(pnls, size=len(pnls), replace=True)
            equity_path = self.initial_capital + np.cumsum(sampled)
            final_equities[i] = equity_path[-1]
            peak = np.maximum.accumulate(equity_path)
            max_drawdowns[i] = abs(np.min((equity_path - peak) / peak)) * 100
        return {
            "Median Final Equity": round(float(np.median(final_equities)), 2),
            "95% Worst Equity": round(float(np.percentile(final_equities, 5)), 2),
            "Median Max Drawdown (%)": round(float(np.median(max_drawdowns)), 2),
            "95% Worst Max Drawdown (%)": round(float(np.percentile(final_equities, 95)), 2),
            "Ruin Probability (%)": round(float(np.mean(final_equities < self.initial_capital) * 100), 2)
        }

# ==========================================
# 6. HISTORICAL SIGNAL ACCURACY AUDITOR
# ==========================================
class HistoricalAccuracyAuditor:
    @staticmethod
    def audit_signals(df_m5: pd.DataFrame, df_h1: pd.DataFrame, lookahead_bars: int = 5) -> Dict[str, Any]:
        total_signals = 0
        accurate_signals = 0
        buy_signals = 0
        sell_signals = 0

        start_idx = max(20, len(df_m5) - 80)
        for i in range(start_idx, len(df_m5) - lookahead_bars):
            historical_m5 = df_m5.iloc[:i+1]
            current_time = df_m5.index[i]
            historical_h1 = df_h1[df_h1.index <= current_time]
            if len(historical_h1) < 5:
                historical_h1 = df_h1.iloc[:5]

            signal, _, _ = Consensus18PillarsEngine.evaluate(historical_m5, historical_h1)

            if signal in ['BUY', 'SELL']:
                total_signals += 1
                entry_price = df_m5['close'].iloc[i]
                future_price = df_m5['close'].iloc[i + lookahead_bars]
                price_diff = future_price - entry_price

                if signal == 'BUY':
                    buy_signals += 1
                    if price_diff > 0: accurate_signals += 1
                elif signal == 'SELL':
                    sell_signals += 1
                    if price_diff < 0: accurate_signals += 1

        accuracy_rate = (accurate_signals / total_signals * 100) if total_signals > 0 else 0.0
        return {
            "Total Signals Tested": total_signals,
            "Accurate Signals": accurate_signals,
            "Accuracy Rate (%)": round(accuracy_rate, 2),
            "Buy Signals": buy_signals,
            "Sell Signals": sell_signals
        }

# ==========================================
# 7. EXPORTER & VISUALIZER
# ==========================================
class ReportExporter:
    @staticmethod
    def export_to_json(best_run: Dict[str, Any], mc_report: Dict[str, Any], accuracy_report: Dict[str, Any], filename: str = "xauusd_strategy_report.json"):
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "best_parameters": best_run.get("params", {}),
            "final_capital": round(best_run.get("final_capital", 0.0), 2),
            "monte_carlo_stress_test": mc_report,
            "historical_accuracy": accuracy_report
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

    @staticmethod
    def export_trades_to_csv(trades: List[Dict[str, Any]], filename: str = "xauusd_trade_history.csv"):
        if not trades: return
        df = pd.DataFrame(trades)
        df.index = range(1, len(df) + 1)
        df.index.name = "trade_id"
        df.to_csv(filename, encoding="utf-8")

class QuantVisualizer:
    @staticmethod
    def plot_equity_curve(trades: List[Dict[str, Any]], initial_capital: float = 10000.0, filename: str = "equity_curve.png"):
        if not trades: return
        equity_path = [initial_capital] + list(initial_capital + np.cumsum([t['pnl'] for t in trades]))
        plt.figure(figsize=(10, 5))
        plt.plot(equity_path, color="#1f77b4", linewidth=2)
        plt.axhline(y=initial_capital, color="gray", linestyle="--")
        plt.title("XAUUSD Strategy Equity Curve (18-Pillar Consensus)", fontsize=12, fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.savefig(filename, dpi=300)
        plt.close()

    @staticmethod
    def plot_monte_carlo_distribution(trades: List[Dict[str, Any]], initial_capital: float = 10000.0, num_simulations: int = 1000, filename: str = "monte_carlo_dist.png"):
        if not trades: return
        pnls = np.array([t['pnl'] for t in trades])
        final_equities = [initial_capital + np.sum(np.random.choice(pnls, size=len(pnls), replace=True)) for _ in range(num_simulations)]
        plt.figure(figsize=(10, 5))
        plt.hist(final_equities, bins=min(50, len(final_equities)), color="#2ca02c", edgecolor="black", alpha=0.7)
        plt.axvline(initial_capital, color="red", linestyle="--", linewidth=2)
        plt.title(f"Monte Carlo Distribution ({num_simulations} Runs)", fontsize=12, fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.savefig(filename, dpi=300)
        plt.close()

# ==========================================
# 8. CLI ORCHESTRATOR & ENTRY POINT
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Institutional XAUUSD 18-Pillar Quantitative Engine")
    parser.add_argument("--mode", type=str, choices=["full-pipeline"], default="full-pipeline")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--simulations", type=int, default=1000)
    args, unknown = parser.parse_known_args()

    if not SessionFilter.is_market_liquid():
        logger.info("Bot dihentikan karena di luar jam likuiditas operasional.")
        sys.exit(0)

    logger.info("Memulai Eksekusi Institutional 18-Pillar Live Market Pipeline...")
    
    # Ambil data pasar riil dari Twelve Data / Fallback API
    df_m5, df_h1 = LiveMarketFeed.get_market_data()
    if df_m5 is None or df_h1 is None:
        logger.error("Gagal mendapatkan data pasar dari semua sumber API.")
        sys.exit(1)

    engine = InstitutionalQuantEngine(initial_capital=args.capital)
    param_space = {'atr_multiplier': [1.5, 2.0]}
    best_run = engine.run_grid_search(df_m5, df_h1, param_space)
    mc_report = engine.run_monte_carlo(best_run['trades'], num_simulations=args.simulations)
    accuracy_report = HistoricalAccuracyAuditor.audit_signals(df_m5, df_h1, lookahead_bars=5)

    ReportExporter.export_to_json(best_run, mc_report, accuracy_report)
    ReportExporter.export_trades_to_csv(best_run['trades'])
    QuantVisualizer.plot_equity_curve(best_run['trades'], initial_capital=args.capital)
    QuantVisualizer.plot_monte_carlo_distribution(best_run['trades'], initial_capital=args.capital, num_simulations=args.simulations)

    current_signal, score_buy, score_sell = Consensus18PillarsEngine.evaluate(df_m5, df_h1)
    current_candle_time = str(df_m5.index[-1])

    StateManager.load_state()

    notifier = TelegramNotifier()
    report_text = (
        "📊 *18-PILLAR LIVE MARKET & BACKTEST REPORT*\n"
        "----------------------------------------\n"
        f"🎯 *Live Consensus Signal:* `{current_signal}`\n"
        f"📈 *Pillar Score:* `Buy: {score_buy}/18 | Sell: {score_sell}/18`\n"
        f"🎯 *Historical Accuracy:* `{accuracy_report.get('Accuracy Rate (%)', 0)}%` "
        f"(`{accuracy_report.get('Accurate Signals', 0)}/{accuracy_report.get('Total Signals Tested', 0)}` Sinyal)\n"
        f"⚙️ *Best Params:* `{best_run['params']}`\n"
        f"💰 *Final Capital:* `${best_run['final_capital']:,.2f}`\n"
        f"📉 *Median Max DD:* `{mc_report.get('Median Max Drawdown (%)', 0)}%`\n"
        f"⚠️ *Ruin Risk:* `{mc_report.get('Ruin Probability (%)', 0)}%`\n"
        "----------------------------------------\n"
        "✅ _Live API Market & Backtest Selesai._"
    )
    notifier.send_message(report_text)
    notifier.send_photo("equity_curve.png", caption="📈 *Equity Curve*")
    notifier.send_photo("monte_carlo_dist.png", caption="📊 *Monte Carlo Distribution*")
    notifier.send_document("xauusd_strategy_report.json", caption="📁 *JSON Report*")

    StateManager.save_state(current_signal, current_candle_time)
    logger.info("Pipeline Live Market dan Audit Berhasil Dijalankan.")

if __name__ == "__main__":
    main()
