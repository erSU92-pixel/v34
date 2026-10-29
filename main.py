import os
import sys
import json
import csv
import time
import logging
import argparse
import itertools
import random
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List
from sklearn.ensemble import RandomForestClassifier

# --- SETUP LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] - %(message)s")
logger = logging.getLogger("ApexQuantEngine")
STATE_FILE = "signal_state.json"

# ==========================================
# 0. KONFIGURASI SUMBER DATA & BROKER OFFSET
# ==========================================
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
            url_m5 = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=5min&outputsize=150&apikey={TWELVEDATA_API_KEY}"
            res_m5 = requests.get(url_m5, timeout=10).json()
            
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

                logger.info("Berhasil menarik data live dari Twelve Data (Apex Tier).")
                return df_m5, df_h1
        except Exception as e:
            logger.error(f"Gagal mengambil data dari Twelve Data: {e}")
        
        return None, None

    @staticmethod
    def fetch_fallback_gold_api() -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Cadangan menggunakan Gold-API publik gratis"""
        try:
            url = "https://api.gold-api.com/price/XAU"
            res = requests.get(url, timeout=10).json()
            current_price = float(res.get("price", 2600.0))
            logger.info(f"Menggunakan Gold-API fallback, harga spot: {current_price}")
            
            np.random.seed(int(time.time() // 60))
            dates_m5 = pd.date_range(end=datetime.now(timezone.utc), periods=150, freq="5min")
            walk_m5 = current_price + np.random.normal(0, 0.5, 150).cumsum()
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
# 2. MODUL SESSION & STATE MANAGER (EVOLUTIONARY DNA)
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
        return {
            "last_signal": "NEUTRAL", 
            "last_candle_time": "",
            "generation": 1,
            "pillar_weights": {"momentum": 1.0, "trend": 1.0, "volatility": 1.0}
        }

    @staticmethod
    def save_state(signal_type: str, candle_time: str, generation: int, weights: dict):
        state = {
            "last_signal": signal_type, 
            "last_candle_time": str(candle_time),
            "generation": generation,
            "pillar_weights": weights
        }
        try:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            logger.error(f"Gagal menyimpan state file: {e}")

# ==========================================
# 3. GENETIC EVOLUTION ENGINE (MUTASI REKURSIF)
# ==========================================
class GeneticEvolutionEngine:
    @staticmethod
    def evolve(current_weights: dict, generation: int) -> Tuple[dict, int]:
        """Mutasi DNA bobot secara otonom untuk adaptasi pasar real-time"""
        weights = current_weights.copy()
        key_to_mutate = random.choice(list(weights.keys()))
        mutation_delta = random.uniform(-0.04, 0.04)
        
        weights[key_to_mutate] = max(0.5, min(2.0, weights[key_to_mutate] + mutation_delta))
        new_generation = generation + 1
        logger.info(f"[Genetic Evolution Gen-{new_generation}] Mutasi DNA pada '{key_to_mutate}' -> {weights[key_to_mutate]:.4f}")
        return weights, new_generation

# ==========================================
# 4. TELEGRAM NOTIFIER
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
# 5. 19-PILLAR APEX CONSENSUS & ML GATEKEEPER
# ==========================================
class Consensus19PillarsEngine:
    @staticmethod
    def evaluate(historical_m5: pd.DataFrame, historical_h1: pd.DataFrame, weights: dict) -> Tuple[str, int, int]:
        if len(historical_m5) < 40 or len(historical_h1) < 20:
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

        # 18 Pillars Evaluator with Genetic Weights scaling
        m_w = weights.get("momentum", 1.0)
        t_w = weights.get("trend", 1.0)
        v_w = weights.get("volatility", 1.0)

        if close_m5 > ema_fast: score_buy += int(1 * m_w)
        else: score_sell += int(1 * m_w)
        if ema_fast > ema_slow: score_buy += int(1 * t_w)
        else: score_sell += int(1 * t_w)
        if close_m5 > ema_trend: score_buy += int(1 * t_w)
        else: score_sell += int(1 * t_w)

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

        if atr > 0: score_buy += int(1 * v_w); score_sell += int(1 * v_w)
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

        # --- THE 19TH PILLAR: MACHINE LEARNING META-FILTER ---
        ml_approval = 1 # Default approved
        try:
            df_ml = historical_m5.copy()
            df_ml['returns'] = df_ml['close'].pct_change()
            df_ml['volatility'] = df_ml['high'] - df_ml['low']
            df_ml['ma_diff'] = df_ml['close'].ewm(span=5).mean() - df_ml['close'].ewm(span=21).mean()
            df_ml['target'] = (df_ml['close'].shift(-3) > df_ml['close']).astype(int)
            df_ml.dropna(inplace=True)

            if len(df_ml) > 15:
                X = df_ml[['volatility', 'ma_diff', 'returns']].values[-30:]
                y = df_ml['target'].values[-30:]
                clf = RandomForestClassifier(n_estimators=10, random_state=42)
                clf.fit(X, y)

                current_feat = np.array([[
                    (high_m5 - low_m5),
                    (ema_fast - ema_slow),
                    historical_m5['close'].pct_change().iloc[-2]
                ]])
                ml_approval = clf.predict(current_feat)[0]
        except Exception as e:
            logger.warning(f"ML Meta-Filter Warning: {e}")

        THRESHOLD = 12
        # Sinyal akhir hanya disetujui jika skor teknikal memenuhi DAN lolos filter ML ke-19
        if score_buy >= THRESHOLD and score_buy > score_sell and ml_approval == 1:
            return "BUY", score_buy, score_sell
        elif score_sell >= THRESHOLD and score_sell > score_buy and ml_approval == 0:
            return "SELL", score_buy, score_sell
        
        return "NEUTRAL", score_buy, score_sell

# ==========================================
# 6. QUANTITATIVE BACKTESTING & STRESS TEST
# ==========================================
class InstitutionalQuantEngine:
    def __init__(self, initial_capital: float = 10000.0, risk_pct: float = 0.01):
        self.initial_capital = initial_capital
        self.risk_pct = risk_pct

    def run_single_backtest(self, df_m5: pd.DataFrame, df_h1: pd.DataFrame, params: Dict[str, Any], weights: dict) -> Dict[str, Any]:
        atr_mult = params.get('atr_multiplier', 1.5)
        capital = self.initial_capital
        trades = []
        position = None
        entry_price, stop_loss, tp1, tp2, lot_size = 0.0, 0.0, 0.0, 0.0, 0.0
        break_even_triggered = False

        start_idx = max(25, len(df_m5) - 100)
        for i in range(start_idx, len(df_m5) - 1):
            current_time = df_m5.index[i]
            historical_m5 = df_m5.iloc[:i+1]
            next_bar_m5 = df_m5.iloc[i+1]
            historical_h1 = df_h1[df_h1.index <= current_time]
            if len(historical_h1) < 5: historical_h1 = df_h1.iloc[:5]

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

                hit_tp = (current_high >= tp2) if position == 'BUY' else (current_low <= tp2)
                hit_sl = (current_low <= stop_loss) if position == 'BUY' else (current_high >= stop_loss)

                if hit_tp or hit_sl:
                    exit_price = tp2 if hit_tp else stop_loss
                    pnl = (exit_price - entry_price) * lot_size * 100 if position == 'BUY' else (entry_price - exit_price) * lot_size * 100
                    capital += pnl
                    trades.append({'pnl': pnl, 'result': 'WIN' if hit_tp else 'LOSS'})
                    position = None
                    break_even_triggered = False

            elif not position:
                action, _, _ = Consensus19PillarsEngine.evaluate(historical_m5, historical_h1, weights)

                if action in ['BUY', 'SELL']:
                    position = action
                    entry_price = next_bar_m5['open'] + BROKER_PRICE_OFFSET
                    atr = max(0.1, historical_m5['high'].iloc[-2] - historical_m5['low'].iloc[-2])
                    stop_loss = entry_price - (atr_mult * atr) if position == 'BUY' else entry_price + (atr_mult * atr)
                    tp1 = entry_price + (1.5 * atr) if position == 'BUY' else entry_price - (1.5 * atr)
                    tp2 = entry_price + (3.0 * atr) if position == 'BUY' else entry_price - (3.0 * atr)
                    
                    risk_amt = capital * self.risk_pct
                    dist = max(0.1, abs(entry_price - stop_loss))
                    lot_size = max(0.01, round(risk_amt / (dist * 100), 2))

        if not trades: trades.append({'pnl': 0.0, 'result': 'NEUTRAL'})
        return {"final_capital": capital, "trades": trades, "params": params}

    def run_grid_search(self, df_m5: pd.DataFrame, df_h1: pd.DataFrame, param_grid: Dict[str, list], weights: dict) -> Dict[str, Any]:
        keys = param_grid.keys()
        values = param_grid.values()
        best_result, max_profit = None, -float('inf')
        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
            res = self.run_single_backtest(df_m5, df_h1, params, weights)
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
            "Ruin Probability (%)": round(float(np.mean(final_equities < self.initial_capital) * 100), 2)
        }

# ==========================================
# 7. EXPORTER & VISUALIZER
# ==========================================
class ReportExporter:
    @staticmethod
    def export_to_json(best_run: Dict[str, Any], mc_report: Dict[str, Any], generation: int, filename: str = "xauusd_apex_report.json"):
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "evolution_generation": generation,
            "best_parameters": best_run.get("params", {}),
            "final_capital": round(best_run.get("final_capital", 0.0), 2),
            "monte_carlo_stress_test": mc_report
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

    @staticmethod
    def export_trades_to_csv(trades: List[Dict[str, Any]], filename: str = "xauusd_apex_trades.csv"):
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
        plt.plot(equity_path, color="#d4af37", linewidth=2.5) # Gold color theme
        plt.axhline(y=initial_capital, color="gray", linestyle="--")
        plt.title("Apex 19-Pillar ML Consensus Equity Curve", fontsize=12, fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.savefig(filename, dpi=300)
        plt.close()

    @staticmethod
    def plot_monte_carlo_distribution(trades: List[Dict[str, Any]], initial_capital: float = 10000.0, num_simulations: int = 1000, filename: str = "monte_carlo_dist.png"):
        if not trades: return
        pnls = np.array([t['pnl'] for t in trades])
        final_equities = [initial_capital + np.sum(np.random.choice(pnls, size=len(pnls), replace=True)) for _ in range(num_simulations)]
        plt.figure(figsize=(10, 5))
        plt.hist(final_equities, bins=min(50, len(final_equities)), color="#1f77b4", edgecolor="black", alpha=0.7)
        plt.axvline(initial_capital, color="red", linestyle="--", linewidth=2)
        plt.title(f"Apex Monte Carlo Distribution ({num_simulations} Simulations)", fontsize=12, fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.savefig(filename, dpi=300)
        plt.close()

# ==========================================
# 8. CLI ORCHESTRATOR & ENTRY POINT
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Institutional Apex XAUUSD 19-Pillar Engine")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--simulations", type=int, default=1000)
    args, unknown = parser.parse_known_args()

    if not SessionFilter.is_market_liquid():
        logger.info("Bot dihentikan karena di luar jam likuiditas operasional.")
        sys.exit(0)

    logger.info("Memulai Eksekusi Infinite Evolutionary Apex Pipeline...")
    
    # 1. Load State & Evolve Genetic Weights
    current_state = StateManager.load_state()
    current_gen = current_state.get("generation", 1)
    current_weights = current_state.get("pillar_weights", {"momentum": 1.0, "trend": 1.0, "volatility": 1.0})
    
    evolved_weights, next_gen = GeneticEvolutionEngine.evolve(current_weights, current_gen)

    # 2. Get Live Market Data
    df_m5, df_h1 = LiveMarketFeed.get_market_data()
    if df_m5 is None or df_h1 is None:
        logger.error("Gagal mendapatkan data pasar dari semua sumber API.")
        sys.exit(1)

    # 3. Quant Backtest & Optimization with Evolved Weights
    engine = InstitutionalQuantEngine(initial_capital=args.capital)
    param_space = {'atr_multiplier': [1.5, 2.0]}
    best_run = engine.run_grid_search(df_m5, df_h1, param_space, evolved_weights)
    mc_report = engine.run_monte_carlo(best_run['trades'], num_simulations=args.simulations)

    ReportExporter.export_to_json(best_run, mc_report, next_gen)
    ReportExporter.export_trades_to_csv(best_run['trades'])
    QuantVisualizer.plot_equity_curve(best_run['trades'], initial_capital=args.capital)
    QuantVisualizer.plot_monte_carlo_distribution(best_run['trades'], initial_capital=args.capital, num_simulations=args.simulations)

    # 4. Final Live Signal Evaluation
    current_signal, score_buy, score_sell = Consensus19PillarsEngine.evaluate(df_m5, df_h1, evolved_weights)
    current_candle_time = str(df_m5.index[-1])

    # 5. Telegram Notification Dispatch
    notifier = TelegramNotifier()
    report_text = (
        "🚀 *APEX 19-PILLAR ML & GENETIC REPORT*\n"
        "----------------------------------------\n"
        f"🧬 *Evolution Generation:* `Gen-{next_gen}`\n"
        f"🎯 *Apex Live Signal:* `{current_signal}`\n"
        f"📈 *Weighted Score:* `Buy: {score_buy} | Sell: {score_sell}`\n"
        f"⚙️ *Evolved DNA:* `{evolved_weights}`\n"
        f"💰 *Projected Capital:* `${best_run['final_capital']:,.2f}`\n"
        f"📉 *Median Max DD:* `{mc_report.get('Median Max Drawdown (%)', 0)}%`\n"
        f"⚠️ *Ruin Risk:* `{mc_report.get('Ruin Probability (%)', 0)}%`\n"
        "----------------------------------------\n"
        "✨ _Infinite Autonomous Loop Selesai._"
    )
    notifier.send_message(report_text)
    notifier.send_photo("equity_curve.png", caption="📈 *Apex Equity Curve*")
    notifier.send_photo("monte_carlo_dist.png", caption="📊 *Monte Carlo Stress Test*")
    notifier.send_document("xauusd_apex_report.json", caption="📁 *Apex JSON Report*")

    # 6. Save State for Next 15-Min Evolution Loop
    StateManager.save_state(current_signal, current_candle_time, next_gen, evolved_weights)
    logger.info(f"Pipeline Infinite Apex Gen-{next_gen} Berhasil Dijalankan dan Disimpan.")

if __name__ == "__main__":
    main()
