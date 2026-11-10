"""
optimizer.py — v7.0 Fuzzy-regime walk-forward optimizer.
======================================================
Backtest simplifies to crisp regimes (bucket) for param grid,
then bot.py applies fuzzy blend at inference.

Gate tambahan: OOS win-rate & PF harus cukup agar Kelly tidak collapse.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from bot import (VERSION, BotConfig, adx, atr, ema, rsi, sma,
                 get_logger, log_event, REGIMES)
from kelly import kelly_fraction


LOG = get_logger("xauusd_optimizer")
CRISP_REGIMES = ("TRENDING", "WEAK_TREND", "RANGING")


def _crisp_regime(adx_val: float) -> str:
    if adx_val >= 25.0: return "TRENDING"
    if adx_val >= 20.0: return "WEAK_TREND"
    return "RANGING"


# ============================================================================
# DATA
# ============================================================================
def fetch_long_history(period="2y") -> pd.DataFrame:
    tk = yf.Ticker("GC=F")
    raw = tk.history(period=period, interval="1h")
    if raw.empty: raise RuntimeError("no data")
    raw = raw.reset_index()
    tcol = "Datetime" if "Datetime" in raw.columns else ("Date" if "Date" in raw.columns else raw.columns[0])
    df = raw.rename(columns={tcol:"time","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume",1000.0), errors="coerce").fillna(1000.0)
    df = df.dropna(subset=["open","high","low","close"]).sort_values("time").reset_index(drop=True)
    log_event(event="data_loaded", bars=len(df), start=str(df["time"].iloc[0]), end=str(df["time"].iloc[-1]))
    return df


# ============================================================================
# PRECOMPUTE
# ============================================================================
def _resample_m15_bull(h1):
    m15 = h1.set_index("time").resample("15min").ffill()
    bull = (ema(m15["close"],9) > ema(m15["close"],21)).astype(int)
    return bull.reindex(h1["time"], method="ffill").fillna(0).astype(int).values


def precompute(df, warmup=250):
    close = df["close"]
    highs, lows = df["high"].values, df["low"].values
    opens, closes = df["open"].values, df["close"].values
    vols = df["volume"].values

    sma_v = {p: sma(close,p).values for p in (5,10,20,50,100,200)}
    e9_v, e21_v = ema(close,9).values, ema(close,21).values
    e12_v, e26_v = ema(close,12).values, ema(close,26).values
    e50_v, e200_v = ema(close,50).values, ema(close,200).values
    rsi_v = rsi(close,14).values
    adx_v, p_di_v, m_di_v = adx(df,14)
    adx_v, p_di_v, m_di_v = adx_v.values, p_di_v.values, m_di_v.values
    a14_v = atr(df,14).values
    vol_ma20 = df["volume"].rolling(20).mean().values
    poc_proxy = close.rolling(50).median().values
    m15_bull = _resample_m15_bull(df)

    n = len(df)
    raw_vote = np.zeros(n, dtype=int)
    forensic = np.zeros(n)
    atr_out = np.zeros(n)
    adx_out = np.zeros(n)

    for i in range(warmup, n):
        c = closes[i]
        v = 0
        for p in (5,10,20,50,100,200):
            v += 1 if c > sma_v[p][i] else -1
        v += 1 if e9_v[i] > e21_v[i] else -1
        v += 1 if e12_v[i] > e26_v[i] else -1
        v += 1 if e50_v[i] > e200_v[i] else -1
        r = rsi_v[i]
        v += 1 if r>50 else -1
        v += 1 if r<70 else -1
        v += 1 if r>30 else -1
        v += 1 if p_di_v[i] > m_di_v[i] else -1
        v += 1 if adx_v[i] > 20 else -1
        v += 1 if c > closes[i-1] else -1
        v += 1 if c > closes[i-5] else -1
        v += 1 if c > closes[i-10] else -1
        for nn in (5,10):
            hi_n = highs[i-nn+1:i+1].max()
            lo_n = lows[i-nn+1:i+1].min()
            pos = (c-lo_n)/max(hi_n-lo_n,1e-10)
            v += 1 if pos > 0.5 else -1
        v += 1 if vols[i] > vol_ma20[i] else -1
        v += 1 if c > poc_proxy[i] else -1
        raw_vote[i] = v

        hi, lo, op, cl = highs[i], lows[i], opens[i], closes[i]
        rng = max(hi-lo, 0.01)
        wick = (max(hi-max(op,cl), min(op,cl)-lo)/rng)*100
        prev_hi = highs[i-20:i].max()
        prev_lo = lows[i-20:i].min()
        sweep = 35 if (hi>prev_hi and cl<prev_hi) or (lo<prev_lo and cl>prev_lo) else 0
        wick_pts = 20 if wick > 40 else 0
        vr = vols[i]/(vols[i-20:i].mean()+1e-10)
        vol_pts = 15 if vr > 1.2 else 0
        forensic[i] = min(30+sweep+wick_pts+vol_pts, 100)
        atr_out[i] = a14_v[i]
        adx_out[i] = adx_v[i]

    voter_agree = (21.0+np.abs(raw_vote))/42.0
    regimes = np.array([_crisp_regime(x) for x in adx_out])

    return pd.DataFrame({
        "time": df["time"].values, "open": opens, "high": highs, "low": lows, "close": closes,
        "raw_vote": raw_vote, "voter_agree": voter_agree, "forensic": forensic,
        "atr": atr_out, "adx": adx_out, "regime": regimes,
        "ema200": e200_v, "m15_bull": m15_bull,
    }).iloc[warmup:].reset_index(drop=True)


# ============================================================================
# SIGNAL & SIM
# ============================================================================
def apply_signals(pre, params):
    combined = (pre["voter_agree"]*0.6 + pre["forensic"]/100.0*0.4)*100
    direction = np.where(pre["raw_vote"] > 0, "BUY", "SELL")
    ema_ok = ((direction=="BUY")&(pre["close"]>pre["ema200"])) | ((direction=="SELL")&(pre["close"]<pre["ema200"]))
    micro_ok = ((direction=="BUY")&(pre["m15_bull"]==1)) | ((direction=="SELL")&(pre["m15_bull"]==0))
    regime = pre["regime"].values
    threshold = np.array([params[r].get("threshold",70.0) for r in regime])
    go = (combined.values >= threshold) & ema_ok & micro_ok
    return np.where(go, direction, "WAIT")


def simulate(pre, signal, params):
    opens, highs, lows = pre["open"].values, pre["high"].values, pre["low"].values
    atrs, regimes = pre["atr"].values, pre["regime"].values
    n = len(pre)
    trades = []
    i = 0
    while i < n-1:
        d = signal[i]
        if d == "WAIT": i += 1; continue
        p = params[regimes[i]]
        sl_d = max(atrs[i]*p.get("sl_mult",1.5), 1.0)
        tp1_m = p.get("tp1_mult",1.5); tp2_m = p.get("tp2_mult",2.5)
        mh = int(p.get("max_hold",48))
        entry = opens[i+1]
        if d == "BUY":
            sl, tp1, tp2 = entry-sl_d, entry+sl_d*tp1_m, entry+sl_d*tp2_m
        else:
            sl, tp1, tp2 = entry+sl_d, entry-sl_d*tp1_m, entry-sl_d*tp2_m
        et, R, bars = "TIME", 0.0, mh
        for j in range(i+1, min(i+1+mh, n)):
            if d == "BUY":
                if lows[j] <= sl: et,R,bars = "SL",-1.0,j-i; break
                if highs[j] >= tp2: et,R,bars = "TP2",tp2_m,j-i; break
                if highs[j] >= tp1: et,R,bars = "TP1",tp1_m,j-i; break
            else:
                if highs[j] >= sl: et,R,bars = "SL",-1.0,j-i; break
                if lows[j] <= tp2: et,R,bars = "TP2",tp2_m,j-i; break
                if lows[j] <= tp1: et,R,bars = "TP1",tp1_m,j-i; break
        trades.append({"time":pre["time"].iloc[i+1],"dir":d,"R":R,"type":et,"bars":bars,"regime":regimes[i]})
        i += bars+1
    return pd.DataFrame(trades)


# ============================================================================
# METRICS
# ============================================================================
def metrics(trades):
    if trades.empty:
        return {"n":0,"expectancy":0.0,"wr":0.0,"sharpe":-999.0,"total_r":0.0,"max_dd":0.0,"pf":0.0}
    R = trades["R"].values
    n = len(R); wins = R>0; losses = R<0
    eq = np.cumsum(R)
    return {
        "n": n, "expectancy": float(R.mean()),
        "wr": float(wins.mean()*100),
        "sharpe": float(R.mean()/(R.std()+1e-6)*np.sqrt(n)),
        "total_r": float(eq[-1]),
        "max_dd": float((eq-np.maximum.accumulate(eq)).min()),
        "pf": float(R[wins].sum()/(abs(R[losses].sum())+1e-6)),
    }


def per_regime_metrics(trades):
    out = {}
    for r in CRISP_REGIMES:
        sub = trades[trades["regime"]==r] if not trades.empty else pd.DataFrame()
        out[r] = metrics(sub) if not sub.empty else {"n":0,"expectancy":0.0,"wr":0.0,"sharpe":-999.0,"total_r":0.0,"max_dd":0.0,"pf":0.0}
    return out


# ============================================================================
# GRID / OPTIMIZE
# ============================================================================
DEFAULT_PARAMS = {"threshold":70.0,"sl_mult":1.5,"tp1_mult":1.5,"tp2_mult":2.5,"max_hold":48}
GRID = {
    "threshold": [65.0,70.0,75.0,80.0],
    "sl_mult":   [1.0,1.5,2.0],
    "tp1_mult":  [1.0,1.5,2.0],
    "tp2_mult":  [2.0,2.5,3.0],
    "max_hold":  [24,48,72],
}
MIN_TRADES_REGIME = 8


def _init_params():
    return {r: dict(DEFAULT_PARAMS) for r in CRISP_REGIMES}


def grid_search_regime(pre, base, target, n_iter=2):
    best = dict(base[target]); best_score = -1e9; best_m = {}
    keys = list(GRID.keys())
    for _ in range(n_iter):
        improved = False
        for combo in product(*GRID.values()):
            cand = dict(zip(keys, combo))
            trial = {r: dict(base[r]) for r in CRISP_REGIMES}
            trial[target] = cand
            tr = simulate(pre, apply_signals(pre, trial), trial)
            sub = tr[tr["regime"]==target] if not tr.empty else pd.DataFrame()
            m = metrics(sub)
            if m["n"] < MIN_TRADES_REGIME: continue
            score = m["sharpe"] + 0.1*m["expectancy"]
            if score > best_score + 1e-6:
                best_score, best, best_m = score, cand, m
                improved = True
        if not improved: break
    return best, best_m


def joint_optimize(pre, n_sweeps=2):
    params = _init_params()
    for _ in range(n_sweeps):
        for r in CRISP_REGIMES:
            best_r, _m = grid_search_regime(pre, params, r, n_iter=1)
            params[r] = best_r
    tr = simulate(pre, apply_signals(pre, params), params)
    return params, metrics(tr)


# ============================================================================
# WALK-FORWARD
# ============================================================================
def walk_forward(pre, n_folds=4, min_train=400):
    n = len(pre); remaining = n-min_train
    if remaining < n_folds*50: raise RuntimeError(f"insufficient data {n}")
    ts = remaining//n_folds
    folds, all_oos = [], []
    for k in range(n_folds):
        tr_end = min_train + k*ts; te_end = min(tr_end+ts, n)
        train = pre.iloc[:tr_end].reset_index(drop=True)
        test = pre.iloc[tr_end:te_end].reset_index(drop=True)
        if len(test) < 100: break
        best, train_m = joint_optimize(train, n_sweeps=2)
        tr = simulate(test, apply_signals(test, best), best)
        test_m = metrics(tr)
        if not tr.empty:
            tr["fold"] = k; all_oos.append(tr)
        folds.append({"fold":k,"test_start":str(test["time"].iloc[0]),
                      "test_end":str(test["time"].iloc[-1]),"test_bars":len(test),
                      "best_params":best,"train_metrics":train_m,
                      "test_metrics":test_m,"test_regime_metrics":per_regime_metrics(tr)})
        log_event(event="fold_done", fold=k, oos_n=test_m["n"], oos_sharpe=round(test_m["sharpe"],2))
    return folds, pd.concat(all_oos, ignore_index=True) if all_oos else pd.DataFrame()


# ============================================================================
# MONTE CARLO
# ============================================================================
def monte_carlo(R, n_sims=1000):
    n = len(R)
    if n < 10: return None
    rng = np.random.default_rng(42)
    finals, dds = np.zeros(n_sims), np.zeros(n_sims)
    for k in range(n_sims):
        s = rng.choice(R, size=n, replace=True)
        eq = np.cumsum(s)
        finals[k] = eq[-1]; dds[k] = (eq-np.maximum.accumulate(eq)).min()
    return {"n_sims":n_sims,"final_R_median":float(np.median(finals)),
            "final_R_p5":float(np.percentile(finals,5)),
            "final_R_p95":float(np.percentile(finals,95)),
            "max_dd_median":float(np.median(dds)),
            "prob_profit":float((finals>0).mean())}


# ============================================================================
# REPORT
# ============================================================================
def format_report(folds, oos, mc, final_params, final_m):
    L = [f"\n{'='*70}",
         f"  XAUUSD v{VERSION} — FUZZY-REGIME WALK-FORWARD",
         f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
         f"{'='*70}"]

    L.append(f"\n{'─'*70}\n  PER-FOLD OOS\n{'─'*70}")
    L.append(f"  {'Fold':<5}{'Start':<12}{'N':>4}{'WR%':>7}{'ExpR':>8}{'Sharpe':>8}{'MaxDD':>8}")
    for f in folds:
        m = f["test_metrics"]
        L.append(f"  {f['fold']:<5}{f['test_start'][:10]:<12}{m['n']:>4}{m['wr']:>7.1f}"
                 f"{m['expectancy']:>+8.3f}{m['sharpe']:>+8.2f}{m['max_dd']:>+8.2f}")

    oos_m = metrics(oos)
    L.append(f"\n{'─'*70}\n  AGGREGATE OOS\n{'─'*70}")
    for k, lbl in [("n","Trades"),("wr","Win %"),("expectancy","Exp (R)"),
                   ("total_r","Total R"),("sharpe","Sharpe"),("max_dd","Max DD"),("pf","PF")]:
        L.append(f"  {lbl:<12}: {oos_m[k]:+.3f}")

    L.append(f"\n{'─'*70}\n  OOS BY REGIME\n{'─'*70}")
    L.append(f"  {'Regime':<12}{'N':>5}{'WR%':>7}{'ExpR':>9}{'Sharpe':>9}{'TotalR':>9}")
    rm = per_regime_metrics(oos)
    for r in CRISP_REGIMES:
        m = rm[r]
        L.append(f"  {r:<12}{m['n']:>5}{m['wr']:>7.1f}{m['expectancy']:>+9.3f}{m['sharpe']:>+9.2f}{m['total_r']:>+9.2f}")

    if mc:
        L.append(f"\n{'─'*70}\n  MONTE CARLO (1000×)\n{'─'*70}")
        L.append(f"  Final R median: {mc['final_R_median']:+.2f} [p5 {mc['final_R_p5']:+.2f} … p95 {mc['final_R_p95']:+.2f}]")
        L.append(f"  Max DD median : {mc['max_dd_median']:+.2f}")
        L.append(f"  P(profit)     : {mc['prob_profit']*100:.1f}%")

    L.append(f"\n{'─'*70}\n  RECOMMENDED PER-REGIME PARAMS\n{'─'*70}")
    for r in CRISP_REGIMES:
        L.append(f"  {r}:")
        for k, v in final_params[r].items():
            L.append(f"    {k:<10}: {v}")

    # Kelly sanity check dari OOS
    L.append(f"\n{'─'*70}\n  KELLY SANITY (OOS R history)\n{'─'*70}")
    if not oos.empty:
        R_arr = oos["R"].values.tolist()
        k_pct, k_reason = kelly_fraction(R_arr, fraction=0.5, cap_pct=3.0)
        L.append(f"  Half-Kelly risk: {k_pct:.2f}%")
        L.append(f"  Reason         : {k_reason}")

    L.append(f"\n{'─'*70}\n  PRE-LIVE GATES\n{'─'*70}")
    gates = [
        ("OOS expectancy ≥ +0.15R", oos_m["expectancy"] >= 0.15),
        ("OOS max DD ≤ 15R", oos_m["max_dd"] >= -15.0),
        ("OOS win rate ≥ 45%", oos_m["wr"] >= 45.0),
        ("OOS trades ≥ 30", oos_m["n"] >= 30),
        ("MC prob_profit ≥ 70%", (mc or {}).get("prob_profit",0) >= 0.70),
        *[(f"{r} OOS trades ≥ 8", rm[r]["n"] >= 8) for r in CRISP_REGIMES],
    ]
    all_ok = True
    for d, ok in gates:
        L.append(f"  [{'✅' if ok else '❌'}] {d}")
        all_ok = all_ok and ok
    L.append(f"\n  VERDICT: {'🚀 SAFE TO DEPLOY' if all_ok else '⛔ DO NOT DEPLOY'}")
    L.append(f"{'='*70}\n")
    return "\n".join(L)


# ============================================================================
# MAIN
# ============================================================================
def main():
    log_event(event="optimizer_start", version=VERSION)
    df = fetch_long_history("2y")
    if len(df) < 800:
        print("Need ≥800 bars"); return 1

    pre = precompute(df, warmup=250)
    log_event(event="precompute_done", bars=len(pre),
              regimes=pre["regime"].value_counts().to_dict())

    folds, oos = walk_forward(pre, n_folds=4, min_train=400)
    if oos.empty:
        print("No OOS trades"); return 1

    mc = monte_carlo(oos["R"].values)
    final_params, final_m = joint_optimize(pre, n_sweeps=2)
    print(format_report(folds, oos, mc, final_params, final_m))

    out = {
        "schema": 2, "version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_bars": len(df), "data_start": str(df["time"].iloc[0]),
        "data_end": str(df["time"].iloc[-1]),
        "per_regime": final_params, "full_metrics": final_m,
        "oos_metrics": metrics(oos), "per_regime_oos": per_regime_metrics(oos),
        "monte_carlo": mc, "folds": folds,
    }
    with open("params.json","w") as f:
        json.dump(out, f, indent=2, default=str)
    log_event(event="params_saved", params=final_params)

    oos_m = metrics(oos); rm = per_regime_metrics(oos)
    gates = (oos_m["expectancy"]>=0.15 and oos_m["max_dd"]>=-15.0
             and oos_m["wr"]>=45.0 and oos_m["n"]>=30
             and (mc or {}).get("prob_profit",0)>=0.70
             and all(rm[r]["n"]>=8 for r in CRISP_REGIMES))
    return 0 if gates else 2


if __name__ == "__main__":
    sys.exit(main())
