"""
supabase_client.py — DEBUG MODE (Mencetak URL ke Log)
=====================================================
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    from supabase import create_client, Client
except ImportError:
    Client = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _log(level: int, msg: str, **kw) -> None:
    lg = logging.getLogger("xauusd")
    rec = lg.makeRecord(lg.name, level, "", 0, msg, (), None)
    rec.extra_data = kw
    lg.handle(rec)


class SupabaseClient:
    def __init__(self) -> None:
        raw_url = (os.getenv("SUPABASE_URL") or "").strip()
        
        clean_url = raw_url.rstrip("/")
        if clean_url.endswith("/rest/v1"):
            clean_url = clean_url.replace("/rest/v1", "")
        if clean_url and not clean_url.startswith("http"):
            clean_url = f"https://{clean_url}"
            
        self.url = clean_url
        self.key = (os.getenv("SUPABASE_KEY") or "").strip()
        self.table = os.getenv("SUPABASE_TABLE", "signals")
        self.enabled = bool(self.url and self.key and Client)
        self.client = None
        
        # --- TAMBAHAN DEBUG ---
        print(f"DEBUG URL YANG DIBACA BOT: {self.url}")
        print(f"DEBUG KEY YANG DIBACA BOT: {self.key[:15]}...")
        
        if self.enabled:
            try:
                self.client = create_client(self.url, self.key)
            except Exception as e:
                _log(logging.ERROR, "supabase_init_fail", err=str(e)[:100])
                self.enabled = False

    def insert(self, row: Dict[str, Any]) -> bool:
        if not self.enabled or not self.client:
            return False
        try:
            self.client.table(self.table).insert(row).execute()
            return True
        except Exception as e:
            _log(logging.ERROR, "supabase_insert_fail", err=str(e)[:200])
            return False

    def fetch_recent_r(self, limit: int = 50) -> Optional[list]:
        if not self.enabled or not self.client:
            return None
        try:
            res = (
                self.client.table(self.table)
                .select("r_multiple")
                .in_("action", ["BUY", "SELL"])
                .not_.is_("r_multiple", "null")
                .order("ts", desc=True)
                .limit(limit)
                .execute()
            )
            return [float(x["r_multiple"]) for x in res.data if x.get("r_multiple") is not None]
        except Exception:
            return None


SUPABASE = SupabaseClient()


def try_insert_signal(
    action: str, spot: float, score: float, regime: str,
    kelly_frac: float, lot: float, risk_usd: float,
    sl: float, tp1: float, tp2: float, notes: str,
) -> bool:
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "spot": round(spot, 2),
        "score": round(score, 1),
        "regime": regime,
        "kelly_fraction": round(kelly_frac, 4),
        "lot_size": lot,
        "risk_usd": risk_usd,
        "sl": sl, "tp1": tp1, "tp2": tp2,
        "notes": notes[:500],
        "r_multiple": None,
    }
    ok = SUPABASE.insert(row)
    _log(logging.INFO if ok else logging.WARNING,
         "supabase_insert", ok=ok, action=action, score=score)
    return ok
