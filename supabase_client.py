"""
supabase_client.py — REST insert via requests (no SDK dependency).
===============================================================
Free tier safe: single insert per signal, no bulk, no reads.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

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
        self.url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
        self.key = os.getenv("SUPABASE_KEY") or ""
        self.table = os.getenv("SUPABASE_TABLE", "signals")
        self.enabled = bool(self.url and self.key)
        self._session = requests.Session()
        self._session.headers.update({
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        })

    def insert(self, row: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        try:
            r = self._session.post(
                f"{self.url}/rest/v1/{self.table}",
                json=row,
                timeout=10,
            )
            if r.status_code not in (200, 201, 204):
                _log(logging.ERROR, "supabase_insert_fail",
                     code=r.status_code, body=r.text[:200])
                return False
            return True
        except Exception as e:
            _log(logging.WARNING, "supabase_exception", err=str(e)[:100])
            return False

    def fetch_recent_r(self, limit: int = 50) -> Optional[list]:
        """Fetch last N closed trades for Kelly stats. Returns list of R floats."""
        if not self.enabled:
            return None
        try:
            r = self._session.get(
                f"{self.url}/rest/v1/{self.table}",
                params={
                    "select": "r_multiple",
                    "action": "in.(BUY,SELL)",
                    "r_multiple": "not.is.null",
                    "order": "ts.desc",
                    "limit": str(limit),
                },
                timeout=10,
            )
            if r.status_code != 200:
                return None
            rows = r.json()
            return [float(x["r_multiple"]) for x in rows if x.get("r_multiple") is not None]
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
