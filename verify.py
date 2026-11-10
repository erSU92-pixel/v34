"""verify.py — cek semua file wajib + import sanity."""
import os
import sys

REQUIRED = [
    "bot.py", "optimizer.py", "news_calendar.py", "fuzzy_regime.py",
    "kelly.py", "supabase_client.py", "requirements.txt",
]
OPTIONAL = ["params.json", ".env", "supabase_schema.sql"]

print("File check:")
missing = []
for f in REQUIRED:
    ok = os.path.exists(f)
    print(f"  {'✅' if ok else '❌'} {f}")
    if not ok:
        missing.append(f)

for f in OPTIONAL:
    ok = os.path.exists(f)
    print(f"  {'✅' if ok else '⚪'} {f} (opsional)")

if missing:
    print(f"\n❌ MISSING FILES: {missing}")
    print("Push these files to GitHub before running workflow.")
    sys.exit(1)

print("\nImport sanity:")
try:
    from news_calendar import NewsResult, check_news_guard, describe_next_events
    from fuzzy_regime import membership, blend_params, dominant
    from kelly import kelly_fraction
    from supabase_client import SUPABASE
    print("  ✅ news_calendar")
    print("  ✅ fuzzy_regime")
    print("  ✅ kelly")
    print(f"  ✅ supabase_client (enabled={SUPABASE.enabled})")
except Exception as e:
    print(f"  ❌ Import failed: {e}")
    sys.exit(1)

# Test fuzzy blend with dummy params
print("\nFuzzy blend sanity:")
from fuzzy_regime import membership, blend_params
test_params = {
    "TRENDING":   {"threshold": 65.0, "sl_mult": 1.5, "tp1_mult": 2.0, "tp2_mult": 3.0, "max_hold": 72},
    "WEAK_TREND": {"threshold": 72.0, "sl_mult": 1.5, "tp1_mult": 1.5, "tp2_mult": 2.5, "max_hold": 48},
    "RANGING":    {"threshold": 78.0, "sl_mult": 1.0, "tp1_mult": 1.0, "tp2_mult": 2.0, "max_hold": 24},
}
for adx_test in (15, 22, 25, 30, 40):
    w = membership(adx_test)
    blended = blend_params(w, test_params)
    dom = max(w, key=w.get)
    print(f"  ADX={adx_test:<3} → {dom:<10} weights={w}  threshold_blend={blended['threshold']:.1f}")

print("\n✅ All checks passed")
