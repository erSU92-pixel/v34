# Setup XAUUSD APEX V38 — Live Trading

## 1. Prasyarat
- Akun OANDA (practice dulu, live setelah lulus paper)
- Bot Telegram (token) + chat_id
- GitHub Gist (untuk state persisten) + PAT `gist` scope

## 2. GitHub Secrets
Set di `Settings → Secrets and variables → Actions`:

| Secret | Keterangan |
|---|---|
| `OANDA_API_TOKEN` | Token OANDA v20 (practice/live) |
| `OANDA_ACCOUNT_ID` | e.g. `101-001-1234567-001` |
| `TELEGRAM_TOKEN` | Token bot |
| `TELEGRAM_CHAT_ID` | Chat ID tujuan |
| `GIST_ID` | Gist untuk state (buat 1 gist dengan file `xauusd_v38_state.json` = `{}`) |
| `GIST_TOKEN` | PAT dengan scope `gist` |
| `CONFIRM_LIVE` | **WAJIB** `I_UNDERSTAND_THE_RISK` hanya untuk `oanda-live` |

## 3. Alur Aman: Paper → Practice → Live

### Fase A — Paper (default, 2–4 minggu)
