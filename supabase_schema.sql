-- Run sekali di Supabase SQL Editor (free tier aman)
CREATE TABLE IF NOT EXISTS signals (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    action TEXT NOT NULL CHECK (action IN ('BUY','SELL','WAIT')),
    spot NUMERIC(12,2),
    score NUMERIC(5,1),
    regime TEXT,
    kelly_fraction NUMERIC(6,4),
    lot_size NUMERIC(6,2),
    risk_usd NUMERIC(10,2),
    sl NUMERIC(12,2),
    tp1 NUMERIC(12,2),
    tp2 NUMERIC(12,2),
    r_multiple NUMERIC(6,3),       -- diisi manual saat trade closed
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals (ts DESC);
CREATE INDEX IF NOT EXISTS idx_signals_action ON signals (action, ts DESC);

-- View untuk Grafana (dashboard siap query)
CREATE OR REPLACE VIEW v_signal_summary AS
SELECT
    date_trunc('day', ts) AS day,
    action,
    COUNT(*) AS n,
    AVG(score) AS avg_score,
    AVG(kelly_fraction) AS avg_kelly,
    SUM(risk_usd) AS total_risk,
    SUM(r_multiple) AS total_r
FROM signals
GROUP BY 1, 2
ORDER BY 1 DESC;

-- RLS: deny all (service key bypass RLS)
ALTER TABLE signals ENABLE ROW LEVEL SECURITY;
-- No policies = public cannot read/write. Only service_role key can.
