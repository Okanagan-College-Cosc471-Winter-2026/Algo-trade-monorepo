-- Bridge 1: DW API reads from historical.market_data_5m
-- Collector writes to core_dbms.market_data_5m
CREATE OR REPLACE VIEW historical.market_data_5m AS
SELECT
    market_data_id,
    symbol,
    ts,
    open,
    high,
    low,
    close,
    volume,
    asset_type,
    source,
    created_at
FROM core_dbms.market_data_5m;

-- Bridge 2: Maverick reads from ml.market_data_15m
-- DW API writes to dw.market_data_15m
CREATE OR REPLACE VIEW ml.market_data_15m AS
SELECT * FROM dw.market_data_15m;
