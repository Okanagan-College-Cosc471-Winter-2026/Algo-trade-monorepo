-- market.stocks (dimension table for tradable instruments)
CREATE TABLE IF NOT EXISTS market.stocks (
    symbol    VARCHAR(10)  PRIMARY KEY,
    name      VARCHAR(255),
    sector    VARCHAR(100),
    industry  VARCHAR(100),
    exchange  VARCHAR(50),
    currency  VARCHAR(10)  DEFAULT 'USD',
    is_active BOOLEAN      DEFAULT true
);

-- market.daily_prices (fact table for daily OHLC + volume)
CREATE TABLE IF NOT EXISTS market.daily_prices (
    id             SERIAL PRIMARY KEY,
    symbol         VARCHAR(10) REFERENCES market.stocks(symbol),
    date           DATE        NOT NULL,
    open           FLOAT,
    high           FLOAT,
    low            FLOAT,
    close          FLOAT,
    volume         INTEGER,
    previous_close FLOAT,
    change         FLOAT,
    change_pct     FLOAT,
    UNIQUE (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_daily_prices_symbol ON market.daily_prices (symbol);
CREATE INDEX IF NOT EXISTS idx_daily_prices_date   ON market.daily_prices (date);
