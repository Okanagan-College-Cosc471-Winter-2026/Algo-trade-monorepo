import os
import io
import time
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path

# Use relative path for .env
ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / '.env')

DB_USER = os.getenv("POSTGRES_USER", "appuser")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "changeme")
DB_NAME = os.getenv("POSTGRES_DB", "algotrade")
DB_HOST = os.getenv("POSTGRES_SERVER", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5433")

engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

DATA_DIR = ROOT / "ml/data/fmp_historical_5min"

def get_active_tickers():
    with engine.connect() as conn:
        res = conn.execute(text("SELECT symbol FROM market.stocks WHERE is_active = true ORDER BY symbol"))
        return [row[0] for row in res]

def clean_and_import():
    t0 = time.time()
    tickers = get_active_tickers()
    if not tickers:
        print("No active tickers found in database.")
        return

    with engine.connect() as conn:
        print("Cleaning previous market schema...")
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS market;"))
        # We don't drop every ticker table individually anymore if we have 500+
        # But we will handle it in the loop
            
        conn.execute(text('DROP TABLE IF EXISTS market.daily_prices CASCADE;'))
        # We don't drop market.stocks here because we just read from it, 
        # but the original script wanted to replace it.
        # Actually, if we are importing new data, we might want to update it.
        conn.commit()
    
    print(f"\nStarting high-performance COPY ingestion for {len(tickers)} tickers...")
    import psycopg2
    # Open raw psycopg2 connection for COPY command
    raw_conn = engine.raw_connection()
    cur = raw_conn.cursor()

    count = 0
    total = len(tickers)
    
    for ticker in tickers:
        filepath = DATA_DIR / f"{ticker}.csv"
        if not filepath.exists():
            continue
            
        print(f"[{count+1}/{total}] Loading {ticker}...", end="", flush=True)
        # We need to create the table structure first:
        try:
            df_head = pd.read_csv(filepath, nrows=0)
            df_head.to_sql(ticker, engine, schema='market', if_exists='replace', index=False)
            
            # Now use copy_expert to insert rows instantly
            with open(filepath, 'r') as f:
                cur.copy_expert(f'COPY market."{ticker}" FROM STDIN WITH CSV HEADER', f)
            raw_conn.commit()
            print(f" done.")
            count += 1
        except Exception as e:
            print(f" failed: {e}")
        
    cur.close()
    raw_conn.close()
    
    t1 = time.time()
    print(f"\nSuccessfully processed {count} tickers in {round(t1-t0, 1)} seconds!")

if __name__ == "__main__":
    clean_and_import()
