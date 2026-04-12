import os
import io
import time
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv('/home/cosc-admin/the-project-maverick/.env')

DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "changethis")
DB_NAME = os.getenv("POSTGRES_DB", "app")

engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@localhost:5432/{DB_NAME}')

DATA_DIR = "/home/cosc-admin/the-project-maverick/ml/data/fmp_historical_5min"

TICKERS = ['AAPL', 'AMD', 'AMZN', 'BA', 'BABA', 'BAC', 'C', 'CSCO', 'CVX', 
           'DIS', 'F', 'GE', 'GOOGL', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 
           'MCD', 'META', 'MSFT', 'NFLX', 'NVDA', 'PFE', 'T', 'TSLA', 
           'VZ', 'WMT', 'XOM']

def clean_and_import():
    t0 = time.time()
    with engine.connect() as conn:
        print("Cleaning previous market schema...")
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS market;"))
        for ticker in TICKERS:
            conn.execute(text(f'DROP TABLE IF EXISTS market."{ticker}" CASCADE;'))
            
        conn.execute(text('DROP TABLE IF EXISTS market.daily_prices CASCADE;'))
        conn.execute(text('DROP TABLE IF EXISTS market.stocks CASCADE;'))
        conn.commit()
    
    # Re-create stocks registry with all columns the backend ORM expects
    stocks_df = pd.DataFrame({
        'symbol':    TICKERS,
        'name':      TICKERS,
        'sector':    None,
        'industry':  None,
        'currency':  'USD',
        'exchange':  None,
        'is_active': True,
    })
    stocks_df.to_sql('stocks', engine, schema='market', if_exists='replace', index=False)

    print("\nStarting high-performance COPY ingestion...")
    import psycopg2
    # Open raw psycopg2 connection for COPY command
    raw_conn = engine.raw_connection()
    cur = raw_conn.cursor()

    count = 0
    total = len(TICKERS)
    
    for ticker in TICKERS:
        filepath = os.path.join(DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(filepath):
            continue
            
        print(f"[{count+1}/{total}] Loading {ticker}...", end="", flush=True)
        # We need to create the table structure first:
        df_head = pd.read_csv(filepath, nrows=0)
        df_head.to_sql(ticker, engine, schema='market', if_exists='replace', index=False)
        
        # Now use copy_expert to insert 150,000+ rows instantly
        with open(filepath, 'r') as f:
            cur.copy_expert(f'COPY market."{ticker}" FROM STDIN WITH CSV HEADER', f)
        raw_conn.commit()
        
        print(f" done.")
        count += 1
        
    cur.close()
    raw_conn.close()
    
    t1 = time.time()
    print(f"\nSuccessfully loaded 4.3 million FMP rows in {round(t1-t0, 1)} seconds!")

if __name__ == "__main__":
    clean_and_import()
