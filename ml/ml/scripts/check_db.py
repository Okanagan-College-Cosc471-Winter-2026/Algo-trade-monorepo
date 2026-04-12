#!/usr/bin/env python3
from sqlalchemy import create_engine, text

engine = create_engine('postgresql://mluser:mlpassword@localhost:5432/market_data')

with engine.connect() as conn:
    # Check tables
    result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'market'"))
    tables = [row[0] for row in result]
    print(f"Tables in market schema: {tables}")
    
    # Check stocks
    result = conn.execute(text("SELECT symbol, name FROM market.stocks LIMIT 5"))
    print("\nSample stocks:")
    for row in result:
        print(f"  {row[0]}: {row[1]}")
    
    # Check data volume
    result = conn.execute(text("SELECT COUNT(*) as total, COUNT(DISTINCT symbol) as stocks, MIN(date)::text as earliest, MAX(date)::text as latest FROM market.daily_prices"))
    row = result.fetchone()
    print(f"\nData statistics:")
    print(f"  Total records: {row[0]}")
    print(f"  Number of stocks: {row[1]}")
    print(f"  Date range: {row[2]} to {row[3]}")
