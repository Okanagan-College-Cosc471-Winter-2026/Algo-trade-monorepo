#!/usr/bin/env python3
"""Quick script to check data granularity and structure."""

from sqlalchemy import create_engine, text
import pandas as pd

engine = create_engine('postgresql://mluser:mlpassword@localhost:5432/market_data')

with engine.connect() as conn:
    # Get sample data for one stock to check time intervals
    query = text("""
        SELECT symbol, date, close 
        FROM market.daily_prices 
        WHERE symbol = (SELECT symbol FROM market.stocks LIMIT 1)
        ORDER BY date 
        LIMIT 50
    """)
    
    df = pd.read_sql(query, conn)
    
    print("Sample data:")
    print(df.head(10))
    print(f"\nTotal records: {len(df)}")
    
    # Check time differences
    df['date'] = pd.to_datetime(df['date'])
    df['time_diff'] = df['date'].diff()
    
    print("\nTime intervals between records:")
    print(df['time_diff'].value_counts().head(10))
    
    # Check if it's intraday or daily
    avg_diff = df['time_diff'].mean()
    print(f"\nAverage time difference: {avg_diff}")
    
    if avg_diff < pd.Timedelta(hours=1):
        print("✓ Confirmed: Intraday data (sub-hourly)")
        bars_per_day = pd.Timedelta(days=1) / avg_diff
        print(f"  Estimated bars per day: ~{int(bars_per_day)}")
    else:
        print("✓ Daily data detected")
