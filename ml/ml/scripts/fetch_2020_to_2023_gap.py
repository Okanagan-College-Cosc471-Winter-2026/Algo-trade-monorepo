import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

load_dotenv('/home/cosc-admin/the-project-maverick/ml/.env')
FMP_API_KEY = os.getenv("FMP_API_KEY")

if not FMP_API_KEY:
    raise ValueError("Please set FMP_API_KEY in your .env file")

# All 29 Tickers from the dataset
TICKERS = ['AAPL', 'AMD', 'AMZN', 'BA', 'BABA', 'BAC', 'C', 'CSCO', 'CVX', 
           'DIS', 'F', 'GE', 'GOOGL', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 
           'MCD', 'META', 'MSFT', 'NFLX', 'NVDA', 'PFE', 'T', 'TSLA', 
           'VZ', 'WMT', 'XOM']

# EXACT bounds of the missing gap for fmp_historical_5min extended hours
START_DATE = "2020-07-17" 
END_DATE = "2023-06-30"

DATA_DIR = "/home/cosc-admin/the-project-maverick/ml/data/fmp_gap_2020_to_2023"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_fmp_window(ticker):
    filepath = os.path.join(DATA_DIR, f"{ticker}.csv")
    
    print(f"Fetching missing early history for {ticker} from {START_DATE} to {END_DATE} with extended hours...")
    
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    all_new_data = []
    current_start = start_dt
    
    # Needs to be chunked as FMP has date range limitations on 5min chart
    while current_start <= end_dt:
        current_end = current_start + timedelta(days=25)
        if current_end > end_dt:
            current_end = end_dt
            
        str_start = current_start.strftime("%Y-%m-%d")
        str_end = current_end.strftime("%Y-%m-%d")
        
        print(f"  Chunk: {str_start} to {str_end}")
        
        # KEY: `extended=true` includes 4:00 AM to 8:00 PM
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/5min/{ticker}?from={str_start}&to={str_end}&extended=true&apikey={FMP_API_KEY}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data:
                df_chunk = pd.DataFrame(data)
                df_chunk = df_chunk.rename(columns={"date": "date", "open": "open", "low": "low", "high": "high", "close": "close", "volume": "volume"})
                df_chunk = df_chunk[['date', 'open', 'low', 'high', 'close', 'volume']]
                df_chunk['date'] = pd.to_datetime(df_chunk['date'])
                all_new_data.append(df_chunk)
            
            time.sleep(0.5) # Prevent rate limiting
            
        except requests.exceptions.RequestException as e:
            print(f"  Error downloading chunk: {e}")
            
        if current_end >= end_dt:
            break
            
        current_start = current_end + timedelta(days=1)
        
    if not all_new_data:
        print(f"No data returned for {ticker}.")
        return False

    df_new = pd.concat(all_new_data, ignore_index=True)
    df_new = df_new.drop_duplicates(subset=['date'], keep='last')
    df_new = df_new.sort_values(by='date', ascending=True)
    
    df_new.to_csv(filepath, index=False)
    print(f"Saved {ticker}! Fetched {len(df_new)} rows spanning the targeted gap.\n")
    return True

if __name__ == "__main__":
    print(f"Target directory: {DATA_DIR}")
    count = 0
    for tick in TICKERS:
        if fetch_fmp_window(tick):
            count += 1
    print(f"Done processing {count} / {len(TICKERS)} tickers.")
