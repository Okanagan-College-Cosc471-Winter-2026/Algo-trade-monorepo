import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")

if not FMP_API_KEY:
    raise ValueError("Please set FMP_API_KEY in your .env file")

TICKERS = ['AAPL', 'AMD', 'AMZN', 'BA', 'BABA', 'BAC', 'C', 'CSCO', 'CVX', 
           'DIS', 'F', 'GE', 'GOOGL', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 
           'MCD', 'META', 'MSFT', 'NFLX', 'NVDA', 'PFE', 'T', 'TSLA', 
           'VZ', 'WMT', 'XOM']

START_DATE = "2020-07-01"
END_DATE = "2023-06-30"

OUTPUT_DIR = "/home/cosc-admin/the-project-maverick/ml/data/fmp_historical_5min"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_fmp_data(ticker, start_date, end_date):
    """Download 5-minute data from FMP API in chunks."""
    print(f"Downloading historical data for {ticker} from {start_date} to {end_date}...")
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    all_data = []
    current_start = start_dt
    
    while current_start <= end_dt:
        current_end = current_start + timedelta(days=25)
        if current_end > end_dt:
            current_end = end_dt
            
        str_start = current_start.strftime("%Y-%m-%d")
        str_end = current_end.strftime("%Y-%m-%d")
        
        print(f"  Fetching {str_start} to {str_end}")
        
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/5min/{ticker}?from={str_start}&to={str_end}&apikey={FMP_API_KEY}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data:
                df = pd.DataFrame(data)
                df = df.rename(columns={"date": "date", "open": "open", "low": "low", "high": "high", "close": "close", "volume": "volume"})
                df = df[['date', 'open', 'low', 'high', 'close', 'volume']]
                all_data.append(df)
            else:
                pass
            
            # Be nice to the API
            time.sleep(0.5)
            
        except requests.exceptions.RequestException as e:
            print(f"  Error downloading chunk: {e}")
            
        if current_end == end_dt:
            break
            
        current_start = current_end + timedelta(days=1)
        
    if not all_data:
        print(f"No data returned for {ticker} across all chunks")
        return False
        
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by date ascending (oldest to newest)
    final_df['date'] = pd.to_datetime(final_df['date'])
    final_df = final_df.sort_values('date').drop_duplicates(subset=['date'])
    
    output_file = os.path.join(OUTPUT_DIR, f"{ticker}.csv")
    final_df.to_csv(output_file, index=False)
    print(f"Saved {len(final_df)} total rows to {output_file}")
    return True

def main():
    success_count = 0
    for ticker in TICKERS:
        if download_fmp_data(ticker, START_DATE, END_DATE):
            success_count += 1
            
    print(f"\nDownload complete! Successfully downloaded data for {success_count}/{len(TICKERS)} tickers.")

if __name__ == "__main__":
    main()
