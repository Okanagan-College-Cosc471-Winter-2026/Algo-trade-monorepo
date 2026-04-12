import pandas as pd
import os

TICKERS = ['AAPL', 'AMD', 'AMZN', 'BA', 'BABA', 'BAC', 'C', 'CSCO', 'CVX', 
           'DIS', 'F', 'GE', 'GOOGL', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 
           'MCD', 'META', 'MSFT', 'NFLX', 'NVDA', 'PFE', 'T', 'TSLA', 
           'VZ', 'WMT', 'XOM']

GAP_DIR = "/home/cosc-admin/the-project-maverick/ml/data/fmp_gap_2020_to_2023"
GDRIVE_DIR = "/home/cosc-admin/the-project-maverick/ml/data/July 1, 2023 to Oct 24, 2025 /FMP API/29-stocks-5-min"
OUT_DIR = "/home/cosc-admin/the-project-maverick/ml/data/fmp_historical_5min"

os.makedirs(OUT_DIR, exist_ok=True)

def merge_datasets(ticker):
    gap_path = os.path.join(GAP_DIR, f"{ticker}.csv")
    gdrive_path = os.path.join(GDRIVE_DIR, f"{ticker}.csv")
    out_path = os.path.join(OUT_DIR, f"{ticker}.csv")
    
    dfs_to_concat = []
    
    # Load 2020-2023 Gap
    if os.path.exists(gap_path):
        df_gap = pd.read_csv(gap_path)
        if not df_gap.empty and 'date' in df_gap.columns:
            df_gap['date'] = pd.to_datetime(df_gap['date'])
            dfs_to_concat.append(df_gap)
            print(f"[{ticker}] Loaded Gap 2020-2023: {len(df_gap)} rows")
    else:
        print(f"[{ticker}] WARNING: Gap file missing.")

    # Load 2023-2025 GDrive Backup
    if os.path.exists(gdrive_path):
        df_gdrive = pd.read_csv(gdrive_path)
        if not df_gdrive.empty and 'date' in df_gdrive.columns:
            df_gdrive['date'] = pd.to_datetime(df_gdrive['date'])
            dfs_to_concat.append(df_gdrive)
            print(f"[{ticker}] Loaded GDrive 2023-2025: {len(df_gdrive)} rows")
    else:
        print(f"[{ticker}] WARNING: GDrive backup file missing.")
        
    if not dfs_to_concat:
        print(f"[{ticker}] Error: No data found to merge.")
        return False
        
    # Merge, deduplicate, and sort
    df_merged = pd.concat(dfs_to_concat, ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=['date'], keep='last')
    df_merged = df_merged.sort_values(by='date', ascending=True)
    
    # Save back to the expected directory
    df_merged.to_csv(out_path, index=False)
    
    start_date = df_merged['date'].min()
    end_date = df_merged['date'].max()
    print(f"[{ticker}] -> Saved to {OUT_DIR} | Total: {len(df_merged)} rows | {start_date} to {end_date}\n")
    return True

if __name__ == "__main__":
    count = 0
    for tick in TICKERS:
        if merge_datasets(tick):
            count += 1
    print(f"Successfully merged {count} / {len(TICKERS)} tickers into {OUT_DIR}.")
