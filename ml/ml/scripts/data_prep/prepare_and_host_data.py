import os
import pandas as pd
import glob
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket
import threading

def process_and_merge_data():
    """
    Merges 2020-2023 (FMP) and 2023-2025 (Drive) data into single Parquet files per ticker.
    """
    fmp_dir = "/home/cosc-admin/the-project-maverick/ml/data/fmp_historical_5min"
    drive_dir = "/home/cosc-admin/the-project-maverick/ml/data/dm_historical"
    output_dir = "/home/cosc-admin/the-project-maverick/ml/data/processed_parquet"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all tickers from the Drive folder
    drive_files = glob.glob(os.path.join(drive_dir, "*.csv"))
    tickers = [os.path.basename(f).replace('.csv', '') for f in drive_files]
    
    print(f"Found {len(tickers)} tickers to process.")
    
    success_count = 0
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        dfs = []
        
        # 1. Load FMP Data (if exists)
        fmp_file = os.path.join(fmp_dir, f"{ticker}.csv")
        if os.path.exists(fmp_file):
            print(f"  - Loading FMP data (2020-2023)")
            try:
                df_fmp = pd.read_csv(fmp_file)
                df_fmp['source'] = 'fmp'
                dfs.append(df_fmp)
            except Exception as e:
                print(f"    Error reading FMP data: {e}")
        else:
            print(f"  - No FMP data found for {ticker} (Did the download script run?)")
            
        # 2. Load Drive Data
        drive_file = os.path.join(drive_dir, f"{ticker}.csv")
        if os.path.exists(drive_file):
            print(f"  - Loading Google Drive data (2023-2025)")
            try:
                # Some files might have 'Date' instead of 'date'
                df_drive = pd.read_csv(drive_file)
                # Standardize column names to lowercase
                df_drive.columns = [col.lower() for col in df_drive.columns]
                
                # Keep only specific columns we need in case there's extra junk
                target_cols = ['date', 'open', 'low', 'high', 'close', 'volume']
                available_cols = [c for c in target_cols if c in df_drive.columns]
                df_drive = df_drive[available_cols]
                
                df_drive['source'] = 'drive'
                dfs.append(df_drive)
            except Exception as e:
                print(f"    Error reading Drive data: {e}")
                
        # 3. Merge and Clean
        if dfs:
            # Combine all pieces
            merged_df = pd.concat(dfs, ignore_index=True)
            
            # Convert date column to actual datetime objects
            merged_df['date'] = pd.to_datetime(merged_df['date'])
            
            # Sort chronologically
            merged_df = merged_df.sort_values('date')
            
            # Remove any exact duplicate rows
            merged_df = merged_df.drop_duplicates(subset=['date'])
            
            # Save to Parquet
            out_file = os.path.join(output_dir, f"{ticker}.parquet")
            
            # PyArrow engine is usually faster and handles data types better
            merged_df.to_parquet(out_file, engine='pyarrow', index=False)
            print(f"  -> Saved {len(merged_df)} rows spanning {merged_df['date'].min().date()} to {merged_df['date'].max().date()} to {out_file}")
            success_count += 1
        else:
            print(f"  ! No data found to merge for {ticker}")
            
    print(f"\nProcessing complete! Successfully created {success_count}/{len(tickers)} Parquet files in {output_dir}")
    return output_dir

def start_http_server(directory, port=8000):
    """Starts a simple HTTP server in a specific directory."""
    class CustomHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
            
    # Try to find an open port
    max_attempts = 10
    current_port = port
    
    for attempt in range(max_attempts):
        try:
            httpd = HTTPServer(("", current_port), CustomHandler)
            print(f"\n" + "="*50)
            print(f"✅ Data HTTP Server is Running!")
            print(f"Directory being served: {directory}")
            print(f"\nTo download a file on DRAC, run this command on DRAC:")
            
            # Get local IP for better UX
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            
            print(f"  wget http://<YOUR_MACHINES_IP>:{current_port}/AAPL.parquet")
            print(f"  (Ensure DRAC can reach this machine's IP, or use an ssh tunnel/ngrok)")
            print("="*50 + "\n")
            
            httpd.serve_forever()
            return
        except OSError as e:
            if e.errno == 98: # Address already in use
                print(f"Port {current_port} is busy, trying {current_port + 1}...")
                current_port += 1
            else:
                raise e
    print("Could not find an open port.")

if __name__ == "__main__":
    print("Step 1: Processing and Merging CSVs to Parquet...")
    
    # Make sure we have the required parqet library
    try:
        import pyarrow
    except ImportError:
        print("Installing pyarrow for fast Parquet saving...")
        os.system("pip install pyarrow")
        
    serve_dir = process_and_merge_data()
    
    print("\nStep 2: Starting HTTP Data Server...")
    start_http_server(serve_dir)

