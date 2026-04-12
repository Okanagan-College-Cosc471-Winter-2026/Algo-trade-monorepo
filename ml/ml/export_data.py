import pandas as pd
from sqlalchemy import create_engine
import os

DB_CONFIG = {
    'host': '10.12.43.135',
    'port': 5432,
    'database': 'market_data',
    'user': 'mluser',
    'password': 'mlpassword'
}

def get_db_engine():
    connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(connection_string)

def export_all_data():
    engine = get_db_engine()
    print("Connected to database. Fetching all data...")
    
    # Query all data from the table
    query = "SELECT symbol, ts as date, open, high, low, close, volume FROM stg_transform.market_data ORDER BY symbol, ts"
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    print(f"Fetched {len(df)} total rows.")
    
    # Save to a single parquet file (efficient and keeps data types)
    output_file = 'market_data_full.parquet'
    df.to_parquet(output_file, index=False)
    print(f"Data successfully exported to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    export_all_data()
