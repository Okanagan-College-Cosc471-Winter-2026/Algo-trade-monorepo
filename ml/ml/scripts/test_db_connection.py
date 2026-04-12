
import psycopg2
import sys

try:
    print("Attempting connection to 127.0.0.1...")
    conn = psycopg2.connect(
        host="127.0.0.1",
        database="market_data",
        user="mluser",
        password="mlpassword",
        port="5432"
    )
    print("Connection successful for 127.0.0.1!")
    conn.close()
except Exception as e:
    print(f"Connection failed for 127.0.0.1: {e}")

print("-" * 20)

try:
    print("Attempting connection to localhost...")
    conn = psycopg2.connect(
        host="localhost",
        database="market_data",
        user="mluser",
        password="mlpassword",
        port="5432"
    )
    print("Connection successful for localhost!")
    conn.close()
except Exception as e:
    print(f"Connection failed for localhost: {e}")
