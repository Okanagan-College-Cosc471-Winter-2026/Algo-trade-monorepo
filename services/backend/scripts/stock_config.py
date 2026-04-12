"""
Stock configuration for seeding and testing.

To add or remove stocks, simply modify the STOCKS list below.
Each stock requires: symbol, name, sector, industry, currency, exchange, is_active, and start_price.
"""

STOCKS = [
    {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "currency": "USD",
        "exchange": "NASDAQ",
        "is_active": True,
        "start_price": 145.0,
    },
    {
        "symbol": "MSFT",
        "name": "Microsoft Corp.",
        "sector": "Technology",
        "industry": "Software — Infrastructure",
        "currency": "USD",
        "exchange": "NASDAQ",
        "is_active": True,
        "start_price": 340.0,
    },
    {
        "symbol": "GOOGL",
        "name": "Alphabet Inc.",
        "sector": "Communication Services",
        "industry": "Internet Content & Information",
        "currency": "USD",
        "exchange": "NASDAQ",
        "is_active": True,
        "start_price": 130.0,
    },
    {
        "symbol": "AMZN",
        "name": "Amazon.com Inc.",
        "sector": "Consumer Cyclical",
        "industry": "Internet Retail",
        "currency": "USD",
        "exchange": "NASDAQ",
        "is_active": True,
        "start_price": 170.0,
    },
    {
        "symbol": "TSLA",
        "name": "Tesla Inc.",
        "sector": "Consumer Cyclical",
        "industry": "Auto Manufacturers",
        "currency": "USD",
        "exchange": "NASDAQ",
        "is_active": True,
        "start_price": 240.0,
    },
]
