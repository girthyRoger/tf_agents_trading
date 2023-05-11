import yfinance as yf
import random
import pandas as pd

def get_random_ticker_history(eval=False):
    
    if eval:
        ticker_codes = (
            "DIS",
            "ACN",
            "NKE",
            "TXN"
        )
    else:    
        ticker_codes = (
            "MSFT",
            "GOOG",
            "AMZN",
            "V",
            "NVDA",
            "META",
            "UPS",
            "AMD",
            "AAPL",
            "BRK-B",
            "JNJ",
            "TSLA",
            "XOM",
            "UNH",
            "JPM",
            "WMT",
            "PG",
            "MA",
            "NSRGY",
            "HD",
            "BAC-PK",
            "KO",
            "PEP",
            "ORCL",
            "LRLCY",
            "AZN",
            "BABA",
            "MCD",
            "BHP",
            "PFE"
        )
    
    ticker_index_random = random.randint(0, len(ticker_codes)-1)
    random_ticker = ticker_codes[ticker_index_random]
    ticker = yf.Ticker(random_ticker)
    print(f"Picked ticker {random_ticker}")
    result = ticker.history(period="5y", interval="1d")
    result.drop(["Dividends", "Stock Splits"], axis=1, inplace=True)
    result.reset_index(drop=False, inplace=True)
    result.loc[:,"Date"] = result.loc[:,"Date"].apply(lambda x: x.timestamp())
    
    return result


